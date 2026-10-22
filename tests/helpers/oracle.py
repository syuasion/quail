"""One fake model for the end-to-end tests: it answers from planted truth.

The oracle stands in for the GPU. It answers every model operator
from the truth it was given and builds each result with the
production builders (filter_result, score_rows, extraction_rows, and
the join answer record the executor writes), so a test never spells
a port name or a result shape.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from quail.backends.base import GpuContext
from quail.backends.quail.executor.operators.filter import filter_result
from quail.backends.quail.reranker import (
    RerankerBatch,
    RerankerModelExecution,
    score_rows,
)
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows, extraction_rows
from quail.execution.spans import locate_span
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiScore,
    ClassifySpec,
)
from quail.specs import H100_SXM, QWEN3_4B_FP8


class Oracle:
    """A model execution whose answers come from planted truth.

    Args:
        filters: alias -> {question key: a bit per document}, where the
            key is matched against the start of the question's tokens;
            or a callable (alias, document, written_position) -> bool.
        joins: {(anchor, *partners): rule(anchor_id, *partner_ids)}; or
            a callable (node, assignment) -> bool over alias -> id.
        labels: A callable (spec, ids) -> the label of one row or None,
            a sequence indexed by the first id, or None.
        scores: A callable (spec, ids) -> the score of one row.
        answers: A callable (spec, document) -> the extracted text or
            None; or a sequence indexed by document for every question.
        probabilities: Whether a classification reports label
            probabilities when its spec asks for them.
        retain: How many of a keep_kv filter's survivors the oracle
            retains in the arena, as the executor would.
        join_metrics: Metrics every join stage reports beside its rows.
        streams: Whether a score call reports each row as it is scored.
    """

    # the oracle runs one node at a time; a pipeline falls back to that
    execute_pipeline = None

    def __init__(self, filters=None, joins=None, labels=None, scores=None,
                 answers=None, *, probabilities=False, retain=0,
                 join_metrics=None, streams=False):
        self.filters = filters or {}
        self.joins = joins or {}
        self.labels = labels
        self.scores = scores or (lambda spec, ids: (sum(ids) + 1) / 10)
        self.answers = answers
        self.probabilities = probabilities
        self.retain = retain
        self.join_metrics = join_metrics
        self.streams_answers = streams
        self.arena = None
        self.texts = {}
        self.documents = {}
        self.calls = []       # (spec name, rows) of every score or label call

    # ---- the executor's model execution interface ----------------

    def execute(self, node, inputs) -> NodeResult:
        if isinstance(node, AiFilter):
            return self._filter(node, inputs)
        if isinstance(node, AiJoin):
            return self.execute_predicates((node,), inputs)[node.node_id]
        if isinstance(node, AiExtract):
            return self._extract(node, inputs)
        if isinstance(node, AiClassify) and node.spec.partner is not None:
            return self._classify_joined(node, inputs)
        if isinstance(node, (AiScore, AiClassify)):
            return self._reranker().execute(node, inputs.get("score_inputs", inputs))
        raise TypeError(f"the oracle has no answers for {node.type_name}")

    def execute_predicates(self, nodes, inputs) -> dict:
        """Answer a group of join stages, each over the survivors of the last."""
        anchor_ids = list(inputs["anchor_ids"])
        lists_for = inputs.get("anchor_partners")
        survivors = set(range(len(anchor_ids)))
        results = {}
        for index, node in enumerate(nodes):
            partners = inputs["partner_indices"][node.written_position]
            members = {} if lists_for and node.pairs_from else None
            rows = {}
            for local in sorted(survivors):
                mine = (range(len(partners)) if members is None
                        else list(lists_for(anchor_ids[local], index)))
                if members is not None:
                    members[local] = mine
                rows[local] = [
                    self._join_answer(node, anchor_ids[local], partners[member])
                    for member in mine]
            if inputs.get("anchor_done") is not None:
                for local, row in rows.items():
                    inputs["anchor_done"](local, row)
            passing = {local for local, row in rows.items() if any(row)}
            survivors = (survivors - passing if node.semantics == "anti"
                         else passing)
            kept = [anchor_ids[local] for local in sorted(survivors)]
            answer = {
                "rows": rows, "anchor_index": anchor_ids,
                "partner_index": partners, "anchor_partners": members,
                "anchor": node.anchor, "partners": list(node.partners),
                "semantics": node.semantics, "selectivity": node.selectivity,
                "written_position": node.written_position,
            }
            metrics = self.join_metrics or NodeMetrics()
            results[node.node_id] = NodeResult(
                {f"rows:{node.anchor}": Rows({node.anchor: kept}),
                 f"join_answers:{node.written_position}": answer},
                NodeMetrics(
                    input_rows=len(anchor_ids), output_rows=len(kept),
                    evaluated_document_pairs=sum(len(row) for row in rows.values()),
                    gpu_s=metrics.gpu_s, peak_gpu_bytes=metrics.peak_gpu_bytes,
                    extension={**metrics.extension, "answers": [rows]}))
        return results

    # ---- the reranker interface, for scores and labels -------------

    def score(self, spec, rows, documents, on_answers=None) -> RerankerBatch:
        rows = np.asarray(rows).reshape(-1, len(spec.aliases))
        self.calls.append((spec.name, rows.tolist()))
        ids = [tuple(int(value) for value in row) for row in rows]
        if isinstance(spec, ClassifySpec):
            values = np.asarray([self._label(spec, row) for row in ids], dtype=object)
            probabilities = None
            if self.probabilities and spec.probabilities:
                probabilities = np.asarray([
                    [np.nan] * len(spec.labels) if value is None
                    else [0.8 if label == value else 0.1 for label in spec.labels]
                    for value in values])
            return RerankerBatch(values, fresh_tokens=len(rows), cached_tokens=0,
                                 suffix_tokens=3 * len(rows),
                                 probabilities=probabilities)
        scores = [self.scores(spec, row) for row in ids]
        if on_answers is not None:
            for position, value in enumerate(scores):
                on_answers(np.array([position]), [value])
        return RerankerBatch(scores, fresh_tokens=len(rows), cached_tokens=0)

    # ---- answering ------------------------------------------------

    def _filter(self, node, inputs) -> NodeResult:
        document_ids = list(inputs["document_ids"])
        answers = {local: [self._filter_answer(node, document)]
                   for local, document in enumerate(document_ids)}
        if inputs.get("document_done") is not None:
            # the real loop reports a chunk's finished documents by position
            inputs["document_done"]([
                (local, 0, bool(row[0])) for local, row in answers.items()])
        if node.keep_kv and self.arena is not None:
            kept = [document for local, document in enumerate(document_ids)
                    if answers[local][0]]
            for document in kept[:self.retain]:
                self.arena.retain((node.alias, document), 16)
        return filter_result(node, answers, 10 * len(answers), document_ids)

    def _filter_answer(self, node, document) -> bool:
        if callable(self.filters):
            return bool(self.filters(node.alias, document, node.written_position))
        truth = self.filters[node.alias]
        question = node.question_token_ids
        for key, bits in truth.items():
            if any(token.startswith(key) for token in question):
                return bool(bits[document])
        raise KeyError(f"no filter truth for {node.alias!r} matches {question[:5]}")

    def _join_answer(self, node, anchor_id, partner_ids) -> bool:
        if callable(self.joins):
            assignment = dict(zip((node.anchor, *node.partners),
                                  (anchor_id, *partner_ids)))
            return bool(self.joins(node, assignment))
        rule = self.joins[(node.anchor, *node.partners)]
        return bool(rule(anchor_id, *partner_ids))

    def _label(self, spec, ids):
        if callable(self.labels):
            return self.labels(spec, ids)
        return None if self.labels is None else self.labels[ids[0]]

    def _classify_joined(self, node, inputs) -> NodeResult:
        """Label the pairs a join kept; a pair without a label has no row."""
        (answers,) = inputs["score_inputs"].values()
        pairs = []
        for local, row in answers["rows"].items():
            anchor = answers["anchor_index"][local]
            members = (answers["anchor_partners"] or {}).get(local)
            for position, bit in enumerate(row):
                if not bit:
                    continue
                member = position if members is None else members[position]
                (partner,) = answers["partner_index"][member]
                pairs.append((int(anchor), int(partner)))
        spec = node.spec
        labels = {pair: self._label(spec, pair) for pair in pairs}
        labeled = [pair for pair in pairs if labels[pair] is not None]
        rows = np.asarray(labeled, dtype=np.int32).reshape(-1, 2)
        return NodeResult(
            {node.rows_port: score_rows(rows, spec.aliases, spec.name,
                                        [labels[pair] for pair in labeled],
                                        pa.string())},
            NodeMetrics(input_rows=len(pairs), output_rows=len(labeled)))

    def _extract(self, node, inputs) -> NodeResult:
        spec = node.spec
        ids = [int(document) for document in inputs["document_ids"]]
        texts = self.texts.get(spec.alias)
        found = [self._answer(spec, document) for document in ids]
        spans = [None if text is None or texts is None
                 else locate_span(str(texts[document]), text)
                 for document, text in zip(ids, found)]
        return NodeResult(
            {node.rows_port: extraction_rows(
                spec.alias, spec.name, spec.span_name, ids, found, spans)},
            NodeMetrics(input_rows=len(ids), output_rows=len(ids),
                        evaluated_documents=len(ids)))

    def _answer(self, spec, document):
        if callable(self.answers):
            return self.answers(spec, document)
        return None if self.answers is None else self.answers[document]

    def _reranker(self) -> RerankerModelExecution:
        return RerankerModelExecution(GpuContext(
            gpu_index=0, gpus=1, model=QWEN3_4B_FP8, device=H100_SXM,
            query_settings={"reranker": self, "documents": self.documents}))


def oracle_executor(oracle: Oracle, functions=None, **report):
    """Return a physical executor that runs each request's graph on the oracle.

    Args:
        oracle: The oracle answering the model operators.
        functions: The apply() functions by name, when the query has any.
        **report: Report fields the response carries, such as wall_s.
    """
    def execute(request):
        from helpers.fakes import graph_state
        from quail.backends.quail.graph import execute_single_graph
        from quail.builtins import built_in_registry
        from quail.execution.types import PhysicalResponse
        from quail.physical import Scan, decode_graph

        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        scans = [node for node in graph.nodes if isinstance(node, Scan)]
        oracle.documents = {node.alias: request.inputs[node.input_id].documents
                            for node in scans}
        oracle.texts = {node.alias: request.inputs[node.input_id].texts
                        for node in scans}
        state = graph_state(oracle, oracle.documents)
        state["texts"] = oracle.texts
        state["functions"] = functions or {}
        oracle.arena = state["gpu"].arena
        if getattr(request, "relations", None):
            state["columns"] = request.column_tables()
        result = execute_single_graph(state, {"filter_limit": None}, graph)
        outputs = result.pop("_outputs")
        return PhysicalResponse(outputs, {
            **result, "wall_s": 1.0, "boot_s": 0.5, "fresh_tokens": 1234, **report})

    return execute


def finish(query, oracle: Oracle, **report):
    """Run a query to its result on the oracle."""
    from quail.api.execute import execute_query

    return execute_query(query, physical_executor=oracle_executor(oracle, **report))
