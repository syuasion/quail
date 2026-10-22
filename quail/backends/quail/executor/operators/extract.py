"""Run a pipeline's extractions and build their answer tables."""

import time
from dataclasses import replace

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.extract import (
    ExtractionStages,
    load_vocabulary,
)
from quail.backends.quail.executor.parts import (
    ExecutionPart,
    execute_parts,
    gpu_seconds,
    shared_prefix_tree,
    stage_spans,
)
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows, extraction_rows
from quail.physical import AiExtract


def execute_extractions(state: QueryExecutionState, nodes, inputs) -> dict:
    """Run a pipeline's extractions over its documents as one stage run.

    Each document is read once and asked every node's question while
    its KV is resident. The first node is charged the documents' tokens
    and the wall time. Each question is charged its own suffix tokens
    and GPU time.

    Args:
        state: The loaded model and current query state.
        nodes: The AiExtract nodes, in pipeline order.
        inputs: The documents' ids, the tokenized documents and texts
            by alias, the query's shared preamble, and the gpu_timing
            setting.

    Returns:
        Node id -> NodeResult with the node's answer table.

    Raises:
        TypeError: A node is not an extraction.
        ValueError: The table's text was not loaded.
    """
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    started = time.perf_counter()
    part = ExtractPart(state, nodes, inputs["document_ids"], inputs)
    stages = part.extraction
    tree = (shared_prefix_tree(nodes[0].spec.name, stages.prefixes,
                               state.loaded_model.arena)
            if nodes[0].spec.share_prefixes else None)
    results = execute_parts(
        state, [part], stages.prefixes, stages.keys,
        keep_kv=nodes[-1].keep_kv, prefix_tree=tree, limit=inputs.get("limit"),
        label=f"extract {', '.join(node.spec.name for node in nodes)}")
    first = nodes[0].node_id
    results[first] = replace(results[first], metrics=replace(
        results[first].metrics, wall_s=time.perf_counter() - started))
    return results


def extraction_results(nodes, stages, document_ids, *, fresh, own,
                       stats, counts, kept=None) -> dict:
    """Build each extraction node's answer table and metrics.

    Args:
        nodes: The AiExtract nodes, in question order.
        stages: The finished ExtractionStages.
        document_ids: The documents' ids.
        fresh: The run's fresh tokens; the first node is charged them
            less the later questions' suffix tokens.
        own: Question index -> (its GPU seconds, its timed chunks).
        stats: The run's packing time and borrowed tokens.
        counts: Per-question counts collected when answers finished.
        kept: The indices of the documents the answer tables hold;
            None holds every document.
    """
    later = sum(count["suffix_tokens"] for count in counts[1:])
    if kept is not None:
        document_ids = document_ids[kept]
    results = {}
    for q, node in enumerate(nodes):
        spec = node.spec
        answers, answer_spans = stages.answers(q, kept)
        answers = extraction_rows(spec.alias, spec.name, spec.span_name,
                                document_ids, answers, answer_spans)
        first = q == 0
        gpu_s, timed = own(q)
        results[node.node_id] = NodeResult(
            {node.rows_port: answers},
            NodeMetrics(
                input_rows=len(document_ids), output_rows=len(document_ids),
                evaluated_documents=len(document_ids),
                fresh_tokens=(fresh - later if first
                              else counts[q]["suffix_tokens"]),
                gpu_s=gpu_s, chunks=timed,
                extension={"output": spec.name, "aliases": list(spec.aliases),
                           "input_rows": len(document_ids), **counts[q],
                           "borrowed_prefix_tokens": (
                               stats.get("borrowed_tokens", 0) if first else 0),
                           "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                      if first else 0.0)}))
    return results


class ExtractPart(ExecutionPart):
    """Extraction stages and per-question results for the shared executor."""

    combine = True
    execute_group = staticmethod(execute_extractions)

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        if nodes is not None:
            inputs = inputs[node.node_id]
        # the one input carries the documents' ids, or the rows of an
        # earlier operator over them
        (value,) = inputs.values()
        ids = (value.ids[node.spec.alias].to_numpy() if isinstance(value, Rows)
               else np.asarray(value, dtype=np.int64))
        return {"document_ids": ids,
                "documents": context.state["documents"],
                "texts": context.state.get("texts", {}),
                "limit": context.state.get("filter_limit")}

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare consecutive questions over the preceding operator's documents.

        Raises:
            ValueError: A document uses a different prompt or token sequence.
        """
        first = parts[0].node if parts else nodes[0]
        part = cls(state, nodes, ids, {
            "documents": context.state["documents"],
            "texts": context.state.get("texts", {}),
            "pipeline_head": first.document_head})
        alias = part.node.spec.alias
        for document, document_id in zip(part.extraction.documents, part.ids):
            if document.key != (alias, document_id):
                raise ValueError(
                    f"AI.EXTRACT {part.node.spec.name!r} shows document "
                    f"{document_id} of {alias!r} unlike the pipeline before it")
        return part

    def __init__(self, state, nodes, ids, inputs):
        self.nodes = list(nodes)
        self.node = self.nodes[0]
        self.ids = np.asarray(ids, dtype=np.int64)
        if any(not isinstance(node, AiExtract) for node in nodes):
            raise TypeError("an extraction pipeline holds AiExtract nodes only")
        alias = self.node.spec.alias
        texts = inputs.get("texts", {}).get(alias)
        if texts is None:
            raise ValueError(f"AI.EXTRACT needs the text of {alias!r}")
        self.extraction = ExtractionStages(
            state, [node.spec for node in nodes], load_vocabulary(state), self.ids,
            inputs["documents"][alias], texts,
            pipeline_head=inputs.get("pipeline_head", self.node.document_head),
            document_positions=inputs.get("document_positions"))
        self.stages = self.extraction.stages
        self._results = {}

    @property
    def appended_tokens(self) -> int:
        return self.extraction.appended_tokens

    @property
    def tested_columns(self) -> tuple[str, ...]:
        """The columns a Filter in the pipeline may test."""
        return tuple(name for node in self.nodes
                     for name in (node.spec.name, node.spec.span_name))

    def tested_values(self, column: str, positions) -> pa.Array:
        """Return the answers or spans settled so far at the positions.

        Args:
            column: An extraction's answer or span column.
            positions: Positions among the part's documents.
        """
        q, span = next(
            (q, column == node.spec.span_name) for q, node in enumerate(self.nodes)
            if column in (node.spec.name, node.spec.span_name))
        answers = [
            self.extraction.documents[position].answers[q] for position in positions
        ]
        if not span:
            return pa.array([answer.answer for answer in answers], pa.string())
        return pa.array(
            [None if answer.span is None
             else {"start": answer.span[0], "end": answer.span[1]}
             for answer in answers],
            pa.struct([("start", pa.int32()), ("end", pa.int32())]))

    def finish(self, every) -> int:
        """Settle every answer; return the tokens the questions fed."""
        self.extraction.finish()
        self.counts = [self.extraction.counts(q) for q in range(len(self.nodes))]
        return sum(count["suffix_tokens"] for count in self.counts)

    def complete(self, fresh, spans, stats, offset):
        """Build each question's answer table and attribute its GPU work."""
        per_question = len(self.stages) // len(self.nodes)

        def own(q):
            start = offset + q * per_question
            selected = stage_spans(spans, start, start + per_question)
            return gpu_seconds(selected), len(selected)

        self._results = extraction_results(
            self.nodes, self.extraction, self.ids, fresh=fresh, counts=self.counts,
            own=own, stats=stats, kept=self.extraction.asked())
        self.result_value = self._results[self.node.node_id]

    def results(self) -> dict:
        """Return the answer table and metrics for each extraction."""
        return self._results
