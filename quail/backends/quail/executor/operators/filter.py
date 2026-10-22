"""Execute LLM filters and conditions on classification results."""

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.parts import (
    SingleResultPart,
    execute_parts,
    execution_prefix_tree,
)
from quail.backends.quail.executor.stages import Stage, filter_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.backends.quail.reranker import filter_scores
from quail.catalog.selection import group_ids
from quail.catalog.tokens import (
    DocumentKeys,
    DocumentPrefixes,
)
from quail.execution.ids import document_ids as arrow_ids
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows
from quail.progress import answer_sink


def execute_filters(state: QueryExecutionState, nodes, inputs) -> dict:
    """Run filter predicates together while each document retains its KV."""
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    node = nodes[0]
    documents, ids = inputs["documents"], inputs["document_ids"]
    parts = [FilterPart(member, ids, state.async_answers,
                        len(state.loaded_model.pipeline.canvas_ids))
             for member in nodes]
    if len(parts) == 1 and "document_done" in inputs:
        parts[0].document_done = inputs["document_done"]
    # the rows a limit keeps follow admission order, which borrowing
    # sorts by prefix; a query without ORDER BY promises no rows
    tree = execution_prefix_tree(node, documents, state.loaded_model.arena)
    return execute_parts(
        state, parts, documents, DocumentKeys(node.alias, ids),
        keep_kv=nodes[-1].keep_kv, prefix_tree=tree,
        attention=node.attention or "unified",
        limit=inputs.get("limit"), stop_groups=inputs.get("stop_groups"),
        paged=node.arena_writes or len(nodes) > 1,
        label=f"filter ({len(nodes)} predicates)")


class FilterPart(SingleResultPart):
    """Filter stages and result tables within a pipeline."""

    execute_group = staticmethod(execute_filters)

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        if nodes is not None:
            inputs = inputs[node.node_id]
        return filter_inputs(context.state, node, next(iter(inputs.values())))

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare one filter's requests and answer notifications."""
        return cls(nodes[0], ids, state.async_answers,
                   len(state.loaded_model.pipeline.canvas_ids))

    def __init__(self, node, ids, async_answers, canvas_tokens=0):
        self.node = node
        self.ids = ids
        self.canvas_tokens = canvas_tokens
        self.stages = filter_stages(
            [list(node.question_token_ids)],
            async_answers)
        self.document_done = filter_document_sink(node, ids)
        self.answers = {}

    @staticmethod
    def prepare_group(parts):
        """Retain the common question preamble across consecutive filters."""
        stages = filter_stages(
            [part.node.question_token_ids for part in parts],
            parts[0].stages[0].readout)
        for part, stage in zip(parts, stages):
            part.stages = [stage]

    def finish(self, every) -> int | None:
        """Record filter answers and leave token accounting to the pipeline."""
        for stage in every:
            for document, row in stage.items():
                self.answers.setdefault(document, []).append(int(row[0]))
        return len(self.answers) * (len(self.stages[0].suffixes[0])
                                    + self.frame_tokens
                                    + self.canvas_tokens)

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_result(
            self.node, self.answers, tokens, self.ids, gpu_s=gpu_s,
            chunks=chunks, borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0))


class ConditionGate(SingleResultPart):
    """Pipeline condition that drops documents whose tested value fails.

    The value is a label a ClassifyPart decoded or an answer or span
    an ExtractPart copied, read as each document's settles; the
    pipeline then skips the document's later stages and frees its KV.
    """

    stages = ()

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Bind a condition to the part computing its column."""
        node = nodes[0]
        source_id = node.inputs[0].source.node_id
        producer = next(part for part in reversed(parts)
                        if node.column in part.tested_columns)
        source = next(part for part in reversed(parts)
                      if any(member.node_id == source_id for member in part.nodes))
        return cls(node, producer, source)

    def __init__(self, node, producer, source):
        self.node = node
        self.producer = producer
        self.source = source  # the part whose rows the Filter reads
        self.position = {document: index
                         for index, document in enumerate(producer.ids)}
        self.accepted = {}

    def gate(self, key):
        # values settle as chunks finish, so each is tested as its
        # document arrives; a repeated value, such as a label, is
        # evaluated once
        values = self.producer.tested_values(self.node.column,
                                             [self.position[key[1]]])
        value = values[0].as_py()
        token = tuple(sorted(value.items())) if isinstance(value, dict) else value
        if token not in self.accepted:
            self.accepted[token] = bool(self.node.predicate.accepts(
                pa.table({self.node.column: values}))[0].as_py())
        return None if self.accepted[token] else Stage.DROP

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        source = self.node.inputs[0].source
        return filter_scores(self.node,
                             self.source.results()[source.node_id].outputs[source.port])


def filter_result(node, answers, tokens, document_ids,
                  gpu_s: float = 0.0, chunks: int = 0,
                  borrowed_tokens: int = 0, pack_s: float = 0.0) -> NodeResult:
    """Build a filter result using global document IDs and execution metrics.

    Args:
        node: Physical filter node.
        answers: Mapping from local document index to Boolean stage answers.
        tokens: Fresh tokens processed by the model.
        document_ids: Global document IDs in local index order.
        gpu_s: Forward-pass GPU seconds.
        chunks: Number of timed forward passes.
        borrowed_tokens: Prefix tokens read from another document's KV.
        pack_s: Host seconds spent building forward chunks.

    Returns:
        A NodeResult containing surviving IDs, filter answers, and metrics.
    """
    local = np.fromiter(answers, dtype=np.int64, count=len(answers))
    passed = np.fromiter((len(row) == 1 and all(row) for row in answers.values()),
                         dtype=bool, count=len(answers))
    positions = np.asarray(document_ids, dtype=np.int64)
    survivors = arrow_ids(np.sort(positions[local[passed]]))
    global_answers = dict(zip(positions[local].tolist(), answers.values()))
    return NodeResult(
        outputs={
            f"rows:{node.alias}": Rows({node.alias: survivors}),
            f"filter_answers:{node.alias}": global_answers,
        },
        metrics=NodeMetrics(
            input_rows=len(document_ids),
            output_rows=len(survivors),
            evaluated_documents=len(global_answers),
            fresh_tokens=tokens,
            gpu_s=gpu_s,
            chunks=chunks,
            extension={
                **({"borrowed_prefix_tokens": borrowed_tokens}
                   if borrowed_tokens else {}),
                **({"pack_s": round(pack_s, 3)} if pack_s else {}),
            },
        ),
    )


def filter_document_sink(node, document_ids):
    """The sequence's document_done callback, or None when nobody listens.

    Reports each chunk's finished documents to the answer sink as
    ``{"kind": "filter", "node", "alias", "stages", "documents"}`` where
    ``documents`` lists ``[row index, last stage asked, passed]`` per
    document. A document that passed its last stage survived the filter;
    every other listed document failed at the stage given.
    """
    sink = answer_sink()
    if sink is None:
        return None
    stages = 1

    def document_done(finished):
        sink({"kind": "filter", "node": node.node_id, "alias": node.alias,
              "stages": stages,
              "documents": [[int(document_ids[position]), stage, passed]
                            for position, stage, passed in finished]})

    return document_done


def stop_groups(node, columns, document_ids) -> list | None:
    """Return each document's stop key group, or None without a stop key.

    Documents with equal values in the node's ``stop_key`` columns
    share a group number; the scheduler reads one document of a group
    at a time and skips the rest once one survives.

    Args:
        node: The AiFilter.
        columns: The alias's value table, one row per document.
        document_ids: The documents' positions in that table.
    """
    if not node.stop_key:
        return None
    if columns is None:
        raise KeyError(
            f"the stop key of {node.alias!r} needs its value columns")
    positions = arrow_ids(document_ids)
    return group_ids([columns.column(name).take(positions) for name in node.stop_key])


def filter_inputs(state, node, document_ids) -> dict:
    """Scheduler inputs for one filter sequence over the given documents."""
    document_ids = id_positions(document_ids)
    return {
        "documents": DocumentPrefixes(
            node.document_head, state["documents"][node.alias], document_ids
        ),
        "document_ids": document_ids,
        "limit": state["filter_limit"],
        "stop_groups": stop_groups(
            node, state.get("columns", {}).get(node.alias), document_ids),
        "document_done": filter_document_sink(node, document_ids),
    }
