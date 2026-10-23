"""Shard a graph group's inputs across GPU workers and merge their outputs."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.execution.ids import document_ids, id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows, rows_of
from quail.physical import ValueType
from quail.planner import balanced_shards


def keyed_shards(groups, document_tokens, gpus: int) -> tuple:
    """Shard documents by stop key group, balanced by token count.

    Every document of a group lands on one worker, so the worker's
    per-key stop sees the whole group.
    """
    tokens = {}
    for group, length in zip(groups, document_tokens):
        tokens[group] = tokens.get(group, 0) + length
    ordered = list(tokens)
    shards, _ = balanced_shards([tokens[group] for group in ordered], gpus)
    placed = {ordered[index]: worker
              for worker, shard in enumerate(shards) for index in shard}
    out = [[] for _ in range(gpus)]
    for document, group in enumerate(groups):
        out[placed[group]].append(document)
    return tuple(tuple(shard) for shard in out)


def shard_input(port, value, alias, ids):
    """Return the part of one input a worker's documents of alias need.

    Rows and pairs of the alias keep the worker's documents only;
    every other input, such as a partner table, goes whole.
    """
    ids = document_ids(ids)
    if isinstance(value, Rows) and alias in value.ids:
        return value.filter(pc.is_in(value.ids[alias], value_set=ids))
    if port.value_type is ValueType.ROWS and port.source.port == f"rows:{alias}":
        values = document_ids(value)
        return pc.filter(values, pc.is_in(values, value_set=ids))
    if port.value_type is ValueType.PAIRS and alias in value.column_names:
        return value.filter(pc.is_in(value[alias], value_set=ids))
    return value


def merge_results(nodes, outputs, alias, ids, wall) -> dict:
    """Merge the workers' results of a group, each output by its port type.

    Rows come back in the order of the group's input documents, so a
    run on several GPUs returns what a run on one would.

    Args:
        nodes: The group's physical nodes.
        outputs: One worker output per GPU, each with the group's results.
        alias: The alias the group's documents belong to.
        ids: The group's input document ids, in their input order.
        wall: The round's wall seconds, charged to the first node.
    """
    position = {int(document): index for index, document in enumerate(ids)}

    def order(documents):
        return np.argsort([position[int(document)] for document in
                           id_positions(documents)], kind="stable")

    def merge(port, values):
        if port.value_type is ValueType.ROWS:
            joined = Rows.concat([rows_of(value, alias) for value in values])
            return joined.take(order(joined.ids[alias]))
        if port.value_type is ValueType.PAIRS:
            joined = pa.concat_tables(values)
            return joined.take(order(joined[alias]))
        if port.value_type is ValueType.FILTER_ANSWERS:
            joined = pa.concat_tables(values)
            return joined.take(order(joined[alias]))
        if port.value_type is ValueType.JOIN_ANSWERS:
            return merge_join_answers(values)
        raise TypeError(f"cannot merge worker output {port.value_type.value!r}")

    results = {}
    for index, node in enumerate(nodes):
        parts = [output["results"][node.node_id] for output in outputs]
        metrics = sum((part.metrics for part in parts), NodeMetrics())
        results[node.node_id] = NodeResult(
            {port.name: merge(port, [part.outputs[port.name] for part in parts])
             for port in node.outputs},
            replace(metrics, wall_s=wall if index == 0 else 0.0))
    return results


def merge_join_answers(parts: list) -> dict:
    """Merge answer rows while translating each worker's local anchor indices."""
    rows, anchor_index = {}, []
    members = None
    for part in parts:
        base = len(anchor_index)
        anchor_index.extend(part["anchor_index"])
        rows.update((base + int(local), row) for local, row in part["rows"].items())
        if part.get("anchor_partners") is not None:
            if members is None:
                members = {}
            members.update((base + int(local), streamed)
                           for local, streamed in part["anchor_partners"].items())
    return dict(parts[0], rows=rows, anchor_index=anchor_index,
                anchor_partners=members)
