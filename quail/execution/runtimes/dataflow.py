"""Runtimes for the dataflow nodes: exchange, barrier, foreign, recombine, and join."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.catalog.pairs import (
    columns_key,
    compared_pairs,
    pair_ids_table,
    pair_table,
)
from quail.execution.ids import document_ids, id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import (
    IndexRelation,
    build_result_declaration,
    joined_survivors,
    true_answer_rows,
)
from quail.execution.rows import Rows, as_table
from quail.logical.expressions import (
    column_name,
    column_references,
    evaluate_expression,
)
from quail.physical import (
    Barrier,
    Exchange,
    Foreign,
    GraphValidationError,
    HashJoin,
    Recombine,
    ValueType,
)


class ExchangeRuntime:
    """Pass anchor ids through; several GPUs route them to their KV."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Exchange):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Exchange takes one survivor input")
        return NodeResult({f"rows:{node.anchor}": next(iter(inputs.values()))})


class BarrierRuntime:
    """Prune survivor IDs using completed join answers."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Barrier):
            raise TypeError(type(node).__name__)
        from quail.execution.types import _join_answers_table

        survivors = {}
        tables = []
        for port in node.inputs:
            value = inputs[port.name]
            if port.value_type is ValueType.ROWS:
                alias = port.source.port.split(":", 1)[1]
                survivors[alias] = document_ids(value)
            elif port.value_type is ValueType.JOIN_ANSWERS:
                tables.append(value if isinstance(value, pa.Table)
                              else _join_answers_table(value))
        survivors = joined_survivors(
            [true_answer_rows(table) for table in tables], survivors)
        return NodeResult({f"rows:{alias}": Rows({alias: survivors[alias]})
                           for alias in node.aliases})


def _ids_of(value) -> np.ndarray:
    """Ids an apply() returned, from a sequence, an array, or a one-column table."""
    try:
        return id_positions(value)
    except ValueError as error:
        raise ValueError(
            "an apply() function returned a null id; an outer join "
            "leaves nulls, use join_type='inner'") from error


def _pairs_of(value, left: str, right: str) -> np.ndarray:
    """(left, right) id pairs, one row each, from a table with both alias columns."""
    if not isinstance(value, pa.Table) or not {left, right} <= set(
            value.column_names):
        raise ValueError(
            f"an apply() returning pairs must return an Arrow table with "
            f"columns {left!r} and {right!r}")
    return np.column_stack(
        [_ids_of(value.column(left)), _ids_of(value.column(right))])


def alias_table(alias: str, ids, columns, expressions):
    """The Arrow table an apply() function sees for one alias.

    Its first column is the alias's row indices under the alias
    name; the rest are the requested columns for those rows, each
    named as the query wrote it, such as ``length(body)``.
    """
    ids = document_ids(ids)
    arrays = {alias: ids}
    for expression in expressions:
        arrays[column_name(expression, qualified=False)] = relation_values(
            columns, expression).take(ids)
    return pa.table(arrays)


def relation_values(columns, expression):
    """Return one key or apply() column's values from the alias's value table."""
    for ref in column_references(expression):
        if columns is None or ref.column not in columns.column_names:
            raise KeyError(f"column {ref.alias}.{ref.column} is not among the "
                           f"request's relations")
    return evaluate_expression(expression, columns)


def foreign_call(node, values, context):
    """Bind a user function to its input columns and collect call metrics.

    Args:
        node: The Foreign node.
        values: alias -> the ids, or id table, the node was given.
        context: The execution context holding the functions and
            column sources.

    Returns:
        (call, metrics): call(ids by alias) -> kept ids or pairs, as the
        node's ``ids`` says, checked against what it was given;
        metrics() -> the NodeMetrics of every call so far.
    """
    if not isinstance(node, Foreign):
        raise TypeError(type(node).__name__)
    function = context.functions.get(node.function)
    if function is None:
        raise KeyError(
            f"apply() function {node.function!r} is not registered on "
            f"this session")
    missing = [alias for alias in node.aliases if alias not in values]
    if missing:
        raise GraphValidationError(
            f"{node.node_id!r} has no id input for {missing}")
    columns = {alias: context.sources.get(columns_key(alias))
               for alias in node.aliases}
    expressions = {alias: [column for column in node.columns
                           if column.alias == alias] for alias in node.aliases}
    counters = dict(calls=0, input_rows=0, output_rows=0)

    def call(ids_by_alias):
        tables = {alias: alias_table(alias, ids, columns[alias],
                                     expressions[alias])
                  for alias, ids in ids_by_alias.items()}
        counters["calls"] += 1
        counters["input_rows"] += sum(
            len(ids) for ids in ids_by_alias.values())
        result = function(tables)
        if node.ids == "pairs":
            left, right = node.aliases
            pairs = _pairs_of(result, left, right)
            outside = ~(np.isin(pairs[:, 0], id_positions(ids_by_alias[left]))
                        & np.isin(pairs[:, 1], id_positions(ids_by_alias[right])))
            if outside.any():
                a, b = pairs[np.flatnonzero(outside)[0]]
                raise ValueError(
                    f"apply() {node.function!r} returned pair "
                    f"({a}, {b}) outside its input ids")
            counters["output_rows"] += len(pairs)
            return pairs
        (alias,) = node.aliases
        kept = _ids_of(result)
        given = np.unique(id_positions(ids_by_alias[alias]))
        if not np.isin(kept, given).all():
            raise ValueError(
                f"apply() {node.function!r} returned ids it was not "
                f"given; a function never invents an id")
        distinct = len(np.unique(kept))
        if node.ids == "preserve" and distinct != len(given):
            raise ValueError(
                f"apply() {node.function!r} preserves ids but dropped "
                f"{len(given) - distinct}")
        counters["output_rows"] += len(kept)
        return kept

    def metrics():
        return NodeMetrics(
            input_rows=counters["input_rows"],
            output_rows=counters["output_rows"],
            extension={"calls": counters["calls"]})

    return call, metrics


def foreign_ids(node, inputs) -> dict:
    """Return the input IDs for a Foreign node, indexed by table alias."""
    values = {}
    for port in node.inputs:
        if port.source.port.startswith("rows:"):
            values[port.source.port.split(":", 1)[1]] = inputs[port.name]
    return values


def foreign_outputs(node, result) -> dict:
    """Build a Foreign node's output ports from its function result."""
    if node.ids == "pairs":
        return {f"pairs:{node.written_position}":
                pair_ids_table(*node.aliases, result)}
    (alias,) = node.aliases
    return {f"rows:{alias}": Rows({alias: document_ids(result)})}


class ForeignRuntime:
    """Call a user function once over every id it was given.

    Inside a pipeline, the executor calls a per-batch Foreign on each
    document.
    """

    def execute(self, node, inputs, context) -> NodeResult:
        values = foreign_ids(node, inputs)
        call, metrics = foreign_call(node, values, context)
        ids_by_alias = {alias: id_positions(values[alias])
                        for alias in node.aliases}
        return NodeResult(foreign_outputs(node, call(ids_by_alias)), metrics())


class RecombineRuntime:
    """Run the configured exact answer relation join."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Recombine):
            raise TypeError(type(node).__name__)
        if context.recombine is None:
            answer_tables = []
            survivors = {}
            for input_port in node.inputs:
                value = as_table(inputs[input_port.name])
                if not isinstance(value, pa.Table):
                    raise TypeError(
                        "Recombine inputs must be Arrow tables"
                    )
                if input_port.value_type is ValueType.JOIN_ANSWERS:
                    answer_tables.append(true_answer_rows(value))
                elif input_port.value_type is ValueType.ROWS:
                    alias = input_port.source.port.split(":", 1)[1]
                    survivors[alias] = value.column(alias).combine_chunks()
                else:
                    raise TypeError(
                        f"Recombine cannot read {input_port.value_type.value}"
                    )
            declaration, schema = build_result_declaration(
                answer_tables,
                survivors,
                node.alias_order[0],
            )
            value = IndexRelation(declaration, schema)
        else:
            value = context.recombine(node, inputs)
        return NodeResult({"tuples": value})


class ModelNodeRuntime:
    """Delegate one model node to the shared model execution object."""

    def execute(self, node, inputs, context) -> NodeResult:
        if context.model_execution is None:
            raise RuntimeError("model node has no model execution object")
        if context.model_inputs is not None:
            inputs = context.model_inputs(node, inputs, context)
        result = context.model_execution.execute(node, inputs)
        if not isinstance(result, NodeResult):
            raise TypeError("model execution must return NodeResult")
        if context.model_result is not None:
            context.model_result(node, result, context)
        return result


def _join_answers(node, inputs, context) -> NodeResult:
    """Give every document the answers of the answered document with its keys."""
    every, *answers = (inputs[port.name] for port in node.inputs)
    alias = node.left
    columns = context.sources.get(columns_key(alias))
    ids = document_ids(every)

    def keys(documents):
        return [relation_values(columns, key).take(documents)
                for key, _ in node.on]

    outputs = {}
    for port, value in zip(node.ports, answers, strict=True):
        if port.value_type is ValueType.FILTER_ANSWERS:
            answered = pa.array(sorted(value), pa.int32())
        else:
            answered = document_ids(value.ids[alias])
        pairs = pair_table("document", keys(ids), "answer", keys(answered))
        documents = pc.take(ids, pairs.column("document"))
        sources = pc.take(answered, pairs.column("answer"))
        if port.value_type is ValueType.FILTER_ANSWERS:
            outputs[port.name] = {
                int(document): value[int(source)]
                for document, source in zip(documents.to_numpy(),
                                            sources.to_numpy())}
            continue
        order = pc.sort_indices(documents)
        documents = pc.take(documents, order)
        if not value.values.num_columns:
            # a filter's rows carry ids only
            outputs[port.name] = Rows({alias: documents})
            continue
        position = pc.index_in(pc.take(sources, order), value_set=value.ids[alias])
        outputs[port.name] = Rows({alias: documents}, value.values.take(position))
    rows = outputs[f"rows:{alias}"]
    return NodeResult(outputs, NodeMetrics(input_rows=len(ids),
                                           output_rows=len(rows)))


class HashJoinRuntime:
    """Pair the rows of two tables whose key columns are equal."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, HashJoin):
            raise TypeError(type(node).__name__)
        if node.ports:
            return _join_answers(node, inputs, context)
        sides = {}
        for alias, keys, compared in (
                (node.left, [left for left, _ in node.on],
                 [left for left, _, _ in node.comparisons]),
                (node.right, [right for _, right in node.on],
                 [right for _, _, right in node.comparisons])):
            port = next(port for port in node.inputs
                        if port.source.port == f"rows:{alias}")
            ids = document_ids(inputs[port.name])
            columns = context.sources.get(columns_key(alias))
            # a key column may carry the alias's name: the ids stay apart
            sides[alias] = (ids, [relation_values(columns, key).take(ids)
                                  for key in keys],
                            [relation_values(columns, value).take(ids)
                             for value in compared])
        positions = pair_table(node.left, sides[node.left][1],
                               node.right, sides[node.right][1])
        positions = compared_pairs(
            positions, [comparison for _, comparison, _ in node.comparisons],
            sides[node.left][2], sides[node.right][2])
        # pair_table pairs row positions; map them back to the ids read
        pairs = pa.table({
            alias: pc.take(sides[alias][0], positions.column(alias))
            for alias in (node.left, node.right)})
        return NodeResult(
            {f"pairs:{node.written_position}": pairs},
            NodeMetrics(
                input_rows=sum(len(side[0]) for side in sides.values()),
                output_rows=pairs.num_rows))
