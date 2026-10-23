"""Runtimes for the dataflow nodes: exchange, barrier, foreign, recombine, and join."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.catalog.pairs import (
    columns_key,
    key_documents,
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
    """Give every document, or pair, the answers of its key documents."""
    from quail.execution.types import _join_answers_table

    keys = {left.alias: key_documents(
        context.sources, left.alias, left.column).to_numpy()
        for left, _ in node.on}
    every = {alias: document_ids(inputs[f"every:{alias}"]).to_numpy()
             for alias in keys}
    pairs = inputs.get("pairs")
    if isinstance(pairs, pa.Table) and "answer" in pairs.column_names:
        # a classification of joined rows read the join's true pairs
        pairs = true_answer_rows(pairs)
    outputs = {}
    # an AI join's anchor rows come from its expanded answers
    ports = sorted(node.ports, key=lambda port: port.name.startswith("rows:"))
    for port in ports:
        value = inputs[f"answers:{port.name}"]
        if port.value_type is ValueType.JOIN_ANSWERS:
            table = (value if isinstance(value, pa.Table)
                     else _join_answers_table(value))
            outputs[port.name] = _expanded(
                table, (node.left, node.right), every, keys, pairs)
        elif port.value_type is ValueType.FILTER_ANSWERS:
            outputs[port.name] = _expanded(value, (node.left,), every, keys, pairs)
        elif node.semantics:
            joined = next(table for table in outputs.values()
                          if isinstance(table, pa.Table))
            outputs[port.name] = Rows({node.left: _join_survivors(
                joined, node.left, node.semantics, every[node.left],
                len(keys[node.left]))})
        else:
            rows = value if isinstance(value, Rows) else Rows(
                {node.left: document_ids(value)})
            table = _expanded(rows.table(), rows.aliases, every, keys, pairs)
            outputs[port.name] = Rows(
                {alias: table.column(alias) for alias in rows.aliases},
                table.drop_columns(list(rows.aliases)))
    return NodeResult(outputs, NodeMetrics(
        input_rows=len(every[node.left]),
        output_rows=len(next(value for value in outputs.values()
                             if isinstance(value, Rows)))))


def _expanded(table: pa.Table, aliases, every, keys, pairs) -> pa.Table:
    """Give each answer row to every document whose key document it names.

    Args:
        table: Answer rows, one int32 id column per alias.
        aliases: The id columns to expand.
        every: alias -> the documents the answers stand for.
        keys: alias -> each document's key document.
        pairs: The pairs the answers stand for when they cover two
            tables, or None for every combination of their documents.
    """
    answered = {alias: table.column(alias).to_numpy() for alias in aliases}
    if pairs is not None and len(aliases) == 2:
        kept = np.ones(pairs.num_rows, dtype=bool)
        for alias in aliases:
            member = np.zeros(len(keys[alias]), dtype=bool)
            member[every[alias]] = True
            kept &= member[pairs.column(alias).to_numpy()]
        found = {alias: pairs.column(alias).to_numpy()[kept] for alias in aliases}
        index, rows = _matches(
            _pair_keys(*(keys[alias][found[alias]] for alias in aliases)),
            _pair_keys(*answered.values()))
        found = {alias: documents[index] for alias, documents in found.items()}
    else:
        rows = np.arange(table.num_rows)
        found = {}
        for alias in aliases:
            index, at = _matches(keys[alias][every[alias]], answered[alias][rows])
            rows = rows[at]
            found = {name: documents[at] for name, documents in found.items()}
            found[alias] = every[alias][index]
    table = table.take(rows)
    for alias, documents in found.items():
        table = table.set_column(
            table.schema.get_field_index(alias), table.schema.field(alias),
            pa.array(documents, pa.int32()))
    return table


def _pair_keys(anchors, partners) -> np.ndarray:
    """One uint64 per pair: the anchor's id, then the partner's."""
    return ((anchors.astype(np.uint64) << np.uint64(32))
            | partners.astype(np.uint64))


def _matches(wanted, answered) -> tuple[np.ndarray, np.ndarray]:
    """Return positions (i, j) with wanted[i] == answered[j], ordered by i."""
    if answered.dtype.kind == "i" and len(answered) and answered.min() >= 0:
        size = int(max(answered.max(), wanted.max(initial=-1))) + 1
        if np.bincount(answered, minlength=size).max() == 1:
            # one answer row per key: look each key up by position
            row_of = np.full(size, -1, dtype=np.int64)
            row_of[answered] = np.arange(len(answered))
            rows = row_of[wanted]
            index = np.flatnonzero(rows >= 0)
            return index, rows[index]
    joined = pa.table({"key": wanted, "i": np.arange(len(wanted))}).join(
        pa.table({"key": answered, "j": np.arange(len(answered))}), keys="key")
    index, rows = (joined.column(name).to_numpy() for name in ("i", "j"))
    order = np.argsort(index, kind="stable")
    return index[order], rows[order]


def _join_survivors(joined: pa.Table, anchor, semantics, reached,
                    num_documents: int) -> np.ndarray:
    """The anchors an AI join keeps, from its answers over every pair."""
    matched = np.zeros(num_documents, dtype=bool)
    matched[true_answer_rows(joined).column(anchor).to_numpy()] = True
    if semantics == "anti":
        return np.sort(reached[~matched[reached]]).astype(np.int32)
    return np.flatnonzero(matched).astype(np.int32)


class HashJoinRuntime:
    """Pair the rows of two tables whose key columns are equal."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, HashJoin):
            raise TypeError(type(node).__name__)
        if node.ports:
            return _join_answers(node, inputs, context)
        sides = {}
        for alias, keys in ((node.left, [left for left, _ in node.on]),
                            (node.right, [right for _, right in node.on])):
            port = next(port for port in node.inputs
                        if port.source.port == f"rows:{alias}")
            ids = document_ids(inputs[port.name])
            columns = context.sources.get(columns_key(alias))
            # a key column may carry the alias's name: the ids stay apart
            sides[alias] = (ids, [relation_values(columns, key).take(ids)
                                  for key in keys])
        positions = pair_table(node.left, sides[node.left][1],
                               node.right, sides[node.right][1])
        # pair_table pairs row positions; map them back to the ids read
        pairs = pa.table({
            alias: pc.take(sides[alias][0], positions.column(alias))
            for alias in (node.left, node.right)})
        return NodeResult(
            {f"pairs:{node.written_position}": pairs},
            NodeMetrics(
                input_rows=sum(len(ids) for ids, _ in sides.values()),
                output_rows=pairs.num_rows))
