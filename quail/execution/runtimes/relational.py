"""Runtimes for the relational nodes: scan, project, limit, aggregate, and sort."""

from __future__ import annotations

from collections.abc import Sized

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog.pairs import columns_key, key_documents
from quail.catalog.selection import equality_keys
from quail.execution.aggregates import prepare_aggregate
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult
from quail.execution.rows import Rows, as_table
from quail.logical.predicates import compare_literal
from quail.physical import Aggregate, Limit, Project, Scan, Sort, Window


class ScanRuntime:
    """Read a prepared source registered by document alias."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Scan):
            raise TypeError(type(node).__name__)
        if node.input_id not in context.sources:
            raise KeyError(
                f"no prepared source for input {node.input_id!r}"
            )
        value = document_ids(
            _tested_documents(node, context) if node.predicates
            else context.sources[node.input_id])
        return NodeResult({f"rows:{node.alias}": Rows({node.alias: value})},
                          NodeMetrics(output_rows=len(value)))


def _tested_documents(node, context) -> pa.Array:
    """Return the positions of the documents that pass a scan's regular predicates."""
    from quail.catalog.selection import selected_rows

    columns = context.sources.get(columns_key(node.alias))
    if columns is None:
        raise KeyError(
            f"the regular predicates of {node.alias!r} need its value columns")
    return selected_rows(columns, node.predicates)


def _num_rows(value) -> int | None:
    """Rows of a table or sequence, or None when counting would run a plan."""
    if isinstance(value, QueryResult):
        return value.known_num_rows()
    return len(value) if isinstance(value, Sized) else None


class ProjectRuntime:
    """Run the configured result projection.

    The first input is the relation to project; any others are label
    tables whose columns the projection may name.
    """

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Project):
            raise TypeError(type(node).__name__)
        if not inputs:
            raise ValueError("Project needs one input")
        first, *rest = node.inputs
        value = as_table(inputs[first.name])
        labels = {port.name: as_table(inputs[port.name]) for port in rest}
        if context.project is not None:
            value = (context.project(node, value, labels) if labels
                     else context.project(node, value))
        # a projection keeps its rows; a lazy result's count is the
        # input's until the result runs
        rows = _num_rows(inputs[first.name]) or 0
        output_rows = _num_rows(value)
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=rows if output_rows is None else output_rows))


class LimitRuntime:
    """Limit a table, batch, or Python sequence."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Limit):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Limit needs one input")
        value = next(iter(inputs.values()))

        rows = _num_rows(value) or 0
        if isinstance(value, QueryResult):
            value = value.with_limit(node.count)
        else:
            value = value[:node.count]
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=_num_rows(value) or 0))


def _comparable(column, clause: str):
    """Return a column as a type Arrow can sort or group by."""
    if pa.types.is_dictionary(column.type):
        column = column.cast(column.type.value_type)
    if pa.types.is_nested(column.type):
        raise TypeError(
            f"{clause} cannot compare values of type {column.type}")
    return column


def _distinct_indices(table: pa.Table, columns) -> pa.Array:
    """Return the first row of each distinct value of the columns, in order."""
    # internal names stay clear of the user's, and are read by position
    keyed = {f"key_{index}": equality_keys(
        _comparable(table.column(name), "DISTINCT"), nan_equal=True)
        for index, name in enumerate(columns)}
    keys = list(keyed)
    keyed["row_position"] = pa.array(np.arange(table.num_rows), pa.int64())
    firsts = pa.table(keyed).group_by(keys, use_threads=False).aggregate(
        [("row_position", "min")]).column(len(keys))
    return pc.take(firsts, pc.sort_indices(firsts))


_AGGREGATE_KERNELS = {
    "count": "count", "count_distinct": "count_distinct", "sum": "sum",
    "avg": "mean", "min": "min", "max": "max",
}
def _rows_table(value) -> pa.Table:
    """Return a node's rows input as one Arrow table."""
    value = as_table(value)
    if isinstance(value, QueryResult):
        return value.collect()
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.Table):
        return value
    raise TypeError(f"expected rows, got {type(value).__name__}")


def _aggregate_kernel(function: str, values) -> str:
    kernel = _AGGREGATE_KERNELS[function]
    # Arrow cannot count the distinct values of an untyped null column
    if kernel == "count_distinct" and pa.types.is_null(values.type):
        return "count"
    return kernel


def aggregate_table(table: pa.Table, node) -> pa.Table:
    """Group a table by the node's keys and compute its aggregates.

    Dictionary key columns are grouped by their values, and numeric
    keys by numeric equality. A null argument value is left out of
    every aggregate but count(*). Groups come out in the order their
    first rows appear.
    """
    arguments = list(dict.fromkeys((*node.keys, *(
        argument for _, _, argument in node.aggregates
        if argument is not None))))
    # Arrow derives aggregate output names from input names. Use private
    # names throughout the intermediate table to avoid user-name collisions.
    names = {name: f"column_{index}" for index, name in enumerate(arguments)}
    keyed = {names[name]: _comparable(table.column(name), "an aggregate")
             for name in arguments}
    hashed = {name: f"key_{index}" for index, name in enumerate(dict.fromkeys((
        *node.keys, *(argument for _, function, argument in node.aggregates
                      if function == "count_distinct"))))}
    keyed.update({key: equality_keys(keyed[names[name]], nan_equal=True)
                  for name, key in hashed.items()})
    keyed["row_position"] = pa.array(np.arange(table.num_rows), pa.int64())
    aggregates = [
        (name, argument, _aggregate_kernel(
            function, None if argument is None else keyed[names[argument]]))
        for name, function, argument in node.aggregates]
    sources, scalars, finishers, specs = {}, {}, {}, []
    for _, argument, kernel in aggregates:
        key = (argument, kernel)
        if key in sources or key in scalars:
            continue
        column = None
        if argument is not None:
            source = hashed if kernel == "count_distinct" else names
            values, kernel, finish = prepare_aggregate(
                keyed[source[argument]], kernel, grouped=bool(node.keys))
            if kernel is None:
                scalars[key] = finish(None)
                continue
            if finish is not None:
                finishers[len(node.keys) + len(specs)] = finish
            column = f"aggregate_{len(specs)}"
            keyed[column] = values
        sources[key] = len(node.keys) + len(specs)
        specs.append(([], "count_all") if argument is None else (column, kernel))
    grouped = pa.table(keyed).group_by(
        [hashed[key] for key in node.keys], use_threads=False
    ).aggregate(specs + [("row_position", "min")])
    firsts = grouped.column(len(node.keys) + len(specs))
    order = pc.sort_indices(firsts)
    for index, finish in finishers.items():
        grouped = grouped.set_column(index, grouped.column_names[index],
                                     finish(grouped.column(index)))
    # the keys keep their own type: the first row of each group supplies them
    columns = {name: pc.take(keyed[names[name]], firsts) for name in node.keys}
    for name, argument, kernel in aggregates:
        key = (argument, kernel)
        columns[name] = (scalars[key] if key in scalars
                         else grouped.column(sources[key]))
    result = pa.table({name: pc.take(column, order)
                       for name, column in columns.items()})
    mask = None
    for name, comparison, value in node.having:
        test = compare_literal(result.column(name), comparison, value)
        mask = test if mask is None else pc.and_kleene(mask, test)
    if mask is not None:
        result = result.filter(pc.fill_null(mask, False))
    return result.select(list(node.columns))


class AggregateRuntime:
    """Group and aggregate a table or a materialized result."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Aggregate):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Aggregate needs one input")
        if node.alias:
            return _aggregate_documents(node, next(iter(inputs.values())), context)
        table = _rows_table(next(iter(inputs.values())))
        result = aggregate_table(table, node)
        return NodeResult({"rows": QueryResult.from_table(result)}, NodeMetrics(
            input_rows=table.num_rows, output_rows=result.num_rows))


def _aggregate_documents(node, value, context) -> NodeResult:
    """Replace documents, or pairs, by their key documents, each once."""
    if node.partner:
        pairs = as_table(value)
        anchors, partners = (
            key_documents(context.sources, alias, column).to_numpy()[
                pairs.column(alias).to_numpy()].astype(np.uint64)
            for alias, column in ((node.alias, node.keys[0]),
                                  (node.partner, node.keys[-1])))
        # one uint64 per pair: the anchor's key, then the partner's
        kept = np.sort((anchors << np.uint64(32)) | partners)
        kept = kept[np.concatenate(([True], kept[1:] != kept[:-1]))]
        return NodeResult(
            {f"pairs:{node.written_position}": pa.table({
                node.alias: (kept >> np.uint64(32)).astype(np.int32),
                node.partner: (kept & np.uint64(0xFFFFFFFF)).astype(np.int32)})},
            NodeMetrics(input_rows=pairs.num_rows, output_rows=len(kept)))
    ids = document_ids(value).to_numpy()
    keys = key_documents(context.sources, node.alias, node.keys[0]).to_numpy()
    kept = np.zeros(len(keys), dtype=bool)
    kept[keys[ids]] = True
    kept = np.flatnonzero(kept).astype(np.int32)
    return NodeResult(
        {f"rows:{node.alias}": Rows({node.alias: kept})},
        NodeMetrics(input_rows=len(ids), output_rows=len(kept)))


def _partition_starts(keys, num_rows: int) -> pa.Array:
    """Flag the rows whose partition keys differ from the row before."""
    starts = pa.array([True] + [False] * (num_rows - 1), pa.bool_())
    for column in keys:
        column = column.combine_chunks()
        previous, current = column[:-1], column[1:]
        # a null key equals a null key, and differs from a value
        differs = pc.or_kleene(
            pc.fill_null(pc.not_equal(current, previous), False),
            pc.xor(pc.is_null(current), pc.is_null(previous)))
        starts = pc.or_(starts, pa.concat_arrays([pa.array([True]), differs]))
    return starts


def _window_columns(table: pa.Table, calls) -> tuple[pa.Array, dict]:
    """Compute calls that share one partition and order over one sort.

    Returns:
        The sort order of the table's rows, and each call's column in
        that order.
    """
    partition, order = calls[0].partition, calls[0].order
    keyed = pa.table({
        **{name: _comparable(table.column(name), "PARTITION BY")
           for name in partition},
        **{name: _comparable(table.column(name), "ORDER BY")
           for name, _, _ in order}})
    indices = pc.sort_indices(keyed, sort_keys=[
        *((name, "ascending", "at_end") for name in partition),
        *((name, "descending" if descending else "ascending",
           "at_start" if nulls_first else "at_end")
          for name, descending, nulls_first in order)])
    starts = _partition_starts(
        [keyed.column(name).take(indices) for name in partition], table.num_rows)
    position = pa.array(np.arange(table.num_rows), pa.int64())
    # each row's partition, and the position its partition starts at
    first = pc.take(pc.indices_nonzero(starts),
                    pc.subtract(pc.cumulative_sum(starts.cast(pa.int64())), 1))
    columns = {}
    for call in calls:
        values = (None if call.argument is None
                  else table.column(call.argument).take(indices).combine_chunks())
        if call.function == "row_number":
            result = pc.add(pc.subtract(position, first), 1)
        elif call.function in ("lag", "lead"):
            null = pa.nulls(1, values.type)
            if call.function == "lag":
                shifted, edge = pa.concat_arrays([null, values[:-1]]), starts
            else:
                shifted = pa.concat_arrays([values[1:], null])
                edge = pa.concat_arrays([starts[1:], pa.array([True])])
            result = pc.if_else(edge, null[0], shifted)
        else:
            counted = (pc.is_valid(values).cast(pa.int64())
                       if call.function == "count" and values is not None
                       else pa.array(np.ones(table.num_rows), pa.int64())
                       if values is None else pc.fill_null(values, 0))
            running = pc.cumulative_sum(counted)
            # the running value before the partition: none for the first one
            before = pc.if_else(
                pc.equal(first, 0), pa.scalar(0, running.type),
                pc.take(running, pc.max_element_wise(pc.subtract(first, 1), 0)))
            result = pc.subtract(running, before)
        columns[call.name] = result
    return indices, columns


def window_table(table: pa.Table, node) -> pa.Table:
    """Append the node's window columns and keep its output columns.

    Calls over one partition and order share one sort. The rows come
    out in the last call's partition and order, which a Sort over the
    same keys relies on.
    """
    groups = {}
    for call in node.calls:
        groups.setdefault((call.partition, call.order), []).append(call)
    columns = {name: table.column(name) for name in table.column_names}
    indices = None
    for calls in groups.values():
        indices, computed = _window_columns(table, calls)
        restore = pc.sort_indices(indices)
        columns.update({name: pc.take(column, restore)
                        for name, column in computed.items()})
    result = pa.table(columns).select(list(node.columns))
    return result if indices is None else result.take(indices)


class WindowRuntime:
    """Compute window functions over a table or a materialized result."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Window):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Window needs one input")
        table = _rows_table(next(iter(inputs.values())))
        result = window_table(table, node)
        return NodeResult({"rows": QueryResult.from_table(result)}, NodeMetrics(
            input_rows=table.num_rows, output_rows=result.num_rows))


class SortRuntime:
    """Sort, deduplicate, and bound a table or a materialized result."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Sort):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Sort needs one input")
        table = _rows_table(next(iter(inputs.values())))
        rows = table.num_rows
        if node.distinct:
            table = table.take(_distinct_indices(table, node.columns))
        if node.keys:
            keyed = pa.table({
                column: _comparable(table.column(column), "ORDER BY")
                for column, _, _ in node.keys})
            indices = pc.sort_indices(keyed, sort_keys=[
                (column, "descending" if descending else "ascending",
                 "at_start" if nulls_first else "at_end")
                for column, descending, nulls_first in node.keys])
            table = table.take(indices)
        stop = None if node.fetch is None else node.offset + node.fetch
        table = table[node.offset:stop].select(list(node.columns))
        return NodeResult({"rows": QueryResult.from_table(table)}, NodeMetrics(
            input_rows=rows, output_rows=table.num_rows))
