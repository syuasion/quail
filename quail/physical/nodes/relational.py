"""Coordinator nodes: Exchange, Recombine, Project, Aggregate, Sort, Limit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from quail.logical.expressions import WindowCall, packed_columns

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class Exchange(PhysicalNode):
    """Route one alias's documents to the GPU that holds their KV.

    Present only in plans for several GPUs. On one GPU it passes the
    ids through.
    """

    anchor: str = ""

    type_name: ClassVar[str] = "quail.exchange"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"rows:{self.anchor}",
            ValueType.ROWS,
            schema=(self.anchor,),
        ),)

@dataclass(frozen=True)
class Recombine(PhysicalNode):
    """Combine answer relations by exact document ids."""

    alias_order: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.recombine"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "tuples",
            ValueType.ROWS,
            schema=self.alias_order,
        ),)

@dataclass(frozen=True)
class Project(PhysicalNode):
    """Select the requested result columns.

    Attributes:
        columns: The selected columns, in order.
        structs: (struct column, ((field, member column), ...)) pairs.
            Each struct column takes the place of its first member and
            holds one field per member; the members leave the result.
    """

    columns: tuple[str, ...] = ()
    structs: tuple = ()

    type_name: ClassVar[str] = "quail.project"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def result_columns(self) -> tuple[str, ...]:
        """Return the output column names, with struct members packed."""
        return packed_columns(self.columns, self.structs)

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "rows",
            ValueType.ROWS,
            schema=self.result_columns,
        ),)

@dataclass(frozen=True)
class Aggregate(PhysicalNode):
    """Group rows and compute aggregates.

    ``keys`` name input columns. ``aggregates`` are (name, function,
    argument) triples, the argument an input column or None for
    ``count(*)``. ``having`` are (name, comparison, value) tests over
    the aggregates. ``columns`` are the output columns, keys and
    aggregate names in result order.

    Without ``alias`` it groups the result rows on the coordinator.
    With ``alias`` it groups one table's documents on the executor
    before a model operator reads them: ``keys`` is one value column
    of that table holding a document id, and its ``rows:<alias>``
    output holds the distinct ids, such as one document per text.
    With ``partner`` too it groups the pairs of the join at
    ``written_position`` instead: each pair becomes its anchor's and
    its partner's key, and ``pairs:<written_position>`` holds the
    distinct pairs.
    """

    keys: tuple[str, ...] = ()
    aggregates: tuple[tuple[str, str, str | None], ...] = ()
    having: tuple[tuple[str, str, float], ...] = ()
    columns: tuple[str, ...] = ()
    alias: str = ""
    partner: str = ""
    written_position: int = -1

    type_name: ClassVar[str] = "quail.aggregate"
    runtime_key: ClassVar[str] = type_name

    @property
    def location(self) -> ExecutionLocation:
        # a document grouping reads value columns only the executor holds
        return (ExecutionLocation.GPU_EXECUTOR if self.alias
                else ExecutionLocation.COORDINATOR)

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if self.partner:
            return (OutputPort(f"pairs:{self.written_position}", ValueType.PAIRS,
                               schema=(self.alias, self.partner)),)
        if self.alias:
            return (OutputPort(f"rows:{self.alias}", ValueType.ROWS,
                               schema=(self.alias,)),)
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)

@dataclass(frozen=True)
class Sort(PhysicalNode):
    """Order, deduplicate, and bound the result rows on the coordinator.

    ``keys`` are (column, descending, nulls_first) triples over the
    input columns. ``columns`` are the output columns; an input column
    not among them is read by a key only. Distinct runs before the
    sort, and ``offset`` and ``fetch`` after it.
    """

    keys: tuple[tuple[str, bool, bool], ...] = ()
    columns: tuple[str, ...] = ()
    distinct: bool = False
    offset: int = 0
    fetch: int | None = None

    type_name: ClassVar[str] = "quail.sort"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)

@dataclass(frozen=True)
class Window(PhysicalNode):
    """Compute window functions over the result rows on the coordinator.

    ``calls`` read input columns and append one column each. ``columns``
    are the output columns, input columns and window names in result
    order. The rows come out in the last call's partition and order.
    """

    calls: tuple[WindowCall, ...] = ()
    columns: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.window"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS, schema=self.columns),)


@dataclass(frozen=True)
class Limit(PhysicalNode):
    """Stop result output after a fixed number of rows."""

    count: int = 0

    type_name: ClassVar[str] = "quail.limit"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS),)
