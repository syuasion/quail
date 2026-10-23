"""Construct logical plans from bound frontend expressions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from quail.logical.expressions import (
    Aggregation,
    ColumnRef,
    CompileError,
    FilterPredicate,
    ModelCall,
    Prompt,
    RegularPredicate,
    model_call,
)
from quail.logical.nodes import (
    Aggregate,
    Apply,
    Filter,
    Join,
    Project,
    Result,
    Scan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
    Window,
)
from quail.logical.plan import (
    LogicalPlan,
)


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""

    aliases: tuple  # newly joined table aliases
    predicate: Any  # ModelCall | RegularPredicate over an AI.SCORE call
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None
    on: tuple = ()  # tuple[Equality, ...] over the tables
    comparisons: tuple = ()  # tuple[ColumnComparison, ...] over the tables
    applies: tuple = ()  # (function, kind, columns) returning pairs

    @property
    def prompt(self) -> Prompt:
        """The prompt asked of every tuple."""
        return model_call(self.predicate).prompt


class LogicalPlanBuilder:
    """Build the built in logical nodes used by both front ends."""

    def __init__(self):
        self._tables = []
        self._nodes = {}
        self._root = None
        self._joins = 0

    def add_scan(
        self,
        alias: str,
        provider: str,
        column: str,
        predicates: tuple[FilterPredicate, ...] = (),
        applies: tuple = (),
        labels: tuple = (),
        label_filters: tuple = (),
        regular_predicates: tuple = (),
        extracts: tuple = (),
        extract_filters: tuple = (),
    ) -> None:
        """Add one table with its filters, applies, classifications, and extractions.

        Args:
            alias: The table's alias in the query.
            provider: The registered provider name.
            column: The document column.
            predicates: The table's AI.IF predicates in written order.
            regular_predicates: RegularPredicate tests the scan applies
                before any operator reads a document.
            applies: (function, kind, ids, columns) per apply, in order.
            labels: (call, name) per one-table classification. A call
                that a label filter tests sits below the first Filter
                testing it; the others sit above the last Filter, in the
                order given.
            label_filters: (name, values, selectivity) per Filter on a
                classification's label column, in written order.
            extracts: (call, name) per extraction, in order. They sit
                above the classifications and label filters.
            extract_filters: (RegularPredicate, selectivity) per Filter
                on an extraction's answer or span column, in written
                order. Each sits right above the extraction it tests.

        Raises:
            CompileError: The alias is taken, a label filter names a
                column no classification in labels computes, or an
                extraction filter names a column no extraction computes.
        """
        if alias in self._nodes:
            raise CompileError(f"duplicate table alias {alias!r}")
        node = Scan(
            provider=provider,
            alias=alias,
            column=column,
            predicates=tuple(regular_predicates),
        )
        if predicates:
            node = SemanticFilter(node, tuple(predicates))
        for function, kind, ids, columns in applies:
            node = Apply(
                node,
                function=function,
                kind=kind,
                ids=ids,
                columns=tuple(columns),
                aliases=(alias,),
            )
        calls = {name: call for call, name in labels}
        classified = []
        for name, values, selectivity in label_filters:
            if name not in calls:
                raise CompileError(
                    f"a filter on a label of {alias!r} tests a "
                    f"classification the query does not name"
                )
            unknown = sorted(set(values) - set(calls[name].labels))
            if unknown:
                raise CompileError(
                    f"the filter on {name!r} lists {unknown}, which are not "
                    f"among its labels {list(calls[name].labels)}")
            if calls[name] not in classified:
                node = SemanticClassify(node, calls[name], name)
                classified.append(calls[name])
            node = Filter(
                node,
                RegularPredicate(ColumnRef(alias, provider, name), "in", tuple(values)),
                selectivity,
            )
        for call, name in labels:
            if call not in classified:
                node = SemanticClassify(node, call, name)
                classified.append(call)
        tests = list(extract_filters)
        for call, name in extracts:
            extraction = SemanticExtract(node, call, name)
            node = extraction
            for predicate, selectivity in extract_filters:
                if predicate.column.column in (name, extraction.span_name):
                    node = Filter(node, predicate, selectivity)
                    tests.remove((predicate, selectivity))
        if tests:
            raise CompileError(
                f"the filter {tests[0][0]} tests an extraction the query "
                f"does not return")
        self._tables.append(alias)
        self._nodes[alias] = node
        if self._root is None:
            self._root = node

    def add_classify(self, call: ModelCall, name: str) -> None:
        """Classify the rows of the join the call reads.

        The node sits directly above the SemanticJoin over the call's
        two tables.
        """
        aliases = set(call.aliases())

        def insert(node):
            if (
                isinstance(node, SemanticJoin)
                and set(model_call(node.predicate).aliases()) == aliases
            ):
                return SemanticClassify(node, call, name)
            if isinstance(node, (SemanticJoin, SemanticClassify, Apply)):
                return replace(node, input=insert(node.input))
            if isinstance(node, Join):
                return replace(node, left=insert(node.left))
            raise CompileError(
                f"the classification of {call.aliases()[0]!r} x "
                f"{call.aliases()[1]!r} joined rows needs a join of the two"
            )

        if self._root is None or len(aliases) != 2:
            raise CompileError(
                "a classification of joined rows reads one document from "
                "each side of a join"
            )
        self._root = insert(self._root)

    def add_join(self, join: JoinSpec) -> None:
        """Join each new table onto the tree, then ask the prompt.

        The equalities go on the Join that brings the last table in;
        both front ends only produce conditions over that table.
        """
        if self._root is None:
            raise CompileError("a logical join needs an input table")
        root = self._root
        for alias in join.aliases:
            root = Join(root, self._nodes[alias])
        present = {field.alias for field in root.output_schema()}
        for condition in (*join.on, *join.comparisons):
            if not set(condition.aliases()) <= present:
                raise CompileError(
                    f"join condition {condition} names a table this join "
                    f"does not bring in ({list(join.aliases)}); put it on "
                    f"the JOIN that introduces the table"
                )
        if join.on or join.comparisons:
            root = replace(root, on=tuple(join.on),
                           comparisons=tuple(join.comparisons))
        written_position = self._joins
        self._joins += 1
        for function, kind, columns in join.applies:
            aliases = tuple(dict.fromkeys(ref.alias for ref in join.prompt.args))
            root = Apply(
                root,
                function=function,
                kind=kind,
                ids="pairs",
                columns=tuple(columns),
                aliases=aliases,
                written_position=written_position,
            )
        self._root = SemanticJoin(
            input=root,
            predicate=join.predicate,
            semantics=join.semantics,
            selectivity=join.selectivity,
            anchor=join.anchor,
        )

    def add_cross_join(self, alias: str) -> None:
        """Add one relational cross join without an AI predicate."""
        if self._root is None:
            raise CompileError("a logical join needs an input table")
        self._root = Join(self._root, self._nodes[alias])

    def project(
        self,
        columns: tuple,
        limit: int | None = None,
        order: tuple = (),
        offset: int = 0,
        distinct: bool = False,
        aggregation: Aggregation | None = None,
        structs: tuple = (),
        windows: tuple = (),
        output: tuple = (),
    ) -> LogicalPlan:
        """Build the plan's result nodes over the projected columns.

        Args:
            columns: The projected value expressions and Aliases.
            limit: LIMIT, or None.
            order: The SortKeys of ORDER BY.
            offset: OFFSET.
            distinct: Whether the result is DISTINCT.
            aggregation: GROUP BY and its aggregates, or None.
            structs: The AI.EXTRACT struct columns the projection packs.
            windows: The WindowCalls over the projected or grouped rows.
            output: The result columns in SELECT order when windows
                are present.
        """
        if self._root is None:
            raise CompileError("a logical plan needs an input table")
        if not columns and aggregation is None:
            raise CompileError("Project needs at least one column")
        if structs and aggregation is not None:
            raise CompileError(
                "GROUP BY does not read AI.EXTRACT struct columns; extract "
                "each field as its own column")
        if structs and distinct:
            raise CompileError(
                "DISTINCT does not compare AI.EXTRACT struct columns; extract "
                "each field as its own column")
        node = Project(self._root, tuple(columns), tuple(structs))
        if aggregation is not None:
            node = Aggregate(
                node,
                aggregation.keys,
                aggregation.aggregates,
                aggregation.output,
                aggregation.having,
            )
        if windows:
            node = Window(node, tuple(windows), tuple(output))
        if limit is not None or order or offset or distinct:
            node = Result(node, limit, tuple(order), offset, distinct)
        plan = LogicalPlan(node)
        plan.validate()
        return plan
