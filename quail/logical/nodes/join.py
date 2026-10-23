"""The Join and SemanticJoin nodes and the above-join test."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    Prompt,
    _explain,
    model_call,
    validate_predicate,
)
from quail.logical.nodes.base import LogicalNode, _subtree


@dataclass(frozen=True)
class Join:
    """One binary relational join; ``on`` empty means a cross join.

    The pairs it produces are the tuples a SemanticJoin above it asks
    the model about. Every Equality and ColumnComparison names one
    column on each side; ``comparisons`` keeps only the pairs ``on``
    pairs that also pass them.
    """

    left: LogicalNode
    right: LogicalNode
    on: tuple = ()  # tuple[Equality, ...]
    comparisons: tuple = ()  # tuple[ColumnComparison, ...]

    type_name: ClassVar[str] = "quail.join"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.left, self.right)

    def expressions(self) -> tuple:
        return (*self.on, *self.comparisons)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = list(self.left.output_schema())
        fields.extend(
            field for field in self.right.output_schema() if field not in fields
        )
        return tuple(fields)

    def validate(self) -> None:
        left = {field.alias for field in self.left.output_schema()}
        right = {field.alias for field in self.right.output_schema()}
        if self.comparisons and not self.on:
            raise CompileError(
                f"join comparison {self.comparisons[0]} needs an equality "
                f"condition beside it")
        for condition in self.expressions():
            sides = set(condition.aliases())
            if not (sides & left and sides & right) or len(sides) != 2:
                raise CompileError(
                    f"join condition {condition} must name one table "
                    f"on each side of the join ({sorted(left)} and "
                    f"{sorted(right)})"
                )

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 2:
            raise CompileError("Join needs two children")
        return replace(self, left=children[0], right=children[1])

    def with_expressions(self, expressions: tuple):
        return replace(self, on=tuple(expressions[:len(self.on)]),
                       comparisons=tuple(expressions[len(self.on):]))

    def explain_fields(self) -> dict:
        return {"on": [str(condition) for condition in self.expressions()]
                or "cross"}


@dataclass(frozen=True)
class SemanticJoin:
    """One model predicate over the pairs its input Join produces.

    A query that joins three or more tables has one SemanticJoin per
    pair, each feeding the Join of the next. The physical planner
    chooses the joins' run order and each join's anchor, the table
    whose KV its stage keeps; it keeps ``anchor`` when one was written.
    """

    input: LogicalNode  # a Join, under any Apply that
    #                                    returns its pairs
    predicate: Any  # ModelCall | RegularPredicate over an AI.SCORE call
    semantics: str = "full"  # full | exists | anti
    selectivity: Optional[float] = None  # fraction of tuples that pass
    anchor: Optional[str] = None  # table alias whose KV is kept;
    #                                    None = planner picks

    type_name: ClassVar[str] = "quail.semantic_join"

    @property
    def prompt(self) -> Prompt:
        """The prompt asked of every pair."""
        return model_call(self.predicate).prompt

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.predicate,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if self.semantics not in {"full", "exists", "anti"}:
            raise CompileError(f"unknown join semantics {self.semantics!r}")
        validate_predicate(self.predicate)
        present = {field.alias for field in self.input.output_schema()}
        aliases = model_call(self.predicate).aliases()
        missing = set(aliases) - present
        if missing:
            raise CompileError(
                f"the join predicate reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})"
            )

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticJoin needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticJoin needs one predicate")
        return replace(self, predicate=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "anchor": self.anchor,
            "expression": _explain(self.predicate),
        }


def _above_joins(nodes) -> frozenset:
    return frozenset(node.alias for node in nodes
                     if any(isinstance(below, SemanticJoin)
                            for below in _subtree(node.input)))
