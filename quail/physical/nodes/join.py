"""The Barrier, AiJoin, Foreign, and HashJoin nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from quail.logical.expressions import Expression

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    Token,
    ValueType,
)


@dataclass(frozen=True)
class Barrier(PhysicalNode):
    """Move or repartition survivor ids between join steps."""

    next_anchor: str = ""
    aliases: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.barrier"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return tuple(
            OutputPort(
                f"rows:{alias}",
                ValueType.ROWS,
                schema=(alias,),
            )
            for alias in self.aliases
        )

@dataclass(frozen=True)
class AiJoin(PhysicalNode):
    """Evaluate one join predicate using an anchor document."""

    anchor: str = ""
    anchor_resident: str = "none"
    keep_kv: bool = False
    written_position: int = 0
    exec_index: int = 0
    partners: tuple[str, ...] = ()
    semantics: str = "full"
    selectivity: float | None = None
    expected_rows: float = 0.0
    # the anchors expected to reach the join, after the filters before it
    anchor_documents: float = 0.0
    anchor_frame_tokens: int = 0
    pair_tail_tokens: int = 0
    tuple_tokens: float = 0.0
    pairs_from: str = ""
    frame_token_ids: tuple[Token, ...] = ()
    label_token_ids: tuple[tuple[str, tuple[Token, ...]], ...] = ()
    tail_token_ids: tuple[Token, ...] = ()
    # the token ids the anchors are shown under, before the document
    document_head: tuple[Token, ...] = ()
    # "unified" or "tree": the attention path the choose_attention_algorithm rule
    # chose; empty leaves the model pipeline's default
    attention: str = ""
    # anchors borrow the KV pages of an anchor that shares their token
    # prefix (the should_share_prefixes rule)
    share_prefixes: bool = False

    type_name: ClassVar[str] = "quail.ai_join"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"


    @property
    def document_aliases(self) -> tuple[str, ...]:
        return (self.anchor,)

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return (f"join_answers:{self.written_position}", f"rows:{alias}")

    def can_precede(self, consumer: PhysicalNode) -> bool:
        # classify.py imports AiJoin, so AiClassify is imported at call time
        from .classify import AiClassify

        if isinstance(consumer, AiJoin):
            return (self.anchor == consumer.anchor
                    and self.semantics == consumer.semantics == "full")
        return isinstance(consumer, AiClassify) and consumer.classifies_join(self)

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [OutputPort(
            f"rows:{self.anchor}",
            ValueType.ROWS,
            schema=(self.anchor,),
        )]
        outputs.append(OutputPort(
            f"join_answers:{self.written_position}",
            ValueType.JOIN_ANSWERS,
            schema=(self.anchor, *self.partners, "answer"),
        ))
        return tuple(outputs)

    def explain_fields(self) -> dict:
        return {
            "written_position": self.written_position,
            "exec_index": self.exec_index,
            "anchor": self.anchor,
            "partners": list(self.partners),
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "expected_rows": self.expected_rows,
            "anchor_frame_tokens": self.anchor_frame_tokens,
            "pair_tail_tokens": self.pair_tail_tokens,
            "anchor_resident": self.anchor_resident,
            "tuple_tokens": self.tuple_tokens,
            "pairs_from": self.pairs_from,
            "keep_kv": self.keep_kv,
            "attention": self.attention,
            "share_prefixes": self.share_prefixes,
        }

@dataclass(frozen=True)
class Foreign(PhysicalNode):
    """Physical operator that calls a user function on document IDs or pairs.

    Per-batch functions run as documents reach them in a pipeline, or once
    over a materialized input. Barrier functions run once over all surviving
    rows and end the preceding pipeline. The function may preserve or remove
    input IDs, or return pairs of input IDs; it cannot create new IDs.
    ``columns`` are the source columns and expressions over them the
    function reads, named for it as written.
    """

    function: str = ""
    kind: str = "per_batch"
    ids: str = "drop"
    columns: tuple[Expression, ...] = ()  # each over one of the aliases
    aliases: tuple[str, ...] = ()
    written_position: int = -1

    type_name: ClassVar[str] = "quail.foreign"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        # an apply reads no KV: it gates the documents of any pipeline
        return True

    @property
    def document_aliases(self) -> tuple[str, ...]:
        if self.kind != "per_batch":
            return ()
        return self.aliases if self.ids == "pairs" else self.aliases[:1]

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        if self.ids == "pairs":
            return (f"pairs:{self.written_position}",)
        return super().document_outputs(alias)

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if self.ids == "pairs":
            return (OutputPort(
                f"pairs:{self.written_position}", ValueType.PAIRS,
                schema=tuple(self.aliases)),)
        (alias,) = self.aliases
        return (OutputPort(f"rows:{alias}", ValueType.ROWS,
                           schema=(alias,)),)

@dataclass(frozen=True)
class HashJoin(PhysicalNode):
    """Pair the rows of two tables whose key columns are equal.

    ``on`` lists (left key, right key) pairs, each a source column or an
    expression over one; a row pair is kept when every listed pair of
    values is equal. The node reads the key columns as values and
    writes the pairs the join at ``written_position``
    asks the model about. ``pair_fraction`` is the planner's estimate
    of the pairs kept over the cross product.

    With ``ports`` it joins a model operator's answers back instead.
    The operator read one key document per text; ``on`` names, for
    each table its answers cover, the value column holding a
    document's key document. Its inputs are named:

    - ``every:<alias>``: every document the operator would have read
      without the grouping, for each of those tables.
    - ``pairs``: the pairs the operator would have read, when it read
      pairs rather than every combination.
    - ``answers:<port>``: the operator's output on that port.

    Each document, or pair, takes the answers of its key documents.
    It writes ``ports``, the operator's own ports. ``semantics``
    names an AI join's semantics, and its anchors' rows are then
    those its expanded answers keep; ``written_position`` is the
    operator's.
    """

    left: str = ""
    right: str = ""
    on: tuple[tuple[Expression, Expression], ...] = ()
    written_position: int = -1
    pair_fraction: float = 1.0
    ports: tuple[OutputPort, ...] = ()
    semantics: str = ""

    type_name: ClassVar[str] = "quail.hash_join"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def alias(self) -> str:
        """The table whose documents an answer join fills, as a filter names it."""
        return self.left

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if self.ports:
            return self.ports
        return (OutputPort(f"pairs:{self.written_position}", ValueType.PAIRS,
                           schema=(self.left, self.right)),)
