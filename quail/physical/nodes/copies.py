"""The Representatives and Expand nodes: one model answer per distinct document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class Representatives(PhysicalNode):
    """Keep one document of each group of documents with identical tokens.

    Reads ``rows:<alias>``. Writes ``rows:<alias>`` with the first
    document of each group, in input order, and ``copies:<alias>``, a
    table with one (copy, representative) row per input document; a
    representative is its own copy.
    """

    alias: str = ""

    type_name: ClassVar[str] = "quail.representatives"
    runtime_key: ClassVar[str] = type_name
    # it reads the token documents, which only the executor holds
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (
            OutputPort(f"rows:{self.alias}", ValueType.ROWS, schema=(self.alias,)),
            OutputPort(f"copies:{self.alias}", ValueType.ANY,
                       schema=("copy", "representative")),
        )


@dataclass(frozen=True)
class Expand(PhysicalNode):
    """Give every copy the answers of the representative it stands for.

    Reads one model operator's outputs over representatives, then the
    ``copies:<alias>`` table of the Representatives node before it, as
    its last input. Writes the operator's ports, in ``ports``, over
    every copy; it replaces the operator's outputs for its consumers
    and its answers in the result.
    """

    alias: str = ""
    # the filter answers' predicate position, for a wrapped AiFilter
    written_position: int = -1
    ports: tuple[OutputPort, ...] = ()

    type_name: ClassVar[str] = "quail.expand"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return self.ports
