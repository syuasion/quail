"""Internal physical execution data types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from quail.catalog.pairs import COLUMNS_PREFIX
from quail.catalog.tokens import decode_token_documents
from quail.execution.result import answer_table
from quail.execution.rows import Rows, rows_of
from quail.physical import (
    OutputPort,
    PhysicalGraph,
    PortRef,
    ValueType,
)


@dataclass(frozen=True)
class TokenizedInput:
    """A random access sequence of tokenized documents."""

    documents: Sequence
    texts: Sequence | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.documents, "__len__") \
                or not hasattr(self.documents, "__getitem__"):
            raise TypeError(
                "a tokenized input needs a random access document sequence"
            )
        if self.texts is not None and len(self.texts) != len(self.documents):
            raise ValueError("document text and token sequences must have equal size")

    def __len__(self) -> int:
        return len(self.documents)


def document_input(tokens, texts=None) -> TokenizedInput:
    """Build one physical input from a token document sequence."""
    return TokenizedInput(decode_token_documents(tokens), texts)


@dataclass(frozen=True)
class PhysicalRequest:
    """A physical plan, its token input bindings, and its relations.

    relations holds one value table per alias a HashJoin or an apply()
    function reads, keyed ``columns:<alias>``; see quail.catalog.pairs.
    """

    plan: Mapping[str, Any]
    inputs: Mapping[str, TokenizedInput]
    relations: Mapping[str, pa.Table] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for input_id, value in self.inputs.items():
            if not input_id:
                raise ValueError("execution input ids cannot be empty")
            if not isinstance(value, TokenizedInput):
                raise TypeError(
                    "physical execution inputs must be TokenizedInput values"
                )
        for key, value in self.relations.items():
            if not key.startswith(COLUMNS_PREFIX):
                raise ValueError(
                    f"execution relations are keyed {COLUMNS_PREFIX}<alias>, "
                    f"got {key!r}")
            if not isinstance(value, pa.Table):
                raise TypeError("execution relations must be Arrow tables")

    def column_tables(self) -> dict[str, pa.Table]:
        """Return the value tables apply() functions read, by alias."""
        return {key[len(COLUMNS_PREFIX):]: table
                for key, table in self.relations.items()
                if key.startswith(COLUMNS_PREFIX)}

    @property
    def gpus(self) -> int:
        """Return the GPU count selected by the physical plan."""
        return int(self.plan["gpus"])


@dataclass(frozen=True)
class PhysicalResponse:
    """Arrow output relations and metrics from one physical plan."""

    outputs: Mapping[PortRef, pa.Table]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for ref, value in self.outputs.items():
            if not isinstance(ref, PortRef):
                raise TypeError("execution output keys must be PortRef values")
            if not isinstance(value, pa.Table):
                raise TypeError("physical execution outputs must be Arrow tables")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("execution metrics must be a mapping")

    def output(self, node_id: str, port: str) -> pa.Table:
        """Return one physical output relation."""
        try:
            return self.outputs[PortRef(node_id, port)]
        except KeyError as error:
            raise KeyError(f"missing physical output {node_id}.{port}") \
                from error


def _join_answers_table(value: Mapping[str, Any]) -> pa.Table:
    """Build the join answer table without visiting pairs in Python.

    A join can hold tens of millions of pairs. Python visits the
    anchors; the pairs are concatenated across anchors and converted by
    Arrow, and the partner ids come from one take.
    """
    anchor = str(value["anchor"])
    partners = tuple(value["partners"])
    anchor_map = value["anchor_index"]
    partner_map = value["partner_index"]
    streamed = value.get("anchor_partners") or {}
    anchor_parts, member_parts, answer_parts = [], [], []
    for raw_local, row in value["rows"].items():
        local = int(raw_local)
        members = streamed.get(local)
        anchor_parts.append([int(anchor_map[local])] * len(row))
        member_parts.append(range(len(row)) if members is None else members)
        answer_parts.append(row)
    members = pa.array(chain.from_iterable(member_parts), type=pa.int32())
    columns = {anchor: pa.array(chain.from_iterable(anchor_parts),
                                type=pa.int32())}
    for index, alias in enumerate(partners):
        documents = pa.array([int(member[index]) for member in partner_map],
                             type=pa.int32())
        columns[alias] = pc.take(documents, members)
    # answers arrive as 0/1 integers or booleans; the cast covers both
    answers = pc.cast(pa.array(chain.from_iterable(answer_parts)), pa.bool_())
    metadata = {
        "anchor": anchor, "semantics": value["semantics"],
        "written_position": value["written_position"],
    }
    if value.get("selectivity") is not None:
        metadata["selectivity"] = value["selectivity"]
    return answer_table(columns, answers, "join_answers", metadata)


def _output_table(node, port: OutputPort, value: Any) -> pa.Table:
    if isinstance(value, Rows):
        return value.table()
    if isinstance(value, pa.Table):
        return value
    if port.name.startswith("rows:"):
        return rows_of(value, port.name.split(":", 1)[1]).table()
    if port.value_type is ValueType.JOIN_ANSWERS:
        return _join_answers_table(value)
    raise TypeError(
        f"physical output {node.node_id}.{port.name} must be an Arrow table"
    )


def export_physical_outputs(graph: PhysicalGraph, run_result) -> dict:
    """Convert physical node outputs from one run to Arrow relations."""
    outputs = {}
    for node in graph.nodes:
        if node.node_id not in run_result.nodes:
            continue
        result = run_result.nodes[node.node_id]
        ports = {port.name: port for port in node.outputs}
        for name, value in result.outputs.items():
            outputs[PortRef(node.node_id, name)] = _output_table(
                node, ports[name], value
            )
    return outputs
