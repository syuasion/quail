"""Codecs for physical plans.

A dataclass is encoded from its fields and decoded from its field
annotations, so a node or a spec needs no serialization code of its
own. Tuples become lists, enums their values, and a dataclass a dict;
where an annotation names a dataclass or a union of them, the dict
carries the class under ``"type"`` so a subclass, or one node of an
expression tree, decodes to the class that was encoded.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

import numpy as np

from .base import PhysicalGraph, PhysicalNode, PortRef
from .nodes import (
    Aggregate,
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiScore,
    Barrier,
    Exchange,
    Expand,
    Filter,
    Foreign,
    HashJoin,
    Limit,
    Project,
    Recombine,
    Representatives,
    Scan,
    Sort,
    Window,
)

TYPE_KEY = "type"
_SCALARS = (str, int, float, bool)


def encode_value(value):
    """Encode a dataclass, tuple, enum, or scalar as JSON-ready data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {TYPE_KEY: _tag(type(value)), **encode_fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, _SCALARS):
        return value
    raise TypeError(f"cannot encode a {type(value).__name__} value")


def encode_fields(value, exclude: tuple[str, ...] = ()) -> dict:
    """Encode a dataclass's fields, without a class tag."""
    return {field.name: encode_value(getattr(value, field.name))
            for field in fields(value) if field.name not in exclude}


def decode_value(annotation, value):
    """Decode what encode_value wrote, guided by a type annotation.

    Raises:
        ValueError: The value does not fit the annotation, or a dict
            is missing fields or has unknown ones.
    """
    origin = get_origin(annotation)
    if annotation is Any or annotation is object:
        return _decode_untyped(value)
    if origin in (Union, types.UnionType):
        return _decode_union(get_args(annotation), value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return decode_fields(annotation, value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is tuple or origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"expected a list for {annotation}, got {value!r}")
        arguments = get_args(annotation)
        if not arguments:
            return tuple(_decode_untyped(item) for item in value)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(decode_value(arguments[0], item) for item in value)
        if len(arguments) != len(value):
            raise ValueError(f"expected {len(arguments)} items, got {len(value)}")
        return tuple(decode_value(item_type, item)
                     for item_type, item in zip(arguments, value))
    if annotation is list or origin is list:
        (item_type,) = get_args(annotation) or (Any,)
        return [decode_value(item_type, item) for item in value]
    if annotation is dict or origin is dict:
        key_type, item_type = get_args(annotation) or (Any, Any)
        return {decode_value(key_type, key): decode_value(item_type, item)
                for key, item in value.items()}
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if annotation in _SCALARS or annotation is type(None):
        if not isinstance(value, annotation):
            raise ValueError(
                f"expected {annotation.__name__}, got {type(value).__name__}")
        return value
    raise TypeError(f"cannot decode a {annotation} value")


def decode_fields(annotation: type, value: Mapping[str, Any], **given):
    """Decode a dataclass from its encoded fields, every field present.

    Args:
        annotation: The dataclass, or a base of the tagged class.
        value: The encoded fields, with a class tag when the class is
            a subclass of the annotation.
        **given: Fields supplied by the caller instead of the value.

    Raises:
        ValueError: A field is missing or unknown, or the tag names a
            class that is not the annotation or a subclass of it.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a {annotation.__name__} dict, got {value!r}")
    kind = _tagged_type(value.get(TYPE_KEY), annotation)
    expected = {field.name for field in fields(kind) if field.name not in given}
    actual = set(value) - {TYPE_KEY}
    if expected - actual:
        raise ValueError(
            f"{kind.__name__} is missing fields {sorted(expected - actual)}")
    if actual - expected:
        raise ValueError(
            f"{kind.__name__} has unknown fields {sorted(actual - expected)}")
    hints = get_type_hints(kind)
    return kind(**given, **{
        name: decode_value(hints.get(name, Any), value[name]) for name in expected})


def _tag(kind: type) -> str:
    return f"{kind.__module__}.{kind.__qualname__}"


def _tagged_type(tag, annotation: type) -> type:
    if tag is None:
        return annotation
    module, _, name = str(tag).rpartition(".")
    kind = getattr(import_module(module), name, None) if module else None
    if not isinstance(kind, type) or not issubclass(kind, annotation):
        raise ValueError(f"{tag!r} is not a {annotation.__name__}")
    return kind


def _decode_union(members, value):
    if value is None and type(None) in members:
        return None
    classes = [member for member in members
               if isinstance(member, type) and is_dataclass(member)]
    if isinstance(value, Mapping) and classes:
        kind = _tagged_type(value.get(TYPE_KEY), object)
        base = next((member for member in classes if issubclass(kind, member)), None)
        if base is None:
            raise ValueError(f"{value.get(TYPE_KEY)!r} is none of {members}")
        return decode_fields(base, value)
    for member in members:
        if member is type(None) or (isinstance(member, type) and is_dataclass(member)):
            continue
        try:
            return decode_value(member, value)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"{value!r} fits none of {members}")


def _decode_untyped(value):
    if isinstance(value, list):
        return tuple(_decode_untyped(item) for item in value)
    return value


@dataclass(frozen=True)
class NodeCodec:
    """Encode and decode one registered physical node type."""

    node_type: type[PhysicalNode]

    def __post_init__(self) -> None:
        if not is_dataclass(self.node_type):
            raise TypeError("a physical node codec needs a dataclass")
        if not isinstance(self.type_name, str) or not self.type_name:
            raise TypeError("a physical node codec needs a nonempty type_name")

    @property
    def type_name(self) -> str:
        return self.node_type.type_name

    def encode(self, node: PhysicalNode) -> dict:
        if not isinstance(node, self.node_type):
            raise TypeError(
                f"codec {self.type_name!r} cannot encode "
                f"{type(node).__name__}")
        return {
            TYPE_KEY: self.type_name,
            "id": node.node_id,
            "inputs": encode_value(node.inputs),
            "attributes": encode_fields(node, exclude=("node_id", "inputs")),
        }

    def decode(self, value: Mapping[str, Any]) -> PhysicalNode:
        expected = {TYPE_KEY, "id", "inputs", "attributes"}
        if expected - set(value):
            raise ValueError(
                f"physical node is missing fields {sorted(expected - set(value))}")
        if set(value) - expected:
            raise ValueError(
                f"physical node has unknown fields {sorted(set(value) - expected)}")
        if value[TYPE_KEY] != self.type_name:
            raise ValueError(
                f"codec {self.type_name!r} cannot decode "
                f"{value[TYPE_KEY]!r}"
            )
        hints = get_type_hints(self.node_type)
        inputs = decode_value(hints["inputs"], value["inputs"])
        return decode_fields(self.node_type, {**value["attributes"]},
                             node_id=value["id"], inputs=inputs)


def built_in_codecs() -> tuple[NodeCodec, ...]:
    """Return codecs for every built in physical node."""
    return tuple(NodeCodec(node_type) for node_type in (
        Scan,
        AiFilter,
        AiScore,
        Filter,
        AiClassify,
        AiExtract,
        Barrier,
        Exchange,
        Foreign,
        HashJoin,
        AiJoin,
        Recombine,
        Representatives,
        Expand,
        Project,
        Aggregate,
        Sort,
        Window,
        Limit,
    ))


def encode_graph(
    graph: PhysicalGraph,
    codecs: Mapping[str, NodeCodec],
) -> dict:
    """Encode one physical graph."""
    return {
        "nodes": [
            codecs[node.type_name].encode(node)
            for node in graph.nodes
        ],
        "root": encode_value(graph.root),
    }


def decode_graph(
    value: Mapping[str, Any], codecs: Mapping[str, NodeCodec]
) -> PhysicalGraph:
    """Decode and validate one physical graph."""
    if set(value) != {"nodes", "root"}:
        raise ValueError(f"a physical graph has nodes and a root, got {sorted(value)}")
    nodes = []
    for encoded in value["nodes"]:
        type_name = encoded[TYPE_KEY]
        if type_name not in codecs:
            raise ValueError(f"unknown physical node type {type_name!r}")
        nodes.append(codecs[type_name].decode(encoded))
    graph = PhysicalGraph(
        nodes=tuple(nodes),
        root=decode_value(PortRef, value["root"]),
    )
    graph.validate()
    return graph
def plan_envelope(
    *,
    backend: str,
    model: str,
    device: str,
    gpus: int,
    graph: PhysicalGraph,
    codecs: Mapping[str, NodeCodec],
    settings: Mapping[str, Any] | None = None,
) -> dict:
    """Encode the physical plan fields sent across a process boundary."""
    graph.validate_backend(backend)
    return {
        "backend": backend,
        "model": model,
        "device": device,
        "gpus": gpus,
        "settings": dict(settings or {}),
        "graph": encode_graph(graph, codecs),
    }


def check_plan_envelope(value: Mapping[str, Any]) -> None:
    """Validate a physical plan envelope."""
    required = {
        "backend", "model", "device", "gpus",
        "settings", "graph",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"physical plan is missing fields {sorted(missing)}"
        )
    extra = set(value) - required
    if extra:
        raise ValueError(
            f"physical plan has unknown fields {sorted(extra)}"
        )
    if not isinstance(value["settings"], Mapping):
        raise TypeError("physical plan settings must be a mapping")
    for name in ("backend", "model", "device"):
        if not isinstance(value[name], str) or not value[name]:
            raise TypeError(f"physical plan {name} must be a nonempty string")
    if not isinstance(value["gpus"], int) or value["gpus"] <= 0:
        raise ValueError("physical plan workers must be positive")
    if not isinstance(value["graph"], Mapping):
        raise TypeError("physical plan graph must be a mapping")
