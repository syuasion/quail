"""Run compatible physical nodes through the shared document scheduler."""

from itertools import groupby

from quail.backends.quail.executor.operators import execution_parts
from quail.backends.quail.executor.parts import execute_parts, execution_prefix_tree
from quail.backends.quail.executor.state import QueryExecutionState
from quail.catalog.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.ids import id_positions
from quail.execution.rows import Rows
from quail.physical import AiFilter, AiJoin


def execute_pipeline(state: QueryExecutionState, nodes, inputs, context) -> dict:
    """Prepare compatible operators, run their stages, and return node results."""
    state.gpu_timing = bool(context.state.get("gpu_timing", False))
    first = nodes[0]
    implementations = execution_parts()
    implementation = implementations[first.type_name]
    if (implementation.execute_group is not None
            and all(node.type_name == first.type_name for node in nodes)):
        prepared = context.model_inputs(first, inputs, context, nodes=nodes)
        return implementation.execute_group(state, nodes, prepared)

    alias = first.document_aliases[0]
    anchor_port = next((port.name for port in first.inputs
                        if port.source.port == f"rows:{alias}"), None)
    source = (inputs[first.node_id][anchor_port] if anchor_port is not None
              else next(iter(inputs[first.node_id].values())))
    ids = id_positions(source.ids[alias] if isinstance(source, Rows) else source)
    last = max(index for index, node in enumerate(nodes)
               if node.backend is not None)

    def group_key(node):
        implementation = implementations[node.type_name]
        return node.type_name, None if implementation.combine else node.node_id

    parts = []
    for _, group in groupby(nodes, group_key):
        members = list(group)
        parts.append(implementations[members[0].type_name].prepare(
            state, members, ids, parts, inputs, context))
    prefixes = DocumentPrefixes(
        first.document_head, context.state["documents"][alias], ids
    )
    # a trailing condition gates the documents before they complete,
    # so a result limit counts only the documents that pass it
    return execute_parts(
        state, parts, prefixes, DocumentKeys(alias, ids),
        keep_kv=nodes[last].keep_kv,
        prefix_tree=execution_prefix_tree(first, prefixes, state.loaded_model.arena),
        limit=context.state.get("filter_limit"),
        attention=(first.attention or None
                   if isinstance(first, (AiFilter, AiJoin)) else None),
        label=f"pipeline {alias}")
