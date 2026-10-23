"""Quail runtimes for the typed model graph."""

from __future__ import annotations

import time

import pyarrow as pa

from quail.backends.quail.executor.memory import memory_report
from quail.backends.quail.executor.operators import execution_parts
from quail.backends.quail.reranker import FilterRuntime
from quail.backends.quail.retention import apply_retention
from quail.catalog.pairs import columns_key
from quail.execution.metrics import NodeResult, scalar_node_metrics
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    compute_subgraph,
)
from quail.execution.runtimes import ModelNodeRuntime
from quail.execution.types import export_physical_outputs
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiInference,
    AiJoin,
    AiScore,
    Filter,
    PhysicalGraph,
    Scan,
)


def quail_runtimes() -> dict:
    """Return runtimes for the Quail backend's physical nodes."""
    model_runtime = ModelNodeRuntime()
    return {
        AiFilter.runtime_key: model_runtime,
        AiJoin.runtime_key: model_runtime,
        AiScore.runtime_key: model_runtime,
        AiClassify.runtime_key: model_runtime,
        AiExtract.runtime_key: model_runtime,
        Filter.runtime_key: FilterRuntime(),
    }


def prepare_model_inputs(node, inputs, context: ExecutionContext,
                         upstream=None, nodes=None):
    """Convert graph input ports into backend scheduler inputs.

    Args:
        node: Physical operator to execute.
        inputs: Port values supplied by preceding operators.
        context: Execution context containing document tokens and settings.
        nodes: Join predicates sharing one scheduler run.
        upstream: Optional preceding pipeline state for a join, containing its
            documents, document IDs, operator parts, and selected pairs.

    Returns:
        A scheduler input mapping for filters, joins, and scores. Other
        operators receive the original input mapping.
    """
    implementation = execution_parts().get(node.type_name)
    prepared = (None if implementation is None
                else implementation.model_inputs(
                    node, inputs, context, upstream, nodes))
    if prepared is None:
        return inputs
    return {**prepared,
            "gpu_timing": context.state.get("gpu_timing", False)}


def record_model_result(node, result: NodeResult,
                        context: ExecutionContext) -> None:
    """Record reuse of anchor KV supplied by preceding operators."""
    state = context.state
    if isinstance(node, AiJoin) and state["join_streamed"]:
        state["kv_stats"]["join_anchor_hits"] += result.metrics.kv_hits


def execution_context(state, payload, graph: PhysicalGraph) -> ExecutionContext:
    """Bind a query's inputs, retention, and runtimes to one GPU executor.

    The arena is emptied first: nothing survives from the last query.
    """
    arena = state["gpu"].arena
    for key in arena.resident_keys():
        arena.free_key(key)
    arena.reset_stats()
    # the peak memory reported is this query's, not the process's
    state["gpu"].torch.cuda.reset_peak_memory_stats()
    retention = payload.get("retention", {})
    apply_retention(arena, retention, retention.get("initial", {}))
    documents = state["documents"]
    sources = {
        alias: range(len(tokens))
        for alias, tokens in documents.items()
    }
    for alias, table in state.get("columns", {}).items():
        sources[columns_key(alias)] = table
    runtime_state = {
        **state,
        "retention": retention,
        "filter_limit": (
            None if any(isinstance(node, AiJoin)
                        for node in graph.nodes)
            else payload.get("filter_limit")
        ),
        "joins_started": False,
        "kv_stats": {
            "retained_after_filters": 0,
            "retained_pages_after_filters": 0,
            "retained_prefix_tokens_after_filters": 0,
            "join_anchor_hits": 0,
            "join_anchor_misses": 0,
        },
    }
    execution = state["gpu"].execution
    return ExecutionContext(
        runtimes=state["runtimes"],
        model_execution=execution,
        sources=sources,
        model_inputs=prepare_model_inputs,
        model_result=record_model_result,
        state=runtime_state,
        functions=state.get("functions", {}),
        run_pipeline=execution.execute_pipeline,
    )


def execution_measurements(context: ExecutionContext) -> dict:
    """Return one GPU's KV counters and memory report for the query so far."""
    gpu = context.state["gpu"]
    arena = gpu.arena
    return {
        "kv_manager": {
            **context.state["kv_stats"],
            "evicted_keys": arena.evicted_keys,
            "evicted_pages": arena.evicted_pages,
            "evicted_prefix_tokens": arena.evicted_prefix_tokens,
        },
        "memory": memory_report(gpu),
    }


def execute_single_graph(state, payload, graph: PhysicalGraph) -> dict:
    """Execute one Quail model graph on one GPU executor."""
    context = execution_context(state, payload, graph)
    torch = state["gpu"].torch
    arena = state["gpu"].arena
    started = time.perf_counter()
    with torch.inference_mode():
        result = GenericRunner().run(compute_subgraph(graph), context)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    for key in arena.resident_keys():
        arena.free_key(key)
    report = graph_report(graph, result, wall, state["device"],
                          gpu_timing=state.get("gpu_timing", False))
    measurements = execution_measurements(context)
    report.update(kv_manager=measurements["kv_manager"], **measurements["memory"])
    return report


def graph_report(graph, result, wall, device, *, gpus=1, gpu_timing=False):
    """Collect graph outputs, execution metrics, and GPU cost."""
    filters, joins = model_answers(graph, result)
    report = {
        "filters": filters,
        "joins": joins,
        "_outputs": export_physical_outputs(compute_subgraph(graph), result),
        "wall_s": round(wall, 2),
        "fresh_tokens": result.metrics.fresh_tokens,
        "cached_tokens": result.metrics.cached_tokens,
        "evaluated_documents": result.metrics.evaluated_documents,
        "evaluated_document_pairs": result.metrics.evaluated_document_pairs,
        "usd_per_query": (
            None if device.usd_per_hour is None
            else wall / 3600 * gpus * device.usd_per_hour
        ),
        "backend_metrics": {"inferences": [
            dict(value.metrics.extension)
            for node_id, value in result.nodes.items()
            if isinstance(graph.node(node_id), AiInference)
        ]},
        "node_metrics": scalar_node_metrics(result.nodes),
        "executed_join_plan": executed_join_plan(graph),
    }
    report.update(throughput(graph, result.metrics, wall))
    if gpu_timing:
        report["gpu_s"] = round(result.metrics.gpu_s, 3)
        report["chunks"] = result.metrics.chunks
    return report


def throughput(graph, metrics, seconds: float) -> dict:
    """Return the run's throughput under one key.

    A run that evaluated document pairs reports evaluated pairs per
    second. Any other run reports input document rows, summed over
    every scan and counted before filters, per second.
    """
    seconds = max(seconds, 1e-9)
    if metrics.evaluated_document_pairs:
        return {
            "document_pairs_per_second":
                metrics.evaluated_document_pairs / seconds,
        }
    documents = sum(
        node.num_documents for node in graph.nodes if isinstance(node, Scan)
    )
    return {"documents_per_second": documents / seconds}


def model_answers(graph, result) -> tuple[dict, list]:
    """Collect predicate answers in execution order."""
    filters, joins = {}, []
    for node in graph.topological_nodes():
        if isinstance(node, AiFilter):
            filters.setdefault(node.alias, []).append(
                result.nodes[node.node_id].outputs[f"filter_answers:{node.alias}"])
        elif isinstance(node, AiJoin):
            joins.append(result.nodes[node.node_id].outputs[
                f"join_answers:{node.written_position}"])
    return {alias: pa.concat_tables(tables) for alias, tables in filters.items()}, joins


def executed_join_plan(graph) -> list[dict]:
    """Describe the join nodes executed from the saved graph."""
    from quail.physical import Barrier

    return [
        {"type": node.type_name, "id": node.node_id,
         **node.explain_fields()}
        for node in graph.topological_nodes()
        if isinstance(node, (AiJoin, Barrier))
    ]
