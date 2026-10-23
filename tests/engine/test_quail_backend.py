"""Quail backend execution tests without a GPU."""

from types import SimpleNamespace

import pyarrow as pa
import pytest
from helpers.fakes import (
    FakeArena,
    fake_pipeline,
    fake_readout,
    fake_torch,
    keep_even,
    run_graph_on_arena,
    same_key,
    two_alias_graph,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.executor.arena import KVArena
from quail.builtins import built_in_registry
from quail.execution.result import filter_answer_table
from quail.physical import (
    AiFilter,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS


def test_filter_execution_and_retention_inputs(monkeypatch):
    from quail.physical import plan_envelope

    filtered = AiFilter(
        node_id="filter:d", inputs=input_ports((PortRef("input:d", "rows:d"),)),
        alias="d", question_token_ids=(9,), written_position=0, question_tokens=1,
        preamble_tokens=1, selectivity=None, expected_rows=4)
    graph = PhysicalGraph((Scan(node_id="input:d", alias="d", input_id="d"),
                           filtered), PortRef("filter:d", "rows:d"))

    def round_fn(kind, subs):
        assert kind == "execute"
        outputs = []
        for sub in subs:
            indices = sub["inputs"]["filter:d"]["input:0"].ids["d"].to_pylist()
            from quail.execution.metrics import NodeMetrics, NodeResult

            result = NodeResult({
                "rows:d": [index for index in indices if index % 2 == 0],
                "filter_answers:d": filter_answer_table(
                    "d", indices, [0] * len(indices),
                    [index % 2 == 0 for index in indices])},
                NodeMetrics(fresh_tokens=len(indices)))
            outputs.append({
                "results": {"filter:d": result}, "retained": (),
                "kv_manager": {"evicted_keys": 1},
                "memory": {"peak_gib": 1.0,
                           "memory_capacity": {"pages": 16, "slots": 2}},
            })
        return outputs

    registry = built_in_registry()
    payload = {
        "model": "qwen3-4b-fp8", "chunk_tokens": 8192, "true_ids": [1],
        "false_ids": [2], "filter_limit": None,
        "order_rule": "by_cost", "documents": {"d": [[1], [2], [3], [4]]},
        "joins": [], "filters": {}, "filter_arena_writes": {},
        "shards": {"d": ((0, 2), (1, 3))},
        "physical_plan": plan_envelope(
            backend="quail", model="qwen3-4b-fp8", device="h100-sxm",
            gpus=2, graph=graph, codecs=registry.codecs),
    }
    result = execute_distributed_graph(
        payload, graph, 2, round_fn, DEVICES["h100-sxm"], registry.runtimes,
        registry)
    answers = result["filters"]["d"].sort_by("d")
    assert answers.column(2).to_pylist() == [True, False, True, False]
    assert result["fresh_tokens"] == 4
    assert result["kv_manager"]["evicted_keys"] == 2 and result["peak_gib"] == 1.0
    assert result["memory_capacity"] == {"pages": 32, "slots": 4}
    assert len(result["memory_workers"]) == 2

    arena = FakeArena()
    arena.owned.add(("d", 10))

    def fake_run_stages(*args, on_settled, **kwargs):
        on_settled(0, True, [True])
        return [{0: [True]}], [], 3

    monkeypatch.setattr("quail.backends.quail.executor.parts.run_stages",
                        fake_run_stages)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(
        model=object(), arena=arena, pipeline=fake_pipeline())
    execution.bind_query(torch=fake_torch(), async_answers=fake_readout(),
                         answer_rows=object(), chunk_tokens=8192)
    result = execution.execute(filtered, {
        "documents": [[1]], "document_ids": [10]})
    assert not arena.owned
    assert result.outputs["rows:d"].ids["d"].to_pylist() == [10]


def _key_columns(r_keys, p_keys):
    return {alias: pa.table({alias: pa.array(range(len(keys)), pa.int32()),
                             "key": pa.array(keys)})
            for alias, keys in (("r", r_keys), ("p", p_keys))}


def _pairs(table):
    return sorted(zip(table.column("r").to_pylist(), table.column("p").to_pylist()))


def _ids(tables):
    (alias,) = tables
    return tables[alias].column(alias).to_pylist()


def test_pair_join_and_gpu_timing_run_through_the_quail_graph(monkeypatch):
    # document d has key d % 4; partners 0 and 3 share key 0, so a
    # document with key 0 pairs with both and one with key 3 with none
    columns = _key_columns([d % 4 for d in range(14)], [0, 1, 2, 0])
    allowed = {d: [i for i in range(4) if d % 4 == i % 3] for d in range(14)}
    for pipelined in (True, False):
        result, _, filter_truth, join_truth = run_graph_on_arena(
            monkeypatch, two_alias_graph(pipelined, hash_join=True),
            columns=columns)
        cross, _, _, _ = run_graph_on_arena(
            monkeypatch, two_alias_graph(pipelined))
        survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
        stage = result["joins"][0]
        assert sorted(stage["anchor_index"]) == survivors
        members = stage["anchor_partners"]
        for local, document in enumerate(stage["anchor_index"]):
            mine = sorted(allowed[document])
            assert list(members[local]) == mine
            assert stage["rows"].get(local, []) == [
                join_truth[("r", document)][i] for i in mine]
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert _pairs(table) == sorted(
            (d, i) for d in survivors for i in allowed[d])
        assert table.column("answer").to_pylist() == [
            bool(join_truth[("r", d)][i])
            for d, i in zip(table.column("r").to_pylist(),
                            table.column("p").to_pylist())]
        metrics = result["node_metrics"]["group:0"]
        assert metrics["evaluated_document_pairs"] == sum(
            len(allowed[d]) for d in survivors)
        assert metrics["fresh_tokens"] < (
            cross["node_metrics"]["group:0"]["fresh_tokens"])
        assert result["node_metrics"]["hash_join:r-p"]["output_rows"] == sum(
            len(mine) for mine in allowed.values())
        # anchors whose pairs all answered FALSE are gone
        kept = [d for d in survivors
                if any(join_truth[("r", d)][i] for i in allowed[d])]
        root = result["_outputs"][PortRef("group:0", "rows:r")]
        assert sorted(root.column("r").to_pylist()) == kept

    # GPU timing sums the chunk events only when asked
    off, _, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48)
    assert "gpu_s" not in off
    assert all(metrics["gpu_s"] == 0.0
               for metrics in off["node_metrics"].values())
    on, model, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48, gpu_timing=True)
    # join chunks run before the filter sequence finishes: one pipeline
    kinds = [kind for kind, _ in model.launched]
    assert kinds.index("join") < len(kinds) - 1 - kinds[::-1].index("filter")
    # the fake torch reports 2 ms per event pair, one pair per chunk; a
    # chunk that packed rows of two nodes counts for each, and its time
    # is split between them by rows, so the nodes' seconds sum to the
    # pipeline's and their chunks to at least its chunk count
    per_node = {name: metrics for name, metrics in on["node_metrics"].items()
                if metrics["chunks"]}
    assert on["gpu_s"] == pytest.approx(
        sum(metrics["gpu_s"] for metrics in per_node.values()), abs=1e-3)
    assert on["gpu_s"] > 0
    assert len(model.launched) == round(on["gpu_s"] / 0.002)
    assert on["chunks"] >= len(model.launched)
    assert all(0 < metrics["gpu_s"] <= metrics["chunks"] * 0.002 + 1e-9
               for metrics in per_node.values())


@pytest.mark.parametrize("semantics, keep_kv",
                         [("full", False), ("anti", True)])
def test_a_join_without_a_filter_settles_anchors_with_no_pairs(
        monkeypatch, semantics, keep_kv):
    retained = []
    retain = KVArena.retain

    def record_retention(arena, key, *args, **kwargs):
        retained.append(key)
        return retain(arena, key, *args, **kwargs)

    monkeypatch.setattr(KVArena, "retain", record_retention)
    # document d has key d % 4 and partner i key i % 3: a document with
    # key 3 has no pairs and settles without loading its KV
    columns = _key_columns([d % 4 for d in range(14)], [0, 1, 2, 0])
    allowed = {d: [i for i in range(4) if d % 4 == i % 3] for d in range(14)}
    result, _, _, join_truth = run_graph_on_arena(
        monkeypatch, two_alias_graph(
            False, hash_join=True, filtered=False, semantics=semantics,
            keep_kv=keep_kv),
        columns=columns, streamed_anchors=False)
    assert len(retained) == len(set(retained))
    assert bool(retained) == keep_kv
    stage = result["joins"][0]
    assert sorted(stage["anchor_index"]) == list(range(14))
    for local, document in enumerate(stage["anchor_index"]):
        assert stage["rows"].get(local, []) == [
            join_truth[("r", document)][i] for i in allowed[document]]
    table = result["_outputs"][PortRef("group:0", "join_answers:0")]
    assert _pairs(table) == sorted(
        (d, i) for d in range(14) for i in allowed[d])


def _foreign_run(monkeypatch, graph, functions):
    # document d has key d % 4; partner i has key i
    result, _, filter_truth, join_truth = run_graph_on_arena(
        monkeypatch, graph, seed=9, functions=functions,
        columns=_key_columns([d % 4 for d in range(14)], list(range(4))))
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    return result, survivors, join_truth


def test_foreign_runs_per_batch_or_once_as_a_barrier_and_are_checked(monkeypatch):
    functions = {"keep_even": keep_even, "same_key": same_key}
    # a per-batch drop inside the pipeline: only even survivors reach the join
    result, survivors, join_truth = _foreign_run(
        monkeypatch, two_alias_graph(True, foreign=("per_batch", "drop")),
        functions)
    assert sorted(result["filters"]["r"].column("r").to_pylist()) == list(range(14))
    stage = result["joins"][0]
    kept = [d for d in survivors if d % 2 == 0]
    assert sorted(stage["anchor_index"]) == kept
    foreign = result["node_metrics"]["apply:keep_even"]
    assert foreign["input_rows"] == len(survivors)
    assert foreign["output_rows"] == len(kept)
    assert result["_outputs"][PortRef("apply:keep_even", "rows:r")].column(
        "r").to_pylist() == kept
    assert result["node_metrics"]["group:0"]["kv_hits"] == len(kept)

    result, survivors, _ = _foreign_run(
        monkeypatch, two_alias_graph(False, foreign=("barrier", "drop")),
        functions)
    assert sorted(result["joins"][0]["anchor_index"]) == kept

    # per-batch and barrier pair functions both pair document d with d % 4
    per_batch, survivors, join_truth = _foreign_run(
        monkeypatch, two_alias_graph(True, foreign=("per_batch", "pairs")),
        functions)
    barrier, _, _ = _foreign_run(
        monkeypatch, two_alias_graph(False, foreign=("barrier", "pairs")),
        functions)
    for result in (per_batch, barrier):
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert _pairs(table) == [(d, d % 4) for d in survivors]
        assert {(r, p): a for r, p, a in zip(
            table.column("r").to_pylist(), table.column("p").to_pylist(),
            table.column("answer").to_pylist())} == {
            (d, d % 4): bool(join_truth[("r", d)][d % 4]) for d in survivors}
        pairs = result["_outputs"][PortRef("apply:same_key", "pairs:0")]
        assert _pairs(pairs) == [(d, d % 4) for d in survivors]
        assert result["node_metrics"]["apply:same_key"]["output_rows"] == len(
            survivors)

    for kind, ids, function, message in [
        ("per_batch", "drop", lambda tables: [99], "never invents an id"),
        ("barrier", "preserve", lambda tables: _ids(tables)[1:],
         "preserves ids but dropped"),
        ("barrier", "drop", lambda tables: [None] + _ids(tables)[1:],
         "returned a null id"),
    ]:
        graph = two_alias_graph(kind == "per_batch", foreign=(kind, ids))
        with pytest.raises(ValueError, match=message):
            _foreign_run(monkeypatch, graph, {"keep_even": function})


def test_predicate_nodes_keep_documents_and_join_frames_in_kv(monkeypatch):
    from dataclasses import replace

    from quail.planner.plan import plan_kv_retention

    graph = two_alias_graph(True, stages=2)
    graph = PhysicalGraph(tuple(
        replace(node, share_prefixes=True)
        if isinstance(node, AiFilter) else node
        for node in graph.nodes), graph.root)
    first_join = graph.nodes[-1]
    second_join = replace(
        first_join, node_id="join:second", written_position=1, exec_index=1,
        inputs=input_ports((PortRef(first_join.node_id, "rows:r"),
                            PortRef("input:p", "rows:p"))))
    graph = PhysicalGraph((*graph.nodes, second_join),
                          PortRef(second_join.node_id, "rows:r"))
    graph = plan_kv_retention(graph)
    result, model, filters, joins = run_graph_on_arena(
        monkeypatch, graph, stages=2, shared_prefix=16)
    prefixes = [(spec["key"], spec["prefix"])
                for _, chunk in model.launched for spec in chunk
                if spec["prefix"] is not None]
    assert len(prefixes) == len({key for key, _ in prefixes}) == len(filters)
    survivors = [document for document, answers in enumerate(filters)
                 if all(answers) and any(joins[("r", document)])]
    metrics = result["node_metrics"]
    assert metrics["filter:r"]["borrowed_prefix_tokens"] > 0
    # The second join uses the same frame and reads only partner suffixes.
    assert metrics["join:second"]["fresh_tokens"] == len(survivors) * 4 * 20
    assert metrics["filter:r:1"]["fresh_tokens"] == sum(row[0] for row in filters)
