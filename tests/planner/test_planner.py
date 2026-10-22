"""Planner ordering, anchor selection, sharding, break-evens, and refusals."""

import itertools
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import keep_even, register_claims_evidence, two_alias_graph
from helpers.planning import _optimize, _plan, node_kinds, token_lengths

import quail
from quail.builtins import built_in_registry
from quail.catalog import Catalog, DocumentProvider
from quail.catalog.tokens import TokenLengths
from quail.cost.filters import filter_cost
from quail.cost.joins import PricedJoin
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, ask, scan, stream
from quail.explain import explain, physical_tree
from quail.frontend.builder import docs
from quail.frontend.expressions import col, prompt
from quail.logical import ModelCall, SemanticJoin
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Foreign,
    PhysicalGraph,
    PortRef,
    Project,
    Scan,
)
from quail.physical.base import input_ports
from quail.physical.codec import encode_fields
from quail.physical.pipelines import build_pipelines
from quail.planner.operators.filter import prepare_filter_costs
from quail.planner.optimize import refine_plan
from quail.planner.plan import EngineConfig, Refusal
from quail.planner.rules import (
    ShouldEliminateDistinct,
    ShouldPushDownLimit,
    ShouldRetainKv,
)
from quail.planner.search.filter_order import order_filters_indexed
from quail.planner.statistics.tables import AliasStats, summarize_alias
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8


def order_filters(predicates, rule, **kwargs):
    costs = prepare_filter_costs(predicates, lower=lambda prompt: prompt, **kwargs)
    return [predicates[i] for i in order_filters_indexed(costs, rule)]

def filter_sequence(plan, alias=None):
    return next(n for n in plan.nodes if isinstance(n, AiFilter)
                and (alias is None or n.alias == alias))


def join_stages(plan):
    return [encode_fields(node, exclude=("node_id", "inputs"))
            for node in plan.nodes if isinstance(node, AiJoin)]


def _sink_source(plan):
    sink = next(node for node in plan.nodes if isinstance(node, Project))
    return sink.inputs[0].source


def _parquet(path, columns):
    pq.write_table(pa.table({c: ["x"] for c in columns}), str(path))
    return str(path)


@pytest.fixture()
def catalog(tmp_path):
    cat = Catalog()
    cat.register("reviews", DocumentProvider.from_parquet(
        _parquet(tmp_path / "r.parquet", ["id", "review"]), id_col="id"))
    cat.register("products", DocumentProvider.from_parquet(
        _parquet(tmp_path / "p.parquet", ["asin", "description"]),
        id_col="asin"))
    cat.register("threads", DocumentProvider.from_parquet(
        _parquet(tmp_path / "t.parquet", ["id", "thread"]), id_col="id"))
    return cat


def _cell(text, title):
    for line in text.splitlines():
        if line.strip().startswith(title):
            return line.split(title, 1)[1].split()
    raise AssertionError(f"no row starts with {title!r}")


def _five_filter_plan(catalog, sels):
    q = docs(catalog, "reviews").alias("r")
    for j, s in enumerate(sels):
        q = q.ai_filter(prompt(f"flag {j} of: {{0}}", col("r.review")),
                        selectivity=s)
    return q.select("r.id")


def test_gpu_copies_and_memory_refusals(catalog):
    from quail.planner.graph.shards import contiguous_shards

    assert contiguous_shards([2, 2, 2], 2) == (((0, 1), (1, 3)), [2, 4])
    assert contiguous_shards([1, 0, 9], 2) == (((0, 1), (1, 3)), [1, 9])
    assert contiguous_shards([0, 2, 0, 4], 3) == (
        ((0, 2), (2, 4), (4, 4)), [2, 4, 0])
    assert contiguous_shards([], 2) == (((0, 0), (0, 0)), [0, 0])
    logical = _five_filter_plan(catalog, (0.5,))
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        for gpus in (1, 2, 4, 8):
            plan = _plan(logical, {"r": [100] * 8}, model=model, gpus=gpus)
            assert plan.model == model.name
            assert plan.gpus == gpus
            scan_node = next(node for node in plan.nodes if isinstance(node, Scan))
            assert len(scan_node.shards) == gpus

    big = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    logical = _five_filter_plan(catalog, (0.9,))
    for gpus in (1, 8):
        result = _plan(logical, {"r": [100] * 10}, model=big, gpus=gpus)
        assert isinstance(result, Refusal)
        assert result.constraint == "weights_need_more_cards"
        assert result.needed > result.available

    r = _plan(logical, {"r": [150_000]})
    assert isinstance(r, Refusal)
    assert r.constraint == "suffix_over_chunk"


def test_filter_ordering_and_kv_writes(catalog):
    # the B3 shape: a 0.2-selectivity filter written third among five
    sels = (0.9, 0.9, 0.2, 0.9, 0.9)
    logical = _five_filter_plan(catalog, sels)
    tokens = {"r": [400] * 100}

    by_cost = _plan(logical, tokens)
    sequence = filter_sequence(by_cost)
    assert by_cost.settings["order_rule"] == "by_cost"
    assert sequence.selectivity == 0.2
    second = by_cost.graph.nodes_by_type(AiFilter.type_name)[1]
    assert second.expected_rows == pytest.approx(20.0)
    # the physical planner chooses the order; the logical plan keeps
    # the predicates as written
    optimized, _ = _optimize(logical, tokens)
    assert sequence.written_position == 2
    assert "predicate 3" in explain(optimized, by_cost)

    as_written = _plan(logical, tokens, order="as_written")
    assert [node.selectivity
            for node in as_written.graph.nodes_by_type(AiFilter.type_name)] \
        == list(sels)

    class Predicate:
        def __init__(self, tail, selectivity):
            self.prompt = type("Prompt", (), {
                "tail_tokens": tail, "preamble_tokens": 0})()
            self.expression = ModelCall(self.prompt)
            self.selectivity = selectivity

    first, second = Predicate(10, 0.5), Predicate(10, 0.5)
    kwargs = dict(mean_tokens=400, model=QWEN3_4B_FP8,
                  device=H100_SXM, chunk_tokens=110_376)
    assert order_filters([first, second], "by_cost", **kwargs) == \
        [first, second]
    never_kills = Predicate(1, 1.0)
    assert order_filters([never_kills, first], "by_cost", **kwargs) == \
        [first, never_kills]

    predicates = [Predicate(12, 0.8), Predicate(50, 0.1), Predicate(8, 1.0),
                  Predicate(25, 0.0)]
    ask_costs = [filter_cost(predicate.prompt.tail_tokens, first=False, **kwargs)
                 for predicate in predicates]
    scan_costs = [filter_cost(predicate.prompt.tail_tokens, first=True, **kwargs)
                  for predicate in predicates]

    candidates = []
    for first in range(len(predicates)):
        def score(index):
            rejected = 1.0 - predicates[index].selectivity
            return ask_costs[index] / rejected if rejected else float("inf")

        remaining = sorted(
            (index for index in range(len(predicates)) if index != first),
            key=score)
        order = [first, *remaining]
        expected = 0.0
        expected_rows = 1.0
        for position, index in enumerate(order):
            expected += expected_rows * (scan_costs[index] if position == 0
                                else ask_costs[index])
            expected_rows *= predicates[index].selectivity
        candidates.append((expected, order))

    expected_order = min(candidates)[1]
    actual = order_filters(predicates, "by_cost", **kwargs)
    assert actual == [predicates[index] for index in expected_order]

    logical = (docs(catalog, "reviews").alias("r")
               .ai_filter(prompt("long " * 100 + "{0}", col("r.review")),
                          selectivity=0.1)
               .ai_filter(prompt("short " * 10 + "{0}", col("r.review")),
                          selectivity=0.5).select("r.id"))
    plan = _plan(logical, {"r": [400] * 100})
    assert [node.written_position
            for node in plan.graph.nodes_by_type(AiFilter.type_name)] == [1, 0]

    # one stage: nothing reads the KV again
    single = _five_filter_plan(catalog, (0.9,))
    plan = _plan(single, {"r": [400] * 100})
    sequence = filter_sequence(plan)
    assert sequence.arena_writes is False
    assert "arena_writes=False" in explain(single, plan, verbose=True)

    plan = _plan(_five_filter_plan(catalog, (0.9, 0.9)), {"r": [400] * 100})
    sequence = filter_sequence(plan)
    assert sequence.arena_writes is True

    logical = (docs(catalog, "reviews").alias("r")
               .ai_filter(prompt("a: {0}", col("r.review")))
               .ai_filter(prompt("b: {0}", col("r.review")),
                          selectivity=0.5)
               .select("r.id"))
    plan = _plan(logical, {"r": [100] * 10})
    assert plan.settings["order_rule"] == "by_cost"
    assert "selectivity 0.2" in plan.settings["order_source"]


ROW_ESTIMATES = [
    ((0.5, 0.25), "12.5"), ((1.0, 0.0), "0"),
    ((None, 0.5), "10"), ((None, 0.0), "0"),
]


def test_explain_row_estimates_limits_and_verbose_fields(catalog):
    for sels, expected in ROW_ESTIMATES:
        logical = _five_filter_plan(catalog, sels)
        plan = _plan(logical, {"r": [400] * 100})
        text = explain(logical, plan)
        physical = text.split("physical:", 1)[1]
        assert _cell(physical, "Project: r.id") == [expected], sels
        assert _cell(physical, "AiFilter: r")[0] == expected, sels
        assert _cell(physical, "AiFilter: r")[-1] in ("ms", "s"), sels
        assert _cell(physical, "Scan reviews as r") == ["100"], sels
        assert "tokens=40,000" in physical, sels
        assert "node_id=" not in text and "arena_writes" not in text, sels
        assert "KV=bf16, chunk budget=" in text and "KV rewind=on" in text, sels
        for stage in plan.graph.nodes_by_type(AiFilter.type_name):
            predicate = logical.root.input.predicates[stage.written_position]
            assert (f"predicate {stage.written_position + 1}  "
                    f"PROMPT({predicate.prompt.template!r})") in physical, sels
        verbose = explain(logical, plan, verbose=True)
        assert "node_id=ai_filter:r" in verbose, sels
        assert "admission_tokens=" in verbose and "expected_rows=" in verbose, sels

    logical = _five_filter_plan(catalog, (0.25,))
    logical = replace(logical, root=replace(logical.result, limit=10))
    plan = _plan(logical, {"r": [400] * 100})
    physical = explain(logical, plan).split("physical:", 1)[1]
    assert _cell(physical, "Limit: 10") == ["10"]
    assert _cell(physical, "Project: r.id") == ["25"]
    assert "KV: not stored" in physical

    rules = (ShouldEliminateDistinct(), ShouldPushDownLimit(), ShouldRetainKv())
    assert "rules" not in explain(logical, plan, rules=rules)
    text = explain(logical, plan, verbose=True, rules=rules)
    assert "  logical: should_eliminate_distinct" in text
    assert "  physical: should_push_down_limit, should_retain_kv" in text
    assert ("refine each graph in order: should_push_down_limit, "
            "should_retain_kv") in text


def mask_pairs(prefix, suffix, window):
    """Count allowed keys for every new query position."""
    return sum(1 for q in range(prefix, prefix + suffix) for k in range(q + 1)
               if not window or q - k < window)


def test_work_matches_attention_masks_and_component_costs():
    from quail.cost.dense_decoder_cost import (
        dense_decoder_components,
        dense_params,
    )
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    for window, prefix, suffix in ((0, 2, 4), (1, 9, 3), (4, 0, 3),
                                  (4, 2, 4), (4, 4, 1), (4, 9, 3)):
        first = scan(prefix, suffix, window=window)
        continuation = ask(prefix, suffix, window=window)
        assert first.pairs == mask_pairs(0, prefix + suffix, 0)
        assert continuation.pairs == mask_pairs(prefix, suffix, 0)
        assert first.sliding_pairs == (
            mask_pairs(0, prefix + suffix, window) if window else 0)
        assert continuation.sliding_pairs == (
            mask_pairs(prefix, suffix, window) if window else 0)
        assert continuation.sliding_kv_read == (
            min(prefix, window - 1) if window else 0)
        branches = stream(prefix, [suffix, suffix + 1], window=window)
        assert branches.sliding_pairs == (
            sum(mask_pairs(prefix, s, window) for s in (suffix, suffix + 1))
            if window else 0)
        assert branches.sliding_kv_read == continuation.sliding_kv_read

    work = ask(400, 50) * 1000
    result = speed_of_light(work, QWEN3_4B_FP8, H100_SXM, 110_376)
    assert result.bound_by == "mixed"
    assert [c.name for c in result.components] == [
        "attn_proj", "mlp", "attention"]
    assert result.seconds == pytest.approx(sum(
        max(c.compute_seconds, c.memory_seconds)
        for c in result.components))
    assert result.component("mlp") is result.components[1]
    assert result.seconds > max(result.compute, result.memory)
    dense_bytes = sum(
        c.bytes_moved for c in result.components if c.name != "attention")
    assert dense_bytes == dense_params(QWEN3_4B_FP8) * result.passes
    larger_resident_copy = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    assert speed_of_light(
        work, larger_resident_copy, H100_SXM, 110_376).seconds \
        == pytest.approx(result.seconds)

    model = DIFFUSION_GEMMA_26B_FP8
    work = ask(4096, 16, window=model.sliding_window)
    attention = dense_decoder_components(work, model, 1)[2]
    assert attention.flops == 4 * 16 * (
        5 * 512 * work.pairs + 25 * 256 * work.sliding_pairs)
    assert attention.bytes_moved == (
        5 * 2 * 2 * 512 * 2 * (16 + 4096)
        + 25 * 2 * 8 * 256 * 2 * (16 + 1023))
    assert work + work == work * 2
    assert not work.dominates(replace(work, sliding_pairs=work.sliding_pairs - 1))
    assert not work.dominates(replace(work, sliding_kv_read=1022))


def _join_sequence(catalog, sels=(0.1, 0.1), anchors=(None, None)):
    return (docs(catalog, "reviews").alias("r")
            .ai_join(docs(catalog, "threads").alias("t"),
                     prompt("m1 {0} {1}", col("r.review"),
                            col("t.thread")),
                     selectivity=sels[0], anchor=anchors[0])
            .ai_join(docs(catalog, "products").alias("p"),
                     prompt("m2 {0} {1}", col("t.thread"),
                            col("p.description")),
                     selectivity=sels[1], anchor=anchors[1])
            .select("r.id", "t.id", "p.asin"))


def test_join_anchors_groups_forced_order_and_later_kv_reuse(catalog):
    logical = (docs(catalog, "reviews").alias("r")
               .ai_join([docs(catalog, "products").alias("p"),
                         docs(catalog, "threads").alias("t")],
                        prompt("m {0} {1} {2}", col("r.review"),
                               col("p.description"), col("t.thread")),
                        selectivity=0.02)
               .select("r.id"))
    tokens = {"r": [3000] * 20, "p": [100] * 30, "t": [50] * 40}
    plan = _plan(logical, tokens)
    stage = join_stages(plan)[0]
    assert stage["expected_rows"] == 20 * 30 * 40
    assert stage["anchor"] == "r"
    assert stage["partners"] == ["p", "t"]

    # streaming r's 3,000-token documents as partners would pay them once
    # per pair, so r anchors stage 1 and p stage 2, with a barrier between
    tokens = {"r": [3000] * 10, "t": [50] * 8, "p": [100] * 6}
    plan = _plan(_join_sequence(catalog), tokens, order="as_written")
    stages = join_stages(plan)
    assert [s["anchor"] for s in stages] == ["r", "p"]
    assert [s["written_position"] for s in stages] == [0, 1]
    # the physical planner chooses the stage order and anchors; the
    # logical plan keeps the joins as written
    optimized, _ = _optimize(_join_sequence(catalog), tokens, order="as_written")
    joins = [node for node in optimized.walk() if isinstance(node, SemanticJoin)]
    assert [join.anchor for join in joins] == [None, None]
    assert [(s["exec_index"], s["anchor"]) for s in stages] == [(0, "r"), (1, "p")]
    text = explain(optimized, plan)
    assert "AiJoin: anchor=r" in text and "AiJoin: anchor=p" in text
    assert stages[0]["partners"] == stages[1]["partners"] == ["t"]
    assert stages[0]["expected_rows"] == 10 * 8
    assert stages[1]["expected_rows"] < 8 * 6
    kinds = node_kinds(plan)
    assert (kinds.count("AiJoin"), kinds.count("Barrier")) == (2, 1)
    barrier = plan.graph.nodes_by_type(Barrier.type_name)[0]
    assert barrier.next_anchor == "p"
    assert set(barrier.aliases) == {"r", "t", "p"}
    group2 = plan.graph.nodes_by_type(AiJoin.type_name)[1]
    assert all(input_port.source.node_id == barrier.node_id
               for input_port in group2.inputs)

    # t, shared by both joins, has the long documents: one group anchors it
    tokens = {"r": [50] * 10, "t": [3000] * 8, "p": [50] * 6}
    plan = _plan(_join_sequence(catalog), tokens, order="as_written")
    assert [s["anchor"] for s in join_stages(plan)] == ["t", "t"]
    kinds = node_kinds(plan)
    assert (kinds.count("AiJoin"), kinds.count("Barrier")) == (2, 0)

    # a forced anchor is honored
    tokens = {"r": [3000] * 10, "t": [50] * 8, "p": [100] * 6}
    plan = _plan(
        _join_sequence(catalog, anchors=("t", None)), tokens, order="as_written"
    )
    assert join_stages(plan)[0]["anchor"] == "t"
    same = _plan(
        _join_sequence(catalog, anchors=("r", None)), tokens, order="as_written"
    )
    assert join_stages(same)[0]["anchor"] == "r"

    # by_cost runs the cheap .01 exists gate before the .9 full join
    logical = (docs(catalog, "reviews").alias("r")
               .ai_join(docs(catalog, "products").alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.9)
               .ai_join(docs(catalog, "threads").alias("t"),
                        prompt("m {0} {1}", col("r.review"),
                               col("t.thread")),
                        selectivity=0.01, semantics="exists")
               .select("r.id"))
    tokens = {"r": [400] * 100, "p": [100] * 100, "t": [100] * 100}
    plan = _plan(logical, tokens)
    assert [(s["written_position"], s["exec_index"], s["anchor"])
            for s in join_stages(plan)] == [(1, 0, "r"), (0, 1, "r")]

    as_written = _plan(logical, tokens, order="as_written")
    assert join_stages(as_written)[0]["selectivity"] == 0.9

    assert _sink_source(plan).node_id == "recombine"
    groups = plan.graph.nodes_by_type(AiJoin.type_name)
    assert len(groups) == 2
    assert [group.anchor for group in groups] == ["r", "r"]
    assert groups[0].keep_kv is True
    assert groups[1].keep_kv is False
    assert groups[1].anchor_resident == "kept"


def test_kv_retention_rule_schedules_the_kv_later_stages_read(
        catalog, monkeypatch):
    import quail.planner.search.join_order as joinsearch

    searches = []
    search_joins = joinsearch.search_joins

    def counting_search(*args, **kwargs):
        searches.append(kwargs.get("honor_forced", True))
        return search_joins(*args, **kwargs)

    monkeypatch.setattr(joinsearch, "search_joins", counting_search)

    # the gate runs first and the full join after it, both anchored on
    # r: the first group keeps r's KV for the second
    logical = (docs(catalog, "reviews").alias("r")
               .ai_join(docs(catalog, "products").alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.9)
               .ai_join(docs(catalog, "threads").alias("t"),
                        prompt("m {0} {1}", col("r.review"),
                               col("t.thread")),
                        selectivity=0.01, semantics="exists")
               .select("r.id"))
    tokens = {"r": [400] * 100, "p": [100] * 100, "t": [100] * 100}
    plan = _plan(logical, tokens)
    # the physical planner runs the join search once
    assert searches == [True]
    retention = plan.settings["retention"]
    assert set(retention) >= {"initial", "before", "after", "cap_pages",
                              "linear_seconds", "pair_seconds"}
    groups = plan.graph.nodes_by_type(AiJoin.type_name)
    assert [group.anchor for group in groups] == ["r", "r"]
    assert set(retention["after"]) == set(retention["before"]) == {
        group.node_id for group in groups}
    assert retention["initial"] == {"r": [1.0, 0]}
    assert retention["after"][groups[0].node_id] == {"r": [1.0, 1]}
    assert retention["after"][groups[1].node_id] == {}
    assert [group.keep_kv for group in groups] == [True, False]

    # on one GPU a filtered anchor streams into its join, so its sequence
    # keeps no KV; on two GPUs the sequence keeps its survivors for the
    # join, and writes KV pages to do so
    filtered = (docs(catalog, "reviews").alias("r")
                .ai_filter(prompt("negative: {0}", col("r.review")),
                           selectivity=0.5)
                .ai_join(docs(catalog, "products").alias("p"),
                         prompt("about {0} {1}", col("r.review"),
                                col("p.description")), selectivity=0.1)
                .select("r.id", "p.asin"))
    one = _plan(filtered, tokens)
    assert one.settings["retention"]["initial"] == {}
    sequence = filter_sequence(one, "r")
    assert (sequence.keep_kv, sequence.arena_writes) == (False, True)
    assert tuple(
        member.node_id for member in build_pipelines(one.graph)[sequence.node_id]) == (
        "ai_filter:r", "ai_join:r")
    two = _plan(filtered, tokens, gpus=2)
    assert two.settings["retention"]["initial"] == {"r": [1.0, 0]}
    sequence = filter_sequence(two, "r")
    assert (sequence.keep_kv, sequence.arena_writes) == (True, True)

    # a forced anchor takes one search too; refining a planned query
    # on exact tokens keeps the schedule it was made with
    searches.clear()
    tokens = {"r": [3000] * 10, "t": [50] * 8, "p": [100] * 6}
    forced = _plan(_join_sequence(catalog, anchors=("t", None)), tokens)
    assert searches == [True]
    refined = refine_plan(
        forced, model=QWEN3_4B_FP8, device=H100_SXM,
        document_tokens=token_lengths(tokens),
        registry=built_in_registry())
    assert refined.settings["retention"] == forced.settings["retention"]
    assert refined.nodes == forced.nodes


def _join_search_spec(position, aliases, anchor):
    return PricedJoin(
        written_position=position, aliases=tuple(aliases), anchor=anchor,
        anchor_free=False, semantics="full", selectivity=None,
        frame_tokens={alias: 5 for alias in aliases},
        label_tokens={alias: 4 for alias in aliases}, tail_tokens=6)


def test_join_survivors_count_candidate_pairs_without_rounding_to_zero():
    import math

    from quail.cost import joins
    from quail.planner.search.join_order import stage_work
    from quail.planner.statistics.tables import surviving_documents, thin

    assert surviving_documents(1, 1e20, 1e-20) == pytest.approx(1 - math.exp(-1))
    assert surviving_documents(5, 0, 0.5) == 0
    # an anchor matches when the equality keeps a pair and the model accepts it
    spec = replace(_join_search_spec(0, ("a", "b"), "a"), semantics="exists",
                   pair_fraction=0.25, selectivity=0.5)
    expected_rows = {"a": 2.0, "b": 3.0}
    thin(expected_rows, spec)
    assert expected_rows == {"a": pytest.approx(2 * (1 - 0.875 ** 3)), "b": 3.0}
    # the join reads only the anchors that have a candidate partner
    lengths = {"a": summarize_alias(TokenLengths([50] * 2)),
               "b": summarize_alias(TokenLengths([30] * 3))}
    expected_rows = {"a": 2.0, "b": 3.0}
    work = stage_work(spec, "a", expected_rows, lengths, 7)
    assert work == joins.stage_work(spec, "a", 2 * (1 - 0.75 ** 3),
                                    2 * 3 * 0.25, lengths, 7)
    assert stage_work(spec, "a", {"a": 2.0, "b": 0.0}, lengths, 7) == Work()


def test_join_search_matches_enumeration_and_tracks_residency():
    from quail.planner.search.join_order import (
        anchor_fits,
        search_joins,
        walk,
    )

    specs = [PricedJoin(written_position=index, aliases=aliases, anchor=aliases[0],
                        anchor_free=True, semantics="full",
                        selectivity=selectivity,
                        frame_tokens={alias: 8 for alias in aliases},
                        label_tokens={alias: 4 for alias in aliases},
                        tail_tokens=2)
             for index, (aliases, selectivity) in enumerate([
                 (("a", "b"), 0.01), (("b", "c"), 0.1),
             ])]
    lengths = {alias: summarize_alias(TokenLengths(tokens)) for alias, tokens in [
        ("a", [800] * 20), ("b", [100] * 30), ("c", [400] * 10),
    ]}
    filtered = {"a", "c"}
    expected_rows = {"a": 10.0, "b": 6.0, "c": 8.0}
    result = search_joins(specs, expected_rows, lengths, filtered, 5, 8192,
                          QWEN3_4B_FP8, H100_SXM)
    costs = []
    for order in itertools.permutations(specs):
        for anchors in itertools.product(*(spec.aliases for spec in order)):
            work, records = walk(list(zip(order, anchors)), expected_rows, lengths,
                                 filtered, 5, QWEN3_4B_FP8, H100_SXM)
            seen = set()
            for record in records:
                if record.anchor in seen:
                    assert record.resident != "filter"
                seen.add(record.anchor)
            costs.append(speed_of_light(work, QWEN3_4B_FP8, H100_SXM, 8192).seconds)
    assert speed_of_light(
        result.work, QWEN3_4B_FP8, H100_SXM, 8192).seconds == min(costs)

    specs = [
        _join_search_spec(0, ("a", "b"), "a"),
        _join_search_spec(1, ("b", "c"), "b"),
        _join_search_spec(2, ("a", "c"), "a"),
    ]
    expected_rows = {"a": 3.0, "b": 3.0, "c": 2.0}
    lengths = {"a": TokenLengths([90, 100, 110]), "b": TokenLengths([400] * 3),
               "c": TokenLengths([50] * 2)}
    assert anchor_fits(specs[0], "a", lengths, 10, 100_000)
    assert not anchor_fits(specs[0], "a", lengths, 10, 500)
    current = search_joins(
        specs, expected_rows, lengths, {"a"}, 10, 100_000,
        QWEN3_4B_FP8, H100_SXM, fixed_order=True)
    # a's filter computed its prefixes: its first anchor use pays no
    # prefix, and neither does its later use (kept after group 0)
    assert current.steps[0].resident == "filter"
    assert current.steps[0].resident_documents == 3
    assert current.steps[1].resident == "none"
    assert current.steps[2].resident == "kept"
    assert current.steps[2].resident_documents == 3

    specs = [
        _join_search_spec(0, ("a", "b"), "b"),
        _join_search_spec(2, ("c", "d"), "c"),
    ]
    expected_rows = {alias: 2.0 for alias in "abcd"}
    lengths = {alias: TokenLengths([100, 120]) for alias in "abcd"}

    assert search_joins(specs, expected_rows, lengths, {}, 10, 100_000,
                        QWEN3_4B_FP8, H100_SXM) is None
    found = search_joins(specs, expected_rows, lengths, {}, 10, 100_000,
                         QWEN3_4B_FP8, H100_SXM, already_joined={"b", "c"})
    assert found is not None
    assert {position for position, _ in found.seq} == {0, 2}

    million = AliasStats(count=1_000_000, total=100_000_000,
                         squared=10_000_000_000, maximum=100)
    stats = {alias: million for alias in "abcd"}
    expected_rows = {alias: 1_000_000.0 for alias in "abcd"}
    specs = [
        _join_search_spec(0, ("a", "b"), "a"),
        _join_search_spec(1, ("b", "c"), "b"),
        _join_search_spec(2, ("c", "d"), "c"),
    ]
    found = search_joins(specs, expected_rows, stats, {}, 10, 100_000,
                         QWEN3_4B_FP8, H100_SXM)
    assert found is not None
    assert len(found.seq) == 3


def test_closed_form_join_work_sums_the_per_row_work():
    """stage_work equals the per-row pricing the estimator uses, on uniform lengths."""
    from quail.cost.joins import partner_stream_work, stage_work
    from quail.cost.work import ask, scan

    spec = _join_search_spec(0, ("a", "b"), "a")
    pre, anchor_len, partner_len, documents, partners = 7, 50, 30, 4, 3
    lengths = {"a": summarize_alias(TokenLengths([anchor_len] * documents)),
               "b": summarize_alias(TokenLengths([partner_len] * partners))}
    frame = spec.frame_tokens["a"] + spec.label_tokens["b"]
    suffixes = [spec.tail_tokens + partner_len] * partners
    for resident in (False, True):
        closed = stage_work(spec, "a", float(documents), float(documents * partners),
                            lengths, pre, resident=resident)
        prefix = pre + anchor_len
        per_row = sum(
            (((ask(prefix, frame) if resident else scan(prefix, frame))
              + partner_stream_work(prefix + frame, suffixes))
             for _ in range(documents)), Work())
        for field in ("tokens", "pairs", "kv_written", "kv_read"):
            assert getattr(closed, field) == pytest.approx(
                getattr(per_row, field)), (field, resident)


def _claims_session(gpus=1, backend="quail"):
    session = quail.Session(
        EngineConfig(gpus=gpus, model="qwen3-4b-fp8", backend=backend,
                     device="h100-sxm"),
        tokenizer=lambda text: list(text.encode()))
    register_claims_evidence(session, claim_words=200, text_words=10)
    return session


def _apply_query(session, kind):
    claims = (session.docs("claims").alias("c")
              .ai_filter(prompt("about a person: {0}", col("c.claim")),
                         selectivity=0.5)
              .apply(keep_even, columns=[col("c.url")], kind=kind))
    return (claims.join(session.docs("evidence").alias("e"))
            .ai_filter(prompt("{1} supports {0}", col("c.claim"),
                              col("e.text")), selectivity=0.5)
            .select("c.id", "e.id"))


def test_foreign_node_placement_and_pipelines():
    # claims are the long side, so the planner anchors on them
    with _claims_session() as session:
        per_batch = _apply_query(session, "per_batch").plan()
        sequence = per_batch.graph.node("ai_filter:c")
        foreign = per_batch.graph.node("apply:keep_even")
        join = per_batch.graph.nodes_by_type(AiJoin.type_name)[0]
        # the sequence, the per-batch apply, and the join run as one pipeline
        assert tuple(
            member.node_id for member in build_pipelines(
                per_batch.graph)[sequence.node_id]) == (
            sequence.node_id, foreign.node_id, join.node_id)
        assert isinstance(foreign, Foreign) and foreign.kind == "per_batch"
        assert foreign.inputs[0].source == PortRef("ai_filter:c", "rows:c")
        assert PortRef("apply:keep_even", "rows:c") in {
            port.source for port in join.inputs}
        assert "keep_even" in session.registry.functions
        text = physical_tree(per_batch.graph)
        assert "Foreign: keep_even (per_batch, drop) on c" in text
        # a barrier needs every survivor at once: the sequence materializes
        barrier = _apply_query(session, "barrier").plan()
        assert tuple(
            member.node_id for member in build_pipelines(
                barrier.graph)["ai_filter:c"]) == (
            "ai_filter:c",)
        assert barrier.graph.node("apply:keep_even").kind == "barrier"
        request = _apply_query(session, "barrier")._prepare_physical()
        assert request.column_tables()["c"].column_names == ["url"]
        assert request.column_tables()["c"].column("url").to_pylist()[:2] == [
            "u0", "u1"]

    # every worker runs the per-batch apply in its own pipeline
    with _claims_session(gpus=2) as session:
        for kind in ("per_batch", "barrier"):
            assert not isinstance(_apply_query(session, kind).plan(), Refusal)
    with _claims_session(backend="stock_vllm") as session:
        plan = _apply_query(session, "barrier").plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "apply_needs_quail_backend"

    def sequences(graph):
        return sorted({tuple(member.node_id for member in pipeline)
                       for pipeline in build_pipelines(graph).values()})

    assert sequences(two_alias_graph(True)) == [("filter:r", "group:0")]
    # two joins, each anchored on the other's filtered table, would
    # each wait for the other's pipeline: the joins run apart
    from quail.physical.pipelines import execution_groups

    graph = two_alias_graph(True)
    r_filter = graph.node("filter:r")
    p_filter = replace(r_filter, node_id="filter:p", alias="p",
                       inputs=input_ports((PortRef("input:p", "rows:p"),)))
    inputs = input_ports((PortRef("filter:r", "rows:r"),
                          PortRef("filter:p", "rows:p")))
    r_join = replace(graph.nodes[-1], inputs=inputs)
    p_join = replace(r_join, node_id="group:1", anchor="p", partners=("r",),
                     written_position=1, exec_index=1)
    graph = PhysicalGraph((*graph.nodes[:2], r_filter, p_filter, r_join, p_join),
                          graph.root)
    groups = execution_groups(graph, build_pipelines(graph))
    assert sorted(len(group) for group in groups) == [1, 1, 1, 1, 2]
    available = set()
    for group in groups:
        for node in group:
            assert all(port.source.node_id in available for port in node.inputs)
            available.add(node.node_id)
    assert sequences(two_alias_graph(True, foreign=("per_batch", "drop"))) == [
        ("filter:r", "apply:keep_even", "group:0")]
    assert sequences(two_alias_graph(True, foreign=("per_batch", "pairs"))) == [
        ("filter:r", "apply:same_key", "group:0")]
    # a barrier apply and a Barrier each need every survivor at once
    assert sequences(two_alias_graph(True, foreign=("barrier", "drop"))) == [
        ("filter:r",), ("group:0",)]
    assert sequences(two_alias_graph(False)) == [("filter:r",), ("group:0",)]
    graph = two_alias_graph(True)
    # a join reading the scan takes nothing from the sequence
    orphan = PhysicalGraph(
        tuple(node for node in graph.nodes if node.node_id != "group:0")
        + (graph.node("group:0").with_inputs(input_ports(
            (PortRef("input:r", "rows:r"), PortRef("input:p", "rows:p")))),),
        graph.root)
    assert sequences(orphan) == [("filter:r",), ("group:0",)]


def _big_plan(catalog):
    logical = (docs(catalog, "reviews").alias("r")
               .ai_filter(prompt("negative: {0}", col("r.review")),
                          selectivity=0.5)
               .ai_join(docs(catalog, "products").alias("p"),
                        prompt("about {0} {1}", col("r.review"),
                               col("p.description")), selectivity=0.1)
               .select("r.id", "p.asin"))
    return logical, _plan(logical, {"r": [400] * 20000, "p": [20] * 10})


def test_node_ids_estimates_and_the_recompute_column(catalog):
    logical, plan = _big_plan(catalog)
    assert [node.node_id for node in plan.nodes] == [
        "scan:r", "scan:p", "ai_filter:r", "ai_join:r", "project"]
    sequence = plan.graph.node("ai_filter:r")
    assert not sequence.keep_kv
    assert tuple(
        member.node_id for member in build_pipelines(plan.graph)["ai_filter:r"]) == (
        "ai_filter:r", "ai_join:r")
    assert plan.graph.node("ai_join:r").anchor_resident == "filter"
    seconds = {node_id: entry["seconds"]
               for node_id, entry in plan.estimates.items()
               if "seconds" in entry}
    assert set(seconds) == {"ai_filter:r", "ai_join:r"}
    # each node priced alone: the parts add up to at least the packed
    # whole, up to float rounding, and to no more than three times it
    assert plan.estimated_seconds * (1 - 1e-12) <= sum(seconds.values()) \
        <= 3 * plan.estimated_seconds
    # 10,000 expected survivors of 400 tokens do not fit the retention
    # pool, so releasing the sequence's KV would recompute most of them
    recompute = plan.estimates["ai_filter:r"]
    assert recompute["release_recompute_tokens"] > 0.9 * 10000 * 401
    assert recompute["release_recompute_seconds"] > 0
    text = explain(logical, plan)
    assert "est. time" in text
    assert "expected recompute at the join if the sequence's KV is released" in text
    assert "do not add up to the plan estimate" in text
    assert "KV: anchor=from filters" in text
    assert _cell(text, "AiJoin: anchor=")[-1] in ("ms", "s")
    assert _cell(text, "join 1 full (")[-1] == "10%"
    assert "expected_tuples=" not in text

    edited = plan.insert(
        Barrier(node_id="barrier:r", next_anchor="r", aliases=("r",)),
        between=("ai_filter:r", "ai_join:r"))
    assert [node.node_id for node in edited.nodes] == [
        "scan:r", "scan:p", "ai_filter:r", "barrier:r", "ai_join:r", "project"]
    new_sequence = edited.graph.node("ai_filter:r")
    assert new_sequence.keep_kv
    assert tuple(
        member.node_id for member in build_pipelines(edited.graph)["ai_filter:r"]) == (
        "ai_filter:r",)
    assert edited.graph.node("barrier:r").inputs[0].source.node_id == "ai_filter:r"
    assert [port.source.node_id for port in edited.graph.node("ai_join:r").inputs] == [
        "barrier:r", "scan:p"]
    assert edited.estimates["ai_filter:r"]["release_recompute_tokens"] == \
        recompute["release_recompute_tokens"]
    assert "expected recompute at the join" in explain(logical, edited)
    assert edited.estimated_seconds == pytest.approx(
        plan.estimated_seconds
        + edited.estimates["ai_filter:r"]["release_recompute_seconds"])
    assert not plan.graph.node("ai_filter:r").keep_kv
    assert edited.remove("barrier:r") == plan
    moved = edited.move("barrier:r", between=("scan:r", "ai_filter:r"))
    assert [node.node_id for node in moved.nodes][:4] == [
        "scan:r", "scan:p", "barrier:r", "ai_filter:r"]
    assert not moved.graph.node("ai_filter:r").keep_kv


def _int_tokens(text):
    return [int(t) for t in text.split()]


def _token_store(path, bodies):
    from quail.catalog.tokens import TokenStore

    schema = pa.schema({"body": pa.string()})
    return TokenStore.write(
        str(path), [pa.record_batch([bodies], schema=schema)],
        document_column="body", tokenizer=_int_tokens, token_type=pa.int32())


def test_prefix_sharing_and_attention_path_follow_the_token_store(
        catalog, tmp_path):
    from quail.planner.rules.physical import (
        page_aligned_shared_tokens,
        sharing_pays,
    )

    shared = " ".join(str(i) for i in range(40))
    store = _token_store(tmp_path / "shared.arrow",
                         [shared, shared + " 99 98", "7 7 7"])
    assert page_aligned_shared_tokens(store) == 32
    plain = _token_store(tmp_path / "plain.arrow", ["1 2 3", "4 5 6"])

    logical = docs(catalog, "reviews").alias("r").ai_filter(
        prompt("flag: {0}", col("r.review")), selectivity=0.5).select("r.id")

    plan = _plan(logical, {"r": store.lengths})
    sequence = filter_sequence(plan)
    assert sequence.share_prefixes and sequence.arena_writes
    assert "share_prefixes=True" in explain(logical, plan, verbose=True)

    plan = _plan(logical, {"r": plain.lengths})
    sequence = filter_sequence(plan)
    assert not sequence.share_prefixes and not sequence.arena_writes

    # one shared page in 20,000 tokens saves less forward-pass time
    # than writing KV pages for every token costs (about 1.2% on
    # Qwen3 4B and an H100), so the filter stays unpaged
    body = " ".join(str(i) for i in range(10_000))
    little = _token_store(
        tmp_path / "little.arrow",
        [body, " ".join(str(i) for i in range(16)) + " 5"])
    assert page_aligned_shared_tokens(little) == 16
    assert not filter_sequence(_plan(logical, {"r": little.lengths})).share_prefixes
    # a filter that already writes pages shares any whole page
    assert sharing_pays(QWEN3_4B_FP8, H100_SXM, shared_tokens=16,
                        total_tokens=10_017, writes_pages=True)

    # 40 records of 8 own tokens after a 2,048-token header: many
    # short readers of one long node, where stacking their reads wins
    header = " ".join(str(i) for i in range(2048))
    records = [header + " " + " ".join([str(9000 + r)] * 8) for r in range(40)]
    store = _token_store(tmp_path / "records.arrow", records)
    # a sequence of snapshots each 3,000 tokens longer than the last: one
    # reader per node with more rows than the node saves
    snapshots = [" ".join(str(i) for i in range(3000 * (k + 1)))
                 for k in range(4)]
    sequence = _token_store(tmp_path / "sequence.arrow", snapshots)
    plan = _plan(logical, {"r": store.lengths})
    node = filter_sequence(plan)
    assert node.share_prefixes and node.attention == "tree"
    assert "attention=tree" in explain(logical, plan, verbose=True)
    plan = _plan(logical, {"r": sequence.lengths})
    node = filter_sequence(plan)
    assert node.share_prefixes and node.attention == "unified"
    # a filter without shared prefixes has one reader per node
    plan = _plan(logical, {"r": [40, 42, 3]})
    assert filter_sequence(plan).attention == "unified"

    shared = " ".join(str(i) for i in range(400))
    reviews = _token_store(tmp_path / "anchors.arrow",
                           [shared + f" {9000 + r}" for r in range(20)])
    logical = (docs(catalog, "reviews").alias("r")
               .ai_join(docs(catalog, "products").alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.5, anchor="r")
               .select("r.id", "p.asin"))
    plan = _plan(logical, {"r": reviews.lengths, "p": [20] * 20})
    join = plan.graph.nodes_by_type(AiJoin.type_name)[0]
    assert join.share_prefixes
    # anchors with nothing in common share nothing
    plan = _plan(logical, {"r": [401] * 20, "p": [20] * 20})
    assert not plan.graph.nodes_by_type(AiJoin.type_name)[0].share_prefixes


def test_attention_setting_forces_every_filter_and_join(catalog):
    filters = _five_filter_plan(catalog, (0.5,))
    joins = _join_sequence(catalog)
    tokens = {"r": [40, 42, 3], "t": [50] * 4, "p": [100] * 3}
    for path in ("tree", "unified"):
        plan = _plan(filters, tokens, attention=path)
        assert filter_sequence(plan).attention == path
        plan = _plan(joins, tokens, attention=path)
        assert {node.attention for node in plan.nodes
                if isinstance(node, AiJoin)} == {path}
    with pytest.raises(ValueError, match="tree"):
        _plan(filters, tokens, attention="both")
