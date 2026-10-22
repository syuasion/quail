"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pyarrow as pa
import pytest
from helpers.fakes import letter_tokens

import quail
from quail.cost import budgets
from quail.planner.plan import Refusal
from quail.specs import (
    DEVICES,
    DIFFUSION_GEMMA_26B_FP8,
    H100_SXM,
    MODELS,
    QWEN3_4B_FP8,
    QWEN3_5_4B_BF16,
    QWEN3_32B_FP8,
    RTX_PRO_6000_BLACKWELL_SERVER,
)


def _hybrid_filter_plan(model, device, body="A good movie."):
    with quail.Session(
        quail.EngineConfig(model=model.name, device=device.name),
        tokenizer=letter_tokens,
    ) as session:
        session.register("documents", quail.DocumentProvider.from_table(
            pa.table({"id": ["1"], "body": [body]}), id_col="id"))
        return session.sql(
            "SELECT d.id FROM documents d WHERE "
            "AI_FILTER(PROMPT('Is it good? {0}', d.body), "
            "{'selectivity': 0.5})").plan()


@pytest.mark.parametrize("name,h100_chunk", [
    ("qwen3.5-27b-fp8", 61_680),
    ("qwen3.6-35b-a3b-fp8", 32_768),
    ("qwen3.8-27b-fp8", 61_680),
])
def test_hybrid_chunks_leave_room_for_workspace_and_checkpoints(
        name, h100_chunk, monkeypatch):
    model = MODELS[name]
    assert budgets.chunk_budget(model, H100_SXM) == h100_chunk
    device = replace(H100_SXM, mem_bytes=48 * 2**30)
    monkeypatch.setitem(DEVICES, device.name, device)
    chunk = budgets.chunk_budget(model, device)
    assert 0 < chunk < h100_chunk
    full, sliding = budgets.arena_pages(model, device, chunk)
    slots = budgets.checkpoint_slots(model, device, chunk, full)
    assert slots >= 1 and sliding == 0
    allocated = (model.W_resident
                 + budgets.ACT_RESERVE_CHUNKS * chunk * model.act_per_token
                 + budgets.checkpoint_pass_bytes(
                     model, budgets.CHECKPOINT_PASS_QUESTIONS, chunk)
                 + full * budgets.PAGE_TOKENS * model.kappa
                 + slots * model.checkpoint_bytes)
    assert allocated <= device.mem_bytes * budgets.POOL_FRACTION
    with pytest.raises(ValueError):
        budgets.arena_pages(model, device, chunk + 1, chunk + 1)
    assert not isinstance(_hybrid_filter_plan(model, device), Refusal)
    assert not isinstance(_hybrid_filter_plan(model, device, "x" * (chunk - 256)),
                          Refusal)


def test_hybrid_planning_refuses_when_even_one_token_cannot_fit(monkeypatch):
    model = MODELS["qwen3.5-27b-fp8"]
    device = replace(H100_SXM,
                     mem_bytes=(model.W_mem + 2**30) / budgets.POOL_FRACTION)
    monkeypatch.setitem(DEVICES, device.name, device)
    assert budgets.minimum_weight_gpus(model, device) == 1
    assert budgets.chunk_budget(model, device) == 0
    plan = _hybrid_filter_plan(model, device)
    assert isinstance(plan, Refusal)
    assert plan.constraint == "execution_needs_more_memory"
    assert plan.needed > plan.available
    assert "checkpoint" in plan.reasons[0]


@pytest.mark.parametrize("model", [QWEN3_4B_FP8, QWEN3_5_4B_BF16])
def test_compute_efficiency_cannot_override_a_chunk_limit(model):
    capped = replace(model, chunk_cap_tokens=16)
    assert budgets.compute_knee(capped, H100_SXM) > 16
    assert budgets.chunk_budget(capped, H100_SXM) == 16


def test_model_weights_and_kv_memory_budgets():
    # Tied output weights must remain available for input embeddings.
    assert QWEN3_4B_FP8.head_mem_bytes == 0.0
    assert QWEN3_4B_FP8.W_resident == QWEN3_4B_FP8.W_mem
    # The separate 32B output head stays resident for AI.CLASSIFY: its
    # 1,555,824,640 bytes are 5,936 KV tokens (262,144 bytes each, in
    # whole 16-token pages) the arena does not get.
    head = QWEN3_32B_FP8.head_mem_bytes
    assert head == 151_936 * 5_120 * 2
    assert QWEN3_32B_FP8.W_resident == QWEN3_32B_FP8.W_mem
    assert budgets.arena_tokens(QWEN3_32B_FP8, H100_SXM) == 112_304 - 5_936
    # the chunk budget does not move: the 32B chunk is capped by the
    # int32 kernel index, not by memory
    kept = replace(QWEN3_32B_FP8, tied_head=True)
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == \
        budgets.chunk_budget(kept, H100_SXM)

    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert tokens == 362_240     # 22,640 pages of 16 tokens
    assert tokens // 400 == 905
    fp8 = budgets.arena_tokens(replace(QWEN3_4B_FP8, kv_bytes=1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * tokens, rel=0.01)
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        assert budgets.arena_tokens(model, RTX_PRO_6000_BLACKWELL_SERVER) \
            > budgets.arena_tokens(model, H100_SXM)


def test_arena_pages_needs_room_for_one_chunk_of_sliding_kv():
    full, sliding = budgets.arena_pages(DIFFUSION_GEMMA_26B_FP8, H100_SXM)
    assert full > 0 and sliding >= budgets.transient_sliding_pages(
        budgets.chunk_budget(DIFFUSION_GEMMA_26B_FP8, H100_SXM),
        DIFFUSION_GEMMA_26B_FP8.sliding_window)
    small = replace(H100_SXM, mem_bytes=40e9)
    with pytest.raises(ValueError, match="one chunk's sliding KV"):
        budgets.arena_pages(DIFFUSION_GEMMA_26B_FP8, small)


def test_layer_kind_follows_the_existing_attention_layout():
    gemma = DIFFUSION_GEMMA_26B_FP8
    # a dense model keeps every token in every layer
    assert {QWEN3_4B_FP8.layer_kind(i) for i in range(QWEN3_4B_FP8.layers)} \
        == {"full_attention"}
    # DiffusionGemma slides in 25 layers and keeps every token in each
    # sixth layer
    kinds = [gemma.layer_kind(i) for i in range(gemma.layers)]
    assert [i for i, kind in enumerate(kinds) if kind == "full_attention"] \
        == [5, 11, 17, 23, 29]
    assert kinds.count("sliding_attention") == 25
    # the kinds agree with the helpers they sit beside
    assert all((kind == "full_attention") == gemma.is_full_layer(i)
               for i, kind in enumerate(kinds))
    assert {i for i, kind in enumerate(kinds) if kind == "sliding_attention"} \
        == gemma.sliding_layer_set


def test_tree_attention_is_refused_for_a_model_with_a_gdn_layer():
    # one Gated DeltaNet layer is the only difference from Qwen3-4B
    kinds = ("gated_delta_net",) + ("full_attention",) * 35
    hybrid = replace(QWEN3_4B_FP8, layer_kinds=kinds)
    assert hybrid.params == QWEN3_4B_FP8.params
    assert not hybrid.canvas_tokens
    assert budgets.tree_attention_allowed(QWEN3_4B_FP8)
    assert not budgets.tree_attention_allowed(hybrid)


def _hybrid_projections():
    """(in, out, layers) of Qwen3.5-4B's dense projections, from its config."""
    return [(2560, (2 * 16 + 2 * 4) * 256, 8),    # qkv with the query gate
            (4096, 2560, 8),                       # attention output
            (2560, 2 * 9216, 32), (9216, 2560, 32),    # MLP
            (2560, 2 * 2048 + 2 * 4096 + 2 * 32, 24),  # GDN qkv, z, b, a
            (4096, 2560, 24)]                      # GDN output


def test_compute_knee_counts_each_projection_in_the_layers_that_have_it():
    ridge = H100_SXM.gemm_flops("bf16") / H100_SXM.hbm_bw
    tot_p = sum(layers * i * o for i, o, layers in _hybrid_projections()) / 32
    tot_io = sum(layers * (i + o) for i, o, layers in _hybrid_projections()) / 32
    expected = ridge * tot_p * 2.0 / (2.0 * tot_p - ridge * tot_io * 2)
    assert budgets.compute_knee(QWEN3_5_4B_BF16, H100_SXM) == pytest.approx(
        expected, rel=1e-12)


def test_attention_time_covers_only_the_attention_layers():
    model, chunk, context = QWEN3_5_4B_BF16, 1024, 4000
    flops = 4.0 * chunk * context * 16 * 256
    moved = context * 32_768 / 8 + 2.0 * chunk * 16 * 256 * budgets.ACT_BYTES
    expected = max(flops / H100_SXM.attn_flops, moved / H100_SXM.hbm_bw) * 8
    assert budgets._attention_time(
        model, H100_SXM, chunk, context) == pytest.approx(expected, rel=1e-12)
    # the same layers with all 32 storing KV take 4 times as long
    dense = replace(model, layer_kinds=())
    assert budgets._attention_time(dense, H100_SXM, chunk, context) \
        == pytest.approx(4 * expected, rel=1e-12)


def test_the_tree_merge_is_charged_for_the_attention_layers_only():
    # tree takes 0.73 of the unified time with the merge over 8 layers,
    # and 1.24 of it if the merge were charged over all 32
    assert budgets.choose_attention_path(
        QWEN3_5_4B_BF16, H100_SXM, readers=2, reader_rows=16,
        node_tokens=500) == "tree"


@pytest.mark.parametrize("tokens,expected_slots", [(200, 611), (16_000, 61)])
def test_hybrid_budget_reserves_checkpoints_and_two_execution_chunks(
        tokens, expected_slots):
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.cost.retention import retention_pages

    model = replace(QWEN3_5_4B_BF16, w_mem_bytes=8_411_152_384)
    chunk = budgets.chunk_budget(model, H100_SXM)
    assert chunk == 116_508
    assert model.checkpoint_bytes == 51_511_296
    assert budgets.checkpoint_pass_bytes(model, 1_024, chunk) == 4_941 * 2**20
    pool = budgets.hybrid_pool_pages(model, H100_SXM, chunk)
    assert pool == 68_060
    kv_pages, sliding = budgets.arena_pages(model, H100_SXM, chunk, tokens)
    assert sliding == 0
    slots = budgets.checkpoint_slots(model, H100_SXM, chunk, kv_pages)
    assert slots == expected_slots
    cost = -(-tokens // 16) + budgets.checkpoint_pages(model)
    assert slots * cost <= pool < (slots + 1) * cost
    retained = retention_pages(kv_pages * 16, chunk, 16)
    assert retained == kv_pages - budgets.execution_pages(chunk)
    assert retained + slots * budgets.checkpoint_pages(model) <= pool
    assert checkpoint_arena_args(model, H100_SXM, chunk, kv_pages) == {
        "num_state_slots": slots,
        "gdn_state_shape": (24, 32, 128, 128),
        "conv_state_shape": (24, 3, 8_192),
    }
    with pytest.raises(ValueError, match="holds no checkpoint"):
        budgets.hybrid_arena_pages(model, H100_SXM, chunk, 1e7)


def test_dense_arena_budget_has_no_checkpoint_reservation():
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.cost.retention import coefficients

    model = QWEN3_4B_FP8
    assert model.checkpoint_bytes == 0
    assert budgets.checkpoint_memory_bytes(model, 1_000) == 0
    assert budgets.checkpoint_slots(model, H100_SXM, 1_000, 5) == 0
    assert checkpoint_arena_args(model, H100_SXM, 1_000, 5) == {}
    assert "checkpoint_pages" not in coefficients(model, H100_SXM)
    assert budgets.arena_pages(model, H100_SXM, 10_000)[0] == (
        budgets.arena_tokens(model, H100_SXM, 10_000) // 16)


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_gemm_rates_fall_back_to_specifications_per_precision(precision):
    device = replace(H100_SXM, mamf_fp8_flops=0, mamf_bf16_flops=0)
    assert device.gemm_flops(precision) == device.arithmetic_bandwidth(precision)
    calibrated = replace(device, **{f"mamf_{precision}_flops": 123e12})
    assert calibrated.gemm_flops(precision) == 123e12
    assert calibrated.arithmetic_bandwidth(precision) == device.arithmetic_bandwidth(
        precision)


def test_fp8_calibration_leaves_bf16_planning_unchanged(monkeypatch):
    from quail.planner.rules.physical import sharing_pays

    model = QWEN3_5_4B_BF16
    before = _hybrid_filter_plan(model, H100_SXM)
    device = replace(H100_SXM, mamf_fp8_flops=H100_SXM.peak_flops / 4)
    monkeypatch.setitem(DEVICES, device.name, device)
    after = _hybrid_filter_plan(model, device)
    assert after.estimated_seconds == before.estimated_seconds
    assert budgets.compute_knee(model, device) == budgets.compute_knee(model, H100_SXM)
    args = dict(shared_tokens=70, total_tokens=1000, writes_pages=False)
    assert sharing_pays(model, device, **args) == sharing_pays(model, H100_SXM, **args)


def test_bf16_calibration_changes_gemms_but_not_attention():
    from quail.cost.sol import speed_of_light
    from quail.cost.work import Work

    device = replace(H100_SXM, mamf_fp8_flops=0, mamf_bf16_flops=0)
    calibrated = replace(device, mamf_bf16_flops=device.attn_flops / 2)
    work = Work(tokens=1000, pairs=500_000, kv_written=1000, kv_read=0)
    before = speed_of_light(work, QWEN3_5_4B_BF16, device, 1000)
    after = speed_of_light(work, QWEN3_5_4B_BF16, calibrated, 1000)
    assert after.component("attention") == before.component("attention")
    for name in ("attn_proj", "mlp"):
        assert after.component(name).compute_seconds == pytest.approx(
            2 * before.component(name).compute_seconds)
