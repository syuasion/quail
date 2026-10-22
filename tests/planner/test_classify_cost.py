"""Classification costs: expected work against label outcomes, priced like filters."""

import itertools
from dataclasses import asdict, replace

import numpy as np
import pytest

from quail.catalog.tokens import TokenLengths
from quail.cost.classify import estimate, estimate_suffixes
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.sol import unrounded_seconds
from quail.cost.work import Work, ask, scan, stream
from quail.labels import decode_round_tokens, label_trie
from quail.specs import DECISION_2_KAI_0_6B_BF16, H100_SXM, QWEN3_4B_FP8

# a one-token label; two that share a first token and split there; two
# that split after a forced token
LABELS = ((10,), (20, 21, 22), (20, 25, 26, 27, 28), (30, 31, 32), (30, 31, 33))
# the tokens a greedy decode feeds for each label, per round
RUNS = ([1], [1, 1], [1, 1], [1, 2], [1, 2])
HEAD, FRAME, CHUNK = 2, 3, 10000


def _cost(lengths, labels=LABELS, scoring="trie_decode", **kwargs):
    options = dict(lengths=lengths, shared=(), chunk=CHUNK,
                   model=QWEN3_4B_FP8, device=H100_SXM)
    options.update(kwargs)
    expected_rows = options.pop("expected_rows", len(lengths))
    return estimate(scoring, expected_rows, HEAD, FRAME, labels, **options)


def _head_bytes():
    return 2 * QWEN3_4B_FP8.hidden * QWEN3_4B_FP8.vocab


def _readout_seconds(rows, passes):
    return max(_head_bytes() * rows / H100_SXM.gemm_flops("bf16"),
               _head_bytes() * passes / H100_SXM.hbm_bw)


def _decode_outcome_work(lengths, runs, window=0, shared=None, resident=False):
    """Count one label outcome's decode work from the first principles.

    The first request is the prefix, frame, cue, and the forced tokens
    to the first choice as one causal segment; each later round feeds
    one run that attends to everything before it and reads the
    document's KV. A shared or resident prefix is read instead of
    computed.
    """
    total = dict.fromkeys(asdict(Work()), 0.0)
    shared = shared or (0,) * len(lengths)
    for length, fed, borrowed in zip(lengths, runs, shared):
        prompt = HEAD + length + FRAME
        kept = HEAD + length if resident else HEAD + borrowed if borrowed else 0
        fresh = range(kept + 1, prompt + sum(fed) + 1)
        total["tokens"] += len(fresh)
        total["kv_written"] += len(fresh)
        total["pairs"] += sum(fresh)
        contexts = [prompt + sum(fed[:round_]) for round_ in range(1, len(fed))]
        total["kv_read"] += kept + sum(contexts)
        # The document remains saved while every round reads and writes
        # the answer's checkpoint. A fresh borrower also reads its parent.
        total["state_read"] += len(fed) + bool(borrowed and not resident)
        total["state_written"] += len(fed) + (not resident)
        if window:
            total["sliding_pairs"] += sum(min(i, window) for i in fresh)
            total["sliding_kv_read"] += sum(
                min(context, window - 1) for context in contexts)
            if kept:
                total["sliding_kv_read"] += min(kept, window - 1)
    return total


def test_decode_feeds_one_run_per_choice():
    assert decode_round_tokens(LABELS) == list(RUNS)


@pytest.mark.parametrize("lengths,labels,window,shared,resident", [
    ((10, 90), LABELS, 0, None, False),
    ((10,), ((10,), (20,), (30, 31, 32, 33, 34)), 16, None, False),
    ((40, 7, 25), LABELS, 16, (30, 0, 20), False),
    ((10, 90), LABELS, 0, None, True),
    ((40, 7, 25), LABELS, 16, None, True),
])
def test_expected_work_matches_every_label_assignment(lengths, labels, window,
                                                      shared, resident):
    kinds = ("sliding_attention", "full_attention") * 18 if window else ()
    model = replace(QWEN3_4B_FP8, sliding_window=window, layer_kinds=kinds)
    cost = _cost(lengths, labels, model=model, shared=shared or (),
                 resident=resident)
    runs = decode_round_tokens(labels)
    outcomes = list(itertools.product(runs, repeat=len(lengths)))
    expected = dict.fromkeys(asdict(Work()), 0.0)
    for picked in outcomes:
        for key, value in _decode_outcome_work(lengths, picked, window,
                                               shared, resident).items():
            expected[key] += value / len(outcomes)
    assert asdict(cost.work) == pytest.approx(expected)
    assert cost.passes == pytest.approx(cost.work.tokens / CHUNK)
    assert cost.suffix_tokens == pytest.approx(
        sum(sum(map(sum, picked)) for picked in outcomes) / len(outcomes))
    assert cost.rounds == max(map(len, runs))


def test_sampled_label_draws_average_to_the_expected_work():
    rng = np.random.default_rng(7)
    lengths = tuple(int(x) for x in rng.integers(5, 60, 40))
    cost = _cost(lengths)
    draws = 4000
    mean = dict.fromkeys(asdict(Work()), 0.0)
    for _ in range(draws):
        picks = rng.integers(0, len(LABELS), len(lengths))
        for key, value in _decode_outcome_work(
                lengths, [RUNS[i] for i in picks]).items():
            mean[key] += value / draws
    assert asdict(cost.work) == pytest.approx(mean, rel=0.02)


def test_label_and_document_order_do_not_change_the_estimate():
    lengths = (17, 21, 99, 105)
    expected = _cost(lengths, chunk=140)
    for labels in itertools.permutations(LABELS):
        assert _cost(lengths[::-1], labels, chunk=140) == expected
    assert expected.suffix_tokens == pytest.approx(4 * 11 / 5)


def test_one_round_rules_are_priced_like_filters():
    lengths = (10, 20, 30)
    for scoring, suffix in (("letters", 1), ("trie_tree", len(label_trie(LABELS)))):
        cost = _cost(lengths, scoring=scoring)
        work = Work()
        for length in lengths:
            prefix = HEAD + length
            # the frame is written first; the request reads it and the document
            work += scan(prefix, FRAME) + stream(prefix + FRAME, [suffix])
        work = replace(work, state_read=2 * len(lengths),
                       state_written=2 * len(lengths))
        assert asdict(cost.work) == pytest.approx(asdict(work))
        assert cost.passes == pytest.approx(work.tokens / CHUNK)
        assert cost.seconds == pytest.approx(
            unrounded_seconds(work, QWEN3_4B_FP8, H100_SXM, CHUNK)
            + _readout_seconds(len(lengths) * suffix, cost.passes))
        assert cost.suffix_tokens == len(lengths) * suffix
        assert cost.rounds == 1


def test_resident_documents_ask_instead_of_scanning():
    labels = ((10,), (20,), (30,))
    cost = _cost((10, 20, 30), labels, scoring="letters", resident=True)
    work = Work()
    for length in (10, 20, 30):
        prefix = HEAD + length
        work += ask(prefix, FRAME) + stream(prefix + FRAME, [1])
    work = replace(work, state_written=3)
    assert asdict(cost.work) == pytest.approx(asdict(work))


def test_one_token_labels_decode_as_a_merged_letters_request():
    labels = ((10,), (20,), (30,))
    lengths = (10, 20, 30)
    letters = _cost(lengths, labels, scoring="letters")
    decode = _cost(lengths, labels, scoring="trie_decode")
    for field in ("tokens", "pairs", "kv_written", "sliding_pairs"):
        assert getattr(decode.work, field) == getattr(letters.work, field)
    # the decode packs the cue into the prefix's causal request; the letters
    # request is a separate entry that reads the prefix and frame from KV
    assert decode.work.kv_read == 0
    assert letters.work.kv_read == sum(HEAD + length + FRAME for length in lengths)
    assert decode.suffix_tokens == letters.suffix_tokens == len(lengths)
    assert decode.passes == letters.passes
    assert decode.seconds <= letters.seconds


def test_smaller_chunks_pay_more_weight_reads_without_changing_work():
    device = replace(H100_SXM, peak_flops=float("inf"), bf16_flops=float("inf"),
                     mamf_fp8_flops=float("inf"), mamf_bf16_flops=float("inf"))
    large = _cost((10, 20, 30), device=device)
    small = _cost((10, 20, 30), device=device, chunk=40)
    assert asdict(small.work) == pytest.approx(asdict(large.work))
    assert small.suffix_tokens == large.suffix_tokens
    assert small.passes == pytest.approx(large.passes * CHUNK / 40)
    assert small.seconds > large.seconds
    weights = sum(component.bytes_moved for component in dense_decoder_components(
        Work(), QWEN3_4B_FP8, passes=1))
    for cost in (large, small):
        kv = QWEN3_4B_FP8.kappa * (cost.work.kv_written + cost.work.kv_read)
        assert cost.seconds == pytest.approx(
            ((weights + _head_bytes()) * cost.passes + kv) / device.hbm_bw)


def test_time_falls_with_chunk_size_and_rises_with_label_choices():
    lengths = (10, 20, 30, 300)
    seconds = [_cost(lengths, chunk=chunk).seconds for chunk in (50, 400, 4000, 40000)]
    assert seconds == sorted(seconds, reverse=True)
    # tokens after a label's last choice are never fed, so they cost
    # nothing; a choice below it adds a round
    longer = (LABELS[0], LABELS[1] + (23,), LABELS[2] + (29, 30), *LABELS[3:])
    assert _cost(lengths, longer) == _cost(lengths)
    deeper = (*LABELS[:4], (30, 31, 33, 34), (30, 31, 33, 35))
    short, long = _cost(lengths), _cost(lengths, deeper)
    assert short.work.dominates(long.work)
    assert short.seconds < long.seconds
    assert short.suffix_tokens < long.suffix_tokens
    assert (short.rounds, long.rounds) == (2, 3)


def test_kv_capacity_does_not_enter_the_estimate():
    from quail.cost import budgets
    from quail.planner.operators.classify import ClassifyScoring

    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    tables = [ClassifyScoring(alias="d", mean=200.0, longest=300, budget=chunk,
                              chunk=chunk, model=QWEN3_4B_FP8, device=H100_SXM,
                              tokenizer=None, capacity=capacity,
                              lengths=(100, 200, 300), tree=True)
              for capacity in (1_000, 1_000_000)]
    for scoring in ("letters", "trie_tree", "trie_decode"):
        first, second = (table.estimate(scoring, 1000, 20, 30, LABELS, False)
                         for table in tables)
        assert first == second


@pytest.mark.parametrize("expected_rows", [0, 0.25, 1.8, 7])
def test_the_estimate_is_linear_in_the_expected_document_count(expected_rows):
    cost = _cost((10,), expected_rows=expected_rows)
    single = _cost((10,))
    assert cost.rounds == (2 if expected_rows else 0)
    assert asdict(cost.work) == pytest.approx(
        {key: value * expected_rows for key, value in asdict(single.work).items()})
    for name in ("seconds", "passes", "suffix_tokens"):
        assert getattr(cost, name) == pytest.approx(
            getattr(single, name) * expected_rows
        )


def _decision_readout_seconds(model, rows, passes):
    width, hidden = model.decision_head_dim, model.hidden
    weights = 4 * (4 * width * hidden + 2 * width + 4 * hidden)
    return max(4 * width * hidden * rows / H100_SXM.fp32_flops,
               weights * passes / H100_SXM.hbm_bw)


@pytest.mark.parametrize("resident", [False, True])
def test_decision_models_read_option_rows_through_their_head(resident):
    model = DECISION_2_KAI_0_6B_BF16
    assert model.decision_head_dim == 256
    lengths, options = (10, 20, 30), 26
    cost = estimate_suffixes(
        HEAD,
        FRAME,
        [50],
        expected_rows=len(lengths),
        lengths=lengths,
        shared=(),
        chunk=CHUNK,
        model=model,
        device=H100_SXM,
        resident=resident,
        answer_rows=options + 1,
    )
    work = Work()
    for length in lengths:
        prefix = HEAD + length
        first = ask(prefix, FRAME) if resident else scan(prefix, FRAME)
        work += first + stream(prefix + FRAME, [50])
    work = replace(work, state_read=2 * len(lengths),
                   state_written=len(lengths) * (1 if resident else 2))
    assert asdict(cost.work) == pytest.approx(asdict(work))
    # one row per option block and the prompt's last row, through the
    # fp32 decision head; the vocabulary head is never read
    assert cost.seconds == pytest.approx(
        unrounded_seconds(work, model, H100_SXM, CHUNK)
        + _decision_readout_seconds(model, (options + 1) * len(lengths),
                                    cost.passes))
    vocabulary = 2 * model.hidden * model.vocab * cost.passes / H100_SXM.hbm_bw
    assert cost.seconds < unrounded_seconds(work, model, H100_SXM, CHUNK) + vocabulary
    without_rate = replace(H100_SXM, fp32_flops=0.0)
    with pytest.raises(ValueError, match="fp32"):
        estimate_suffixes(HEAD, FRAME, [50], expected_rows=3, lengths=lengths,
                        shared=(), chunk=CHUNK, model=model, device=without_rate,
                        answer_rows=options + 1)


def test_one_token_labels_read_only_their_target_rows():
    lengths = (10, 20, 30)
    labels = ((10,), (20,), (30,), (30,))
    cost = _cost(lengths, labels, scoring="letters")
    work = Work()
    for length in lengths:
        work += scan(HEAD + length, FRAME) + stream(HEAD + length + FRAME, [1])
    targets = 2 * QWEN3_4B_FP8.hidden * 3
    assert cost.seconds == pytest.approx(
        unrounded_seconds(work, QWEN3_4B_FP8, H100_SXM, CHUNK)
        + max(targets * len(lengths) / H100_SXM.gemm_flops("bf16"),
              targets * cost.passes / H100_SXM.hbm_bw))
    longer = _cost(lengths, ((10,), (20, 21), (30,)), scoring="letters")
    assert longer.work == cost.work
    assert longer.seconds == pytest.approx(
        unrounded_seconds(work, QWEN3_4B_FP8, H100_SXM, CHUNK)
        + _readout_seconds(len(lengths), cost.passes))


def test_classify_and_score_capacity_follow_the_plans_arena_split():
    from types import SimpleNamespace

    from quail.cost import budgets
    from quail.planner.operators.classify import classify_statistics
    from quail.planner.statistics.memory import arena_capacity
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    model = DIFFUSION_GEMMA_26B_FP8
    chunk = budgets.chunk_budget(model, H100_SXM)
    # documents four times the sliding window: the split gives the
    # every-token pool more pages than the split for unknown lengths
    context = SimpleNamespace(
        document_tokens={"short": TokenLengths([300] * 8),
                         "long": TokenLengths([4096] * 8)}, model=model,
        device=H100_SXM, canvas_draws=1, memo={}, gpus=1)
    planned = budgets.arena_tokens(model, H100_SXM, chunk, 4096)
    assert planned > budgets.arena_tokens(model, H100_SXM, chunk)
    # the longest table's mean, as plan_statistics splits the pools
    assert arena_capacity(context, chunk) == planned
    assert classify_statistics(context, "short").capacity == planned
    # a model without sliding layers keeps one pool whatever the lengths
    dense = SimpleNamespace(**{**vars(context), "model": QWEN3_4B_FP8, "memo": {}})
    assert arena_capacity(dense, chunk) == budgets.arena_tokens(
        QWEN3_4B_FP8, H100_SXM, chunk)


@pytest.mark.parametrize("num_documents,gpus", [(8, 2), (1, 8), (2, 8)])
def test_parallel_classification_divides_time_by_the_copies_in_use(
        num_documents, gpus):
    lengths = [100] * num_documents
    serial = _cost(lengths)
    parallel = _cost(lengths, gpus=gpus)
    assert parallel.work == serial.work
    assert parallel.seconds == pytest.approx(
        serial.seconds / min(num_documents, gpus))
