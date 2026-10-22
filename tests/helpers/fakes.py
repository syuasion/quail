"""CPU fakes the executor tests share: a page arena, a planted model, graphs.

Token ids encode what a suffix is: a document packs [DOC + d] * length,
filter stage s asks [QUESTION + s], partner i streams [PARTNER + i] *
length, and the join frame is [FRAME] * length.
"""

import random
from contextlib import nullcontext
from itertools import takewhile
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

import quail
from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.models.base import ModelPipeline
from quail.backends.quail.executor.pages import PageArena
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.logical import ColumnRef
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Foreign,
    HashJoin,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS

DOC = 1000
QUESTION = 2000
PARTNER = 3000
FRAME = 4000
SETTINGS = {"filter_limit": None, "retention": {}}


def bare_arena(arena, pages):
    """Give a tensor-free KVArena the accounting of a single-pool arena."""
    arena.accounting = PageArena(pages, 16)
    arena.sliding = None
    arena.window = 0
    arena.sliding_layers = frozenset()
    arena.pinned = False
    arena._rows = {}
    arena._capacity_rows = {}
    arena._sliding_rows = {}
    arena._base = {}
    arena._sliding_start = {}
    arena._holds = {}
    arena._reading = {}
    arena._deferred = set()
    arena._window_floor = {}
    arena._trimmed = set()
    arena._refresh_rows = lambda *args: None
    arena.reset_stats()
    return arena


def letter_tokens(text):
    """Tokenize text as bytes, treating short answer letters as one token."""
    import re
    if re.fullmatch(r" [A-Za-z]{1,2}", text):
        return [10_000 * (len(text) - 1) + sum(map(ord, text[1:]))]
    return list(text.encode())


def cpu_staging(monkeypatch):
    """Stage packed chunks as plain CPU tensors; returns torch."""
    import torch

    def staged(torch_, data, dtype, pinned=True):
        if isinstance(data, np.ndarray) or torch.is_tensor(data):
            return torch.as_tensor(data, dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def token_parts(torch_, sequences, total, pinned=True, staging=None):
        ids = [int(t) for seq in sequences for part in chunk_mod._token_parts(seq)
               for t in part]
        assert len(ids) == total
        return torch.tensor(ids, dtype=torch.int64)

    monkeypatch.setattr(chunk_mod, "_staged", staged)
    monkeypatch.setattr(chunk_mod, "_staged_token_parts", token_parts)
    return torch


def fake_pipeline(**attributes):
    """A ModelPipeline with the contract's defaults and the given overrides."""
    pipeline = ModelPipeline()
    for name, value in attributes.items():
        setattr(pipeline, name, value)
    return pipeline


def cpu_arena(pages):
    arena = bare_arena(KVArena.__new__(KVArena), pages)

    def allocate(key, tokens, capacity_tokens=None, base_tokens=None,
                 sliding_tokens=None, borrow=None):
        borrowed = (arena.accounting.shared_pages(*borrow) if borrow else ())
        got = arena.accounting.alloc(key, tokens, capacity_tokens,
                                     borrowed=borrowed)
        if got is not None:
            arena._rows[key] = None
            arena._capacity_rows[key] = None
            arena._base[key] = tokens if base_tokens is None else base_tokens
            arena._sliding_start[key] = 0
        return got

    arena.alloc = allocate
    return arena


class FakeModel:
    """Answer packed suffixes from planted truth tables."""

    def __init__(self, filter_truth, join_truth, streamed_anchors=True):
        self.filter_truth = filter_truth
        self.join_truth = join_truth
        self.streamed_anchors = streamed_anchors
        self.launched = []      # ("filter" | "join", specs)

    def forward_chunk(self, chunk):
        bits = []
        kind = None
        for spec in chunk.specs:
            for suffix in spec["suffixes"]:
                head = suffix[0]
                if head >= FRAME:
                    bits.append(0)
                    kind = "join"
                elif head >= PARTNER:
                    bits.append(self.join_truth[spec["key"]][head - PARTNER])
                    kind = "join"
                else:
                    document = spec["key"][1]
                    bits.append(self.filter_truth[document][head - QUESTION])
                    kind = "filter"
            if (self.streamed_anchors and kind == "join"
                    and spec["prefix"] is not None):
                raise AssertionError("a streamed anchor packed its prefix")
        self.launched.append((kind, chunk.specs))
        return bits


def fake_torch():
    return SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            Event=lambda **kw: SimpleNamespace(
                record=lambda: None, elapsed_time=lambda other: 2.0),
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0, reset_peak_memory_stats=lambda: None))


def fake_pack(torch, arena, specs, **kw):
    def own_rows(spec):
        own = spec.get("canvas")
        return len(kw.get("canvas", ())) if own is None else np.shape(own)[-1]

    tokens = sum(
        (len(spec["prefix"]) if spec["prefix"] is not None else 0)
        + sum(len(suffix) + own_rows(spec) for suffix in spec["suffixes"])
        for spec in specs)

    # one row per suffix, every row of a suffix read whole, or every
    # row of the canvas after it
    def canvas_rows(spec):
        own = spec.get("canvas")
        return len(kw.get("canvas") or ()) if own is None else own_rows(spec)

    rows_per_answer = tuple(
        int(spec["read_rows"][index]) if spec.get("read_rows") is not None
        else int(np.sum(spec["read_mask"])) if spec.get("read_mask") is not None
        else (canvas_rows(spec) or len(suffix)) if spec.get("read_all_rows")
        else 1
        for spec in specs for index, suffix in enumerate(spec["suffixes"]))
    if all(rows == 1 for rows in rows_per_answer):
        rows_per_answer = None
    else:
        rows_per_answer = np.asarray(rows_per_answer, dtype=np.int64)
    meta = {}
    if arena.num_state_slots:
        from quail.backends.quail.executor.gdn import plan_phases

        groups, offset = [], 0
        for spec in specs:
            fresh = spec["prefix"] is not None
            prefix_end = offset + (len(spec["prefix"]) if fresh else 0)
            lengths = np.asarray([len(s) for s in spec["suffixes"]])
            ends = prefix_end + np.cumsum(lengths)
            kept = spec.get("write_suffix_tokens", 0)
            groups.append(dict(
                key=spec["key"], fresh=fresh, f=spec["f"], row0=offset,
                prefix_end=prefix_end, start=spec.get("start", 0),
                kept=kept, keeps=bool(kept),
                suffix_spans=(ends - lengths, ends),
                borrow_key=spec.get("borrow_key"),
                share_points=spec.get("share_points", ())))
            offset = int(ends[-1])
        meta["gdn"] = plan_phases(
            arena, groups, lambda name, values, dtype: np.asarray(values))
    return SimpleNamespace(specs=specs, tokens=tokens, layout=[], temporary_keys=(),
                           fresh_keys=(), rows_per_answer=rows_per_answer,
                           meta=meta)


def expected_filter_rows(filter_truth):
    """Each document's answers up to and including its first FALSE."""
    rows = {}
    for d, truth in enumerate(filter_truth):
        passed = list(takewhile(bool, truth))
        rows[d] = passed + truth[len(passed):len(passed) + 1]
    return rows


def two_alias_graph(pipelined, *, stages=1, hash_join=False, foreign=None,
                    filtered=True, semantics="full", keep_kv=False):
    """R's filter sequence into a join with p.

    Args:
        pipelined: Whether the sequence and the join run as one pipeline;
            otherwise a Barrier between them materializes the
            survivors and the sequence retains their KV.
        stages: Filter stages on r.
        hash_join: Pair r and p on their "key" columns with a HashJoin
            over the scans that feeds the join its pairs port.
        foreign: (kind, ids) of a Foreign node between the sequence and
            the join; ids "pairs" feeds the join its pairs port.
        filtered: Whether r's filter sequence runs at all; otherwise the
            join reads r's scan and loads every anchor itself.
        semantics: The join stage's semantics, "full" or "anti".
        keep_kv: Whether the join retains surviving anchors' KV.
    """
    nodes = [Scan(node_id="input:r", alias="r", input_id="r"),
             Scan(node_id="input:p", alias="p", input_id="p")]
    anchor_src = PortRef("input:r", "rows:r")
    if filtered:
        for index in range(stages):
            node_id = "filter:r" if index == 0 else f"filter:r:{index}"
            nodes.append(AiFilter(
                node_id=node_id, inputs=input_ports((anchor_src,)), alias="r",
                arena_writes=True, keep_kv=not pipelined, written_position=index,
                question_tokens=1, preamble_tokens=0,
                selectivity=0.8 - 0.1 * index, expected_rows=14 * 0.8 ** index,
                question_token_ids=(QUESTION + index,)))
            anchor_src = PortRef(node_id, "rows:r")
    if filtered and not pipelined:
        nodes.append(Barrier(node_id="barrier:r",
                             inputs=input_ports((anchor_src,)),
                             next_anchor="r", aliases=("r",)))
        anchor_src = PortRef("barrier:r", "rows:r")
    join_inputs = []
    pairs_from = ""
    if hash_join:
        nodes.append(HashJoin(
            node_id="hash_join:r-p",
            inputs=input_ports((PortRef("input:r", "rows:r"),
                                PortRef("input:p", "rows:p"))),
            left="r", right="p", written_position=0,
            on=((ColumnRef("r", "r", "key"), ColumnRef("p", "p", "key")),)))
        join_inputs.append(PortRef("hash_join:r-p", "pairs:0"))
        pairs_from = "hash_join:r-p"
    if foreign and foreign[1] == "pairs":
        nodes.append(Foreign(
            node_id="apply:same_key",
            inputs=input_ports((anchor_src, PortRef("input:p", "rows:p"))),
            function="same_key", kind=foreign[0], ids="pairs",
            columns=(ColumnRef("r", "r", "key"), ColumnRef("p", "p", "key")),
            aliases=("r", "p"),
            written_position=0))
        join_inputs.append(PortRef("apply:same_key", "pairs:0"))
        pairs_from = "apply:same_key"
    elif foreign:
        nodes.append(Foreign(
            node_id="apply:keep_even",
            inputs=input_ports((anchor_src,)),
            function="keep_even", kind=foreign[0], ids=foreign[1],
            columns=(), aliases=("r",)))
        anchor_src = PortRef("apply:keep_even", "rows:r")
    resident = "filter" if filtered and pipelined else "none"
    nodes.append(AiJoin(
        node_id="group:0", anchor="r", anchor_resident=resident, keep_kv=keep_kv,
        inputs=input_ports((anchor_src, PortRef("input:p", "rows:p"),
                            *join_inputs)),
        written_position=0, exec_index=0, partners=("p",), semantics=semantics,
        selectivity=0.5, expected_rows=1, anchor_frame_tokens=1,
        pair_tail_tokens=0, tuple_tokens=0, pairs_from=pairs_from,
        frame_token_ids=(FRAME,), label_token_ids=(("p", ()),), tail_token_ids=()))
    return PhysicalGraph(tuple(nodes), PortRef("group:0", "rows:r"))


def run_graph_on_arena(monkeypatch, graph, *, num_documents=14, num_partners=4,
                       seed=5, pages=64, stages=1, partner_tokens=20,
                       streamed_anchors=True, shared_prefix=0, **extra):
    """Run a graph over random documents through the real graph runtime.

    Returns:
        (result, model, filter_truth, join_truth); the arena is empty
        afterwards.
    """
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    rng = random.Random(seed)
    documents = {
        "r": [[DOC - 1] * shared_prefix + [DOC + d] * rng.randrange(10, 40)
              for d in range(num_documents)],
        "p": [[PARTNER + i] * partner_tokens for i in range(num_partners)],
    }
    filter_truth = [[1 if rng.random() < p else 0
                     for p in (0.8, 0.7)[:stages]] for _ in range(num_documents)]
    join_truth = {
        ("r", d): [1 if rng.random() < 0.5 else 0 for _ in range(num_partners)]
        for d in range(num_documents)
    }
    model = FakeModel(filter_truth, join_truth, streamed_anchors)
    torch = fake_torch()
    arena = cpu_arena(pages)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=object(), arena=arena,
                                pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=fake_readout(),
        answer_rows=object(),
        chunk_tokens=120)
    state = {
        "gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                               execution=execution, spec=MODELS["qwen3-4b-fp8"]),
        "runtimes": built_in_registry().runtimes,
        "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "documents": documents, **extra,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    return result, model, filter_truth, join_truth


def register_claims_evidence(session, claim_words=20, text_words=100):
    """Four claims and three evidence rows sharing a url column."""
    session.register("claims", quail.DocumentProvider.from_table(pa.table({
        "id": [f"c{i}" for i in range(4)],
        "claim": [f"{i} " + "word " * claim_words for i in range(4)],
        "url": ["u0", "u1", "u1", "u9"],
    }), id_col="id"))
    session.register("evidence", quail.DocumentProvider.from_table(pa.table({
        "id": [f"e{i}" for i in range(3)],
        "text": [f"{i} " + "word " * text_words for i in range(3)],
        "url": ["u1", "u0", "u1"],
    }), id_col="id"))


def keep_even(tables):
    (alias,) = tables
    table = tables[alias]
    return [document for document in table.column(alias).to_pylist()
            if document % 2 == 0]


def same_key(tables):
    left, right = tables["r"], tables["p"]
    return left.join(right, keys=["key"], join_type="inner").select(["r", "p"])


class FakeArena:
    num_pages = 100
    page_tokens = 16
    retention_cap_pages = None
    evicted_keys = evicted_pages = evicted_prefix_tokens = 0

    def __init__(self):
        self.accounting = self
        self.owned = set()
        self.retained = set()

    @property
    def retained_pages(self):
        return len(self.retained)

    @property
    def retained_prefix_tokens(self):
        return 16 * len(self.retained)

    def pages_needed(self, tokens):
        return -(-tokens // self.page_tokens)

    def configure_retention(self, policy, cap_pages):
        self.retention_policy = policy
        self.retention_cap_pages = cap_pages

    def is_resident(self, key):
        return key in self.owned

    def resident_keys(self):
        return list(self.owned)

    def retained_keys(self):
        return list(self.retained)

    def free_key(self, key):
        self.owned.discard(key)
        self.retained.discard(key)

    def retain(self, key, prefix_tokens, priority=None):
        self.owned.add(key)
        self.retained.add(key)

    def evict_retained(self, pages):
        return ()

    def reset_stats(self):
        pass


def graph_state(model_execution, documents):
    return {
        "gpu": SimpleNamespace(torch=fake_torch(), arena=FakeArena(),
                               pipeline=fake_pipeline(), execution=model_execution,
                               spec=MODELS["qwen3-4b-fp8"]),
        "runtimes": built_in_registry().runtimes,
        "device": DEVICES["h100-sxm"],
        "chunk_tokens": 8192,
        "documents": documents,
    }


def plain_arena(pages=64):
    import torch

    return KVArena(
        num_layers=1,
        num_pages=pages,
        page_tokens=16,
        num_kv_heads=1,
        d_head=2,
        dtype=torch.float32,
        device="cpu",
    )


def fake_readout(**attributes):
    """A readout with the base class's defaults and the given attributes."""
    return SimpleNamespace(**{
        "trailing_rows": 1, "normalize": False, "dtype": None,
        "submit": lambda value, rows_per_answer=None: value,
        "result": lambda value: value, **attributes})
