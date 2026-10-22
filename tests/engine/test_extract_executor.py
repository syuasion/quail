"""The Quail executor's AI.EXTRACT: its steps on a scripted model."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from helpers.extract_fakes import (
    LEASE,
    NONE_HELD,
    NOTHING,
    QUOTED,
    TAIL,
    TERM,
    make_spec,
)
from helpers.fakes import cpu_arena, fake_pack, fake_pipeline, fake_readout, fake_torch

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.distributed import (
    execute_distributed_graph,
)
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.extract.candidates import Stream
from quail.backends.quail.executor.extract.decode import AnswerState, Decoder
from quail.backends.quail.executor.extract.stages import ExtractionStages
from quail.backends.quail.executor.extract.vocabulary import same_word
from quail.backends.quail.executor.operators.extract import (
    execute_extractions,
)
from quail.backends.quail.executor.readout import split_rows
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.backends.quail.graph import execute_single_graph
from quail.backends.quail.worker import ChildState, _child_execute
from quail.builtins import built_in_registry
from quail.catalog.tokens import TokenOffsets, TokenStore
from quail.logical import ColumnRef, GetField, RegularPredicate
from quail.physical import (
    AiExtract,
    AiFilter,
    PhysicalGraph,
    PortRef,
    Scan,
    plan_envelope,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS, QWEN3_4B_FP8


@pytest.mark.parametrize(("scores", "stop", "chosen", "done"), [
    ([-1, -1, -9, -10], -8, 1, False),
    ([-2, -1, -9, -10], -1, None, True),
    ([-2, -1, -9, -0.5], -8, None, True),
])
def test_continuation_arrays_handle_duplicates_ties_and_stops(
        scores, stop, chosen, done):
    tokens = [33, 1, 1]
    vocabulary = SimpleNamespace(vocab=range(64), is_stop=np.zeros(64, dtype=bool))
    vocabulary.is_stop[62] = True
    decoder = Decoder(vocabulary, 4, 9, 1, make_spec()[0])
    text = "abcdefghij"
    streams = [Stream(at, [0, token], [at + 1, at + 2], text, exhausted=True)
               for at, token in enumerate(tokens)]
    streams.append(Stream(5, [0], [6], text, exhausted=True))
    answer = AnswerState(written=[0], streams=streams)
    document = SimpleNamespace(answers=[answer], text=text)
    actual = decoder.choose(document, 0, [stop, 33, 1, 40, 50, *scores])
    assert actual == (chosen if not done else None)
    assert answer.done == done
    assert answer.written == ([0, chosen] if chosen is not None else [0])
    if chosen is not None:
        assert [s.char for s in answer.streams] == [
            index for index, token in enumerate(tokens) if token == chosen]


def test_readout_scores_reset_between_rows_and_keep_token_order():
    _, vocabulary = make_spec()
    decoder = Decoder(vocabulary, 4, 9, 1, make_spec()[0])
    scores, stop = decoder.lp([-5, 5, 2, 3, 4, -1, -1, -2, -3])
    assert decoder.top == 5 and stop == -5
    assert scores[5] == scores[2] == -1
    reused, _ = decoder.lp([-7, 6, 1, 0, 0, -0.5, -1, -20, -30])
    assert reused is scores
    assert decoder.top == 6
    assert reused[5] == reused[2] == -np.inf
    assert reused[0] == -30


def test_writing_that_begins_outside_the_document_answers_none(monkeypatch):
    # at the cue the model prefers a word that begins no position: its
    # answer is not in the document, and neither extraction answers
    state, model, run, _ = _setup(monkeypatch)
    tables = model.tables
    stray = tables.encode_ids(" five")[0]
    cue_row = model._cue_row

    def straying(key):
        if key[1] != 0:
            return cue_row(key)
        march = tables.encode_ids("March")[0]
        return model._cue({stray: -0.05, march: -0.5}, -6.0)

    model._cue_row = straying
    result, stages = run()
    assert result.outputs["rows:c"].column("ends").to_pylist() == [
        None, "five years", None]
    assert stages.documents[0].answers[0].none_wins


def test_writing_that_leaves_the_document_later_ends_the_answer(monkeypatch):
    # after "March" the model would rather write a word the document
    # does not continue with: the answer is what it wrote within the
    # document, as a contract's first party before an "and" the
    # contract does not have
    state, model, run, _ = _setup(monkeypatch)
    tables = model.tables
    stray = tables.encode_ids(" lease")[0]
    answer_row = model._answer_row

    def straying(key, written):
        row = answer_row(key, written)
        if key[1] == 0 and len(written) == 1:
            gold = model._gold_ids(key)
            return model._cue({stray: -0.05, gold[1]: -1.0}, -4.0)
        return row

    model._answer_row = straying
    result, stages = run()
    assert result.outputs["rows:c"].column("ends").to_pylist() == [
        "March", "five years", None]
    assert not stages.documents[0].answers[0].none_wins


def _extract_one(monkeypatch, text, gold):
    """Ask the scripted model one question of one document; return its answer."""
    spec, tables = make_spec()
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    stages = ExtractionStages(state, [spec], tables, [0],
                              [tables.encode_ids(text)], [text])
    key = stages.documents[0].key
    model.documents[key] = text
    model.gold[key] = gold
    from quail.backends.quail.executor.stages import run_stages

    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys,
               appended_tokens=stages.appended_tokens)
    stages.finish()
    return stages.documents[0].answers[0]


def test_an_answer_copies_the_quotes_inside_its_span(monkeypatch):
    asked = _extract_one(monkeypatch, QUOTED, ' the "Premises" for five years')
    assert asked.answer == 'the "Premises" for five years'
    assert asked.span == (9, 38)


def test_an_answer_may_begin_with_none(monkeypatch):
    asked = _extract_one(monkeypatch, NONE_HELD, "None of the parties")
    assert asked.answer == "None of the parties"
    assert asked.span == (0, 19)


@pytest.mark.parametrize("recurrent", [False, True])
def test_a_rejected_draft_is_replaced_by_the_next_round(monkeypatch, recurrent):
    # "five" starts two positions; the draft follows the first, "five
    # days", and the model chooses " years" instead: the round ends
    # there, and the next round feeds " years" in the draft's place
    text = "five days or five years"
    spec, tables = make_spec(text)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    state = _state(model, recurrent=recurrent)
    state.loaded_model.vocabulary = tables
    stages = ExtractionStages(state, [spec], tables, [0],
                           [tables.encode_ids(text)], [text])
    key = stages.documents[0].key
    model.documents[key] = text
    model.gold[key] = "five years"
    from quail.backends.quail.executor.stages import run_stages

    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys,
               appended_tokens=stages.appended_tokens)
    stages.finish()
    asked = stages.documents[0].answers[0]
    assert asked.answer == "five years" and asked.span == (13, 23)
    assert asked.written == tables.encode_ids(" five years")
    five, days, years = (tables.encode_ids(w)[0] for w in (" five", " days", " years"))
    fed = [(suffix, f) for k, suffix, kept, _, f in model.packed if k == key]
    # the cue, then the two rounds: the second starts where the first
    # verified, one token past the first round's position
    assert fed[1][0][:2] == [five, days]
    if recurrent:
        assert fed[2][0][:2] == [five, years]
        assert fed[2][1] == fed[1][1]
        kept = [entry[2] for entry in model.packed if entry[0] == key]
        assert kept[1:] == [0, 1]
    else:
        assert fed[2][0][:1] == [years] and fed[2][1] == fed[1][1] + 1
    assert asked.steps == 3
    assert asked.fed_tokens == (len(tables.encode_ids(TAIL)) + len(fed[1][0])
                                + len(fed[2][0]))

def test_rows_with_one_live_stream_are_decided_together(monkeypatch):
    # with one live stream, a round's rows are decided at once; the
    # answers, steps, and fed tokens match choosing row by row
    counts = []
    accepted = Decoder.accepted

    def counting(self, answer, rows, fed):
        counts.append(accepted(self, answer, rows, fed))
        return counts[-1]

    def outcome(run):
        result, stages = run()
        return (result.outputs["rows:c"].column("ends").to_pylist(),
                [(a.written, a.steps, a.fed_tokens, a.span)
                 for document in stages.documents for a in document.answers])

    monkeypatch.setattr(Decoder, "accepted", counting)
    fast = outcome(_setup(monkeypatch)[2])
    assert max(counts) > 0
    # the adaptive policy changes draft lengths, not answers or spans
    rows, states = outcome(_setup(monkeypatch, policy="adaptive")[2])
    assert rows == fast[0]
    assert [(w, span) for w, _, _, span in states] == [
        (w, span) for w, _, _, span in fast[1]]
    monkeypatch.setattr(Decoder, "accepted", lambda self, answer, rows, fed: 0)
    assert outcome(_setup(monkeypatch)[2]) == fast


class ScriptedModel:
    """Writes the gold answer: the lines that hold it, its tokens, then stops.

    At the cue the model prefers the gold answer's first word, written
    with its space; at each later step it prefers the gold answer's
    next token while the tokens written so far are a prefix of it,
    and ends its turn once the whole answer is written. As in the
    model's readout, the stop value includes the probabilities of
    the stop tokens among the top tokens.
    """

    def __init__(self, tables, documents, gold, k):
        self.tables = tables
        self.documents = documents            # key -> text
        self.gold = gold            # key -> gold answer text, or None
        self.k = k
        self.filter_truth = {}      # key -> a filter question's answer
        self.reads = Counter()      # key -> times its prefix was packed
        self.width = 1 + 2 * k
        self.cue_token = tables.encode_ids(TAIL)[-1]
        self.kv = {}                # key -> tokens appended after the prefix
        self.prefix_len = {}        # key -> the prefix's tokens
        self.packed = []

    def _row(self, values):
        row = np.full(self.width, np.nan, dtype=np.float32)
        row[:len(values)] = values
        return row

    def _top(self, lps, extras):
        """A row: the extras, then the k likeliest ids and values."""
        best = sorted(lps.items(), key=lambda item: -item[1])[:self.k]
        while len(best) < self.k:
            best.append((0, -30.0))
        return self._row(list(extras) + [float(i) for i, _ in best]
                         + [v for _, v in best])

    def _cue(self, lps, stop_lp):
        """A row: the stop, then the top tokens."""
        for token, value in lps.items():
            if self.tables.is_stop[token]:
                stop_lp = np.logaddexp(stop_lp, value)
        return self._top(lps, [stop_lp])

    def _gold_ids(self, key):
        gold = self.gold[key]
        return self.tables.encode_ids(gold) if gold else []

    def forward_chunk(self, chunk):
        rows = []
        for spec in chunk.specs:
            key = spec["key"]
            for index, suffix in enumerate(spec["suffixes"]):
                suffix = [int(t) for t in suffix]
                read_all = bool(spec.get("read_all_rows"))
                self.packed.append((key, suffix, spec.get("write_suffix_tokens", 0),
                                    read_all, spec.get("f")))
                # the request's rows follow position f: a frame, a
                # rewinding stage, or a discarding one drops what an
                # earlier stage appended
                if spec.get("prefix") is not None:
                    self.prefix_len[key] = len(spec["prefix"])
                    self.reads[key] += 1
                held = self.kv.get(key, [])
                if spec.get("f") is not None and key in self.prefix_len:
                    held = held[:spec["f"] - self.prefix_len[key]]
                path = held + suffix
                # every row of a draft is read; otherwise the last
                count = (spec["read_rows"][index] if spec.get("read_rows") is not None
                         else len(suffix))
                for end in (range(len(path) - count + 1, len(path) + 1) if read_all
                            else [len(path)]):
                    rows.append(self._row_after(key, path[:end]))
                if spec.get("write_suffix_tokens", 0):
                    self.kv[key] = path
        return np.stack(rows)

    def _row_after(self, key, path):
        if path[-1] == self.cue_token:
            return self._cue_row(key)
        if self.cue_token in path:
            cue = len(path) - 1 - path[::-1].index(self.cue_token)
            return self._answer_row(key, path[cue + 1:])
        if key in self.filter_truth:
            # a filter's question: the row's first value is the answer
            row = self._cue({}, -9.0)
            row[:] = float(self.filter_truth[key])
            return row
        raise AssertionError(f"unexpected request {path}")

    def _cue_row(self, key):
        tables = self.tables
        gold = self._gold_ids(key)
        lps = {}
        if gold:
            # the model writes the first word with its space after the
            # cue, and every other document word is far behind
            first = same_word(tables.vocab[gold[0]]) or gold[0]
            for index in tables.variants.get(first, [gold[0]]):
                lps[index] = -0.5 if tables.vocab[index].startswith("Ġ") else -0.9
            for index in tables.variants.get("the", []):
                lps[index] = -2.0
            return self._cue(lps, -6.0)
        # without an answer the model leaves the phrase empty
        return self._cue({}, -0.2)

    def _answer_row(self, key, written):
        """The answer's next token while the writing fits the gold, then the stop."""
        tables = self.tables
        gold = self._gold_ids(key)
        keys = [same_word(tables.vocab[t]) or t for t in gold]
        fed = [same_word(tables.vocab[t]) or t for t in written]
        if fed == keys:
            return self._cue({}, -0.1)
        if fed == keys[:len(fed)]:
            return self._cue({gold[len(fed)]: -0.2}, -4.0)
        return self._cue({}, -0.5)


def _fake_readout(tables, k):
    """A readout over scripted rows."""
    width = 1 + 2 * k
    value = fake_readout(k=k, width=width, ragged=True, dtype=np.dtype(object))

    def submit(rows, rows_per_answer=None):
        values = np.asarray(rows, dtype=np.float32)[:, :width]
        return split_rows(values, rows_per_answer)

    value.submit, value.result = submit, lambda handle: handle
    return value


def _state(model, chunk_tokens=400, recurrent=False):
    from quail.backends.quail.executor.pages import PageArena

    arena = cpu_arena(256)
    if recurrent:
        arena.accounting = PageArena(256, 16, num_slots=8)
    return QueryExecutionState(
        loaded_model=LoadedModelState(
            model_spec=QWEN3_4B_FP8,
            arena=arena,
            pipeline=fake_pipeline(forward_chunk=model.forward_chunk),
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object()),
        torch=fake_torch(), chunk_tokens=chunk_tokens, answer_rows=object(),
        async_answers=fake_readout())


DOCS = [LEASE, TERM, NOTHING]
GOLD = ["March 3, 2027", "five years", None]


def _setup(monkeypatch, documents=None, token_documents=None,
           share_prefixes=False, document_ids=None, gpu_timing=False,
           policy="fixed", recurrent=False):
    """A state, its scripted model, and a node builder for one spec name."""
    spec, tables = make_spec()
    spec = replace(spec, draft_policy=policy, round_cost_per_token=0.05)
    documents = DOCS if documents is None else documents
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    state = _state(model, recurrent=recurrent)
    state.loaded_model.vocabulary = tables
    tokens = {"c": ([tables.encode_ids(text) for text in documents]
                    if token_documents is None else token_documents)}
    stages_seen = {}
    original = ExtractionStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for document in self.documents:
            model.documents[document.key] = document.text
            model.gold[document.key] = GOLD[documents.index(document.text)]
        stages_seen["stages"] = self

    monkeypatch.setattr(ExtractionStages, "__init__", init)

    def node(name):
        return AiExtract(node_id=f"ai-extract:{name}", backend_name="quail",
                         model="qwen3-4b-fp8", spec=replace(
                             spec, name=name, share_prefixes=share_prefixes))

    inputs = {"document_ids": (list(range(len(documents))) if document_ids is None
                               else document_ids), "documents": tokens,
              "texts": {"c": documents}, "gpu_timing": gpu_timing}

    def run(name="ends", keep_kv=False):
        result = execute_extractions(
            state, [replace(node(name), keep_kv=keep_kv)], inputs)[node(name).node_id]
        return result, stages_seen["stages"]

    def run_pipeline(*names):
        results = execute_extractions(state, [node(name) for name in names],
                                      inputs)
        return results, stages_seen["stages"]

    return state, model, run, run_pipeline


def test_extraction_borrows_prefix_pages_and_keeps_decode_kv(monkeypatch):
    documents = [LEASE, LEASE, NOTHING]
    _, _, plain, _ = _setup(monkeypatch, documents=documents)
    unshared, _ = plain()
    _, model, shared, _ = _setup(
        monkeypatch, documents=documents, share_prefixes=True)
    result, stages = shared()
    assert result.outputs["rows:c"].table().equals(unshared.outputs["rows:c"].table())
    borrowed = result.metrics.extension["borrowed_prefix_tokens"]
    assert borrowed > 0
    assert result.metrics.fresh_tokens == unshared.metrics.fresh_tokens - borrowed
    for document in stages.documents:
        rounds = [entry for entry in model.packed if entry[0] == document.key]
        if document.answers[0].answer is not None:
            assert len(rounds) > 1
        assert model.reads[document.key] <= 1
        assert all(entry[2] > 0 for entry in rounds)



def test_extraction_accepts_empty_input_and_sparse_document_ids(monkeypatch):
    document_ids = []
    state, _, run, _ = _setup(monkeypatch, document_ids=document_ids)
    result, _ = run()
    table = result.outputs["rows:c"]
    assert table.ids["c"].to_pylist() == document_ids
    assert table.column("ends").to_pylist() == [GOLD[i] for i in document_ids]
    assert table.column("ends_span").is_null().to_pylist() == [
        GOLD[i] is None for i in document_ids]
    spec, vocabulary = make_spec()
    stages = ExtractionStages(
        state, [spec], vocabulary, [10_000_000, 2], None, DOCS,
        document_positions=[0, 1])
    assert stages.document_ids.nbytes + stages.document_order.nbytes == 32
    for document in stages.documents:
        assert stages._document(document.key) is document


def _run(monkeypatch, documents=None):
    _, model, run, _ = _setup(monkeypatch, documents)
    result, stages = run()
    return result, stages, model


def test_extraction_reuses_arrow_tokens_and_offsets(monkeypatch, tmp_path):
    _, vocabulary = make_spec()

    offsets = TokenOffsets({index: text.encode() for index, text
                            in enumerate(vocabulary.tokenizer.tokens)})

    store = TokenStore.write(
        str(tmp_path / "extract.arrow"), [pa.record_batch({"text": DOCS})],
        document_column="text", tokenizer=vocabulary.encode_ids,
        token_type=pa.int32(), offsets_encoder=offsets)
    _, model, run, _ = _setup(monkeypatch, token_documents=store)
    encode_batch = model.tables.encode_batch

    def windows_only(texts):
        assert all(text not in DOCS for text in texts)
        return encode_batch(texts)

    monkeypatch.setattr(model.tables, "encode_batch", windows_only)
    result, stages = run()
    assert result.outputs["rows:c"].column("ends").to_pylist() == GOLD
    assert result.outputs["rows:c"].column("ends_span").to_pylist() == [
        {"start": LEASE.index("March"),
         "end": LEASE.index("March") + len(GOLD[0])},
        {"start": 6, "end": 16}, None]
    for index, document in enumerate(stages.documents):
        assert np.shares_memory(document.scan.tokens, store[index].numpy())
        assert np.shares_memory(document.scan.starts, store[index].character_offsets)
    store.close()


@pytest.mark.parametrize("gpu_timing,recurrent", [(False, False), (True, True)])
def test_a_pipeline_of_extractions_reads_each_document_once(
        monkeypatch, gpu_timing, recurrent):
    state, model, _, run_pipeline = _setup(
        monkeypatch, gpu_timing=gpu_timing, recurrent=recurrent)
    timing_calls = []
    event = state.torch.cuda.Event

    def timed_event(**kwargs):
        timing_calls.append("event")
        return event(**kwargs)

    monkeypatch.setattr(state.torch.cuda, "Event", timed_event)
    monkeypatch.setattr(state.torch.cuda, "synchronize",
                        lambda: timing_calls.append("sync"))
    results, stages = run_pipeline("ends", "starts")
    first, second = results["ai-extract:ends"], results["ai-extract:starts"]
    assert second.outputs["rows:c"].column("starts").to_pylist() == \
        first.outputs["rows:c"].column("ends").to_pylist()
    assert second.outputs["rows:c"].column("starts_span").to_pylist() == \
        first.outputs["rows:c"].column("ends_span").to_pylist()
    assert dict(model.reads) == {document.key: 1 for document in stages.documents}
    # the documents are read once: the second extraction is charged
    # only the tokens fed after them
    prefixes = sum(len(document.prefix) for document in stages.documents)
    fed = [stages.counts(q)["suffix_tokens"] for q in range(2)]
    assert first.metrics.fresh_tokens == prefixes + fed[0]
    assert second.metrics.fresh_tokens == fed[1] > 0
    assert (first.metrics.gpu_s > 0) == gpu_timing
    assert (second.metrics.gpu_s > 0) == gpu_timing
    if gpu_timing:
        assert "event" in timing_calls and timing_calls.count("sync") == 1
    else:
        assert timing_calls == []


@pytest.mark.parametrize("gpus", [2, 4])
def test_distributed_extraction_preserves_spans_order_and_kv(monkeypatch, gpus):
    import pickle

    state, model, _, run_pipeline = _setup(monkeypatch, share_prefixes=True)
    expected, _ = run_pipeline("ends", "starts")
    spec, tables = make_spec()
    nodes = [Scan(node_id="input:c", alias="c", input_id="c")]
    for name in ("ends", "starts"):
        previous = nodes[-1]
        nodes.append(AiExtract(
            node_id=f"ai-extract:{name}", backend_name="quail", model="qwen3-4b-fp8",
            inputs=input_ports((PortRef(previous.node_id, "rows:c"),)),
            spec=replace(spec, name=name, share_prefixes=True),
            keep_kv=name == "starts"))
    graph = PhysicalGraph(tuple(nodes), PortRef(nodes[-1].node_id, "rows:c"))
    registry = built_in_registry()
    children = []
    for _ in range(gpus):
        torch, arena = fake_torch(), cpu_arena(256)
        pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
        execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
        execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
        execution.loaded_model.vocabulary = tables
        execution.loaded_model.input_staging = SimpleNamespace(fixed_tokens=set())
        execution.bind_query(torch=torch, answer_rows=object(),
                             chunk_tokens=state.chunk_tokens,
                             async_answers=fake_readout())
        children.append(ChildState(gpu_index=0, registry=registry, gpu=SimpleNamespace(
            execution=execution, torch=torch, arena=arena, pipeline=pipeline,
            spec=MODELS["qwen3-4b-fp8"])))
    payload = {
        "model": "qwen3-4b-fp8", "chunk_tokens": state.chunk_tokens,
        "true_ids": [1], "false_ids": [2],
        "documents": {"c": tables.document_batch(DOCS, [None] * len(DOCS))},
        "texts": {"c": pa.array(DOCS)},
        "physical_plan": plan_envelope(
            backend="quail", model="qwen3-4b-fp8", device="h100-sxm",
            gpus=gpus, graph=graph, codecs=registry.codecs),
    }
    rounds = []

    def round_fn(kind, subs):
        assert kind == "execute"
        subs = pickle.loads(pickle.dumps(subs))
        rounds.append(subs)
        return [_child_execute(child, sub) for child, sub in zip(children, subs)]

    result = execute_distributed_graph(
        payload, graph, gpus, round_fn, DEVICES["h100-sxm"], registry.runtimes,
        registry)
    assert len(rounds) == 1
    # every worker ran the whole pipeline over its own documents
    shards = [sub["inputs"]["ai-extract:ends"]["input:0"].ids["c"].to_pylist()
              for sub in rounds[0]]
    assert sorted(document for shard in shards for document in shard) == [0, 1, 2]
    assert all(sub["pipeline"] and sub["final"] for sub in rounds[0])
    for node in nodes[1:]:
        actual = result["_outputs"][PortRef(node.node_id, "rows:c")]
        assert actual.equals(expected[node.node_id].outputs["rows:c"].table())
    assert result["fresh_tokens"] == sum(
        item.metrics.fresh_tokens for item in expected.values())
    assert result["evaluated_documents"] == 2 * len(DOCS)
    assert result["backend_metrics"]["inferences"][0]["none"] == 1
    assert all(not child.gpu.arena.accounting.owned for child in children)


def test_an_extraction_after_a_filter_asks_the_filtered_documents_once(
        monkeypatch):
    # the filter writes each document under the query's shared head;
    # the plain extraction continues its pipeline, so a document is
    # packed once and the extraction feeds only its question
    spec, tables = make_spec()
    plain = spec
    head = list(plain.prompt_token_parts[0])
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    for index, text in enumerate(DOCS):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = GOLD[index]
        model.filter_truth[("c", index)] = index != 2
    question = tables.encode_ids(" question")
    graph = PhysicalGraph(
        (
            Scan(node_id="input:c", alias="c", input_id="c"),
            AiFilter(
                node_id="ai_filter:c",
                inputs=input_ports((PortRef("input:c", "rows:c"),)),
                alias="c",
                arena_writes=True,
                question_token_ids=tuple(question),
                written_position=0,
                question_tokens=len(question),
                preamble_tokens=len(head),
                document_head=tuple(head),
                selectivity=0.7,
                expected_rows=2.0,
            ),
            AiExtract(
                node_id="ai-extract:ends",
                inputs=input_ports((PortRef("ai_filter:c", "rows:c"),)),
                backend_name="quail",
                model="qwen3-4b-fp8",
                spec=plain,
            ),
            AiExtract(
                node_id="ai-extract:starts",
                inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
                backend_name="quail",
                model="qwen3-4b-fp8",
                spec=replace(plain, name="starts"),
            ),
        ),
        PortRef("ai-extract:starts", "rows:c"),
    )
    torch, arena = fake_torch(), cpu_arena(256)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = SimpleNamespace(fixed_tokens=set())
    execution.bind_query(
        torch=torch, answer_rows=object(), chunk_tokens=400,
        async_answers=fake_readout(
            submit=lambda rows: [float(np.asarray(r).ravel()[0]) for r in rows],
            result=lambda v: v, dtype=None))
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"],
             "chunk_tokens": 400,
             "documents": {"c": [tables.encode_ids(text) for text in DOCS]},
             "texts": {"c": list(DOCS)}, "gpu_timing": True}
    report = execute_single_graph(
        state, {"filter_limit": None, "retention": {}}, graph)
    table = report["_outputs"][PortRef("ai-extract:ends", "rows:c")].to_pydict()
    assert table["c"] == [0, 1]
    assert table["ends"] == ["March 3, 2027", "five years"]
    # every document was packed once: the extraction's tail followed
    # the filter's question over the resident KV
    assert dict(model.reads) == {("c", 0): 1, ("c", 1): 1, ("c", 2): 1}
    metrics = report["node_metrics"]
    documents = sum(len(head) + len(tokens) for tokens in state["documents"]["c"])
    assert metrics["ai-extract:ends"]["fresh_tokens"] < documents
    assert metrics["ai_filter:c"]["fresh_tokens"] >= documents
    assert metrics["ai-extract:ends"]["evaluated_documents"] == 2
    later = report["_outputs"][PortRef("ai-extract:starts", "rows:c")].to_pydict()
    assert later["starts"] == table["ends"]
    assert later["starts_span"] == table["ends_span"]
    assert metrics["ai-extract:ends"]["gpu_s"] > 0
    assert metrics["ai-extract:starts"]["gpu_s"] > 0
    assert not arena.accounting.owned


def test_a_second_extraction_reads_the_documents_resident_kv(monkeypatch):
    state, model, run, _ = _setup(monkeypatch)
    first, stages = run("ends", keep_kv=True)
    arena = state.loaded_model.arena
    # every document's KV stays resident after the extraction
    assert all(arena.is_resident(document.key) for document in stages.documents)
    assert all(arena.prefix_tokens(document.key) == len(document.prefix)
               for document in stages.documents)
    model.kv = {}
    second, again = run("starts")
    assert [document.key for document in again.documents] == [
        document.key for document in stages.documents
    ]
    assert second.outputs["rows:c"].column("starts").to_pylist() == \
        first.outputs["rows:c"].column("ends").to_pylist()
    # the second extraction feeds no document: only its tail, lines,
    # cue, and spans are fresh
    prefixes = sum(len(document.prefix) for document in stages.documents)
    assert first.metrics.fresh_tokens > prefixes
    assert second.metrics.fresh_tokens == first.metrics.fresh_tokens - prefixes
    assert not arena.resident_keys()


@pytest.mark.parametrize("limit,draft", [(5, 16), (3, 0), (1, 0)])
def test_extractor_stops_at_the_answer_token_cap(monkeypatch, limit, draft):
    # a model that never ends its turn keeps writing the document;
    # the answer ends at stages.spec.max_answer_tokens
    long = " ".join(["word"] * 200)
    spec, tables = make_spec(long)
    spec = replace(spec, max_answer_tokens=limit, draft_tokens=draft)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    state = _state(model, chunk_tokens=2000)
    state.loaded_model.vocabulary = tables
    stages = ExtractionStages(state, [spec], tables, [0],
                           [tables.encode_ids(long)], [long])
    model.documents[stages.documents[0].key] = long
    model.gold[stages.documents[0].key] = long
    from quail.backends.quail.executor.stages import run_stages

    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys)
    stages.finish()
    asked = stages.documents[0].answers[0]
    assert asked.done and len(asked.written) == stages.spec.max_answer_tokens
    assert asked.answer == " ".join(["word"] * stages.spec.max_answer_tokens)
    assert all(len(suffix) <= 1 + draft
               for _, suffix, _, whole, _ in model.packed if whole)


def test_top_logprob_readout_sums_a_group_of_extras():
    import torch

    from quail.backends.quail.executor.readout import AsyncTopLogprobs

    head = torch.eye(6, dtype=torch.float32)
    readout = AsyncTopLogprobs(torch, torch.nn.functional, head, 2,
                               [(0, 1), 2])
    normed = torch.tensor([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]])
    (row,) = readout.values(normed).tolist()
    logprobs = torch.log_softmax(normed[0], dim=0).tolist()
    assert row[0] == pytest.approx(np.logaddexp(logprobs[0], logprobs[1]))
    assert row[1] == pytest.approx(logprobs[2])
    assert row[2:4] == [0.0, 1.0] and row[4:] == pytest.approx(logprobs[:2])


def test_a_filter_on_answers_drops_documents_before_the_next_question(
        monkeypatch):
    # the second question is asked of the documents whose first answer
    # passed, over the KV the first question left; the dropped
    # document is packed once and its KV freed
    from quail.physical import Filter
    from quail.physical.pipelines import build_pipelines

    spec, tables = make_spec()
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    for index, text in enumerate(DOCS):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = GOLD[index]
    graph = PhysicalGraph((
        Scan(node_id="input:c", alias="c", input_id="c"),
        AiExtract(node_id="ai-extract:ends",
                  inputs=input_ports((PortRef("input:c", "rows:c"),)),
                  backend_name="quail", model="qwen3-4b-fp8", spec=spec),
        Filter(node_id="filter:c:0",
               inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
               predicate=RegularPredicate(
                   ColumnRef("c", "contracts", "ends"), "not like", "five%"),
               aliases=("c",), written_position=0),
        AiExtract(node_id="ai-extract:starts",
                  inputs=input_ports((PortRef("filter:c:0", "rows:c"),)),
                  backend_name="quail", model="qwen3-4b-fp8",
                  spec=replace(spec, name="starts")),
        Filter(node_id="filter:c:1",
               inputs=input_ports((PortRef("ai-extract:starts", "rows:c"),)),
               predicate=RegularPredicate(
                   GetField(ColumnRef("c", "contracts", "starts_span"), "start"),
                   "<", 40),
               aliases=("c",), written_position=1),
    ), PortRef("filter:c:1", "rows:c"))
    assert [member.node_id for member in build_pipelines(graph)["ai-extract:ends"]
            ] == ["ai-extract:ends", "filter:c:0", "ai-extract:starts", "filter:c:1"]
    torch, arena = fake_torch(), cpu_arena(256)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = SimpleNamespace(fixed_tokens=set())
    execution.bind_query(torch=torch, answer_rows=object(), chunk_tokens=400,
                         async_answers=fake_readout())
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"], "chunk_tokens": 400,
             "documents": {"c": [tables.encode_ids(text) for text in DOCS]},
             "texts": {"c": list(DOCS)}}
    report = execute_single_graph(
        state, {"filter_limit": None, "retention": {}}, graph)
    outputs = report["_outputs"]
    first = outputs[PortRef("ai-extract:ends", "rows:c")].to_pydict()
    assert first["ends"] == ["March 3, 2027", "five years", None]
    kept = outputs[PortRef("filter:c:0", "rows:c")].to_pydict()
    assert kept["c"] == [0] and kept["ends"] == ["March 3, 2027"]
    # the second question was asked of the survivor only
    second = outputs[PortRef("ai-extract:starts", "rows:c")].to_pydict()
    assert second == {"c": [0], "starts": ["March 3, 2027"],
                      "starts_span": [first["ends_span"][0]]}
    assert first["ends_span"][0] == {"start": 37, "end": 50}
    assert outputs[PortRef("filter:c:1", "rows:c")].to_pydict()["c"] == [0]
    assert outputs[PortRef("filter:c:1", "rows:c")].column("c").to_pylist() == [0]
    assert dict(model.reads) == {("c", 0): 1, ("c", 1): 1, ("c", 2): 1}
    metrics = report["node_metrics"]
    assert metrics["ai-extract:starts"]["evaluated_documents"] == 1
    assert metrics["filter:c:0"]["output_rows"] == 1
    assert not arena.accounting.owned


def test_a_filter_after_an_extraction_rewinds_the_kv_past_the_answer(
        monkeypatch):
    # the filter's question follows the document in KV, in place of
    # the answer the extraction wrote; the document is packed once
    from quail.physical import Filter
    from quail.physical.pipelines import build_pipelines

    spec, tables = make_spec()
    head = list(spec.prompt_token_parts[0])
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    for index, text in enumerate(DOCS):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = GOLD[index]
        model.filter_truth[("c", index)] = index != 1
    question = tables.encode_ids(" question")
    graph = PhysicalGraph(
        (
            Scan(node_id="input:c", alias="c", input_id="c"),
            AiExtract(
                node_id="ai-extract:ends",
                inputs=input_ports((PortRef("input:c", "rows:c"),)),
                backend_name="quail",
                model="qwen3-4b-fp8",
                spec=spec,
            ),
            Filter(
                node_id="filter:c:1",
                inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
                predicate=RegularPredicate(
                    ColumnRef("c", "contracts", "ends"), "is not null"
                ),
                aliases=("c",),
                written_position=1,
            ),
            AiFilter(
                node_id="ai_filter:c",
                inputs=input_ports((PortRef("filter:c:1", "rows:c"),)),
                alias="c",
                arena_writes=True,
                question_token_ids=tuple(question),
                written_position=0,
                question_tokens=len(question),
                preamble_tokens=len(head),
                document_head=tuple(head),
                selectivity=0.7,
                expected_rows=2.0,
            ),
        ),
        PortRef("ai_filter:c", "rows:c"),
    )
    assert [member.node_id for member in build_pipelines(graph)["ai-extract:ends"]
            ] == ["ai-extract:ends", "filter:c:1", "ai_filter:c"]
    torch, arena = fake_torch(), cpu_arena(256)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = SimpleNamespace(fixed_tokens=set())
    execution.bind_query(
        torch=torch, answer_rows=object(), chunk_tokens=400,
        async_answers=fake_readout(
            submit=lambda rows: [float(np.asarray(r).ravel()[0]) for r in rows],
            result=lambda v: v, dtype=None))
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"], "chunk_tokens": 400,
             "documents": {"c": [tables.encode_ids(text) for text in DOCS]},
             "texts": {"c": list(DOCS)}}
    report = execute_single_graph(
        state, {"filter_limit": None, "retention": {}}, graph)
    outputs = report["_outputs"]
    assert outputs[PortRef("ai-extract:ends", "rows:c")].to_pydict()["ends"] == [
        "March 3, 2027", "five years", None]
    # the test drops the third document; the filter asks the other two
    # and keeps the first
    assert outputs[PortRef("filter:c:1", "rows:c")].column("c").to_pylist() == [0, 1]
    assert outputs[PortRef("ai_filter:c", "rows:c")].column("c").to_pylist() == [0]
    assert dict(model.reads) == {("c", 0): 1, ("c", 1): 1, ("c", 2): 1}
    # the filter's question was fed right after the document, in place
    # of the extraction's cue and answer
    fed = [(key, suffix, f) for key, suffix, kept, _, f in model.packed
           if suffix == question]
    assert [(key, f) for key, _, f in fed] == [
        (("c", index), len(head) + len(state["documents"]["c"][index]))
        for index in (0, 1)]
    assert report["node_metrics"]["ai_filter:c"]["evaluated_documents"] == 2
    assert not arena.accounting.owned


@pytest.mark.parametrize("limit", [1, 2])
def test_a_limit_ends_the_pipeline_once_enough_documents_pass(monkeypatch, limit):
    # the result's limit stops the run after that many documents
    # passed every operator, the trailing test included; the documents
    # still queued are not read
    from quail.physical import Filter

    spec, tables = make_spec()
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    # the first four documents have no answer, so the test drops them;
    # some of the rest are still queued when enough have passed
    texts = [NOTHING] * 4 + [LEASE] * 4 + [TERM] * 4
    gold = [None] * 4 + [GOLD[0]] * 4 + [GOLD[1]] * 4
    for index, text in enumerate(texts):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = gold[index]
    graph = PhysicalGraph((
        Scan(node_id="input:c", alias="c", input_id="c"),
        AiExtract(node_id="ai-extract:ends",
                  inputs=input_ports((PortRef("input:c", "rows:c"),)),
                  backend_name="quail", model="qwen3-4b-fp8", spec=spec),
        Filter(node_id="filter:c:0",
               inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
               predicate=RegularPredicate(
                   ColumnRef("c", "contracts", "ends"), "is not null"),
               aliases=("c",), written_position=0),
    ), PortRef("filter:c:0", "rows:c"))
    torch, arena = fake_torch(), cpu_arena(256)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = SimpleNamespace(fixed_tokens=set())
    documents = [tables.encode_ids(text) for text in texts]
    # a chunk holds about one document and its question
    chunk = max(len(document) for document in documents) + len(
        spec.prompt_token_parts[0]) + len(spec.prompt_token_parts[1]) + 8
    execution.bind_query(torch=torch, answer_rows=object(), chunk_tokens=chunk,
                         async_answers=fake_readout())
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"], "chunk_tokens": chunk,
             "documents": {"c": documents}, "texts": {"c": list(texts)}}
    report = execute_single_graph(
        state, {"filter_limit": limit, "retention": {}}, graph)
    outputs = report["_outputs"]
    kept = outputs[PortRef("filter:c:0", "rows:c")].to_pydict()
    # the documents that passed before the run ended have their whole
    # answers; the queued ones are never read
    assert limit <= len(kept["c"]) < len(texts)
    assert all(answer is not None for answer in kept["ends"])
    assert kept["ends"][:limit] == [gold[index] for index in kept["c"][:limit]]
    assert sum(model.reads.values()) < len(texts)
    assert not arena.accounting.owned


def test_operators_after_grouped_extractions_read_the_last_member(monkeypatch):
    from quail.backends.quail.executor.operators import classify as classify_module
    from quail.backends.quail.executor.operators.apply import ApplyGate
    from quail.physical import AiClassify, ClassifySpec, Foreign, PortRef

    spec, _ = make_spec()
    extracts = [AiExtract(node_id=f"ai-extract:{name}", backend_name="quail",
                          model="qwen3-4b-fp8", spec=replace(spec, name=name))
                for name in ("ends", "starts")]
    group = SimpleNamespace(nodes=extracts)
    last = PortRef("ai-extract:starts", "rows:c")
    monkeypatch.setattr(classify_module, "ClassifyStages",
                        lambda *args, **kwargs: SimpleNamespace(
                            partners=None, stages=(), labels=[]))
    monkeypatch.setattr(classify_module, "answer_sink", lambda: None)
    classify = AiClassify(
        node_id="classify:c", backend_name="quail", model="qwen3-4b-fp8",
        inputs=input_ports((last,)),
        spec=ClassifySpec(name="topic", aliases=("c",), query_template="",
                          arguments=(), expected_rows=3, estimated_seconds=0.0,
                          labels=("a", "b"), label_token_ids=((13,), (14,)),
                          scoring="letters"))
    part = classify_module.classification_part(None, classify, [0, 1, 2], [group], {})
    assert part.priors == [(group, last)] and part.prior_tables == []
    monkeypatch.setattr("quail.backends.quail.executor.operators.apply.foreign_call",
                        lambda node, values, context: (values, None))
    apply = Foreign(node_id="apply:keep", inputs=input_ports((last,)),
                    function="keep", kind="per_batch", ids="drop", columns=(),
                    aliases=("c",))
    gate = ApplyGate.prepare(None, [apply], [0, 1, 2], [group], {"apply:keep": {}},
                             None)
    assert gate.call == {"c": [0, 1, 2]}
