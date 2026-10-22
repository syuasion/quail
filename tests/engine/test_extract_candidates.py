"""The extraction's candidates: word keys, first tokens, and streams."""

import numpy as np
import pytest
from helpers.extract_fakes import RANKED, make_vocabulary, squeezed

from quail.backends.quail.executor.extract.candidates import (
    DocumentTokens,
    SqueezedText,
    Stream,
    ensure_tokens,
    first_tokens,
    make_streams,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.catalog.tokens import GigatokenEncoder


def test_gigatoken_vocabulary_encodes_unicode_without_transformers(monkeypatch):
    import sys

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    texts = ['  Café\n\n東京 🙂', 'none "March"', "", "a  b"]
    native = Tokenizer(models.BPE())
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=256, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|im_end|>"]))
    expected = native.encode_batch(texts, add_special_tokens=False)
    encoder = GigatokenEncoder(native)
    monkeypatch.setitem(sys.modules, "transformers", None)
    tables = Vocabulary(encoder, "<|im_end|>\nassistant\n")
    monkeypatch.setattr(encoder, "batch_arrow", lambda _: pytest.fail(
        "continuations must not convert through Arrow token documents"))
    # the answer ends at the turn close alone: a quote is document text
    assert tables.stop_ids == (native.token_to_id("<|im_end|>"),)
    for (ids, offsets), original in zip(tables.encode_batch(texts), expected):
        assert list(ids) == original.ids
        assert offsets.tolist() == [list(pair) for pair in original.offsets]
    text = 'n  Café\n東京 🙂 ' * 80
    stream = Stream(0, tables.encode_ids("n"), [1], text)
    stream.next(100_000, tables)
    normalized, character_ends = squeezed(text)
    expected = native.encode(normalized, add_special_tokens=False)
    assert stream.stream == expected.ids
    assert stream.ends == [character_ends[end - 1] for _, end in expected.offsets]
    encode = encoder.batch_numpy
    monkeypatch.setattr(encoder, "batch_numpy", lambda texts: encode(["ab"]))
    with pytest.raises(ValueError, match="preserve the source text"):
        tables.encode("a b")


def test_first_tokens_follow_the_models_spelling():
    tables = make_vocabulary()
    text = RANKED
    ids, offsets = tables.encode(text)
    scan = DocumentTokens.from_tokens(tables, ids, offsets, 1)
    the, march = tables.encode_ids("The")[0], tables.encode_ids("March")[0]
    space_march = tables.encode_ids(" March")[0]
    lp = {the: -0.5, march: -1.0, space_march: -3.0}
    starts = first_tokens(tables, _scores(tables, lp), scan)
    by_char = {scan.starts[i - 1]: value for i, value in starts.items(tables)}
    # the model writes "March" without its space; the document's token
    # has one, and the likelier spelling starts it; "The" twice
    assert by_char[text.index(" March")] == (march, [])
    assert by_char[0] == (the, []) and by_char[26] == (the, [])
    assert set(by_char) == {0, 26, text.index(" March")}
    assert {scan.starts[i - 1] for i in starts} == set(by_char)
    # the chosen first token names the streams, each the text from
    # its start as the model would write it, with its character ends
    streams = make_streams(tables, text, starts, the)
    assert [stream.char for stream in streams] == [0, 26]
    # a stream holds its first token until the writing reaches it
    assert streams[1].stream == [the]
    streams[1].extend(tables)
    assert streams[1].stream[:3] == tables.encode_ids("The rent is")
    # a start's offset includes its token's leading space
    assert text[streams[1].char:streams[1].ends[2]] == " The rent is"
    (march_stream,) = make_streams(tables, text, starts, march)
    march_stream.extend(tables)
    assert march_stream.ends[:4] == [23, 24, 25, 26]
    assert text[march_stream.char:march_stream.ends[3]] == " March 3."


def test_a_first_token_that_begins_a_longer_word_carries_its_rest():
    text = "The lease is their lease."
    tables = make_vocabulary(text, "their", "ir")
    ids, offsets = tables.encode(text)
    scan = DocumentTokens.from_tokens(tables, ids, offsets, 1)
    the = tables.encode_ids("The")[0]
    starts = first_tokens(tables, _scores(tables, {the: -0.5}), scan)
    by_char = {scan.starts[i - 1]: value for i, value in starts.items(tables)}
    # "the" begins "their" too: that start continues with "ir"
    assert by_char[0] == (the, [])
    assert by_char[12] == (the, tables.encode_ids("ir"))
    streams = make_streams(tables, text, starts, the)
    assert [stream.char for stream in streams] == [0, 12]
    assert streams[1].stream[:2] == [the, tables.encode_ids("ir")[0]]
    assert text[streams[1].char:streams[1].ends[1]] == " their"

@pytest.mark.parametrize("first_window,window", [(64, 512), (9, 19)])
def test_stream_batches_preserve_tokens_and_offsets_across_windows(
        monkeypatch, first_window, window):
    texts = [" ".join(["word"] * 300), "word   \n word  ", "word"]
    tables = make_vocabulary(*texts)
    streams = []
    expected = []
    word = tables.encode_ids("word")[0]
    for text in texts:
        ids, offsets = tables.encode(text)
        scan = DocumentTokens.from_tokens(tables, ids, offsets)
        starts = first_tokens(tables, _scores(tables, {word: -0.5}), scan)
        streams.append(make_streams(tables, text, starts, word,
                                    first_window_chars=first_window,
                                    window_chars=window)[0])
        rest, ends = squeezed(text)
        ids, offsets = tables.encode(rest)
        expected.append((ids, [ends[b - 1] for _, b in offsets]))
    batches = []
    encode_batch = tables.encode_batch

    def recorded(texts):
        batches.append(tuple(texts))
        return encode_batch(texts)

    def scalar(text):
        raise AssertionError("stream windows must be tokenized together")

    monkeypatch.setattr(tables, "encode_batch", recorded)
    monkeypatch.setattr(tables, "encode", scalar)
    ensure_tokens(streams, 80, tables)
    assert len(batches[0]) == 2
    largest_window = max(len(text) for batch in batches for text in batch)
    assert largest_window <= max(first_window, window) + 4
    assert not streams[0].exhausted
    assert streams[1].exhausted and streams[2].exhausted
    ensure_tokens(streams, 10_000, tables)
    assert [(stream.stream, stream.ends) for stream in streams] == expected
    assert all(stream.exhausted for stream in streams)
    before = list(batches)
    ensure_tokens(streams, 10_000, tables)
    assert batches == before


def _scores(vocabulary, pairs):
    values = np.full(len(vocabulary.vocab), -np.inf, dtype=np.float32)
    for token, score in pairs.items():
        values[token] = score
    return values


def test_normalized_windows_preserve_partial_whitespace_runs():
    text = " a\t\tbc\r\n  d\u2003e\u00a0 f   "
    normalized = SqueezedText(text)
    for start in range(len(text) + 1):
        for end in range(start, len(text) + 1):
            want_text, want_ends = squeezed(text[start:end])
            got_text, got_ends = normalized.window(start, end)
            assert got_text == want_text
            assert got_ends.tolist() == [start + value for value in want_ends]


def test_candidate_ties_prefer_longer_spelling():
    text = "their"
    tables = make_vocabulary(text, "the", "The", "ir")
    ids, offsets = tables.encode(text)
    whole, short = (tables.encode_ids(word)[0] for word in ("their", "The"))
    scan = DocumentTokens.from_tokens(tables, ids, offsets)
    tied = first_tokens(tables, _scores(tables, {whole: -1, short: -1}),
                        scan)
    assert tied.best_token == whole
    higher = first_tokens(tables, _scores(tables, {whole: -2, short: -1}),
                          scan)
    assert higher.best_token == short
    assert list(higher.items(tables)) == [(0, (short, tables.encode_ids("ir")))]
