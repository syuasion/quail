"""The decode of an answer: the token rule applied to each readout row.

The answer is decoded after the phrase cue as a decoder writes it, but
each token is chosen among the tokens that continue some position of
the document: the span is the document's own text wherever the
model's writing leads. Each round feeds the chosen token and the
document's next tokens as a draft, and the rows after them verify it.
The answer ends where the model would end its turn, or where its
writing would leave the document.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.extract.candidates import (
    NEG,
    DocumentTokens,
    ensure_tokens,
    first_tokens,
    make_streams,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.cost.draft_policy import DraftPolicy
from quail.physical import ExtractSpec


@dataclass
class AnswerState:
    """One document's rounds and result for one question."""

    asked: bool = False     # the document reached the question's stages
    best_lp: float = NEG
    written: list = field(default_factory=list)   # the answer's tokens so far
    streams: list = field(default_factory=list)      # the drafts the writing fits
    fed: list = field(default_factory=list)       # the last round's tokens
    discard: int = 0    # tokens of the last round past what it verified
    done: bool = False
    steps: int = 0
    fed_tokens: int = 0
    answer: str | None = None
    span: tuple | None = None
    none_wins: bool = False


@dataclass
class Document:
    """One document's prompt and its answer to each question."""

    index: int
    text: str
    prefix: Sequence
    scan: DocumentTokens
    key: tuple
    answers: list       # one AnswerState per question
    rho: float = 0.0    # the document's KV read per round, in fed tokens


class Decoder:
    """The token rule over one model's vocabulary, for a pipeline's questions.

    Args:
        vocabulary: The model's vocabulary index.
        gpus: How many likeliest tokens each readout row carries.
        width: The values in each readout row: the stop, then k ids
            and k log probabilities.
        questions: How many questions the pipeline asks.
        spec: Extraction plan carrying the execution settings.
    """

    def __init__(self, vocabulary: Vocabulary, k: int, width: int, questions: int,
                 spec: ExtractSpec):
        self.spec = spec
        self.vocabulary = vocabulary
        self.k = k
        self.width = width
        self.scores = np.full(len(vocabulary.vocab), NEG, dtype=np.float32)
        self.previous_ids = np.empty(0, dtype=np.int64)
        # one answer-length curve per question under the adaptive policy
        self.policies = ([DraftPolicy(spec.draft_tokens, spec.max_answer_tokens,
                                      spec.expected_answer_tokens)
                          for _ in range(questions)]
                         if spec.draft_policy == "adaptive" else None)

    def rows(self, row) -> np.ndarray:
        """Return a stage's answer as its rows of readout values."""
        values = np.asarray(row)
        if values.dtype == object:
            values = np.asarray(values.reshape(-1)[0], dtype=np.float32)
        return values.reshape(-1, self.width)

    def draft(self, answer: AnswerState, rho: float = 0.0, q: int = 0) -> list:
        """Return the tokens the next round feeds: the chosen token, then a draft.

        The draft is the document's next tokens along the first
        position the writing fits; the rows after them verify it, so a
        round writes as many tokens as the model would have chosen one
        at a time. Its length is the plan's, or under the adaptive
        policy the length that costs the least per written token for
        this document's KV read and the answer so far.

        Args:
            answer: The answer's state.
            rho: The document's KV read per round, in fed tokens.
            q: The question's index.
        """
        at = len(answer.written)
        stream = min(answer.streams, key=lambda s: s.char)
        length = self.spec.draft_tokens
        if self.policies is not None:
            length = self.policies[q].draft_length(rho, at, len(answer.streams))
        draft = []
        for i in range(length):
            token = stream.next(at + i, self.vocabulary)
            if token is None:
                break
            draft.append(token)
        answer.fed = [answer.written[-1], *draft]
        return answer.fed

    def lp(self, values) -> tuple:
        """Return a row's log probabilities by token id, and the stop's."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        k = self.k
        self.scores[self.previous_ids] = NEG
        ids = values[-2 * k:-k].astype(np.int64)
        self.scores[ids] = values[-k:]
        self.previous_ids = ids
        self.top = int(ids[np.argmax(self.scores[ids])])
        return self.scores, float(values[0])

    def verify(self, document: Document, q: int, rows, fed) -> None:
        """Choose the answer's tokens from a round's rows, in order.

        Row i follows token i of the round's fed tokens, so the token
        chosen from it is written and, when it is not the token fed
        next, the rest of the round is discarded: the next round feeds
        the chosen token in its place. With one live stream the rows
        that write the draft are decided together.

        Args:
            document: The document.
            q: The question's index.
            rows: The readout rows of the round.
            fed: The round's fed tokens.
        """
        answer = document.answers[q]
        answer.steps += 1
        answer.discard = 0
        rows = np.asarray(rows)
        # a draft along one of several live positions tells the policy
        # whether that position was the answer's source
        num_streams = (
            len(answer.streams) if answer.written and len(fed) > 1 else 1
        )
        accepted = 0
        if answer.written and len(answer.streams) == 1:
            accepted = self.accepted(answer, rows, fed)
            rows = rows[accepted:]
        if accepted:
            self.observe(q, num_streams, True)
            written = min(accepted, self.spec.max_answer_tokens - len(answer.written))
            answer.written.extend(fed[1:written + 1])
            if len(answer.written) >= self.spec.max_answer_tokens:
                self.settle(document, q)
                return
        for i, values in enumerate(rows):
            token = self.choose(document, q, values)
            if answer.done:
                return
            after = accepted + i + 1
            if after == 1 and num_streams > 1:
                self.observe(q, num_streams, token == fed[1])
            if after < len(fed) and token != fed[after]:
                answer.discard = len(fed) - after
                return

    def observe(self, q: int, num_streams: int, accepted: bool) -> None:
        if self.policies is not None and num_streams > 1:
            self.policies[q].observe(num_streams, accepted)

    def accepted(self, answer: AnswerState, rows: np.ndarray, fed) -> int:
        """Return how many leading rows write the draft token fed after them.

        With one live stream the draft is that stream's next tokens, so
        a row writes the draft token exactly when choose() would: the
        token beats the stop and the model's likeliest token stays in
        the document. Those tests read a few values per row, so every row
        of the round is decided at once; the first row that fails, and
        the rows after it, go through choose().
        """
        k = self.k
        vocabulary = self.vocabulary
        n = min(len(rows), len(fed) - 1,
                self.spec.max_answer_tokens - len(answer.written))
        if n <= 0:
            return 0
        values = rows[:n]
        ids = values[:, -2 * k:-k].astype(np.int64)
        lps = values[:, -k:]
        draft = np.asarray(fed[1:n + 1], dtype=np.int64)
        index = np.arange(n)
        hit = ids == draft[:, None]
        lp_draft = np.where(hit.any(axis=1), lps[index, hit.argmax(axis=1)], NEG)
        top_col = lps.argmax(axis=1)
        top, lp_top = ids[index, top_col], lps[index, top_col]
        stop = values[:, 0]
        leaves = ((lp_top > np.maximum(lp_draft, stop)) & (top != draft)
                  & ~vocabulary.is_stop[top])
        writes = (lp_draft > stop) & ~leaves
        return int(np.argmin(writes)) if not writes.all() else n

    def choose(self, document: Document, q: int, values):
        """Choose the answer's next token among the document's continuations.

        The answer is the document text the model writes before its
        writing would leave the document: a token it likes better than
        every continuation and the stop, and that continues no
        position, ends the answer, and at the cue it answers none, as
        does the model ending its turn at once, the empty phrase the
        prompt asks for when there is no answer. At the cue, the token is the
        likeliest first token of any position in the document;
        afterwards, the likeliest continuation of the streams the
        writing so far fits. The answer also ends when ending the turn
        is at least as likely as every continuation, when no stream
        continues, or at the configured answer limit.

        Returns:
            The token written, or None when the answer ended.
        """
        answer = document.answers[q]
        vocabulary = self.vocabulary
        lp, stop_lp = self.lp(values)
        if not answer.written:
            starts = first_tokens(vocabulary, lp, document.scan)
            answer.best_lp = starts.best_lp
            answer.none_wins = (stop_lp > answer.best_lp
                                or self.leaves(lp, starts.allows(self.top),
                                               answer.best_lp, stop_lp))
            if answer.none_wins:
                self.settle(document, q)
                return None
            token = starts.best_token
            answer.streams = make_streams(
                vocabulary, document.text, starts, token,
                first_window_chars=self.spec.first_window_chars,
                window_chars=self.spec.window_chars)
        else:
            at = len(answer.written)
            ensure_tokens(answer.streams, at, vocabulary)
            best, allowed, next_tokens = self.continuations(answer, at, lp)
            best_lp = NEG if best is None else lp[best]
            if (best is None or stop_lp >= best_lp
                    or self.leaves(lp, allowed, best_lp, stop_lp)):
                self.settle(document, q)
                return None
            token = best
            if next_tokens is not None:
                answer.streams = [
                    s for s, keep in zip(answer.streams, next_tokens == token) if keep
                ]
        answer.written.append(token)
        if len(answer.written) >= self.spec.max_answer_tokens:
            self.settle(document, q)
        return token

    def continuations(self, answer: AnswerState, at: int, lp) -> tuple:
        """Return the likeliest next token of the live streams.

        Returns:
            The token, or None when no stream continues; whether the
            model's likeliest token continues a stream; and each
            stream's next token, or None with one stream.
        """
        if len(answer.streams) == 1:
            stream = answer.streams[0].stream
            best = stream[at] if at < len(stream) else None
            return best, self.top == best, None
        next_tokens = np.fromiter(
            (s.stream[at] if at < len(s.stream) else -1 for s in answer.streams),
            dtype=np.int64, count=len(answer.streams))
        following = np.unique(next_tokens[next_tokens >= 0])
        best = int(following[np.argmax(lp[following])]) if len(following) else None
        return best, bool(np.any(following == self.top)), next_tokens

    def leaves(self, lp, top_allowed: bool, best_lp, stop_lp) -> bool:
        """Return whether the model's likeliest token leaves the document."""
        vocabulary = self.vocabulary
        top = self.top
        return (lp[top] > max(best_lp, stop_lp) and not top_allowed
                and not vocabulary.is_stop[top])

    def settle(self, document: Document, q: int) -> None:
        """Set the question's answer for the document: its span, or none.

        Among the streams the writing fits, the first position of the
        document wins.
        """
        answer = document.answers[q]
        answer.done = True
        if answer.none_wins or not answer.written or not answer.streams:
            answer.answer, answer.span = None, None
            return
        if self.policies is not None:
            self.policies[q].settle(len(answer.written))
        stream = min(answer.streams, key=lambda s: s.char)
        start, end = stream.char, stream.ends[len(answer.written) - 1]
        text = document.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        answer.answer, answer.span = text[start:end], (start, end)
