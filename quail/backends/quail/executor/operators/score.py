"""Numeric reranker scores through Quail's forward loop."""

import numpy as np

from quail.backends.quail.executor.operators.classify import QuailClassifier
from quail.backends.quail.executor.parts import (
    AnswerStream,
    SingleResultPart,
    execute_parts,
    shared_prefix_tree,
)
from quail.backends.quail.executor.readout import AsyncScores
from quail.backends.quail.executor.stages import Stage
from quail.backends.quail.executor.state import QueryExecutionState
from quail.backends.quail.reranker import RerankerBatch
from quail.catalog.tokens import DocumentPrefixes, concat_tokens
from quail.physical import ClassifySpec
from quail.specs.base import CANVAS_ENTROPY_NATS, CANVAS_SEED


class QuailScorer:
    """Score document rows with Quail's token admission and KV arena."""

    # score() reports answers through on_answers while it runs
    streams_answers = True

    def __init__(self, state: QueryExecutionState):
        self.state = state

    def score(self, spec, rows, documents, on_answers=None):
        """Score or classify the rows.

        Args:
            spec: Score or classification specification.
            rows: Document indices, one column per alias.
            documents: Tokenized documents indexed by table alias and document ID.
            on_answers: Optional callable(row positions, values) run as each
                chunk's answers are read.

        Returns:
            A RerankerBatch with one value per row, in row order.
        """
        if isinstance(spec, ClassifySpec):
            return QuailClassifier(self.state).classify(
                spec, rows, documents, on_answers=on_answers)
        state = self.state
        rows = np.asarray(rows, dtype=np.int32)
        parts = spec.prompt_token_parts
        if len(parts) != len(spec.aliases) + 1:
            raise ValueError("AI.SCORE needs tokenized prompt parts")
        if (len(spec.aliases) == 1 and spec.draws > 1
                and state.loaded_model.pipeline.canvas_ids):
            return self._drawn(spec, rows, documents, on_answers)
        if len(spec.aliases) == 1 and spec.share_prefixes:
            return self._shared(spec, rows, documents, on_answers)
        if len(spec.aliases) == 1:
            prefixes = [parts[0]]
            suffixes = [concat_tokens(documents[spec.aliases[0]][document_id], parts[1])
                        for document_id in rows[:, 0]]
            partners = None
            order = np.arange(len(rows))
            offsets = np.array([0, len(rows)])
            total = len(rows) * len(parts[0]) + sum(map(len, suffixes))
        else:
            left, right = spec.aliases
            anchors, anchor_index = np.unique(rows[:, 0], return_inverse=True)
            candidates, candidate_index = np.unique(rows[:, 1], return_inverse=True)
            prefixes = [concat_tokens(parts[0], documents[left][document_id], parts[1])
                        for document_id in anchors]
            suffixes = [concat_tokens(documents[right][document_id], parts[2])
                        for document_id in candidates]
            order = np.argsort(anchor_index, kind="stable")
            offsets = np.concatenate(([0], np.cumsum(np.bincount(anchor_index))))
            partners = [candidate_index[order[start:end]]
                        for start, end in zip(offsets[:-1], offsets[1:])]
            prefix_lengths = np.asarray([len(prefix) for prefix in prefixes])
            suffix_lengths = np.asarray([len(suffix) for suffix in suffixes])
            total = int(prefix_lengths[anchor_index].sum()
                        + suffix_lengths[candidate_index].sum())
        scores = np.full(len(rows), np.nan, dtype=np.float32)
        stages = [Stage(suffixes=suffixes, readout=self._scores(),
                        requests=(None if partners is None else
                                  lambda key: partners[key[2]]))]
        batch = ScoreBatch(spec.name, stages, scores, lambda: total, on_answers)

        batch.document_done = None

        def report(anchor, stage, start, end, values):
            first = offsets[anchor]
            positions = order[first + start:first + end]
            scores[positions] = values
            if on_answers is not None:
                batch.stream.send(positions, lambda position: scores[position])

        return self._execute(spec, batch, prefixes, on_answers=report)

    def _execute(self, spec, batch, prefixes, *, prefix_tree=None, on_answers=None):
        """Execute prepared scores and collect common timing and token metrics."""
        return execute_parts(
            self.state, [batch], prefixes,
            [("score", spec.name, index) for index in range(len(prefixes))],
            keep_kv=False, prefix_tree=prefix_tree, label=f"score {spec.name}",
            on_answers=on_answers)[spec.name]

    def _scores(self) -> AsyncScores:
        """Return the score readout for the current model."""
        state = self.state
        if state.score_readout is not None:
            return state.score_readout
        answer_rows = state.answer_rows
        async_scores = state.async_scores
        if async_scores is None or async_scores.rows is not answer_rows:
            async_scores = AsyncScores(state.torch, answer_rows)
            state.async_scores = async_scores
        return async_scores

    def _shared(self, spec, rows, documents, on_answers=None) -> RerankerBatch:
        """Score documents that borrow the KV pages of shared prefixes.

        Each document's prompt head and tokens are its prefix; a
        document whose prefix shares whole pages with another's reads
        those pages instead of computing them. The prompt tail follows
        each document, its last token read for the score.

        Args:
            spec: Single-document score specification.
            rows: Input row indices with shape (documents, 1).
            documents: Tokenized documents indexed by table alias and document ID.
            on_answers: Optional callable(row positions, values) run with each
                chunk's finished scores.

        Returns:
            A RerankerBatch of scores and execution metrics.

        Raises:
            ValueError: The prompt has no tokens after the document.
        """
        state = self.state
        (alias,) = spec.aliases
        head, tail = spec.prompt_token_parts
        if not tail:
            raise ValueError("AI.SCORE needs a question after the document")
        document_ids = rows[:, 0]
        scores = np.full(len(document_ids), np.nan, dtype=np.float32)

        def keep(anchor, row):
            scores[anchor] = float(row[0])
            return True

        prefixes = DocumentPrefixes(head, documents[alias], document_ids)
        stages = [Stage(suffixes=[[tail[-1]]], readout=self._scores(),
                        frame=list(tail[:-1]), decide=keep, single=True,
                        label=f"score {spec.name}")]
        total = sum(len(head) + len(documents[alias][document_id]) + len(tail)
                    for document_id in document_ids)
        batch = ScoreBatch(spec.name, stages, scores, lambda: total, on_answers)
        return self._execute(spec, batch, prefixes, prefix_tree=shared_prefix_tree(
            spec.name, prefixes, state.loaded_model.arena))

    def _drawn(self, spec, rows, documents, on_answers=None) -> RerankerBatch:
        """Score documents with reproducible diffusion draws.

        A low-entropy first answer uses one draw. Other documents run the remaining
        draws and use the mean TRUE probability. Draws share each document's KV.

        Args:
            spec: Single-document score specification.
            rows: Input row indices with shape (documents, 1).
            documents: Tokenized documents indexed by table alias and document ID.
            on_answers: Optional callable(row positions, values) run with each
                chunk's finished scores.

        Returns:
            A RerankerBatch of scores and execution metrics.
        """
        state = self.state
        pipeline = state.loaded_model.pipeline
        vocab = state.loaded_model.model_spec.vocab
        (alias,) = spec.aliases
        head, tail = spec.prompt_token_parts
        document_ids = rows[:, 0]
        width = len(pipeline.canvas_ids)
        more = spec.draws - 1
        scores = np.full(len(document_ids), np.nan, dtype=np.float32)
        first = np.zeros(len(document_ids), dtype=np.float64)
        extended = [0]

        def canvas(anchor, draw):
            rng = np.random.default_rng((CANVAS_SEED, int(document_ids[anchor]), draw))
            return rng.integers(0, vocab, width)

        def settle(anchor, row):
            p = float(row[0])
            first[anchor] = p
            q = min(max(p, 1e-12), 1 - 1e-12)
            if -(q * np.log(q) + (1 - q) * np.log(1 - q)) <= CANVAS_ENTROPY_NATS:
                scores[anchor] = p
            return True

        def average(anchor, row):
            extended[0] += 1
            scores[anchor] = (first[anchor] + float(np.sum(row))) / spec.draws
            return True

        cue = [tail[-1]]
        readout = self._scores()
        stages = [
            Stage(suffixes=[cue], readout=readout, frame=list(tail[:-1]),
                  decide=settle, single=True, label=f"score {spec.name}",
                  canvas=lambda anchor: canvas(anchor, 0), canvas_rows=width),
            Stage(suffixes=[cue] * more, readout=readout,
                  frame=list(tail[:-1]), decide=average,
                  requests=lambda key: (None if np.isnan(scores[key[2]])
                                        else Stage.SKIP),
                  label=f"score {spec.name} draws",
                  canvas=lambda anchor: np.stack(
                      [canvas(anchor, draw) for draw in range(1, spec.draws)]),
                  canvas_rows=width),
        ]
        prefixes = DocumentPrefixes(head, documents[alias], document_ids)
        tree = (shared_prefix_tree(spec.name, prefixes, state.loaded_model.arena)
                if spec.share_prefixes else None)
        read = len(cue) + width
        base = sum(
            len(head) + len(documents[alias][document_id])
            for document_id in document_ids
        ) + len(document_ids) * (len(tail) - 1 + read)
        batch = ScoreBatch(spec.name, stages, scores,
                           lambda: base + extended[0] * more * read, on_answers)
        return self._execute(spec, batch, prefixes, prefix_tree=tree)


class ScoreBatch(SingleResultPart):
    """Prepared score requests, streamed answers, and execution metrics."""

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        # scored through the reranker readout over the documents
        return {"score_inputs": inputs, "documents": context.state["documents"]}

    def __init__(self, name, stages, scores, total_tokens, on_answers):
        self.name = name
        self.stages = stages
        self.scores = scores
        self.total_tokens = total_tokens
        self.stream = AnswerStream(on_answers)
        self.document_done = self.stream.chunk(lambda anchor: (
            None if np.isnan(scores[anchor]) else scores[anchor]))

    def finish(self, every) -> int:
        """Deliver remaining scores and return their full prompt token count."""
        self.stream.finish(range(len(self.scores)), lambda index: self.scores[index])
        return self.total_tokens()

    def result(self, tokens, gpu_s, chunks, stats) -> RerankerBatch:
        """Build scores with the metrics collected by the common executor."""
        return RerankerBatch(
            self.scores, fresh_tokens=tokens,
            cached_tokens=self.total_tokens() - tokens,
            borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0), gpu_s=gpu_s, chunks=chunks)

    def results(self) -> dict:
        return {self.name: self.result_value}
