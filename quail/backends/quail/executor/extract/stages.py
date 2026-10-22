"""The stages that ask a pipeline's questions of its documents.

The extractions of one pipeline run as one stage list over the
documents: a question's tail is the frame of its stages, written right
after the document in place of the tokens the question before it
appended, so a document is read once however many questions it takes.
Each question is the tail stage, whose row chooses the answer's first
token, then up to the configured answer limit in rounds, each feeding the chosen
token and a draft of the document's next tokens and verifying them.
"""

from functools import partial

import numpy as np

from quail.backends.quail.executor.extract.candidates import DocumentTokens
from quail.backends.quail.executor.extract.decode import (
    AnswerState,
    Decoder,
    Document,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.readout import AsyncTopLogprobs
from quail.backends.quail.executor.stages import Stage, StageRequests
from quail.catalog.tokens import concat_tokens


class ExtractionStages:
    """The stages of one pipeline's extractions and the answers they produce.

    Args:
        state: The loaded model and current query state.
        specs: The ExtractSpecs, in order; they share a table and a
            prompt head.
        vocabulary: The model's vocabulary index.
        document_ids: The documents' ids in their table.
        documents: The table's tokenized documents.
        texts: The table's document texts.
        readout: Optional readout in place of the model's. Its one
            extra is the stop tokens as a group.
        pipeline_head: The token ids the pipeline shows the documents
            under; an extraction under the same head reads the KV the
            operators before it wrote.
        document_positions: Positions in the supplied documents and texts.
            None uses document_ids.

    Raises:
        ValueError: The specs differ in their table, head, or decoding settings.
    """

    def __init__(self, state, specs, vocabulary: Vocabulary, document_ids,
                 documents, texts, readout=None, pipeline_head=None,
                 document_positions=None):
        self.specs = list(specs)
        first = self.specs[0]
        self.spec = first
        self.recurrent = bool(getattr(state.loaded_model.arena, "num_state_slots", 0))
        if any(not first.same_decoding(spec) for spec in self.specs):
            raise ValueError("a pipeline's extractions share decoding settings")
        if any(spec.alias != first.alias for spec in self.specs):
            raise ValueError("a pipeline's extractions share a table")
        parts = [spec.prompt_token_parts for spec in self.specs]
        self.head = list(parts[0][0])
        if any(list(head) != self.head for head, _ in parts):
            raise ValueError("a pipeline's extractions share a prompt head")
        self.tails = [list(tail) for _, tail in parts]
        self.pipeline_head = (None if pipeline_head is None
                              else [int(t) for t in pipeline_head])
        if any(len(tail) < 2 for tail in self.tails):
            raise ValueError("an extraction's tail ends in its cue after the question")
        self.vocabulary = vocabulary
        positions = (document_ids if document_positions is None
                     else document_positions)
        shown = [text.as_py() if hasattr(text, "as_py") else str(text)
                 for text in (texts[position] for position in positions)]
        stored = [None if documents is None else documents[document_id]
                  for document_id in positions]
        encoded = vocabulary.document_batch(shown, stored)
        self.documents = [self._prepare(index, document_id, text, tokens, prior)
                     for index, (document_id, text, tokens, prior) in enumerate(
                         zip(document_ids, shown, encoded, stored))]
        document_ids = np.asarray(document_ids, dtype=np.int64)
        self.document_order = np.argsort(document_ids, kind="stable")
        self.document_ids = document_ids[self.document_order]
        self.prefixes = [document.prefix for document in self.documents]
        self.keys = [document.key for document in self.documents]
        self.readout = self._readout(state) if readout is None else readout
        self.decoder = Decoder(vocabulary, self.readout.k, self.readout.width,
                               len(self.specs), self.spec)
        self.stages = []
        for q, (spec, tail) in enumerate(zip(self.specs, self.tails)):
            self.stages.extend(self._question_stages(q, spec.name, tail))

    def _question_stages(self, q: int, name: str, tail: list) -> list:
        """Build one question's stages: its tail, then the answer's rounds.

        The tail but its last token is the frame of every stage, so
        the scheduler writes it once after the document; the first
        stage rewinds past what the question before appended. The
        last token of the tail, the cue, is the stage's suffix, whose
        row chooses the answer's first token.
        """
        frame, last = tail[:-1], tail[-1]
        stages = [Stage(
            suffixes=[[last]], frame=frame, rewind=True,
            readout=self.readout, decide=partial(self._tail_decide, q),
            single=True, append=True, label=f"{name} tail")]
        stages.extend(
            Stage(suffixes=partial(self._step_suffix, q),
                  suffix_tokens=(1 + self.spec.draft_tokens)
                  * (2 if self.recurrent else 1), frame=frame,
                  readout=self.readout, decide=partial(self._step_decide, q),
                  read_all_rows=True,
                  single=True, append=True, label=f"{name} round {t}")
            for t in range(1, self.spec.max_answer_tokens + 1))
        return stages

    @property
    def appended_tokens(self) -> int:
        """The most tokens a question leaves in KV after its frame.

        The cue, and the answer's tokens with the last round's draft
        past them.
        """
        return 1 + self.spec.max_answer_tokens + self.spec.draft_tokens

    def _readout(self, state):
        """The readout of every row: the stop, then the top tokens."""
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        return AsyncTopLogprobs(torch, torch.nn.functional, head,
                                self.spec.top_tokens, [self.vocabulary.stop_ids],
                                ragged=True)

    def _prepare(self, index, document_id, text, ids, stored) -> Document:
        """Build a document's prefix: the head and its text.

        The key names the document under the extraction's head, so a
        later extraction of the same table reads the KV this one
        leaves resident. A document shown under the query's shared
        head, whose tokens are the table's, keeps the table's key, so
        the KV a filter wrote is read.
        """
        alias = self.specs[0].alias
        key = (("extract", alias), int(document_id))
        shared = self.head == self.pipeline_head
        tokens = ids.numpy()
        same_tokens = stored is ids
        if shared and stored is not None and not same_tokens:
            prior = stored.numpy() if hasattr(stored, "numpy") else np.asarray(stored)
            same_tokens = np.array_equal(prior, tokens)
        if shared and stored is not None and same_tokens:
            key = (alias, int(document_id))
        prefix = concat_tokens(self.head, ids)
        scan = DocumentTokens.from_tokens(
            self.vocabulary, tokens, ids.character_offsets, len(self.head))
        # the round's KV read covers the prompt through the first tail
        rho = self.spec.round_cost_per_token * (len(prefix) + len(self.tails[0]))
        return Document(index=index, text=text, prefix=prefix, scan=scan, key=key,
                        answers=[AnswerState() for _ in self.specs], rho=rho)

    # ---- the tail, then the answer, a round of tokens at a time ------------

    def _tail_decide(self, q, anchor, row):
        """Choose the answer's first token from the row after the cue."""
        document = self.documents[anchor]
        answer = document.answers[q]
        answer.asked = True
        # a tail equal to the question's before it is in KV already:
        # only its cue token is fed
        same = (not self.recurrent and q > 0
                and self.tails[q] == self.tails[q - 1])
        answer.fed_tokens += 1 if same else len(self.tails[q])
        self.decoder.verify(document, q, self._rows(row), [])
        return True

    def _rows(self, row):
        """Return a readout's rows for one answer."""
        # the admission hands over the document's one answer in a list
        # or an object array
        if isinstance(row, list) and len(row) == 1:
            row = row[0]
        elif isinstance(row, np.ndarray) and row.dtype == object and row.size == 1:
            row = row.reshape(-1)[0]
        return self.decoder.rows(row)

    def _step_suffix(self, q, key):
        """Feed the chosen token, then the document's next tokens as a draft.

        A recurrent model replays the accepted tokens to advance its
        checkpoint, so they are fed again and written as the suffix.
        """
        document = self._document(key)
        answer = document.answers[q]
        if answer.done:
            return Stage.SKIP
        accepted = answer.fed[:len(answer.fed) - answer.discard]
        tokens = self.decoder.draft(answer, document.rho, q)
        if self.recurrent:
            answer.fed_tokens += len(accepted)
            return StageRequests([accepted + tokens],
                                 write_suffix_tokens=len(accepted),
                                 read_rows=[len(tokens)])
        return StageRequests([tokens], discard=answer.discard)

    def _document(self, key):
        position = np.searchsorted(self.document_ids, key[1], side="right") - 1
        return self.documents[self.document_order[position]]

    def _step_decide(self, q, anchor, row):
        """Verify the round's draft by the row after each of its tokens."""
        document = self.documents[anchor]
        answer = document.answers[q]
        answer.fed_tokens += len(answer.fed)
        self.decoder.verify(document, q, self._rows(row), answer.fed)
        return True

    def finish(self) -> None:
        """Answer every question the steps left open."""
        for document in self.documents:
            for q, answer in enumerate(document.answers):
                if not answer.done:
                    self.decoder.settle(document, q)

    def answers(self, q: int, kept=None) -> tuple:
        """Return each document's answer and span for question q.

        Args:
            q: The question's index.
            kept: The indices of the documents to return, in order;
                None returns every document.
        """
        documents = (
            self.documents if kept is None else [self.documents[i] for i in kept]
        )
        return ([document.answers[q].answer for document in documents],
                [document.answers[q].span for document in documents])

    def asked(self) -> list:
        """Return the indices of the documents the questions were asked of."""
        return [index for index, document in enumerate(self.documents)
                if document.answers[0].asked]

    def counts(self, q: int) -> dict:
        """Return how many documents answered none and how much was fed."""
        states = [document.answers[q] for document in self.documents]
        return {
            "none": sum(answer.answer is None for answer in states),
            "steps": sum(answer.steps for answer in states),
            "answer_tokens": sum(len(answer.written) for answer in states),
            "suffix_tokens": sum(answer.fed_tokens for answer in states),
        }
