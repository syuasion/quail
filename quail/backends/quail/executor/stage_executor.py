"""Run one stage list: admission, packed forward chunks, and answer readout."""

import time
from functools import partial
from itertools import groupby

import numpy as np

from quail.backends.quail.executor.admission import (
    DROP,
    SKIP,
    JoinAdmission,
    OwnRequests,
)
from quail.backends.quail.executor.readout_groups import ReadoutGroups
from quail.backends.quail.executor.stages import (
    PendingAnswers,
    _advance_stage,
    frame_writes,
    state_share_points,
)
from quail.catalog.tokens import PrefixTree
from quail.progress import Progress, logger


def attention_path(pipeline, requested, default_attention="unified") -> str:
    """The path a filter stream or join runs: the plan's request where the model has it.

    None asks for the default_attention. A model without tree attention, and a
    canvas model (whose rows the tree path does not pack), run unified.
    """
    if requested is None:
        requested = default_attention
    if requested == "tree" and pipeline.tree_attention and not pipeline.canvas_ids:
        return "tree"
    return "unified"


def forward_chunk(pipeline, arena, chunk):
    """Run the forward pass, then release the chunk's temporary pages.

    The loop owns every page it hands the forward pass, so it frees the
    temporaries whether the pass returns or raises.
    """
    try:
        if pipeline.decode_graphs is not None:
            out = pipeline.decode_graphs.forward(chunk)
        else:
            out = pipeline.forward_chunk(chunk)
        arena.sample_memory(chunk.tokens, len(chunk.layout))
        return out
    finally:
        keys, chunk.temporary_keys = chunk.temporary_keys, ()
        for key in keys:
            arena.free_key(key)


def borrow_check(arena, keys, borrowing):
    """can_borrow for Borrowing: whether the arena can serve a borrow.

    A fresh parent admitted in the same chunk is not allocated yet;
    its sliding pages will start at the window origin below the tokens
    it borrows itself. A parent already resident, admitted in this
    chunk or earlier, has the pages it has: it may have trimmed its
    window in an earlier query.
    """
    def can_borrow(document, parent, same_chunk):
        shared = borrowing.tree_shared[document]
        if not same_chunk or arena.is_resident(keys[parent]):
            return arena.can_borrow(keys[parent], shared)
        return arena.origin(borrowing.shared(parent)) <= arena.origin(shared)
    return can_borrow


def lowest_borrows(borrowing):
    """Per document, the fewest tokens any of its borrowers shares."""
    lowest = {}
    for document, parent in enumerate(borrowing.tree_parent):
        if parent is not None:
            shared = borrowing.tree_shared[document]
            lowest[parent] = min(lowest.get(parent, shared), shared)
    return lowest


class StageExecutor:
    """Admission state, packed inputs, pending answers, and metrics for one run."""

    def __init__(self, torch, arena, pipeline, stages, anchor_prefixes, budget, *,
                 anchor_keys, on_settled, staging, attention_mode, prefix_tree,
                 stats, limit, paged, unit, count_answers, label,
                 default_attention, on_chunk, on_answers, stop_groups=None,
                 appended_tokens=None, gpu_timing=False, survival=None):
        from quail.backends.quail.executor.chunk import Suffixes


        self.torch = torch
        self.arena = arena
        self.pipeline = pipeline
        self.stages = stages
        self.on_settled = on_settled
        self.staging = staging
        self.stats = stats
        self.paged = paged
        self.on_chunk = on_chunk
        self.on_answers = on_answers
        self.label = label
        self.gpu_timing = gpu_timing

        k = len(self.stages)
        if not self.paged and (
                k > 1 or prefix_tree is not None
                or any(stage.own_suffixes or len(stage.suffixes) != 1
                       for stage in self.stages)):
            raise ValueError("the unpaged path runs one stage of one suffix")
        if any(stage.append and not stage.single for stage in self.stages):
            raise ValueError("an append stage sends one request per document")
        self.prefixes = anchor_prefixes
        # a prefix is built on each access, so its length is read once
        self.prefix_lengths = [len(prefix) for prefix in self.prefixes]
        self.keys = (list(range(len(self.prefixes))) if anchor_keys is None
                     else anchor_keys)
        if len(self.keys) != len(self.prefixes):
            raise ValueError("anchor_keys must match anchor_prefixes")
        self.frames = [list(stage.frame) for stage in self.stages]
        self.frame_ids = [np.asarray(frame, dtype=np.int64) for frame in self.frames]
        self.writes = frame_writes(self.stages)
        if getattr(self.arena, "num_state_slots", 0):
            self.writes = [write or (stage.rewind and bool(stage.frame))
                           for write, stage in zip(self.writes, self.stages)]
        # a stage whose suffixes are each document's own is priced by
        # one sequence of its most tokens until a document enters it
        self.suffixes = [Suffixes.of([[0] * stage.suffix_tokens])
                         if stage.own_suffixes else Suffixes.of(stage.suffixes)
                         for stage in self.stages]
        self.own = {}      # (stage, document) -> its StageRequests
        self.read_rows = [None if stage.read_rows is None
                          else np.asarray(stage.read_rows, dtype=np.int64)
                          for stage in self.stages]
        self.read_masks = [None if stage.read_mask is None
                           else np.asarray(stage.read_mask, dtype=bool)
                           for stage in self.stages]
        # Stages packing label paths always need tree attention.
        if any(stage.paths is not None for stage in self.stages):
            attention_mode = "tree"
        self.mode = attention_path(
            self.pipeline, attention_mode, default_attention=default_attention)
        self.canvas = tuple(self.pipeline.canvas_ids)
        self.answer_row = self.pipeline.canvas_answer_row
        self.canvas_rows = [len(self.canvas) if stage.canvas is None
                            else stage.canvas_rows for stage in self.stages]

        # A frame's canvas occupies document pages after the frame.
        frame_max = max(len(f) + (len(self.canvas) if f else 0) for f in self.frames)
        self.capacity_extra = frame_max
        # A single suffix also occupies document pages on the unified path.
        if self.paged and self.mode == "unified":
            self.capacity_extra = max(
                [frame_max] + [len(frame) + int(other.lengths.max()) + rows
                               for frame, stage, other, rows in zip(
                                   self.frames, self.stages, self.suffixes,
                                   self.canvas_rows)
                               if stage.single])
        appended = sorted(
            (tuple(frame), int(suffixes.lengths.max()))
            for frame, suffixes, stage in zip(self.frames, self.suffixes, self.stages)
            if stage.append)
        for frame, group in groupby(appended, key=lambda entry: entry[0]):
            kept = sum(tokens for _, tokens in group)
            if appended_tokens is not None:
                kept = min(kept, int(appended_tokens))
            self.capacity_extra = max(self.capacity_extra, len(frame) + kept)
        # append-stage tokens each document holds after its frame
        self.appended = [0] * len(self.prefixes)
        # documents whose exact copies borrow their question's KV
        self.copied = set()
        self.question_tokens = 0
        prefix_tree = self._copies_borrow_questions(prefix_tree, stop_groups)

        resident = {}
        if self.paged:
            for a, key in enumerate(self.keys):
                if not self.arena.is_resident(key):
                    continue
                if self.arena.prefix_tokens(key) != self.prefix_lengths[a]:
                    # KV an earlier operator left under this key was
                    # built for another prefix; it is fed afresh
                    self.arena.free_key(key)
                    continue
                resident[a] = self._held_pages(key, self.prefix_lengths[a])
        # Pin resident documents while fresh admissions evict retained KV.
        for a in resident:
            self.arena.pin(self.keys[a])

        self.asking = any(stage.requests is not None or stage.own_suffixes
                          for stage in self.stages)
        if prefix_tree is not None and not prefix_tree.shared_tokens:
            prefix_tree = None
        self.sched = JoinAdmission(
            self.prefix_lengths,
            [s.lengths + rows
             for s, rows in zip(self.suffixes, self.canvas_rows)],
            budget,
            self.arena.num_pages if self.paged else 1 << 62,
            self.arena.page_tokens,
            frame_tokens=[len(f) for f in self.frames], resident=resident,
            anchor_partners={a: self._requests_of(a, key)
                             for a, key in enumerate(self.keys)},
            # a windowed model may pack any chunk unified, so it reserves
            # the unified path's temporary pages throughout
            temporary_suffix_pages=self.paged and self.mode == "unified",
            answer_dtype=[stage.readout.dtype for stage in self.stages],
            frame_canvas_tokens=len(self.canvas),
            page_cost=self.arena.page_cost if self.paged else (lambda *args: 0),
            tree=prefix_tree,
            advance=partial(_advance_stage, stages),
            frame_writes=self.writes,
            limit=limit, survival=survival,
            stop_groups=stop_groups,
            stop_width=self._stop_width(stop_groups, budget),
            extra_tokens=self.capacity_extra,
            reserve=self._reserve if self.paged else None,
        )
        self.borrowing = self.sched.borrowing
        self.borrowing.can_borrow = borrow_check(self.arena, self.keys, self.borrowing)
        # a Gated DeltaNet layer starts a borrower from the parent's state
        # at the shared point, so a parent writes one slot per point
        self.share_points = [()] * len(self.keys)
        self._extends = False
        spec = getattr(self.pipeline, "spec", None)
        if (self.paged and getattr(spec, "state_layers", ())
                and not getattr(self.arena, "num_state_slots", 0)):
            raise ValueError(
                "the arena has no checkpoint slots: the pool cannot hold "
                "one document's state beside a chunk's KV, so a document "
                "this long needs a smaller chunk or a larger pool")
        if self.paged and getattr(self.arena, "num_state_slots", 0):
            self._extends = (any(self.writes)
                             or any(s.append for s in self.stages))
            # a parent holds one slot per share length beside its own and
            # its extension, and its borrowers need slots of their own, so
            # a small pool keeps only the lengths that save the most
            base = 1 + int(self._extends)
            room = max(0, (self.arena.num_state_slots - base) // 3)
            self.share_points = state_share_points(
                self.borrowing.tree_parent, self.borrowing.tree_shared,
                limit=room)
            need = base + max(map(len, self.share_points), default=0)
            if self.arena.num_state_slots < need:
                raise ValueError(
                    f"the arena has {self.arena.num_state_slots} checkpoint "
                    f"slots and this query holds {need} per document in "
                    f"flight; a document this long needs a smaller chunk "
                    f"or a larger pool")
            self.borrowing.can_borrow = self._state_borrow_check(
                self.borrowing.can_borrow)
        # The admission detaches residents from borrowing: they pack no prefix.
        self.borrowers = self.borrowing.borrowers()
        self.lowest_borrow = lowest_borrows(self.borrowing)
        self.held = [self.keys[a] for a, n in enumerate(self.borrowers) if n]

        self._held_parents = set()
        self._reserved_parents = {}
        self.spans = []
        self.tokens = 0
        self.pack_s = 0.0
        self.outstanding = []     # PendingAnswers in launch order
        self.split_answers = any(s.append and s.own_suffixes for s in stages)

        if self.label is None:
            self.label = " > ".join(stage.label or f"stage {j}"
                                   for j, stage in enumerate(self.stages))
        self.counting_answers = count_answers
        total = (sum(self.sched.num_partners(a, j) for a in range(len(self.prefixes))
                     for j in range(k))
                 if self.counting_answers else len(self.prefixes))
        self.progress = Progress(self.label, total=total, unit=unit)
        self.finished = 0

        self.readouts = ReadoutGroups(self.torch, self.stages)

    def _copies_borrow_questions(self, tree, stop_groups):
        """Let each exact copy borrow its first copy's question KV too.

        A copy's answer needs one fresh row, so it borrows the whole
        pages of its first copy's document and question and computes
        the rest. Only a run of one single-question stage qualifies:
        any later write past the document would land in borrowed
        pages. Hybrid, windowed, and canvas models keep their
        borrows short of the document's last token.

        Returns:
            The prefix tree with each copy's parent and shared tokens.
        """
        stage = self.stages[0]
        if (tree is None or not self.paged or len(self.stages) != 1
                or stop_groups is not None or stage.frame or not stage.single
                or stage.own_suffixes or stage.requests is not None
                or stage.append or stage.paths is not None
                or stage.canvas is not None or self.canvas
                or stage.read_all_rows or stage.read_rows is not None
                or stage.read_mask is not None or len(self.suffixes[0]) != 1
                or getattr(getattr(self.pipeline, "spec", None), "state_layers", ())
                or getattr(self.arena, "num_state_slots", 0)
                or getattr(self.arena, "has_sliding", False)):
            return tree
        question = int(self.suffixes[0].lengths[0])
        page = self.arena.page_tokens
        parent, shared = list(tree.parent), list(tree.shared)
        for a, first in enumerate(tree.copy_of):
            if first is None:
                continue
            # the last question row is the answer, so a copy computes it
            borrowed = (self.prefix_lengths[a] + question - 1) // page * page
            if borrowed <= shared[a]:
                continue
            parent[a], shared[a] = first, borrowed
            self.copied.add(first)
        if not self.copied:
            return tree
        self.question_tokens = question
        # the first copy keeps its question's KV for the copies to read
        self.capacity_extra = max(self.capacity_extra, question)
        return PrefixTree(tree.order, parent, shared, tree.copy_of)

    def _stop_width(self, stop_groups, budget) -> int:
        """Anchors of one stop key group in flight at once.

        One per group is the most a group can save, but with fewer
        groups than a chunk holds it would leave the chunk mostly
        empty, so the width grows to keep about one chunk in flight.
        """
        if stop_groups is None or not self.prefixes:
            return 1
        groups = len(set(stop_groups)) or 1
        mean = (sum(self.prefix_lengths) / len(self.prefix_lengths)
                + float(np.mean(self.suffixes[0].lengths)))
        per_chunk = max(1, int(budget // max(1.0, mean)))
        return max(1, -(-per_chunk // groups))

    def _suffixes_of(self, j, a):
        """Return the suffixes document a requests at stage j."""
        return (self.own[(j, a)].suffixes if self.stages[j].own_suffixes
                else self.suffixes[j])

    def _entry_rows(self, a, j, start, end, carried):
        f = self.prefix_lengths[a]
        indices = self.sched.partner_indices(a, j, start, end)
        lengths = self._suffixes_of(j, a).lengths_at(indices)
        rows = ((f if carried else 0) + int(lengths.sum())
                + self.canvas_rows[j] * len(lengths))
        if self.writes[j] and start == 0:
            rows += len(self.frames[j]) + len(self.canvas)
        return rows

    def _held_pages(self, key, prefix_tokens):
        # the admission prices a document at page_cost(prefix + extra)
        # less what it holds; a trimmed window holds fewer sliding
        # pages than that price assumes, so count what growing takes
        capacity = prefix_tokens + self.capacity_extra
        return (self.arena.page_cost(capacity)
                - self.arena.growth_cost(key, capacity))

    def _requests_of(self, a, key):
        """Return a stage request selector, or None if every request is selected.

        At a stage whose suffixes are each document's own, the selector
        asks the stage for the document's sequences, keeps them for the
        chunk builder, and hands the admission their token counts.
        """
        if not self.asking:
            return None
        stages = self.stages

        def ask(j):
            stage = stages[j]
            if not stage.own_suffixes:
                return None if stage.requests is None else stage.requests(key)
            got = stage.suffixes(key)
            if got is DROP or got is SKIP:
                return got
            own = got.suffixes
            if len(own) and int(own.lengths.max()) > stage.suffix_tokens:
                raise ValueError(
                    f"{stage.label or f'stage {j}'}: a document's suffix has "
                    f"{int(own.lengths.max())} tokens, over the stage's "
                    f"{stage.suffix_tokens}")
            if len(own):
                self.own[(j, a)] = got
            return OwnRequests(own.lengths)

        return ask

    def _state_borrow_check(self, can_borrow):
        """Narrow can_borrow to borrows a parent's saved state can serve.

        The borrower must start after the parent's own start and before
        its own end, at a point the parent saves a state for or has
        saved already.
        """
        def check(doc, parent, same_chunk):
            shared = self.borrowing.tree_shared[doc]
            if not shared < self.prefix_lengths[doc]:
                return False
            if not shared > self.borrowing.shared(parent):
                return False
            # the parent saves a state only at the lengths it kept
            if (shared != self.prefix_lengths[parent]
                    and shared not in self.share_points[parent]):
                return False
            if not same_chunk and self.arena.shared_state(
                    self.keys[parent], shared) is None:
                return False
            return can_borrow(doc, parent, same_chunk)
        return check

    def _reserve(self, anchor, borrow):
        """Reserve a document's KV and checkpoints before admitting its work."""
        key = self.keys[anchor]
        prefix = self.prefix_lengths[anchor]
        parent, shared = (None, 0) if borrow is None else borrow
        points = tuple(t for t in self.share_points[anchor]
                       if shared < t < prefix)
        if not self.arena.reserve(
                key, prefix, capacity_tokens=prefix + self.capacity_extra,
                borrow=(self.keys[parent], shared) if shared else None,
                extension=self._extends, share_points=points):
            return False
        if parent is not None:
            self.arena.read_state((self.keys[parent],))
            self._reserved_parents[anchor] = self.keys[parent]
        if anchor in self.copied:
            # its chunk writes the question into its pages before any
            # copy's rows read them, in this chunk or a later one
            self.arena.mark_readable(key, prefix + self.question_tokens)
        if (anchor not in self._held_parents and anchor < len(self.borrowers)
                and self.borrowers[anchor]):
            self._held_parents.add(anchor)
            self.arena.hold(key, self.borrowers[anchor])
            self.arena.keep_window(key, self.lowest_borrow[anchor])
        return True

    def _merged(self, j, start, end):
        """Return whether the group packs its frame and suffix as one entry."""
        return self.stages[j].single and end - start == 1

    def _build(self, chunk_groups):
        """Pack a chunk and identify each entry's stage and answer row count."""
        from quail.backends.quail.executor.chunk import Suffixes, pack_chunk

        specs = []
        entries = []
        appended = {}
        for a, j, start, end, carried in chunk_groups:
            key = self.keys[a]
            f = self.prefix_lengths[a]
            if not carried and self.borrowing.shared(a) >= f:
                # a copy reads its first copy's document and question
                # pages and computes the question's last rows
                borrowed = self.borrowing.shared(a)
                question = self.suffixes[j].take([0])
                rest = question.ids[borrowed - f:]
                specs.append(dict(key=key, prefix=None, f=borrowed,
                                  suffixes=Suffixes(rest, [len(rest)]),
                                  single=True))
                entries.append((j, 1))
                continue
            frame = self.frames[j]
            append = self.stages[j].append
            request = self.own.get((j, a))
            # a written frame, or a rewinding stage, replaces any path
            # an earlier stage kept
            rewinds = self.writes[j] or self.stages[j].rewind
            path = 0 if rewinds and start == 0 else self.appended[a]
            if request is not None:
                path = max(0, path - request.discard)
            parent, shared = ((self.borrowing.parent(a), self.borrowing.shared(a))
                              if carried else (None, 0))
            prefix = None
            if carried:
                prefix = self.prefixes[a][shared:] if shared else self.prefixes[a]
            indices = self.sched.partner_indices(a, j, start, end)
            sufs = self._suffixes_of(j, a).take(indices)
            read_all = self.stages[j].read_all_rows
            rows = int(sufs.lengths.sum()) if read_all else len(sufs)
            own = {}
            read_rows = (request.read_rows if request is not None
                         else self.read_rows[j])
            if read_rows is not None:
                own["read_rows"] = np.asarray(read_rows)[np.asarray(list(indices))]
                rows = int(own["read_rows"].sum())
            if self.read_masks[j] is not None:
                own["read_mask"] = self.read_masks[j]
                rows = int(own["read_mask"].sum())
            if self.stages[j].canvas is not None:
                ids = np.asarray(self.stages[j].canvas(a))
                if ids.ndim == 2:
                    ids = ids[np.asarray(list(indices), dtype=np.int64)]
                own = dict(canvas=ids)
                if read_all:
                    rows = self.stages[j].canvas_rows * len(sufs)
            # under tree attention a borrowing document with one
            # suffix reads its parent's pages stacked with its siblings
            read_key = self.keys[parent] if shared and self.stages[j].single else None
            carry = {}
            if prefix is not None:
                carry = dict(
                    borrow_key=self.keys[parent] if shared else None,
                    share_points=self.share_points[a])
            kept = int(sufs.lengths.sum()) if append else 0
            if request is not None and request.write_suffix_tokens is not None:
                kept = request.write_suffix_tokens
            if a in self.copied and start == 0:
                kept = self.question_tokens
            if append or (self.writes[j] and start == 0):
                appended[a] = path + kept
            if self.writes[j] and start == 0 and self._merged(j, start, end):
                # the frame and the one suffix are one entry; the
                # frame's rows are scattered into KV after the document
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f, suffixes=Suffixes(
                        np.concatenate([self.frame_ids[j], sufs.ids]),
                        [len(frame) + int(sufs.lengths[0])]),
                    write_suffix_tokens=len(frame) + kept, single=True,
                    read_all_rows=read_all, **carry, **own))
            elif self.writes[j] and start == 0:
                # frame entry: scatter the frame into KV after the
                # document rows; its own answer row means nothing
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f, suffixes=Suffixes(self.frame_ids[j], [len(frame)]),
                    write_suffix_tokens=len(frame), **carry))
                entries.append((j, 1))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs, read_all_rows=read_all,
                    write_suffix_tokens=kept,
                    paths=self.stages[j].paths, **own))
            else:
                specs.append(dict(
                    key=key, prefix=prefix, start=shared, read_key=read_key,
                    f=f + len(frame) + path,
                    suffixes=sufs, read_all_rows=read_all,
                    write_suffix_tokens=kept,
                    single=self.stages[j].single and end - start == 1,
                    paths=self.stages[j].paths, **carry, **own))
            entries.append((j, rows))
        chunk = pack_chunk(
            self.torch, self.arena, specs, attention_mode=self.mode,
            staging=self.staging, canvas=self.canvas, answer_row=self.answer_row)
        for a, tokens in appended.items():
            self.appended[a] = tokens
        return chunk, entries

    def _settle(self, settlement):
        anchor = settlement.anchor
        if self.on_settled is None:
            if self.arena.is_resident(self.keys[anchor]):
                self.arena.free_key(self.keys[anchor])
        else:
            self.on_settled(
                anchor, settlement.survived, self.sched.answers[-1].get(anchor, []))

    def _event(self, settlement):
        # Progress counts settled documents: a dropped one is done too.
        self.finished += 1
        if settlement.kind == "finished":
            self._settle(settlement)
        elif self.arena.is_resident(self.keys[settlement.anchor]):
            self.arena.free_key(self.keys[settlement.anchor])

    def _report(self, entry):
        if entry.values is None:
            entry.values = self.readouts.results(entry.handles)
            entry.positions = np.zeros(len(self.readouts), dtype=np.int64)
        groups = entry.groups[entry.at:entry.at + entry.step]
        values, position = entry.values, entry.positions
        transitions = []
        for a, j, start, end, _ in groups:
            key = self.readouts.group_key[j]
            if self.writes[j] and start == 0 and not self._merged(j, start, end):
                position[key] += 1        # the frame entry's answer means nothing
            count = end - start
            rows = values[key][position[key]:position[key] + count]
            result = self.sched.report(a, j, start, end, rows)
            if self.stages[j].own_suffixes and end == self.sched.num_partners(a, j):
                del self.own[(j, a)]
            if self.on_answers is not None:
                self.on_answers(a, j, start, end, rows)
            for settlement in result.settlements:
                self._event(settlement)
            transitions.extend(result.transitions)
            position[key] += count
        entry.at += len(groups)
        self.progress.update(
            self.progress.done + sum(end - start for _, _, start, end, _ in groups)
            if self.counting_answers else self.finished)
        if transitions and self.on_chunk is not None:
            self.on_chunk(transitions)
        self._release_skipped()

    def _report_next(self, partial=False):
        """Decide the oldest results, splitting append stages when no work is ready."""
        entry = self.outstanding[0]
        if not entry.step:
            entry.step = len(entry.groups)
            if partial and self.split_answers and all(
                    self.stages[j].append for _, j, _, _, _ in entry.groups):
                entry.step = (entry.step + 1) // 2
        self._report(entry)
        if entry.at == len(entry.groups):
            self.outstanding.pop(0)

    def _release_skipped(self):
        """Let go of what the documents a stop key skipped would have used.

        A skipped document never packs, so its parent's hold for it is
        released, or not counted when the parent is still queued, and
        its own borrowers pack their whole prefix.
        """
        for a in self.sched.take_skipped():
            self.borrowing.skip(a)
            parent = (self.borrowing.tree_parent[a]
                      if a < len(self.borrowing.tree_parent) else None)
            if parent is not None:
                if parent in self.borrowing.admitted:
                    self.arena.release(self.keys[parent])
                else:
                    self.borrowers[parent] -= 1
            if self.arena.is_resident(self.keys[a]):
                self.arena.free_key(self.keys[a])

    def _log_memory_event(self, kind, **detail):
        log = getattr(self.arena, "log_event", None)
        if log is not None:
            reason = detail.get("reason", "")
            if reason:
                detail["reason"] = ("slots" if "checkpoint slot" in reason
                                    else "pages" if "pages" in reason
                                    else "rows")
            log(kind, **detail)

    def _run_part(self, part):
        """Run one chunk of groups, halving it when its pages do not fit."""
        from quail.backends.quail.executor.chunk import ArenaFullError

        try:
            self._run_one(part)
        except ArenaFullError as error:
            # retained KV nothing here reads makes room first
            need = self.arena.page_cost(sum(self._entry_rows(*e) for e in part))
            if self.arena.evict_retained(need):
                self._log_memory_event("evict_retry", groups=len(part))
                logger.info("chunk of %d groups retried after evicting "
                            "retained KV", len(part))
                self._run_part(part)
                return
            if len(part) < 2:
                raise
            self._log_memory_event("split", groups=len(part), reason=str(error))
            logger.info("chunk of %d groups split: %s", len(part), error)
            half = len(part) // 2
            self._run_part(part[:half])
            self._run_part(part[half:])

    def _run_one(self, part):
        started = time.perf_counter()
        chunk, entries = self._build(part)
        self.pack_s += time.perf_counter() - started
        self.tokens += chunk.tokens
        if self.gpu_timing:
            e0 = self.torch.cuda.Event(enable_timing=True)
            e1 = self.torch.cuda.Event(enable_timing=True)
            e0.record()
        normed = forward_chunk(self.pipeline, self.arena, chunk)
        if self.gpu_timing:
            e1.record()
            by_stage = {}
            for entry in part:
                stage = entry[1]
                by_stage[stage] = by_stage.get(stage, 0) + self._entry_rows(*entry)
            self.spans.append((by_stage, chunk.tokens, e0, e1))
        for key in chunk.fresh_keys:
            self.arena.trim_window(key)
        # the earlier passes' readouts finished before this forward pass
        # started, so their answers are decided while the GPU runs it;
        # a readout of many rows launches enough kernels to fill the
        # launch queue, and a host blocked there could not decide them
        while self.outstanding:
            self._report_next()
        handles = self.readouts.submit(
            normed, entries, chunk.rows_per_answer)
        self.outstanding.append(PendingAnswers(part, handles))

    def run(self):
        """Drive admission and drain the launched answer handles in order."""
        while True:
            # documents with nothing to send at their first stage never run
            for settlement in self.sched.take_settled():
                self._event(settlement)
            if self.sched.done():
                break
            if len(self.outstanding) > 1:
                self._report_next()
            if self.sched.blocked_pages:
                # the free list is short for the next fresh document:
                # retained KV nothing here reads makes room
                self.arena.evict_retained(self.sched.blocked_pages)
            groups = self.sched.next_chunk(
                self.arena.free_pages if self.paged else 1 << 62)
            if not groups:
                if self.outstanding:
                    self._report_next(partial=True)
                    continue
                if (self.sched.blocked_pages
                        and self.arena.evict_retained(self.sched.blocked_pages)):
                    continue
                # parents freed but held for queued children hold the
                # pages: the children pack their whole prefixes
                if self.arena.drop_holds(self.held):
                    continue
                raise AssertionError("nothing buildable and nothing in flight")
            self._run_part(groups)
            # the packed children hold their parents' pages now
            for a, j, start, _, carried in groups:
                parent = (self.borrowing.tree_parent[a]
                          if a < len(self.borrowing.tree_parent) else None)
                # a copy that borrows its whole document packs no prefix
                packed = carried or (
                    j == 0 and self.borrowing.shared(a) >= self.prefix_lengths[a])
                if packed and start == 0 and parent is not None:
                    self.arena.release(self.keys[parent])
                if a in self._reserved_parents:
                    self.arena.done_reading((self._reserved_parents.pop(a),))
        while self.outstanding:
            self._report_next()
        self._release_skipped()
        # a limit ends the run with documents still queued
        for anchor in self.sched.drain():
            if self.arena.is_resident(self.keys[anchor]):
                self.arena.free_key(self.keys[anchor])
        self.progress.finish(f"{self.label} done", f"{self.tokens:,} fresh tokens")
        self.arena.drop_holds(self.held)
        if self.stats is not None:
            self.stats["borrowed_tokens"] = self.borrowing.borrowed_tokens
            self.stats["pack_s"] = self.pack_s
        if self.spans:
            self.torch.cuda.synchronize()
        return self.sched.answers, self.spans, self.tokens
