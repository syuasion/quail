"""Schedule document requests within token and KV page budgets.

JoinAdmission serves joins, filter sequences, and classifications.
It prioritizes unfinished requests, then documents entering their next
stage, then new documents. Suffix requests are never split across chunks.
Tensor construction is handled by chunk.py.
"""

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np


class OwnRequests(list):
    """A document's requests at a stage whose suffixes are its own.

    The indices run over the document's own sequences, and lengths
    holds their token counts, which the admission prices in place of
    the stage's shared list.
    """

    def __init__(self, lengths):
        super().__init__(range(len(lengths)))
        self.lengths = np.asarray(lengths, dtype=np.int64)

def pages_for(tokens: int, page_tokens: int) -> int:
    """Pages needed for `tokens` rows (suffix KV is never paged)."""
    return -(-tokens // page_tokens)


# --------------------------------------------- join admission

_DONE = -2      # the anchor's last stage answered, or it failed a gate


def partner_pages(page_cost, page_tokens, prefix_tokens, frame_tokens,
                  suffix_rows):
    """Temporary pages one partner's rows take on the unified path.

    They follow the anchor's last partial page, whose rows are copied
    in first.
    """
    return page_cost((prefix_tokens + frame_tokens) % page_tokens
                     + suffix_rows, 0)


class Borrowing:
    """Which parent each document borrows its KV prefix from, as admitted.

    The prefix tree names each document's parent and shared tokens.
    decide() makes the choice when the document is admitted: wait
    while the parent is still queued, borrow the parent's first shared
    tokens, or pack the whole document when the arena cannot serve the
    borrow (the parent has left, or trimmed its window below the
    shared tokens) or the parent never runs (see skip()). record()
    keeps the choice. Filter and join admission both use it.

    Args:
        n: Documents.
        tree: A PrefixTree over the documents, or None.
        can_borrow: Callable(document, parent, same_chunk) -> bool, asked
            before a borrow; same_chunk says the parent is admitted in
            the chunk being built. None allows every borrow.
    """

    WAIT = "wait"

    def __init__(self, n, tree=None, can_borrow=None):
        self.tree_parent = [None] * n if tree is None else list(tree.parent)
        self.tree_shared = [0] * n if tree is None else list(tree.shared)
        self.can_borrow = can_borrow
        self.admitted = set()
        self.skipped = set()       # documents that never run a chunk
        self.chosen = {}           # admitted borrower -> (parent, shared)
        self.borrowed_tokens = 0   # prefix tokens read from a parent

    def detach(self, document):
        """The document's KV is already resident, so it borrows nothing."""
        self.tree_parent[document] = None
        self.tree_shared[document] = 0

    def skip(self, document):
        """The document never runs a chunk, so its borrowers pack whole."""
        self.skipped.add(document)

    def borrowers(self):
        """Per document, how many documents the tree has borrow from it."""
        counts = [0] * len(self.tree_parent)
        for parent in self.tree_parent:
            if parent is not None:
                counts[parent] += 1
        return counts

    def decide(self, document, chunk):
        """WAIT, None to pack the whole document, or (parent, shared)."""
        parent = self.tree_parent[document]
        if parent is None or parent in self.skipped:
            return None
        if parent not in self.admitted:
            return self.WAIT
        if self.can_borrow is not None and not self.can_borrow(
                document, parent, parent in chunk):
            return None
        return parent, self.tree_shared[document]

    def record(self, document, choice):
        """The document was admitted with decide()'s choice."""
        self.admitted.add(document)
        if choice is not None:
            self.chosen[document] = choice
            self.borrowed_tokens += choice[1]

    def parent(self, document):
        """The parent an admitted document borrows from, or None."""
        return self.chosen.get(document, (None, 0))[0]

    def shared(self, document):
        """The tokens an admitted document borrows; 0 when it packs whole."""
        return self.chosen.get(document, (None, 0))[1]


# an anchor's partner request that takes it out of the run at a stage
DROP = object()
# an anchor's partner request that passes it on to the next stage
# with nothing asked at this one
SKIP = object()


@dataclass(frozen=True)
class Settlement:
    """Completion record indicating whether a document survived the last stage."""

    kind: str
    anchor: int
    survived: bool


@dataclass(frozen=True)
class AdmissionReport:
    """Stage transitions and document completions caused by a group's answers.

    Each transition is (anchor, reported stage, passed). Passing a stage
    remains true when requests at the next stage drop the document.
    Skipped stages do not add transitions, and late answers for a stage
    that already advanced do not repeat its transition.
    """

    settlements: tuple[Settlement, ...] = ()
    transitions: tuple[tuple[int, int, bool], ...] = ()


class JoinAdmission:
    """Scheduler that mixes documents at different stages within each chunk.

    Args:
        prefix_tokens: Per-anchor prefix token counts.
        stage_suffixes: Per stage, the partner suffix token counts of
            every partner tuple at that stage.
        chunk_budget: Tokens per forward pass.
        arena_pages: Pages in the KV arena.
        page_tokens: Tokens per arena page.
        frame_tokens: Per stage, tokens of the framing written into
            the anchor's KV ahead of its first suffix.
        resident: anchor -> pages already held. A resident anchor's
            prefix KV is in the arena, so it packs no prefix tokens.
        anchor_partners: anchor -> per stage, the indices into that
            stage's partner list the anchor streams, or None for the
            whole list. Omitted anchors stream the whole list at every
            stage. An anchor's value may instead be a callable(stage)
            -> list, None, DROP, or SKIP, asked when the anchor enters
            the stage; DROP takes the anchor out of the run there,
            SKIP passes it to the next stage with nothing asked (a
            stage after the first only).
        frame_canvas_tokens: Rows a diffusion model adds after a frame
            entry. They take chunk room and page room but are never
            kept in the anchor's KV. A suffix's canvas rows are part of
            its count in stage_suffixes.
        page_cost: Callable(tokens, base_tokens) giving the pages a
            key of that many rows takes, in the arena's every-token pages; None
            prices one pool of page_tokens pages.
        tree: A PrefixTree over the anchors given at construction, or
            None. Fresh anchors are admitted in tree order; one whose
            parent is still queued waits, one whose parent's pages can
            be borrowed packs and pays for only the tokens past its
            shared length, and one whose parent has left packs whole.
        can_borrow: Callable(anchor, parent, same_chunk) -> bool, asked
            before a borrow; same_chunk says the parent is admitted in
            the chunk being built. None allows every borrow.
        advance: Callable(anchor, stage, row) -> bool deciding, once an
            anchor's whole row at a stage is in, whether it goes on to
            the next stage, or at the last stage whether it survives.
            None advances an anchor as soon as one answer in the row
            is true.
        frame_writes: Per stage, whether its frame is written; a stage
            whose frame is already in the anchor's KV from the stage
            before packs no frame rows. None writes every frame.
        limit: Stop admitting anchors once this many survived the last
            stage; None runs every anchor.
        survival: The fraction of anchors expected to survive every
            stage, or None. Under a limit, anchors are admitted only
            while the ones in flight are expected to fall short of it;
            None admits every anchor that fits.
        stop_groups: Per anchor, its stop key group, or None. At most
            stop_width anchors of a group are admitted at a time, and
            once one survives the last stage the group's queued anchors
            are skipped: they never run and take_skipped() lists them.
        stop_width: Anchors of one group in flight at once; the caller
            widens it when the groups are too few to fill a chunk.
        extra_tokens: Rows past the prefix an anchor's pages must
            cover, when more than its largest frame: a stage whose
            one suffix is written into the anchor's own pages.
        answer_dtype: The answer array type, one for every stage or a
            list per stage; None records 0/1 lists.
        reserve: Optional callable(anchor, borrow) that reserves the anchor's
            storage before admission. borrow is None or (parent, shared tokens).
            False leaves the anchor queued; True commits the reservation.

    Each chunk fills in priority order: partner streams cut by the
    previous chunk, then anchors starting their next stage, then
    fresh anchors whose pages fit the free list. Requests skipped for token
    space take priority in the next chunk. Pages are granted in queue order.
    An anchor advances to its next stage once its whole stream is launched
    and one partner has answered TRUE; the remaining answers fill in
    while the next stage runs. It drops out when every partner
    answered FALSE. An anchor with no partner at a stage settles
    without a chunk: finished with an empty row at the last stage,
    dropped at any earlier one.
    """

    def __init__(self, prefix_tokens, stage_suffixes, chunk_budget,
                 arena_pages, page_tokens, frame_tokens=None,
                 resident=None, anchor_partners=None, temporary_suffix_pages=False,
                 answer_dtype=None, frame_canvas_tokens=0, page_cost=None,
                 tree=None, can_borrow=None, advance=None,
                 frame_writes=None, limit=None, survival=None, extra_tokens=None,
                 stop_groups=None, stop_width=1, reserve=None):
        k = len(stage_suffixes)
        self.answer_dtypes = (list(answer_dtype) if isinstance(answer_dtype, list)
                              else [answer_dtype] * k)
        if len(self.answer_dtypes) != k:
            raise ValueError("answer_dtype must match stage_suffixes")
        self.advance = advance
        self.limit = limit
        self.survival = survival
        self.survivors = 0
        n = len(prefix_tokens)
        self.groups = None if stop_groups is None else list(stop_groups)
        if self.groups is not None and len(self.groups) != n:
            raise ValueError("stop_groups must match prefix_tokens")
        self._group_width = max(1, int(stop_width))
        self._group_busy = {}       # group -> its anchors admitted, unsettled
        self._group_done = set()    # groups with a survivor
        self._group_members = {}    # group -> its anchors still queued
        self._skipped = []          # queued anchors a done group skipped
        self._dead = 0              # skipped anchors still in the pending deque
        self.borrowing = Borrowing(n, tree, can_borrow)
        # page_cost(tokens, base_tokens) prices a key in the arena's
        # every-token pages; the default is one pool of page_tokens pages
        self.page_cost = page_cost or (
            lambda tokens, base_tokens=None: pages_for(tokens, page_tokens))
        self._answer_counts = [{} for _ in stage_suffixes]
        self.temporary_suffix_pages = temporary_suffix_pages
        self._page_cums = {}
        self._page_reserve = 0
        self.prefix = []
        self.stages = [np.asarray(s, dtype=np.int64) for s in stage_suffixes]
        # frames: the rows a frame entry keeps in the anchor's KV;
        # frame_rows: the rows it packs, canvas included
        self.frames = (list(frame_tokens) if frame_tokens
                       else [0] * len(self.stages))
        if len(self.frames) != len(self.stages):
            raise ValueError("frame_tokens must match stage_suffixes")
        writes = ([True] * len(self.stages) if frame_writes is None
                  else list(frame_writes))
        if len(writes) != len(self.stages):
            raise ValueError("frame_writes must match stage_suffixes")
        self.frame_writes = writes
        self.frame_rows = [f + frame_canvas_tokens if f and write else 0
                           for f, write in zip(self.frames, writes)]
        if not self.stages:
            raise ValueError("a join needs at least one stage")
        self.chunk_budget = chunk_budget
        self.arena_pages = arena_pages
        self.page_tokens = page_tokens
        k = len(self.stages)
        self._cum = []
        for j, (lens, frame) in enumerate(zip(self.stages, self.frame_rows)):
            self._cum.append(np.concatenate(([0], np.cumsum(lens))))
            longest = int(lens.max()) if len(lens) else 0
            if len(lens) and frame + longest > chunk_budget:
                raise ValueError(
                    f"stage {j}: a {frame + longest}-token partner "
                    f"exceeds the {chunk_budget}-token chunk budget "
                    f"(suffixes are atomic)")
        self._extra = max(self.frame_rows)
        if extra_tokens is not None:
            self._extra = max(self._extra, extra_tokens)
        self._lists = []           # per anchor, per stage: indices or None
        self._cums = []            # per anchor, per stage: cumulative sums
        self._lazy = []            # per anchor: callable(stage) or None
        self._page_cost = []
        self._carried = []
        self._first = []           # per anchor: its first partner's cost
        self._min_fresh = float("inf")
        self._zero_cost = 0
        self.pending = deque()
        self.ready = deque()       # placed anchors with partners left
        self._stage = []           # current stage, or _DONE
        self._next = []            # next partner index at the stage
        self._true = []
        self._settled = []         # events for anchors that never ran
        self.in_flight = 0         # launched groups not yet reported
        self.blocked_pages = 0
        self.reserve = reserve
        self.answers = [dict() for _ in range(k)]
        resident = dict(resident or {})
        anchor_partners = dict(anchor_partners or {})
        for a, prefix in enumerate(prefix_tokens):
            self._register(prefix, resident.get(a), anchor_partners.get(a))
        # resident anchors first: they cost no prefix tokens and few
        # or no pages, so they never wait behind a page-blocked anchor
        n = len(self.prefix)
        fresh_order = range(n) if tree is None else tree.order
        self.pending.extend(a for a in range(n)
                            if a in resident and self._stage[a] == -1)
        self.pending.extend(a for a in fresh_order
                            if a not in resident and self._stage[a] == -1)
        for a in resident:
            self.borrowing.detach(a)
        if self.groups is not None:
            for a in self.pending:
                self._group_members.setdefault(self.groups[a], set()).add(a)


    def num_partners(self, a, j):
        """Return the number of partner requests anchor a streams at stage j.

        A lazy stage not yet entered counts its full suffix list.
        """
        lst = self._lists[a][j]
        return len(self.stages[j]) if lst is None else len(lst)

    def _enter(self, a, j):
        """Advance a document past skipped stages to its next request.

        Args:
            a: Anchor document index.
            j: Stage index to enter.

        Returns:
            None if the document has requests to run. Otherwise, the completion
            event: "dropped" for removal or an empty intermediate stage, and
            "finished" when the document completes its final stage.
        """
        k = len(self.stages)
        ask = self._lazy[a]
        while True:
            self._stage[a] = j
            self._next[a] = 0
            lst = None if ask is None else ask(j)
            if lst is DROP:
                self._stage[a] = _DONE
                return "dropped"
            if lst is SKIP:
                # nothing asked here: the anchor passes this stage
                self._true[a][j] = True
                if j + 1 == k:
                    self._stage[a] = _DONE
                    self.survivors += 1
                    return "finished"
                j += 1
                continue
            if ask is not None:
                self._lists[a][j], self._cums[a][j] = self._partner_list(
                    a, j, lst)
            if self.num_partners(a, j):
                return None
            # an empty row at the last stage, dropped otherwise
            self._stage[a] = _DONE
            return "finished" if j + 1 == k else "dropped"

    def _partner_list(self, a, j, lst):
        """Return partner indices and cumulative request token counts for a stage.

        A list carrying its own lengths (OwnRequests) prices those in
        place of the stage's shared list.
        """
        if lst is None:
            return None, None
        lengths = lst.lengths if isinstance(lst, OwnRequests) else None
        lst = np.asarray(lst, dtype=np.int64)
        if lengths is None:
            invalid = (lst < 0) | (lst >= len(self.stages[j]))
            if invalid.any():
                i = lst[invalid][0]
                raise ValueError(
                    f"anchor {a} stage {j}: partner {i} is out of range")
            lengths = self.stages[j][lst]
        return lst, np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

    def _cum_of(self, a, j):
        return self._cum[j] if self._lists[a][j] is None else self._cums[a][j]

    def partner_indices(self, a, j, start, end):
        """Indices into stage j's partner list for one launched group."""
        lst = self._lists[a][j]
        return range(start, end) if lst is None else lst[start:end]

    def _register(self, prefix, resident_pages, partners):
        """Record one anchor's costs; returns its index."""
        a = len(self.prefix)
        k = len(self.stages)
        lazy = None
        if partners is None:
            lists = [None] * k
            cums = [None] * k
        elif callable(partners):
            # stage 0 now, later stages when the anchor reaches them
            lazy = partners
            lists = [None] * k
            cums = [None] * k
            first = partners(0)
            if first is SKIP:
                raise ValueError(
                    f"anchor {a}: the first stage cannot be skipped")
            if first is DROP:
                first = []
            lists[0], cums[0] = self._partner_list(a, 0, first)
        else:
            if len(partners) != k:
                raise ValueError(
                    f"anchor {a}: partner lists for {len(partners)} stages, "
                    f"the join has {k}")
            lists, cums = [], []
            for j, lst in enumerate(partners):
                lst, cum = self._partner_list(a, j, lst)
                lists.append(lst)
                cums.append(cum)
        self._lazy.append(lazy)
        need = self.page_cost(prefix + self._extra)
        if resident_pages is not None:
            need = max(0, need - resident_pages)
        elif need > self.arena_pages:
            raise ValueError(
                f"anchor {a} needs {need} KV pages; the arena "
                f"holds {self.arena_pages}")
        carried = 0 if resident_pages is not None else prefix
        self.prefix.append(prefix)
        self._lists.append(lists)
        self._cums.append(cums)
        self._page_cost.append(need)
        self._carried.append(carried)
        self._stage.append(-1)
        self._next.append(0)
        self._true.append([False] * k)
        if self.num_partners(a, 0) == 0:
            # nothing to evaluate: the anchor settles without a chunk
            self._first.append(0)
            self._stage[a] = _DONE
            self._settled.append(Settlement(
                "finished" if k == 1 else "dropped", a, False))
            self.borrowing.skip(a)
            return a
        first = self.frame_rows[0] + self._suffix(a, 0, 0)
        if resident_pages is None and prefix + first > self.chunk_budget:
            raise ValueError(
                f"anchor {a}: prefix {prefix} tokens leaves no "
                f"room for a partner in a {self.chunk_budget}-token "
                f"chunk")
        if self.temporary_suffix_pages:
            # a lazy anchor's later stages are bounded by the stage's
            # largest suffix
            largest = max(
                (self._suffix_page_cost(a, j, i)
                 for j in range(k) if lazy is None or j == 0
                 for i in range(self.num_partners(a, j))),
                default=0,
            )
            if lazy is not None:
                largest = max([largest] + [
                    partner_pages(self.page_cost, self.page_tokens, prefix,
                                  self.frames[j], int(self.stages[j].max()))
                    for j in range(1, k) if len(self.stages[j])])
            needed = self.page_cost(prefix + self._extra) + largest
            if needed > self.arena_pages:
                raise ValueError("anchor and one suffix exceed the KV arena")
            self._page_reserve = max(self._page_reserve, largest)
        self._first.append(first)
        # the fewest tokens any fresh anchor can pack: a borrower
        # packs only the tokens past its shared prefix
        tree_shared = self.borrowing.tree_shared
        shared = tree_shared[a] if a < len(tree_shared) else 0
        self._min_fresh = min(self._min_fresh, max(0, carried - shared) + first)
        if not need:
            self._zero_cost += 1
        return a

    def _suffix(self, a, j, i):
        cum = self._cum_of(a, j)
        return cum[i + 1] - cum[i]

    def take_settled(self):
        """Remove and return documents settled without a forward pass.

        Returns:
            Settlement records, using the same format as report().settlements.
        """
        events, self._settled = self._settled, []
        for event in events:
            self._settle_group(event.anchor, event.survived)
        return events

    def _first_cost(self, a, j, i):
        return self._suffix(a, j, i) + (self.frame_rows[j] if i == 0 else 0)

    def _suffix_page_cost(self, a, j, i):
        return partner_pages(self.page_cost, self.page_tokens, self.prefix[a],
                             self.frames[j], self._suffix(a, j, i))

    def _take(self, a, j, i, room, page_room):
        """Return the partner run that fits the token and temporary page budgets."""
        cum = self._cum_of(a, j)
        frame = self.frame_rows[j] if i == 0 else 0
        end = bisect_right(cum, cum[i] + room - frame) - 1
        pages = 0
        if self.temporary_suffix_pages:
            key = (a, j)
            if key not in self._page_cums:
                costs = [0]
                for index in range(self.num_partners(a, j)):
                    costs.append(costs[-1] + self._suffix_page_cost(a, j, index))
                self._page_cums[key] = costs
            costs = self._page_cums[key]
            end = min(end, bisect_right(costs, costs[i] + page_room) - 1)
            pages = costs[end] - costs[i]
        return end, frame + cum[end] - cum[i], pages

    def _launch(self, a, j, end):
        """Record a launched group; True when the stream continues."""
        self._next[a] = end
        self.in_flight += 1
        return end < self.num_partners(a, j)

    def next_chunk(self, free_pages):
        """Groups for the next chunk: [(anchor, stage, start, end, carried)].

        start and end index the anchor's own partner list at the
        stage; partner_indices() maps them to the stage's list.
        carried means the anchor's prefix tokens are packed and its KV
        written to its pages. free_pages is the arena's free list;
        fresh anchors admit against it. Returns [] when nothing is
        buildable.
        """
        self.blocked_pages = 0
        if self.limit_reached():
            return []
        room = self.chunk_budget
        groups = []
        continued = []
        deferred = []
        temporary_pages = 0
        # 1) placed anchors: cut streams (front) and next-stage starts
        for _ in range(len(self.ready)):
            if room == 0:
                break
            a = self.ready.popleft()
            j, i = self._stage[a], self._next[a]
            if self._first_cost(a, j, i) > room:
                deferred.append(a)
                continue
            end, tokens, pages = self._take(a, j, i, room, free_pages)
            if end == i:
                self.blocked_pages = self._suffix_page_cost(a, j, i) - free_pages
                self.ready.appendleft(a)
                break
            free_pages -= pages
            temporary_pages += pages
            groups.append((a, j, i, end, False))
            room -= tokens
            if self._launch(a, j, end):
                continued.append(a)
        # 2) fresh anchors: pages in queue order, chunk room may skip
        held = []
        blocked = False
        admitted = set()
        paced = self.limit is not None and self.survival is not None
        expected_rows = self._live() if paced else 0
        while self.pending and room >= self._min_fresh:
            if paced and self.survivors + expected_rows * self.survival >= self.limit:
                break       # the anchors in flight should reach the limit
            a = self.pending.popleft()
            if self._stage[a] == _DONE:
                self._dead -= 1     # skipped by its group's survivor
                continue
            if self.groups is not None and self._group_busy.get(
                    self.groups[a], 0) >= self._group_width:
                held.append(a)      # the group's anchors in flight are enough
                continue
            borrow = self.borrowing.decide(a, admitted)
            if borrow == Borrowing.WAIT:
                held.append(a)      # its parent is still queued
                continue
            need = (self._page_cost[a] if borrow is None
                    else self.page_cost(self.prefix[a] - borrow[1] + self._extra))
            if blocked and need:
                held.append(a)
                if not self._zero_cost:
                    break
                continue
            required = need + max(0, self._page_reserve - temporary_pages)
            if required > free_pages:
                blocked = True
                self.blocked_pages = required - free_pages
                held.append(a)
                if not self._zero_cost:
                    break
                continue
            # a copy may borrow past its document, into the question
            carried = (self._carried[a] if borrow is None
                       else max(0, self.prefix[a] - borrow[1]))
            if carried + self._first[a] > room:
                held.append(a)      # chunk room only; retry next chunk
                continue
            end, tokens, pages = self._take(
                a, 0, 0, room - carried, free_pages - need)
            if end == 0:
                held.append(a)
                self.blocked_pages = (
                    need + self._suffix_page_cost(a, 0, 0) - free_pages)
                break
            if self.reserve is not None and not self.reserve(a, borrow):
                held.append(a)
                break
            temporary_pages += pages
            groups.append((a, 0, 0, end, carried > 0))
            room -= carried + tokens
            free_pages -= need + pages
            if not self._page_cost[a]:
                self._zero_cost -= 1
            self._stage[a] = 0
            admitted.add(a)
            expected_rows += 1
            if self.groups is not None:
                group = self.groups[a]
                self._group_busy[group] = self._group_busy.get(group, 0) + 1
                self._group_members[group].discard(a)
            self.borrowing.record(a, borrow)
            if self._launch(a, 0, end):
                continued.append(a)
        self.pending.extendleft(reversed(held))
        self.ready.extendleft(reversed(continued))
        self.ready.extendleft(reversed(deferred))
        return groups

    # ---- gating --------------------------------------------------------

    def report(self, a, j, start, end, bits):
        """Record a group's answers and update document progress.

        Returns:
            An AdmissionReport containing stage transitions and document
            settlements. A transition records the stage's decision even if the
            next stage immediately removes the document. Settlements record
            whether the document survived its final stage.
        """
        dtype = self.answer_dtypes[j]
        if dtype is None:
            row = self.answers[j].setdefault(a, [])
            received = len(row)
        else:
            if a not in self.answers[j]:
                self.answers[j][a] = np.empty(self.num_partners(a, j), dtype=dtype)
            row = self.answers[j][a]
            received = self._answer_counts[j].get(a, 0)
        if received != start or len(bits) != end - start:
            raise AssertionError(
                f"anchor {a} stage {j}: answers for partners "
                f"{start}:{end} arrived with {received} recorded")
        if dtype is None:
            row.extend(bits)
        else:
            row[start:end] = bits
            self._answer_counts[j][a] = end
        self.in_flight -= 1
        k = len(self.stages)
        num_partners = self.num_partners(a, j)
        complete = end == num_partners
        if self.advance is not None:
            if complete:
                self._true[a][j] = bool(self.advance(a, j, row))
        elif (any(bits) if dtype is None else np.any(bits)):
            self._true[a][j] = True
        settlements = []
        transitions = []
        if j == self._stage[a] and j + 1 < k:
            if self._true[a][j] and self._next[a] == num_partners:
                # the whole stream is launched, so every later chunk
                # is behind it on the stream and the next stage's
                # frame write cannot race a read of this stage's
                transitions.append((a, j, True))
                settled = self._enter(a, j + 1)
                if settled is None:
                    self.ready.append(a)
                else:
                    settlements.append(Settlement(
                        settled, a, bool(self._true[a][-1])))
            elif complete and not self._true[a][j]:
                self._stage[a] = _DONE
                transitions.append((a, j, False))
                settlements.append(Settlement("dropped", a, False))
        if j == k - 1 and complete:
            self._stage[a] = _DONE
            if self._true[a][j]:
                self.survivors += 1
            survived = bool(self._true[a][j])
            transitions.append((a, j, survived))
            settlements.append(Settlement("finished", a, survived))
        for settlement in settlements:
            self._settle_group(settlement.anchor, settlement.survived)
        return AdmissionReport(tuple(settlements), tuple(transitions))

    def _settle_group(self, a, survived):
        """Free the anchor's group for its next anchor, or skip the rest.

        A skipped anchor stays in the pending deque until the admission
        pops it, so a skip costs the group's size, not the queue's.
        """
        if self.groups is None:
            return
        group = self.groups[a]
        busy = self._group_busy.get(group, 0) - 1
        if busy > 0:
            self._group_busy[group] = busy
        else:
            self._group_busy.pop(group, None)
        if not survived or group in self._group_done:
            return
        self._group_done.add(group)
        skipped = sorted(self._group_members.pop(group, ()))
        for b in skipped:
            self._stage[b] = _DONE
            if not self._page_cost[b]:
                self._zero_cost -= 1
        self._dead += len(skipped)
        self._skipped.extend(skipped)

    def take_skipped(self):
        """Remove and return the queued anchors a done group skipped."""
        skipped, self._skipped = self._skipped, []
        return skipped

    def _live(self) -> int:
        """Anchors admitted and not yet settled."""
        return sum(1 for stage in self._stage if stage >= 0)

    def limit_reached(self) -> bool:
        """Return whether enough documents survived to stop new admissions."""
        return self.limit is not None and self.survivors >= self.limit

    def drain(self):
        """Remove queued documents after the result limit is reached.

        Returns:
            Document indices whose KV the caller must release. These documents
            have not run through the remaining stages.
        """
        out = [a for a in self.ready if self._stage[a] != _DONE]
        out.extend(a for a in self.pending if self._stage[a] != -1)
        for a in out:
            self._stage[a] = _DONE
        self.ready.clear()
        self.pending.clear()
        self._dead = 0
        return out

    # ---- progress ------------------------------------------------------

    def done(self):
        if self.limit_reached():
            return not self.in_flight and not self._settled
        return len(self.pending) == self._dead and not self.ready \
            and not self.in_flight and not self._settled
