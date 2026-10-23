"""Page accounting for the KV arena: a free list, page lists per key, retention."""

import heapq


class PageArena:
    """Page accounting: a free list and per-document page lists.

    A key may borrow the leading pages of another resident key's
    prefix: it reads them and never writes them. A page returns to
    the free list when its last holder frees it.
    """

    def __init__(self, num_pages: int, page_tokens: int, num_slots: int = 0):
        self.num_pages = num_pages
        self.page_tokens = page_tokens
        self.num_slots = num_slots
        self.free_slots = list(range(num_slots - 1, -1, -1))    # stack
        self.slots = {}    # key -> the checkpoint slot it holds
        self.slot_tokens = {}    # key -> tokens its checkpoint covers
        self.ext_slots = {}    # key -> slot of the state past retained suffix
        self.ext_tokens = {}    # key -> tokens that second checkpoint covers
        self.share_slots = {}    # (key, tokens) -> slot of the state there
        self.free = list(range(num_pages - 1, -1, -1))    # stack
        self.owned = {}       # key -> list of page ids it alone writes
        self.borrowed = {}    # key -> leading page ids read from a parent
        self.holds = [0] * num_pages    # page id -> keys holding it
        self.tokens = {}      # key -> resident token count, borrowed included
        # key -> tokens past its prefix that its admitted chunk writes and
        # nothing rewrites, so an exact copy may borrow them too
        self.readable = {}
        self.pinned = set()    # current operators depend on these keys
        self._retained_sizes = {}
        self._retained_pages = 0
        self.retained = {}     # key -> reusable prefix tokens
        self._retained_heap = []
        self._retained_versions = {}
        self._retention_version = 0
        self.retention_policy = None
        self.retention_cap_pages = None

    def pages_needed(self, tokens: int) -> int:
        return -(-tokens // self.page_tokens)

    def alloc(self, key, tokens: int, capacity_tokens: int | None = None,
              borrowed=()):
        """The document's own pages, or None when the free list is short.

        The admission scheduler treats None as "wait". borrowed are
        resident pages of another key the new key reads as its
        leading pages (see shared_pages); tokens and capacity_tokens
        count them.
        """
        if key in self.owned:
            raise KeyError(f"{key!r} already resident")
        capacity_tokens = tokens if capacity_tokens is None else capacity_tokens
        if capacity_tokens < tokens:
            raise ValueError("capacity_tokens must cover logical tokens")
        borrowed = list(borrowed)
        if len(borrowed) * self.page_tokens > capacity_tokens:
            raise ValueError("a key cannot borrow past its own capacity")
        need = self.pages_needed(capacity_tokens) - len(borrowed)
        if need > len(self.free):
            return None
        pages = [self.free.pop() for _ in range(need)]
        self.owned[key] = pages
        self.borrowed[key] = borrowed
        self.tokens[key] = tokens
        for page in borrowed + pages:
            self.holds[page] += 1
        return pages

    def shared_pages(self, parent, shared: int, start: int = 0) -> list:
        """The parent's pages for its tokens [start, shared), to borrow.

        Both bounds are whole pages. The parent's first page must hold
        token `start` or an earlier one.
        """
        if (shared - start) % self.page_tokens or start % self.page_tokens:
            raise ValueError("a borrowed prefix is a whole number of pages")
        if parent not in self.owned:
            raise KeyError(f"borrow parent {parent!r} is not resident")
        if shared > self.readable_tokens(parent):
            raise ValueError(f"{parent!r} holds fewer tokens than borrowed")
        first = start // self.page_tokens
        return self.table_pages(parent)[first:shared // self.page_tokens]

    def readable_tokens(self, key) -> int:
        """The leading tokens of a key another key may borrow."""
        return max(self.tokens[key], self.readable.get(key, 0))

    def drop_leading(self, key, pages: int) -> int:
        """Drop a key's first pages, borrowed ones first; returns pages freed."""
        borrowed = self.borrowed[key]
        from_borrowed = min(pages, len(borrowed))
        dropped = borrowed[:from_borrowed] + self.owned[key][:pages - from_borrowed]
        self.borrowed[key] = borrowed[from_borrowed:]
        self.owned[key] = self.owned[key][pages - from_borrowed:]
        return self.release_pages(dropped)

    def table_pages(self, key) -> list:
        """The pages a key's block table reads: borrowed, then its own."""
        return self.borrowed[key] + self.owned[key]

    def borrowed_tokens(self, key) -> int:
        return len(self.borrowed[key]) * self.page_tokens

    def release_pages(self, pages) -> int:
        """Drop one hold on each page; returns how many went free."""
        freed = 0
        for page in pages:
            self.holds[page] -= 1
            if not self.holds[page]:
                self.free.append(page)
                freed += 1
        return freed

    def alloc_slot(self, key):
        """A checkpoint slot for a resident key, or None when none is free.

        A key that already holds a slot keeps it.
        """
        if key not in self.owned:
            raise KeyError(key)
        if key in self.slots:
            return self.slots[key]
        if not self.free_slots:
            return None
        self.slots[key] = self.free_slots.pop()
        return self.slots[key]

    def alloc_extension(self, key):
        """A second checkpoint slot for a resident key, or None when none is free.

        Holds the state past tokens the key's KV retains beyond its
        document, so the document-end checkpoint stays for a later
        operator that asks at the document's length.
        """
        if key not in self.owned:
            raise KeyError(key)
        if key in self.ext_slots:
            return self.ext_slots[key]
        if not self.free_slots:
            return None
        self.ext_slots[key] = self.free_slots.pop()
        return self.ext_slots[key]

    def alloc_share(self, key, tokens: int):
        """A slot for the key's state after `tokens` tokens, or None.

        A borrower of the key's first `tokens` tokens starts from it.
        The slot lives until the key is freed.
        """
        if key not in self.owned:
            raise KeyError(key)
        point = (key, tokens)
        if point in self.share_slots:
            return self.share_slots[point]
        if not self.free_slots:
            return None
        self.share_slots[point] = self.free_slots.pop()
        return self.share_slots[point]

    def drop_extension(self, key) -> None:
        """Free the key's second checkpoint slot, if it has one."""
        slot = self.ext_slots.pop(key, None)
        self.ext_tokens.pop(key, None)
        if slot is not None:
            self.free_slots.append(slot)

    def free_key(self, key) -> int:
        """Drop a document's pages and slots; returns the pages that went free."""
        slot = self.slots.pop(key, None)
        self.slot_tokens.pop(key, None)
        self.drop_extension(key)
        for point in [p for p in self.share_slots if p[0] == key]:
            self.free_slots.append(self.share_slots.pop(point))
        if slot is not None:
            self.free_slots.append(slot)
        pages = self.owned.pop(key)
        borrowed = self.borrowed.pop(key)
        self.tokens.pop(key)
        self.readable.pop(key, None)
        self.pinned.discard(key)
        self._forget_retained(key)
        return self.release_pages(borrowed + pages)

    def pin(self, key) -> None:
        """Protect a resident key while an operator uses it."""
        if key not in self.owned:
            raise KeyError(key)
        self._forget_retained(key)
        self.pinned.add(key)

    def _forget_retained(self, key):
        self._retained_pages -= self._retained_sizes.pop(key, 0)
        self.retained.pop(key, None)
        self._retained_versions.pop(key, None)

    def retain(self, key, priority=None) -> None:
        """Make a resident key evictable after its current use."""
        if key not in self.owned:
            raise KeyError(key)
        self.pinned.discard(key)
        prefix_tokens = self.tokens[key]
        self._forget_retained(key)
        # the pages evicting the key would free: those it alone holds
        # now; a child freed later leaves more, so the count is a
        # lower bound, and admission short of pages evicts anyway
        pages = sum(1 for page in self.table_pages(key) if self.holds[page] == 1)
        self.retained[key] = prefix_tokens
        self._retained_sizes[key] = pages
        self._retained_pages += pages
        self._retention_version += 1
        version = self._retention_version
        self._retained_versions[key] = version
        if priority is None:
            # a key whose pages its children all share frees none alone
            priority = (self.retention_policy.priority(
                key, prefix_tokens, max(1, pages))
                if self.retention_policy
                else (prefix_tokens / max(1, pages),))
        heapq.heappush(
            self._retained_heap,
            (priority, version, key, pages, prefix_tokens),
        )

    def configure_retention(self, policy, cap_pages: int) -> None:
        """Update priorities without making active prefixes evictable."""
        if not 0 <= cap_pages <= self.num_pages:
            raise ValueError("retention capacity must fit inside the arena")
        self.retention_policy = policy
        self.retention_cap_pages = cap_pages
        keys = list(self.retained)
        self._retained_heap.clear()
        for key in keys:
            self.retain(key)

    def pop_retained_victim(self, keep=frozenset()):
        """Remove the retained prefix with the lowest planned reuse priority.

        Keys in keep stay retained and are passed over.
        """
        passed = []
        victim = None
        while self._retained_heap:
            entry = heapq.heappop(self._retained_heap)
            _, version, key, pages, prefix_tokens = entry
            if self._retained_versions.get(key) != version:
                continue
            if key in keep:
                passed.append(entry)
                continue
            self._forget_retained(key)
            victim = key, pages, prefix_tokens
            break
        for entry in passed:
            heapq.heappush(self._retained_heap, entry)
        return victim

    def rewind(self, key, tokens: int) -> int:
        """Keep the first tokens and return unused trailing pages."""
        if key not in self.owned:
            raise KeyError(key)
        borrowed = self.borrowed_tokens(key)
        capacity = borrowed + len(self.owned[key]) * self.page_tokens
        if tokens < borrowed or tokens > capacity:
            raise ValueError("rewind tokens leave the resident capacity")
        keep = self.pages_needed(tokens) - len(self.borrowed[key])
        released = self.owned[key][keep:]
        self.owned[key] = self.owned[key][:keep]
        self.tokens[key] = tokens
        self.readable.pop(key, None)
        freed = self.release_pages(released)
        if key in self.retained:
            self.retain(key)
        return freed

    def grow(self, key, capacity_tokens: int) -> int | None:
        """Add pages for capacity_tokens without changing logical tokens."""
        if key not in self.owned:
            raise KeyError(key)
        need = (self.pages_needed(capacity_tokens) - len(self.borrowed[key])
                - len(self.owned[key]))
        if need <= 0:
            return 0
        if need > len(self.free):
            return None
        added = [self.free.pop() for _ in range(need)]
        self.owned[key].extend(added)
        for page in added:
            self.holds[page] += 1
        return need

    def row_indices(self, key, tokens=None):
        """Flat row positions of the document's tokens inside a pool.

        The pool is viewed as (num_pages * page_tokens, ...): page p holds
        rows p*page_tokens .. p*page_tokens + page_tokens.
        """
        rows = []
        left = self.tokens[key] if tokens is None else tokens
        for p in self.table_pages(key):
            take = min(left, self.page_tokens)
            base = p * self.page_tokens
            rows.extend(range(base, base + take))
            left -= take
        return rows

    @property
    def free_pages(self) -> int:
        return len(self.free)

    @property
    def resident_tokens(self) -> int:
        return sum(self.tokens.values())

    @property
    def retained_pages(self) -> int:
        return self._retained_pages

    @property
    def retained_prefix_tokens(self) -> int:
        return sum(self.tokens[key] for key in self.retained)


def _evict_event(key, prefix_tokens):
    """The answer-sink entry for one evicted document prefix.

    Returns None when the key is not an alias and a document index.
    """
    if not (isinstance(key, tuple) and len(key) == 2):
        return None
    alias, document = key
    if not isinstance(alias, str):
        return None
    try:
        document = int(document)
    except (TypeError, ValueError):
        return None
    return {"kind": "evict", "alias": alias, "document": document,
            "tokens": int(prefix_tokens)}
