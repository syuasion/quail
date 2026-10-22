"""Rules that change executable graphs."""

from __future__ import annotations

from dataclasses import replace

from quail.cost.budgets import choose_attention_path, tree_attention_allowed
from quail.cost.retention import coefficients
from quail.physical import (
    Aggregate as PhysicalAggregate,
)
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiInference,
    AiJoin,
    AiScore,
    Limit,
    PhysicalGraph,
    PortRef,
    Sort,
    Window,
)
from quail.physical.base import input_ports
from quail.physical.pipelines import build_pipelines
from quail.planner.graph import retention
from quail.planner.operators.classify import classify_scoring
from quail.planner.optimizer.rules import PhysicalRule
from quail.planner.plan import plan_kv_retention
from quail.planner.search.statistics import (
    cached_statistics,
    expected_rows_after_filters,
)
from quail.planner.statistics.prefixes import page_tree
from quail.planner.statistics.tables import summarize_alias


class ShouldPushDownLimit(PhysicalRule):
    """Let a LIMIT end the filter run once enough documents survived.

    The rule puts the limit in the ``filter_limit`` setting, which the
    executor reads to stop admitting documents once that many passed
    every operator of their pipeline. The pushdown keeps the first
    survivors to finish, so it is sound only when those are the rows the
    query returns: the graph ends in a Limit, its documents run through
    one pipeline, and it has no Sort, PhysicalAggregate, or Window (an
    ORDER BY, DISTINCT, OFFSET, GROUP BY, or window function needs
    every survivor) and no AiJoin
    (survivors are settled per join round). Otherwise the rule removes
    the setting.
    """

    name = "should_push_down_limit"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        limit = next((node for node in graph.nodes
                      if isinstance(node, Limit)), None)
        pipelines = {id(members) for members
                     in build_pipelines(graph).values()}
        blocked = len(pipelines) > 1 or any(
            isinstance(node, (Sort, PhysicalAggregate, Window, AiJoin))
            for node in graph.nodes)
        if limit is None or blocked:
            context.settings.pop("filter_limit", None)
        else:
            context.settings["filter_limit"] = limit.count
        return None


def _window_sort_keys(call) -> tuple:
    """The Sort keys that order rows as the window call's partition and order do."""
    return tuple((name, False, False) for name in call.partition) + tuple(call.order)


class ShouldEliminateWindowSort(PhysicalRule):
    """Drop a Sort whose order the Window before it already produced.

    A Window's rows come out in its last call's partition and order, so
    a Sort over the same keys, with no DISTINCT, OFFSET, or LIMIT, has
    nothing to do; the Window returns the Sort's columns instead.
    """

    name = "should_eliminate_window_sort"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        sort = next((node for node in graph.nodes if isinstance(node, Sort)), None)
        if sort is None or sort.distinct or sort.offset or sort.fetch is not None:
            return None
        window = graph.node(sort.inputs[0].source.node_id)
        if not isinstance(window, Window) or graph.root.node_id != sort.node_id \
                or sort.keys != _window_sort_keys(window.calls[-1]):
            return None
        nodes = tuple(replace(node, columns=sort.columns) if node is window else node
                      for node in graph.nodes if node is not sort)
        return PhysicalGraph(nodes, sort.inputs[0].source)


class ShouldPushDownLimitThroughWindow(PhysicalRule):
    """Turn a LIMIT after an unpartitioned window into a top-k before it.

    A window with an ORDER BY and no PARTITION BY gives each row a value
    that depends on the rows before it only, so the first k rows in that
    order get the same values whether or not the rest are there. The
    Limit becomes a Sort with a fetch under the Window. lead reads the
    row after, so it keeps the Limit.
    """

    name = "should_push_down_limit_through_window"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        limit = next((node for node in graph.nodes if isinstance(node, Limit)), None)
        if limit is None or graph.root.node_id != limit.node_id:
            return None
        window = graph.node(limit.inputs[0].source.node_id)
        if not isinstance(window, Window):
            return None
        orders = {call.order for call in window.calls}
        if len(orders) != 1 or not next(iter(orders)) or any(
                call.partition or call.function == "lead" for call in window.calls):
            return None
        below = window.inputs[0].source
        top = Sort(node_id="top", inputs=input_ports((below,)),
                   keys=tuple(next(iter(orders))),
                   columns=graph.node(below.node_id).outputs[0].schema,
                   fetch=limit.count)
        nodes = []
        for node in graph.nodes:
            if node is limit:
                continue
            if node is window:
                nodes.append(top)
                node = replace(node, inputs=input_ports((PortRef("top", "rows"),)))
            nodes.append(node)
        return PhysicalGraph(tuple(nodes), limit.inputs[0].source)


class ShouldRetainKv(PhysicalRule):
    """Keep the KV that later operators read resident, within the page cap.

    The schedule (quail.planner.graph.retention.schedule) records, at each
    execution boundary, which aliases a later stage anchors on and how
    likely each document is to survive to that use. The executor's
    retention pool keeps their KV by that priority, up to the page cap.

    - A filter sequence keeps its survivors' KV (keep_kv) when a later join
      anchors on its table, unless the sequence streams into that join in
      one pipeline on one GPU.
    - A join keeps its anchor's KV (keep_kv) when a later group
      anchors on the same table.
    - A sequence writes KV pages only when a later stage, the next operator
      of its pipeline, or the retention pool reads them.

    The rule needs the logical plan on the context and fires only on a
    plan without a ``retention`` setting, so a later pass keeps the
    first schedule. It puts the schedule, the retention coefficients,
    and the page cap in the ``retention`` setting.
    """

    name = "should_retain_kv"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None or context.logical_plan is None \
                or "retention" in context.settings:
            return None
        logical = context.logical_plan
        statistics = cached_statistics(
            logical, context.memo, model=context.model, device=context.device,
            document_tokens=context.document_tokens,
            pair_fractions=context.pair_fractions,
            scan_fractions=context.scan_fractions, context=context)
        joins = [node for node in graph.nodes if isinstance(node, AiJoin)]
        grouped = []
        for node in joins:
            if (not grouped or grouped[-1][-1].anchor != node.anchor
                    or grouped[-1][-1].semantics != "full"
                    or node.semantics != "full"):
                grouped.append([])
            grouped[-1].append(node)
        groups = [[(statistics.specs[node.written_position], node.anchor)
                   for node in group] for group in grouped]
        schedule = retention.schedule(
            [predicate for group in groups for predicate in group],
            expected_rows_after_filters(logical, statistics),
            [group[0].node_id for group in grouped])
        before, after = schedule["before"], schedule["after"]
        schedule["before"] = {
            node.node_id: before[group[0].node_id]
            for group in grouped for node in group}
        schedule["after"] = {
            node.node_id: (after[group[0].node_id] if node is group[-1]
                           else before[group[0].node_id])
            for group in grouped for node in group}
        paths = build_pipelines(graph)
        streamed = {
            node.alias for node in graph.nodes if isinstance(node, AiFilter)
            and any(isinstance(member, AiJoin)
                    for member in paths[node.node_id])}
        schedule["initial"] = {
            alias: use for alias, use in schedule["initial"].items()
            if alias not in streamed}
        schedule.update(**coefficients(context.model, context.device),
                        cap_pages=statistics.cap_pages)
        context.settings["retention"] = schedule
        return plan_kv_retention(graph)


class ChooseClassifyImplementation(PhysicalRule):
    """Choose each classification's implementation by estimated time.

    Fires for every AiClassify whose spec has no scoring rule yet and
    leaves chosen ones alone, so a later pass keeps the first choice.
    The candidates are letters, when the prompt has a one-token letter
    per label; trie_tree, under the tree attention path; and
    trie_decode, when no label starts another.
    The documents count as resident, with their KV in the arena, when
    the classification continues its table's filter sequence on one GPU
    before any join, or follows a classification with the same prompt
    head. A classification of joined rows always uses letters and
    arrives chosen.

    Graph refinement updates the plan cost from the chosen specs.
    The rule needs the logical plan on the context for the calls' prompts.
    """

    name = "choose_classify_implementation"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        pending = [node for node in graph.nodes
                   if isinstance(node, AiClassify) and node.spec is not None
                   and not node.spec.scoring]
        if not pending:
            return None
        if context.logical_plan is None:
            raise ValueError(
                "choose_classify_implementation needs the logical plan "
                "on the planning context "
                "to score a classification without a scoring rule")

        logical = context.logical_plan
        calls = {node.name: node.call for node in logical.operators().classifies}
        paths = build_pipelines(graph)
        chosen = {}
        for node in pending:
            table = classify_scoring(context, node.spec.anchor)
            resident = paths[node.node_id][0] is not node
            chosen[node.node_id] = replace(node, spec=table.choose_scoring(
                node.spec, calls[node.spec.name], resident))
            if "estimated_fresh_tokens" in context.settings:
                work = table.estimate_spec(chosen[node.node_id].spec, resident).work
                context.settings["estimated_fresh_tokens"] += work.tokens
                context.settings["estimated_attention_pairs"] += work.pairs
        return PhysicalGraph(
            tuple(chosen.get(node.node_id, node) for node in graph.nodes),
            graph.root)


def page_aligned_shared_tokens(store) -> int:
    """Tokens a prefix tree over the store saves in whole KV pages."""
    return page_tree(store).shared_tokens


def filter_attention(model, device, store, lengths, question_tokens: int) -> str:
    """The attention path for a filter whose documents share prefixes.

    Documents borrowing the same pages of one parent are that node's
    readers; each reads with its own tokens plus the question. The
    path that wins on the most borrowed tokens wins the filter.
    """
    tree = page_tree(store)
    votes = {"tree": 0, "unified": 0}
    for shared, readers, rows in zip(*tree.reader_groups(lengths)):
        path = choose_attention_path(
            model, device, readers=readers,
            reader_rows=rows / readers + question_tokens, node_tokens=shared)
        votes[path] += readers * shared
    return "tree" if votes["tree"] > votes["unified"] else "unified"


def sharing_pays(model, device, *, shared_tokens: int, total_tokens: int,
                 writes_pages: bool) -> bool:
    """Whether borrowing shared prefixes saves more than it costs.

    Borrowing saves the forward pass over the shared tokens. For a
    filter that did not write KV pages, it costs writing them for every
    token of the corpus.
    """
    if shared_tokens <= 0:
        return False
    saved = shared_tokens * coefficients(model, device)["linear_seconds"]
    cost = (0.0 if writes_pages
            else total_tokens * model.kappa / device.hbm_bw)
    return saved > cost


class ShouldSharePrefixes(PhysicalRule):
    """Let documents borrow the KV pages of a document sharing their prefix.

    Fires for a filter whose documents share whole pages of prefix
    worth more forward-pass time than the page writes the filter takes
    on. The filter then writes pages even when it has one stage, since
    borrowed pages must exist. A join always writes its anchors' KV,
    so it fires for any join whose anchors share a whole page; a
    classification or extraction writes its documents' KV the same way.
    A score of
    one table is priced like a filter that writes no pages, since
    without sharing its documents follow one shared prompt head.
    """

    name = "should_share_prefixes"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        nodes = []
        changed = False
        summaries = {
            alias: summarize_alias(lengths, context.model.sliding_window,
                                   memo=context.memo)
            for alias, lengths in context.document_tokens.items()
        }
        shared_by_alias = {}
        for node in graph.nodes:
            if (not isinstance(node, (AiFilter, AiJoin, AiClassify, AiExtract, AiScore))
                    or (isinstance(node, AiInference) and node.spec is None)
                    or node.share_prefixes
                    or (isinstance(node, AiScore) and len(node.spec.aliases) != 1)):
                nodes.append(node)
                continue
            alias = (node.spec.aliases[0] if isinstance(node, AiScore)
                     else node.document_aliases[0])
            lengths = context.document_tokens.get(alias)
            store = None if lengths is None else lengths.store
            if store is None:
                nodes.append(node)
                continue
            if alias not in shared_by_alias:
                shared_by_alias[alias] = page_aligned_shared_tokens(store)
            shared = shared_by_alias[alias]
            writes_pages = (node.arena_writes if isinstance(node, AiFilter)
                            else not isinstance(node, AiScore))
            if not sharing_pays(
                    context.model, context.device, shared_tokens=shared,
                    total_tokens=summaries[alias].total, writes_pages=writes_pages):
                nodes.append(node)
                continue
            # the plan's pricer prices a sharing node over its fresh tokens
            if isinstance(node, AiFilter):
                node = replace(node, share_prefixes=True, arena_writes=True)
            elif isinstance(node, AiJoin):
                node = replace(node, share_prefixes=True)
            else:
                node = replace(node, spec=replace(node.spec, share_prefixes=True))
            nodes.append(node)
            changed = True
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class ChooseAttentionAlgorithm(PhysicalRule):
    """Annotate each join with the attention path the cost model prefers.

    A join stage has every partner of an anchor reading the anchor's
    KV: under tree attention those reads are stacked into one, under
    unified attention each partner reads the anchor itself. The choice
    is by roofline (choose_attention_path). A diffusion model and a
    model under 2B dense parameters stay unified
    (tree_attention_allowed).

    A filter sharing prefixes is annotated from its tree: the
    documents borrowing one parent's pages are that node's readers
    (filter_attention). A filter that does not share has one reader
    per node and stays on the unified path.

    The choice is recomputed on every pass, so a plan first made on
    estimated lengths gets the path its exact tokens call for.
    """

    name = "choose_attention_algorithm"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        model, device = context.model, context.device
        forced = context.attention
        if forced is not None and forced not in ("tree", "unified"):
            raise ValueError(
                f"attention must be 'tree' or 'unified', got {forced!r}")
        tree_available = tree_attention_allowed(model)
        nodes = []
        changed = False
        for node in graph.nodes:
            path = None
            if isinstance(node, AiFilter):
                lengths = context.document_tokens.get(node.alias)
                store = None if lengths is None else lengths.store
                path = "unified"
                if tree_available and node.share_prefixes and store is not None:
                    path = filter_attention(
                        model, device, store, lengths,
                        node.question_tokens)
            elif isinstance(node, AiJoin):
                lengths = context.document_tokens.get(node.anchor, ())
                path = "unified"
                if tree_available:
                    stage = node
                    # the partners each live anchor reads, on average
                    readers = (stage.expected_rows / stage.anchor_documents
                               if stage.anchor_documents else 1.0)
                    # a partner's rows: its document and the question tail
                    rows = stage.pair_tail_tokens + sum(
                        _mean(context.document_tokens.get(partner, ()))
                        for partner in stage.partners)
                    path = choose_attention_path(
                        model, device, readers=readers, reader_rows=rows,
                        node_tokens=_mean(lengths))
            if path is not None and forced is not None:
                path = forced
            if path is not None and path != node.attention:
                node = replace(node, attention=path)
                changed = True
            nodes.append(node)
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)
