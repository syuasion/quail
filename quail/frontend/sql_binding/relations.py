"""Bind FROM, JOIN, and WHERE clauses."""

from dataclasses import dataclass, field

from sqlglot import exp

from quail.frontend.sql_binding.expressions import (
    JOIN_OPTION_KEYS,
    SCORE_OPTION_KEYS,
    ExpressionBinder,
    _conjuncts,
    _is_ai_score_comparison,
    _is_call,
    _label_test,
)
from quail.frontend.sql_binding.predicates import (
    COLUMN_NODES,
    PLAIN_COMPARISONS,
    ai_aliases,
    bind_plain_test,
    extract_test,
    parse_plain_test,
    unwrap_hint,
)
from quail.logical import (
    Alias,
    ColumnComparison,
    CompileError,
    Equality,
    FilterPredicate,
    JoinSpec,
)
from quail.logical.expressions import JOIN_COMPARISONS


def _parse_column_condition(b, term, joined_alias: str):
    """Parse one ordinary ON condition: two columns compared.

    Returns:
        An Equality for ``=``, otherwise a ColumnComparison.
    """
    comparison = PLAIN_COMPARISONS.get(type(term))
    if (comparison not in ("=", *JOIN_COMPARISONS) or not all(
            isinstance(side, COLUMN_NODES)
            for side in (term.this, term.expression))):
        raise CompileError(
            f"ON supports comparisons of two columns and AI predicates, "
            f"got {term.sql()}"
        )
    left, right = b.resolve_value(term.this), b.resolve_value(term.expression)
    condition = (Equality(left, right) if comparison == "="
                 else ColumnComparison(left, comparison, right))
    if joined_alias not in condition.aliases() or len(set(condition.aliases())) != 2:
        raise CompileError(
            f"JOIN {joined_alias} ON {term.sql()} must relate "
            f"{joined_alias!r} to a table already in the query"
        )
    return condition


def _from_clause(select):
    return select.args.get("from_")


@dataclass
class JoinBinder:
    """Bind table references and assign predicates to joins."""

    binder: ExpressionBinder
    aliases: list[str] = field(default_factory=list)
    # Equality and ColumnComparison conditions by the alias they join
    conditions: dict[str, list] = field(default_factory=dict)
    claimed: set[str] = field(default_factory=set)

    def bind_tables(self, tree: exp.Select) -> None:
        """Bind FROM and JOIN clauses in table order."""
        b = self.binder
        from_ = _from_clause(tree)
        if from_ is None or not isinstance(from_.this, exp.Table):
            raise CompileError("FROM must name one registered provider")
        b.add_table(from_.this)

        on_preds = []
        for join in tree.args.get("joins") or []:
            if join.side or (join.kind and join.kind.upper() not in ("INNER", "CROSS")):
                raise CompileError(
                    f"only plain JOIN is supported, got "
                    f"{join.side or ''} {join.kind or ''} JOIN".strip()
                )
            if not isinstance(join.this, exp.Table):
                raise CompileError("JOIN must name one registered provider")
            alias = b.add_table(join.this)
            self.aliases.append(alias)
            on = join.args.get("on")
            if on is None:
                continue  # a bare/cross-joined table: some join
                #             predicate must cover it (checked below)
            # ON: column conditions choose the pairs; at most one AI
            # predicate
            column_conditions, predicates = [], []
            for term in _conjuncts(on):
                if _is_call(term, "AI_FILTER"):
                    predicates.append(
                        b.bind_ai_filter(term, JOIN_OPTION_KEYS, join=True)
                    )
                elif _is_ai_score_comparison(term):
                    predicates.append(
                        b.bind_ai_score(term, SCORE_OPTION_KEYS, join=True)
                    )
                else:
                    if any(_is_call(call, "AI_SCORE") for call in term.walk()):
                        raise CompileError(
                            "AI.SCORE must be compared with <, <=, >, or >="
                        )
                    test = None if _has_ai_call(term) else parse_plain_test(
                        term, b.dialect)
                    if test is not None:
                        plain = bind_plain_test(b, test)
                        b.regular_predicates.setdefault(plain.column.alias, []).append(
                            plain
                        )
                        continue
                    column_conditions.append(
                        _parse_column_condition(b, term, alias))
            if len(predicates) > 1:
                raise CompileError(
                    f"JOIN {alias} has {len(predicates)} AI predicates in "
                    f"ON; ask one question per JOIN"
                )
            if column_conditions:
                if not any(isinstance(item, Equality)
                           for item in column_conditions):
                    raise CompileError(
                        f"JOIN {alias} ON compares columns without an equality "
                        f"beside the comparison")
                self.conditions[alias] = column_conditions
            on_preds.extend(predicates)

        for predicate in on_preds:
            self.add_predicate(*predicate)

    def add_predicate(self, predicate, options, aliases) -> None:
        """Assign a join predicate to the tables it introduces."""
        b = self.binder
        joinable = {b.tables[0][0], *self.aliases}
        outside = [a for a in aliases if a not in joinable]
        if outside:
            raise CompileError(
                f"the join prompt references {outside}, which are not "
                f"the FROM table or JOINed tables of this query "
                f"({sorted(joinable)})"
            )
        anchor = options.get("anchor")
        if anchor is not None and anchor not in aliases:
            raise CompileError(
                f"anchor {anchor!r} is not a table of this join ({aliases})"
            )
        # each spec carries the joined tables its prompt references
        # that no earlier spec carried, so the logical builder adds
        # every table to the join tree exactly once
        news = tuple(a for a in self.aliases if a in aliases and a not in self.claimed)
        self.claimed.update(news)
        on = tuple(condition for a in news for condition in self.conditions.pop(a, ()))
        if on and len(aliases) != 2:
            raise CompileError(
                f"join conditions are supported on two-table AI "
                f"predicates; this one names {aliases}"
            )
        for condition in on:
            if not set(condition.aliases()) <= set(aliases):
                raise CompileError(
                    f"join condition {condition} names a table the AI "
                    f"predicate over {aliases} does not"
                )
        b.joins.append(
            JoinSpec(
                aliases=news,
                predicate=predicate,
                semantics="full",
                selectivity=options.get("selectivity"),
                anchor=anchor,
                on=tuple(item for item in on if isinstance(item, Equality)),
                comparisons=tuple(item for item in on
                                  if isinstance(item, ColumnComparison)),
            )
        )


def _has_ai_call(term) -> bool:
    """Return whether a term asks the model, counting AI.EXTRACT only bare."""
    return any(
        _is_call(node, "AI_FILTER")
        or _is_call(node, "AI_SCORE")
        or isinstance(node, (exp.AIClassify, exp.Exists))
        or (_is_call(node, "AI_EXTRACT") and not isinstance(node.parent, exp.Dot))
        for node in term.walk()
    )


def _name_select_labels(b: ExpressionBinder, term, selected: dict):
    """Replace a bare column named after an AI.CLASSIFY in SELECT with the call.

    Raises:
        CompileError: The name is also a column of a table in scope.
    """
    for column in list(term.find_all(exp.Column)):
        if column.table or column.name not in selected:
            continue
        if not isinstance(selected[column.name], exp.AIClassify):
            continue
        if any(column.name in b.catalog.get(provider).columns
               for _, provider in b.tables):
            raise CompileError(
                f"{column.name!r} is both a column of a table in scope and "
                f"an AS name in SELECT; qualify or rename it")
        if column is term:
            return selected[column.name].copy()
        column.replace(selected[column.name].copy())
    return term


def _bind_plain_term(b: ExpressionBinder, term, selected: dict,
                     selectivity: float | None) -> bool:
    """Bind a test of a source column or of an AI.EXTRACT field.

    Args:
        b: The binder.
        term: The term.
        selected: The AI calls in SELECT, by AS name.
        selectivity: The term's SELECTIVITY hint, or None.

    Returns:
        Whether the term was a plain test; False leaves an AI term to
        its binder.

    Raises:
        CompileError: A test of a source column carries a hint; the
            scan applies it before any model call, so it has no cost.
    """
    if _has_ai_call(term):
        return False
    test = parse_plain_test(term, b.dialect)
    if test is None:
        return False
    found = extract_test(b, test, selected)
    if found is not None:
        alias, predicate = found
        b.extract_tests.setdefault(alias, []).append((predicate, selectivity))
    else:
        if selectivity is not None:
            raise CompileError(
                f"HINT estimates a test of a label or an answer; "
                f"{term.sql()} tests a source column, applied at the scan")
        plain = bind_plain_test(b, test)
        b.regular_predicates.setdefault(plain.column.alias, []).append(plain)
    return True


def bind_where(b: ExpressionBinder, tree: exp.Select, joins: JoinBinder) -> None:
    """Bind WHERE conditions to scans and joins."""
    where = tree.args.get("where")
    selected = ai_aliases(tree)
    for term in _conjuncts(where.this) if where else []:
        term, hint = unwrap_hint(b, term)
        term = _name_select_labels(b, term, selected)
        if _bind_plain_term(b, term, selected, hint):
            continue
        anti = False
        node = term
        if isinstance(node, exp.Not):
            node, anti = node.this, True
        if isinstance(node, exp.Exists):
            _compile_exists(b, node, anti)
            continue
        if anti:
            raise CompileError(
                f"NOT is only supported as NOT EXISTS, got NOT {node.sql()}"
            )
        tested = _label_test(node)
        if tested is not None:
            classify_node, items = tested
            call, options, aliases = b.bind_ai_classify(classify_node)
            if len(aliases) != 1:
                raise CompileError(
                    f"a filter on a label tests a one-document "
                    f"classification; this one classifies pairs of {aliases}"
                )
            accepted = []
            for item in items:
                if not (isinstance(item, exp.Literal) and item.is_string):
                    raise CompileError("AI.CLASSIFY is compared with string literals")
                accepted.append(str(item.this))
            if len(set(accepted)) != len(accepted):
                raise CompileError("a filter on a label lists a label twice")
            b.label_tests.setdefault(aliases[0], []).append(
                (call, tuple(accepted),
                 options.get("selectivity") if hint is None else hint)
            )
            continue
        if any(isinstance(call, exp.AIClassify) for call in term.walk()):
            raise CompileError(
                "AI.CLASSIFY in WHERE is tested with = 'label' or IN ('label', ...)"
            )
        if any(_is_call(call, "AI_EXTRACT") for call in term.walk()):
            raise CompileError(
                "an AI.EXTRACT call is not a predicate; test one of its "
                "fields, such as AI.EXTRACT(...).end_date IS NOT NULL, or "
                "its AS name from SELECT")
        if _is_ai_score_comparison(term):
            predicate, options, aliases = b.bind_ai_score(term, SCORE_OPTION_KEYS)
        else:
            if any(_is_call(call, "AI_SCORE") for call in term.walk()):
                raise CompileError("AI.SCORE must be compared with <, <=, >, or >=")
            predicate, options, aliases = b.bind_ai_filter(term, JOIN_OPTION_KEYS)
        if len(aliases) == 1:
            if "anchor" in options:
                raise CompileError(
                    "anchor is a join option; a one-provider "
                    "AI_FILTER takes only selectivity"
                )
            b.filters.setdefault(aliases[0], []).append(
                FilterPredicate(
                    expression=predicate, selectivity=options.get("selectivity")
                )
            )
            continue
        # a multi-provider WHERE predicate is a join predicate,
        # BigQuery style: tables cross-joined in FROM, filtered here
        joins.add_predicate(predicate, options, aliases)


def check_join_coverage(
    b: ExpressionBinder,
    joined_aliases: list,
    projected_scores: tuple[Alias, ...] = (),
) -> None:
    """Check that the join predicates cover and connect every JOINed table.

    Every JOINed table must appear in a join predicate, and all
    predicates must form one connected graph with the FROM table.
    """
    if not joined_aliases:
        return
    preds = [{r.alias for r in j.prompt.args} for j in b.joins if j.semantics == "full"]
    preds.extend(
        set(score.expression.aliases())
        for score in projected_scores
        if len(score.expression.aliases()) > 1
    )
    uncovered = [a for a in joined_aliases if not any(a in p for p in preds)]
    if uncovered:
        raise CompileError(
            f"JOINed tables {uncovered} appear in no join predicate: "
            f"every JOINed table must appear in at least one "
            f"AI predicate on a JOIN's ON "
            f"or as a WHERE term"
        )
    root = b.tables[0][0]
    reached = {root}
    grew = True
    while grew:
        grew = False
        for p in preds:
            if p & reached and not p <= reached:
                reached |= p
                grew = True
    disconnected = sorted(a for a in joined_aliases if a not in reached)
    if disconnected:
        raise CompileError(
            f"join predicates do not connect {disconnected} to the "
            f"FROM table {root!r}: the predicates' tables must form "
            f"one connected graph with it"
        )


def _compile_exists(b: ExpressionBinder, node: exp.Exists, anti: bool) -> None:
    inner = node.this
    if not isinstance(inner, exp.Select):
        raise CompileError(
            "EXISTS must wrap SELECT 1 FROM provider WHERE AI_FILTER(...)"
        )
    inner_from = _from_clause(inner)
    if inner_from is None or not isinstance(inner_from.this, exp.Table):
        raise CompileError("the EXISTS subquery must scan one registered provider")
    if (
        inner.args.get("joins")
        or inner.args.get("group")
        or len(inner.expressions) != 1
    ):
        raise CompileError(
            "the EXISTS subquery must be exactly "
            "SELECT 1 FROM provider WHERE AI_FILTER(...)"
        )
    alias = b.add_table(inner_from.this)
    where = inner.args.get("where")
    terms = _conjuncts(where.this) if where else []
    if len(terms) != 1:
        raise CompileError("the EXISTS subquery takes exactly one AI_FILTER predicate")
    predicate, options, aliases = b.bind_ai_filter(
        terms[0], JOIN_OPTION_KEYS, join=True
    )
    if len(aliases) != 2 or alias not in aliases:
        raise CompileError(
            "the EXISTS predicate must reference the inner provider "
            "and exactly one outer provider"
        )
    outer = next(a for a in aliases if a != alias)
    anchor = options.get("anchor")
    if anchor is not None and anchor != outer:
        raise CompileError(
            f"exists/anti always anchor on the outer table {outer!r} "
            f"- the gate applies to its documents - got anchor "
            f"{anchor!r}"
        )
    b.joins.append(
        JoinSpec(
            aliases=(alias,),
            predicate=predicate,
            semantics="anti" if anti else "exists",
            selectivity=options.get("selectivity"),
            anchor=outer,
        )
    )
