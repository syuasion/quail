"""Python builder that mirrors AI SQL construct for construct."""

from collections.abc import Mapping
from typing import Optional

from quail.catalog import Catalog
from quail.frontend.expressions import (
    AggSpec,
    AndSpec,
    ColSpec,
    EqualsSpec,
    HavingSpec,
    PredicateSpec,
    PromptSpec,
    SortSpec,
    WindowSpec,
    col,
)
from quail.frontend.label_tables import read_label_table
from quail.frontend.prompts import PromptBinder
from quail.frontend.scope import Scope
from quail.logical import (
    AggregateCall,
    Aggregation,
    Alias,
    ColumnComparison,
    ColumnRef,
    CompileError,
    Equality,
    ExtractStruct,
    FilterPredicate,
    HavingTest,
    JoinSpec,
    LogicalPlan,
    LogicalPlanBuilder,
    ModelCall,
    RegularPredicate,
    SortKey,
    WindowCall,
)
from quail.logical.expressions import (
    JOIN_COMPARISONS,
    SPAN_SUFFIX,
    aliased_call,
    column_expression,
    column_name,
    validate_task_description,
)


class Query:
    def __init__(self, catalog: Catalog, provider: str,
                 turn: tuple[str, str] = ("", ""), layout: str = "ai-if",
                 prompts: PromptBinder | None = None):
        self._catalog = catalog
        self._prompts = prompts or PromptBinder(turn=turn, layout=layout)
        self._scope_ = Scope(catalog)
        self._scope_.add_table(provider, provider)
        self._tables = self._scope_.tables      # (alias, provider)
        self._document_columns = self._scope_.document_columns
        self._filters = {}
        self._label_filters = {}     # alias -> [(name, labels, selectivity)]
        self._joins = []
        # (new aliases, (equalities, comparisons), applies)
        self._pending_join = None
        #                              awaiting the AI predicate over
        #                              its pairs
        self._applies = {}           # alias -> [(name, kind, ids, refs)]
        self._functions = {}         # name -> the Python function
        self._labels = {}            # name -> Alias of an AI.CLASSIFY call
        self._scores = {}            # name -> Alias of an AI.SCORE call
        # name -> an AI.EXTRACT's ExtractStruct and its fields' Aliases
        self._extracts = {}
        self._regular_predicates = {}  # alias -> [RegularPredicate]
        # alias -> [(RegularPredicate on an answer or span, selectivity)]
        self._extract_filters = {}
        self._limit = None
        self._order = []             # (column or name, descending, nulls_first)
        self._offset = 0
        self._distinct = False
        self._group = []             # key columns or names
        self._aggs = {}              # name -> AggSpec
        self._having = []            # HavingSpec
        self._windows = {}           # name -> WindowSpec

    # ---- scope -------------------------------------------------------

    def alias(self, a: str) -> "Query":
        if (self._joins or self._filters or self._label_filters
                or self._regular_predicates or self._scores or self._extracts
                or self._applies or self._document_columns or self._labels
                or len(self._tables) != 1):
            raise CompileError("alias() must come right after docs()")
        self._tables[0] = (a, self._tables[0][1])
        return self

    def _scope(self) -> dict:
        return dict(self._tables)

    def _resolve_value(self, c: ColSpec, *, named: bool = False):
        """Resolve a col() to a value expression over a source column.

        Args:
            c: The column as written.
            named: Whether the column may carry a result name from
                ``named()``; only a selected, sorted, or grouped column
                may.

        Raises:
            CompileError: The column is named where it is read as an
                input, or its name is a table alias or an earlier name.
        """
        if c.name is not None and not named:
            raise CompileError(
                f"{c.name!r} names a result column; a prompt argument, a "
                f"join key, an apply() column, and a where() test read a "
                f"column, not a name")
        if c.name is not None:
            self._claim_name(c.name)
        return self._scope_.resolve_value(c.column, c.alias, path=c.path,
                                          function=c.function, name=c.name)

    def _note_document_column(self, column) -> None:
        self._scope_.note_document_column(column)

    def _bind(self, p: PromptSpec, join: bool = False):
        refs = tuple(self._resolve_value(c) for c in p.cols)
        bound = self._prompts.bind("join" if join else "filter", p.template, refs)
        for r in refs:
            self._note_document_column(r)
        aliases = []
        for r in refs:
            if r.alias not in aliases:
                aliases.append(r.alias)
        return bound, aliases

    # ---- the query surface --------------------------------------------

    def ai_filter(self, p: PromptSpec,
                  selectivity: Optional[float] = None) -> "Query":
        """Ask the prompt of every document, or of every joined pair.

        A prompt over one table filters its documents. A prompt over
        the tables of the preceding join() is the AI predicate asked
        of that join's pairs.
        """
        if self._pending_join is not None:
            return self._ai_join_pairs(p, selectivity)
        bound, aliases = self._bind(p)
        if len(aliases) != 1:
            raise CompileError(
                f"ai_filter must reference exactly one provider, got "
                f"{aliases}; a two-provider predicate follows join() or "
                f"is ai_join")
        self._filters.setdefault(aliases[0], []).append(
            FilterPredicate(ModelCall(bound), selectivity=selectivity))
        return self

    ai_if = ai_filter

    def ai_score(self, p: PromptSpec, *, name: str) -> "Query":
        """Add a named AI.SCORE column: the model's belief that a prompt is TRUE.

        Args:
            p: A prompt over one or two columns.
            name: The score column's name, used in select(), order_by(),
                group_by() arguments, and agg() arguments.
        """
        if not name or "." in name:
            raise CompileError(
                f"a score needs a column name without a dot, got {name!r}")
        self._claim_name(name)
        refs = tuple(self._resolve_value(c) for c in p.cols)
        bound = self._prompts.bind("score", p.template, refs)
        for ref in refs:
            self._note_document_column(ref)
        call = ModelCall(bound, "score")
        call.validate()
        self._scores[name] = Alias(call, name)
        return self

    def ai_classify(self, p: PromptSpec, labels, *, name: str,
                    descriptions=None,
                    task_description: str = "",
                    probabilities: bool = False) -> "Query":
        """Give each document one of the labels, in a named result column.

        Use select() to return the column and label_in() to keep documents
        with chosen labels. The planner chooses how to score the labels.

        Args:
            p: The prompt over one document column, or over the anchor and
                partner columns of a join's rows.
            labels: Label strings, (label, description) pairs, or the name
                of a registered label table.
            name: Result column name, without a dot.
            descriptions: Optional descriptions in the same order as labels.
                Leave empty when labels holds (label, description) pairs.
            task_description: Additional instructions, up to 50 words.
            probabilities: Whether to also return a map of label probabilities
                in the column named name + "_probabilities".

        Returns:
            This query builder, with the classification added.

        Raises:
            CompileError: The name, labels, descriptions, or prompt are
                invalid, or a pending join has no AI predicate yet.
        """
        if self._pending_join is not None:
            raise CompileError(
                "join() is waiting for the ai_filter over its pairs; "
                "classify before joining")
        if not name or "." in name:
            raise CompileError(
                f"a classification needs a column name without a dot, "
                f"got {name!r}")
        self._claim_name(name)
        if isinstance(labels, str):
            labels, table_descriptions = read_label_table(self._catalog, labels)
            descriptions = descriptions or table_descriptions
        else:
            labels = tuple(labels)
            if labels and all(isinstance(label, (tuple, list))
                              for label in labels):
                if descriptions:
                    raise CompileError(
                        "give descriptions in the pairs or separately, not both")
                descriptions = tuple(description for _, description in labels)
                labels = tuple(label for label, _ in labels)
        descriptions = tuple(descriptions or ())
        validate_task_description(task_description)
        refs = tuple(self._resolve_value(c) for c in p.cols)
        bound = self._prompts.bind(
            "label", p.template, refs, labels=labels, descriptions=descriptions,
            task_description=task_description)
        for ref in refs:
            self._note_document_column(ref)
        call = ModelCall(bound, "label", labels, descriptions,
                         probabilities=probabilities)
        call.validate()
        self._labels[name] = Alias(call, name)
        return self

    def ai_extract(self, c, questions, *, name: str) -> "Query":
        """Copy the words that answer each question from each document.

        ``questions`` maps field names to questions. The result column
        ``name`` is a struct with one field per question, holding the
        copied words, or null when the document does not answer it. The
        column ``name`` plus "_span" is a struct of the answers'
        character spans in the document, each a (start, end) pair. Use
        select() to return the columns.

        For example, ``ai_extract("c.text", {"end": "When does the lease
        end?", "signer": "Who signed the lease?"}, name="lease")`` adds
        ``lease``, a struct of the fields ``end`` and ``signer``, and
        ``lease_span``. Each field is its own extraction; the fields of
        one document read the document's KV, written once.

        Args:
            c: The document column, as "alias.column" or a col().
            questions: A mapping from field names to the questions their
                answers are copied for.
            name: Result column name, without a dot.

        Returns:
            This query builder, with the extraction added.

        Raises:
            CompileError: The name, column, fields, or a question is
                invalid, or a pending join has no AI predicate yet.
        """
        if self._pending_join is not None:
            raise CompileError(
                "join() is waiting for the ai_filter over its pairs; "
                "extract before joining")
        if not isinstance(questions, Mapping):
            raise CompileError(
                "ai_extract takes a mapping from field names to questions, "
                "such as {'end': 'When does the lease end?'}")
        if not name or "." in name:
            raise CompileError(
                f"an extraction needs a column name without a dot, "
                f"got {name!r}")
        self._claim_name(name)
        self._claim_name(name + SPAN_SUFFIX)
        ref = self._resolve_value(col(c) if isinstance(c, str) else c)
        struct = ExtractStruct(name, tuple(questions))
        struct.validate()
        members = []
        for field in struct.fields:
            bound = self._prompts.bind("extract", questions[field], (ref,))
            call = ModelCall(bound, "extract", question=questions[field])
            call.validate()
            members.append(Alias(call, struct.member(field)))
        self._note_document_column(ref)
        self._extracts[name] = (struct, tuple(members))
        return self

    def _claim_name(self, name: str) -> None:
        """Check that no earlier call, table alias, or source column uses the name."""
        if (name in self._labels or name in self._scores
                or name in self._extracts
                or any(name == n + SPAN_SUFFIX for n in self._extracts)
                or name in self._scope()):
            raise CompileError(f"the name {name!r} is already used")
        for alias, provider in self._tables:
            if name in self._catalog.get(provider).columns:
                raise CompileError(
                    f"the name {name!r} is a column of {alias!r}; pick another")

    def label_in(self, name: str, labels,
                 selectivity: Optional[float] = None) -> "Query":
        """Keep documents whose label is one of the given labels.

        This uses the same membership check as SQL IN.

        Args:
            name: Column name from an earlier ai_classify() call.
            labels: Labels to keep.
            selectivity: Estimated fraction of documents kept. None uses 0.2.

        Returns:
            This query builder, with the condition added.

        Raises:
            CompileError: No classification has this name, the classification
                refers to document pairs, or labels contains duplicates.
        """
        if name not in self._labels:
            raise CompileError(f"no classification is named {name!r}")
        call = self._labels[name].expression
        if len(call.aliases()) != 1:
            raise CompileError(
                f"a filter on a label tests a one-document classification; "
                f"{name!r} classifies pairs of {call.aliases()}")
        (alias,) = call.aliases()
        labels = tuple(labels)
        RegularPredicate(ColumnRef(alias, self._scope()[alias], name), "in",
                         labels).validate()
        self._label_filters.setdefault(alias, []).append(
            (name, labels, selectivity))
        return self

    def join(self, other: "Query", on=None) -> "Query":
        """Join one table on ordinary column conditions.

        The next ai_filter() must name this table and one already in
        the query; the model then sees only the pairs the conditions
        allow. With on=None every pair is a candidate.

        Args:
            other: One docs() query, optionally filtered.
            on: ``col("c.url") == col("e.url")`` or a list of conditions.
                Beside at least one equality, a condition may compare
                two columns with ``<``, ``<=``, ``>``, ``>=``, or
                ``!=``; for example, ``col("r1.id") < col("r2.id")``
                keeps each unordered pair of a self-join once.
        """
        if self._pending_join is not None:
            raise CompileError(
                "join() is waiting for the ai_filter over its pairs; "
                "add that predicate before joining another table")
        new_aliases = self._absorb([other])
        conditions = [] if on is None else (
            [on] if isinstance(on, (EqualsSpec, PredicateSpec)) else list(on))
        resolved = []
        for condition in conditions:
            if isinstance(condition, EqualsSpec):
                resolved.append(Equality(self._resolve_value(condition.left),
                                         self._resolve_value(condition.right)))
            elif (isinstance(condition, PredicateSpec)
                    and isinstance(condition.value, ColSpec)
                    and condition.comparison in JOIN_COMPARISONS):
                resolved.append(ColumnComparison(
                    self._resolve_value(condition.col), condition.comparison,
                    self._resolve_value(condition.value)))
            else:
                raise CompileError(
                    "join(on=...) takes conditions that compare two "
                    "columns, such as col(...) == col(...)")
            sides = set(resolved[-1].aliases())
            if new_aliases[0] not in sides or len(sides) != 2:
                raise CompileError(
                    f"join condition {resolved[-1]} must relate the joined "
                    f"table {new_aliases[0]!r} to a table already in the "
                    f"query")
        equalities = tuple(item for item in resolved if isinstance(item, Equality))
        comparisons = tuple(item for item in resolved
                            if isinstance(item, ColumnComparison))
        if comparisons and not equalities:
            raise CompileError(
                f"join comparison {comparisons[0]} needs an equality "
                f"condition beside it")
        self._pending_join = (new_aliases, (equalities, comparisons), [])
        return self

    def _ai_join_pairs(self, p: PromptSpec, selectivity):
        new_aliases, conditions, applies = self._pending_join
        bound, aliases = self._bind(p, join=True)
        if len(aliases) != 2 or new_aliases[0] not in aliases:
            raise CompileError(
                f"the predicate after join() must name the joined "
                f"table {new_aliases[0]!r} and one other table, got "
                f"{aliases}")
        for name, _kind, refs in applies:
            outside = sorted({ref.alias for ref in refs} - set(aliases))
            if outside:
                raise CompileError(
                    f"apply {name!r} reads tables {outside} that the join "
                    f"predicate over {aliases} does not join")
        self._pending_join = None
        equalities, comparisons = conditions
        self._joins.append(JoinSpec(aliases=tuple(new_aliases),
                                    predicate=ModelCall(bound),
                                    selectivity=selectivity,
                                    on=equalities, comparisons=comparisons,
                                    applies=tuple(applies)))
        return self

    def apply(self, fn, columns=(), *, name=None, ids=None,
              kind="per_batch") -> "Query":
        """Call a Python function between two operators.

        The function receives a dict of Arrow tables keyed by alias:
        each table holds the alias's row indices under the alias name
        plus the listed columns. After ``join()`` it returns the pairs
        the next AI predicate is asked about, as a table with both
        alias columns. Otherwise it works on one table and returns the
        ids to keep. A function never invents an id.

        Args:
            fn: The function. Registered on the session under name.
            columns: ``col(...)`` references the function reads.
            name: Registered name; the function's name by default.
            ids: "drop" (default, may leave ids out), "preserve"
                (returns every id), or "pairs" (after join()).
            kind: "per_batch" runs on each batch a streaming operator
                hands over; "barrier" runs once over every survivor.
        """
        if not callable(fn):
            raise CompileError("apply() needs a callable")
        name = name or getattr(fn, "__name__", None)
        if not name or name == "<lambda>":
            raise CompileError("apply() needs a name for a lambda")
        if kind not in ("per_batch", "barrier"):
            raise CompileError(
                f"apply kind must be per_batch or barrier, got {kind!r}")
        known = self._functions.get(name)
        if known is not None and known is not fn:
            raise CompileError(
                f"apply name {name!r} is already used by another function")
        columns = [columns] if isinstance(columns, ColSpec) else list(columns)
        refs = tuple(self._resolve_value(spec) for spec in columns)
        if self._pending_join is not None:
            if ids not in (None, "pairs"):
                raise CompileError(
                    "an apply() after join() returns the pairs to ask "
                    "about; ids must be 'pairs'")
            self._pending_join[2].append((name, kind, refs))
            self._functions[name] = fn
            return self
        if ids == "pairs":
            raise CompileError(
                "an apply() returning pairs follows join()")
        ids = ids or "drop"
        if ids not in ("preserve", "drop"):
            raise CompileError(
                f"apply ids must be preserve, drop, or pairs, got {ids!r}")
        owners = {ref.alias for ref in refs}
        if len(owners) != 1:
            raise CompileError(
                f"apply {name!r} must work on one table; its columns "
                f"name {sorted(owners) or 'none'}")
        (alias,) = owners
        self._applies.setdefault(alias, []).append((name, kind, ids, refs))
        self._functions[name] = fn
        return self

    def apply_table(self, fn, columns=(), *, name=None, ids=None) -> "Query":
        """Call a function once over every survivor: apply() as a barrier."""
        return self.apply(fn, columns, name=name, ids=ids, kind="barrier")

    @property
    def functions(self) -> dict:
        """The Python functions apply() calls, by registered name."""
        return dict(self._functions)

    def _absorb(self, others) -> list:
        """Bring other single-table queries into scope; returns aliases."""
        new_aliases = []
        for other in others:
            if not isinstance(other, Query) or other._joins \
                    or other._pending_join is not None \
                    or len(other._tables) != 1:
                raise CompileError(
                    "every joined side must be a single (optionally "
                    "filtered) docs(...) query")
            alias, provider = other._tables[0]
            if alias in self._scope():
                raise CompileError(f"duplicate table alias {alias!r}")
            self._claim_name(alias)
            for name in (*other._labels, *other._scores, *other._extracts):
                self._claim_name(name)
            self._labels.update(other._labels)
            self._scores.update(other._scores)
            self._extracts.update(other._extracts)
            self._scope_.add_table(alias, provider)
            for a, preds in other._filters.items():
                self._filters.setdefault(a, []).extend(preds)
            for a, tests in other._label_filters.items():
                self._label_filters.setdefault(a, []).extend(tests)
            for a, preds in other._regular_predicates.items():
                self._regular_predicates.setdefault(a, []).extend(preds)
            for a, tests in other._extract_filters.items():
                self._extract_filters.setdefault(a, []).extend(tests)
            for a, applies in other._applies.items():
                self._applies.setdefault(a, []).extend(applies)
            for name, fn in other._functions.items():
                if self._functions.get(name, fn) is not fn:
                    raise CompileError(
                        f"apply name {name!r} is used by two functions")
                self._functions[name] = fn
            for c in other._document_columns.values():
                self._note_document_column(c)
            new_aliases.append(alias)
        return new_aliases

    def ai_join(self, others, p: PromptSpec,
                selectivity: Optional[float] = None,
                anchor: Optional[str] = None,
                semantics: str = "full") -> "Query":
        """Join one or more tables with a single prompt over every tuple.

        Shorthand for join(other) followed by ai_filter(p) when every
        pair is a candidate.

        Args:
            others: One docs() query or a list of them.
            p: Prompt with one placeholder per table it references.
            selectivity: Fraction of tuples expected to pass.
            anchor: Table alias whose KV is kept across tuples.
            semantics: "full", "exists", or "anti".
        """
        if self._pending_join is not None:
            raise CompileError(
                "join() is waiting for the ai_filter over its pairs; "
                "add that predicate before ai_join")
        if semantics not in ("full", "exists", "anti"):
            raise CompileError(f"semantics must be full, exists, or "
                               f"anti, got {semantics!r}")
        others = [others] if isinstance(others, Query) else list(others)
        if semantics != "full" and len(others) != 1:
            raise CompileError(
                "an exists/anti gate takes exactly one inner table; "
                "use one ai_join call per gate")
        new_aliases = self._absorb(others)
        bound, aliases = self._bind(p, join=True)
        if semantics == "full":
            # each call's prompt must cover the tables that call
            # joins and touch at least one table already in the
            # query, so every join connects to the join graph
            missing = [a for a in new_aliases if a not in aliases]
            if missing:
                raise CompileError(
                    f"the join prompt must reference every table this "
                    f"call joins; {missing} joined but not referenced "
                    f"(got {aliases})")
            if all(a in new_aliases for a in aliases):
                raise CompileError(
                    f"the join prompt must reference at least one "
                    f"table already in the query, so this join "
                    f"connects to it; got only new tables {aliases}")
            if anchor is not None and anchor not in aliases:
                raise CompileError(
                    f"anchor {anchor!r} is not a table of this join "
                    f"({aliases})")
        else:
            inner = new_aliases[0]
            if len(aliases) != 2 or inner not in aliases:
                raise CompileError(
                    f"an exists/anti prompt must reference the inner "
                    f"table {inner!r} and exactly one outer table, "
                    f"got {aliases}")
            outer = next(a for a in aliases if a != inner)
            if anchor is not None and anchor != outer:
                raise CompileError(
                    f"exists/anti always anchor on the outer table "
                    f"{outer!r} - the gate applies to its documents - "
                    f"got anchor {anchor!r}")
            anchor = outer
        self._joins.append(JoinSpec(aliases=tuple(new_aliases),
                                    predicate=ModelCall(bound),
                                    semantics=semantics,
                                    selectivity=selectivity,
                                    anchor=anchor))
        return self

    def where(self, *tests) -> "Query":
        """Keep the rows that pass the tests of columns against literals.

        A test of a source column runs at the scan, before any model
        call. A test of a label from ai_classify() or of an answer from
        ai_extract() runs right after the call that computes it, so the
        later operators read fewer documents; the model runs once for
        the test and the returned column. Such a test takes a
        selectivity hint, ``col(...).like(...).hint(selectivity=0.4)``,
        the estimated fraction of documents it keeps; 0.2 without one.

        Args:
            tests: Comparisons of a ``col(...)`` with a literal, such as
                ``col("r.year") > 2020``, ``col(...).isin([...])``,
                ``col(...).is_null()``, ``col(...).is_not_null()``,
                ``col(...).like(...)``, ``col(...).ilike(...)``,
                ``col(...).contains(...)``, ``col(...).startswith(...)``,
                ``col(...).endswith(...)``, ``col(...).rlike(...)``, or
                ``~`` before a text test, combined with ``&``; several
                arguments are combined the same way.
                ``col("lease.end_date")`` names an extraction's field,
                ``col("lease_span.end_date").field("start")`` its
                span's start, and ``col(...).length()`` the number of
                characters.
        """
        for group in tests:
            if not isinstance(group, (PredicateSpec, AndSpec)):
                raise CompileError(
                    f"where() takes tests such as col('r.year') > 2020, "
                    f"combined with &, got {group!r}")
            for test in group.tests():
                self._where_test(test)
        return self

    filter = where

    def _where_test(self, test: PredicateSpec) -> None:
        """Route one test to the scan, a label filter, or an answer filter."""
        spec = test.col
        if spec.alias is None and spec.column in self._labels:
            if test.comparison not in ("=", "in"):
                raise CompileError(
                    f"a label is tested with == or isin(), got {test.comparison}")
            values = test.value if test.comparison == "in" else (test.value,)
            self.label_in(spec.column, values, test.selectivity)
            return
        extracted = self._extract_column(spec)
        if extracted is not None:
            predicate = RegularPredicate(
                column_expression(extracted, spec.path, spec.function),
                test.comparison, test.value)
            if spec.path:
                predicate.validate()
            else:
                predicate.validate_text_column()
            self._extract_filters.setdefault(extracted.alias, []).append(
                (predicate, test.selectivity))
            return
        if test.selectivity is not None:
            raise CompileError(
                f"a hint estimates a test of a label or an answer; "
                f"{spec.alias}.{spec.column} is a source column, tested at "
                f"the scan")
        predicate = RegularPredicate(self._resolve_value(spec), test.comparison,
                                     test.value)
        predicate.validate()
        self._regular_predicates.setdefault(predicate.column.alias, []).append(
            predicate)

    def _extract_column(self, spec: ColSpec) -> Optional[ColumnRef]:
        """Resolve a col() naming an extraction's answer or span, or None.

        Raises:
            CompileError: The name is both a table alias and an
                extraction's name, or the field or span part is unknown.
        """
        if spec.alias is None:
            return None
        name = spec.alias
        # an extraction may itself be named with the span suffix
        span = name not in self._extracts and name.endswith(SPAN_SUFFIX)
        if span:
            name = name[:-len(SPAN_SUFFIX)]
        if name not in self._extracts:
            return None
        if spec.alias in self._scope():
            raise CompileError(
                f"{spec.alias!r} is both a table alias and an extraction's "
                f"name; rename one")
        struct, members = self._extracts[name]
        if spec.column not in struct.fields:
            raise CompileError(
                f"ai_extract {name!r} has no field {spec.column!r}; its "
                f"fields are {list(struct.fields)}")
        if span and tuple(spec.path) not in (("start",), ("end",)):
            raise CompileError(
                f"{spec.alias}.{spec.column} is a span; test its start or "
                f"end with .field('start') or .field('end')")
        if span and spec.function is not None:
            raise CompileError(
                f"{spec.alias}.{spec.column}.{spec.path[0]} is a number; "
                f"{spec.function}() takes text")
        if not span and spec.path:
            raise CompileError(f"{spec.alias}.{spec.column} has no fields")
        member = next(m for m in members if m.name == struct.member(spec.column))
        (alias,) = member.expression.aliases()
        column = struct.member(spec.column) + (SPAN_SUFFIX if span else "")
        return ColumnRef(alias, self._scope()[alias], column)

    def limit(self, n: int) -> "Query":
        if not isinstance(n, int) or n <= 0:
            raise CompileError("LIMIT must be a positive integer")
        self._limit = n
        return self

    def order_by(self, *keys) -> "Query":
        """Sort the result rows.

        Args:
            keys: Each a column name such as ``"r.year"`` or a projected
                name such as ``"score"``, sorted ascending, or a
                ``col(...)`` with ``.asc()`` or ``.desc()`` and an
                optional ``.nulls_first()`` or ``.nulls_last()``.
        """
        self._order.extend(_sort_keys(keys))
        return self

    def group_by(self, *keys) -> "Query":
        """Group the result rows by columns or by named AI outputs."""
        self._group.extend(keys)
        return self

    def agg(self, **aggregates) -> "Query":
        """Add named aggregates, such as ``n=count()`` or ``s=avg("r.stars")``."""
        for name, spec in aggregates.items():
            if not isinstance(spec, AggSpec):
                raise CompileError(
                    f"agg() takes count(), count_distinct(), sum_(), avg(), "
                    f"min_(), or max_(), got {spec!r} for {name!r}")
            self._aggs[name] = spec
        return self

    def window(self, **calls) -> "Query":
        """Add named window functions, such as ``n=row_number().over(...)``."""
        for name, spec in calls.items():
            if not isinstance(spec, WindowSpec):
                raise CompileError(
                    f"window() takes row_number(), lag(), lead(), sum_(), or "
                    f"count() with .over(...), got {spec!r} for {name!r}")
            self._windows[name] = spec
        return self

    def having(self, *tests) -> "Query":
        """Keep the groups whose aggregates pass tests such as ``count() > 2``."""
        for test in tests:
            if not isinstance(test, HavingSpec):
                raise CompileError(
                    f"having() takes tests such as count() > 2, got {test!r}")
            self._having.append(test)
        return self

    def offset(self, n: int) -> "Query":
        if not isinstance(n, int) or n < 0:
            raise CompileError("OFFSET must be a nonnegative integer")
        self._offset = n
        return self

    def distinct(self) -> "Query":
        self._distinct = True
        return self

    def select(self, *cols) -> LogicalPlan:
        if self._pending_join is not None:
            raise CompileError(
                f"join() of {self._pending_join[0]} has no AI predicate "
                f"over its pairs; a plain join belongs in the database "
                f"the ids came from")
        if not (self._joins or self._filters or self._labels or self._scores
                or self._extracts):
            raise CompileError("the query has no AI predicate; a plain "
                               "scan belongs in the database the ids "
                               "came from")
        columns = []
        structs = []
        aggregation = None
        if self._group or self._aggs or self._having:
            if any(c in self._extracts for c in cols if isinstance(c, str)):
                raise CompileError(
                    "GROUP BY does not read AI.EXTRACT struct columns; "
                    "extract each field as its own column")
            if self._windows:
                raise CompileError(
                    "a window function and group_by() in one query are not "
                    "supported")
            columns, aggregation = self._aggregation(cols)
        windows, output = self._windowing(columns, cols)
        for c in () if aggregation is not None or windows else cols:
            if isinstance(c, str) and c in self._labels:
                columns.append(self._labels[c])
                continue
            if isinstance(c, str) and c in self._scores:
                columns.append(self._scores[c])
                continue
            if isinstance(c, str) and c in self._extracts:
                struct, members = self._extracts[c]
                structs.append(struct)
                columns.extend(members)
                continue
            if isinstance(c, str) and c == "*":
                for alias, provider in self._tables:
                    for name in self._catalog.get(provider).columns:
                        columns.append(ColumnRef(alias=alias,
                                                 provider=provider,
                                                 column=name))
                continue
            columns.append(self._resolve_value(col(c) if isinstance(c, str) else c,
                                               named=True))
        # a classification is planned when the query returns or tests
        # its label
        tested = {name for tests in self._label_filters.values()
                  for name, _, _ in tests}
        wanted = [column for column in self._labels.values()
                  if column in columns or column.name in tested]
        # an extraction is planned when the query returns its answer
        extracted = [column for column in columns
                     if aliased_call(column, "extract") is not None]
        logical = LogicalPlanBuilder()
        for alias, provider in self._tables:
            logical.add_scan(
                alias,
                provider,
                self._document_columns.get(alias, ""),
                tuple(self._filters.get(alias, ())),
                tuple(self._applies.get(alias, ())),
                tuple((column.expression, column.name) for column in wanted
                      if column.expression.aliases() == (alias,)),
                tuple(self._label_filters.get(alias, ())),
                regular_predicates=tuple(
                    self._regular_predicates.get(alias, ())),
                extracts=tuple((column.expression, column.name)
                               for column in extracted
                               if column.expression.aliases() == (alias,)),
                extract_filters=tuple(self._extract_filters.get(alias, ())),
            )
        for join in self._joins:
            logical.add_join(join)
        for column in wanted:
            if len(column.expression.aliases()) == 2:
                logical.add_classify(column.expression, column.name)
        named = {column_name(column): column for column in columns}
        if aggregation is not None:
            named.update({a.name: a for a in aggregation.aggregates})
        named.update({window.name: window for window in windows})
        order = []
        for spec, descending, nulls_first in self._order:
            if spec.alias is None and spec.column in self._extracts:
                raise CompileError(
                    f"ORDER BY does not compare the AI.EXTRACT struct column "
                    f"{spec.column!r}; extract the field as its own column")
            expression = (named[spec.column]
                          if spec.alias is None and spec.column in named
                          else self._resolve_value(spec, named=True))
            order.append(SortKey(expression, descending=descending,
                                 nulls_first=nulls_first))
        return logical.project(tuple(columns), self._limit, order=tuple(order),
                               offset=self._offset, distinct=self._distinct,
                               aggregation=aggregation, structs=tuple(structs),
                               windows=windows, output=output)

    def _project(self, columns: list, c) -> str:
        """Add the column or named output ``c`` to ``columns``; return its name."""
        if isinstance(c, str) and c in self._labels:
            column = self._labels[c]
        elif isinstance(c, str) and c in self._scores:
            column = self._scores[c]
        else:
            column = self._resolve_value(col(c) if isinstance(c, str) else c,
                                         named=True)
        named = next((c for c in columns if isinstance(c, Alias)
                      and c.expression == column), None)
        if named is not None:
            return named.name
        if column not in columns:
            columns.append(column)
        return column_name(column)

    def _windowing(self, columns: list, cols) -> tuple:
        """Bind window() to the selected names.

        Returns:
            The WindowCalls and the result columns in select() order.
        """
        if not self._windows:
            return (), ()
        if any(c in self._extracts or c == "*" for c in cols if isinstance(c, str)):
            raise CompileError(
                "window() does not read AI.EXTRACT struct columns or *; "
                "name each column")
        # the selected columns come first, then what only a window reads
        output = tuple(c if isinstance(c, str) and c in self._windows
                       else self._project(columns, c) for c in cols)
        windows = []
        for name, spec in self._windows.items():
            argument = None if spec.col is None else self._project(columns, spec.col)
            windows.append(WindowCall(
                spec.function, argument,
                tuple(self._project(columns, c) for c in spec.partition),
                tuple((self._project(columns, key), descending, nulls_first)
                      for key, descending, nulls_first in _sort_keys(spec.order)),
                name))
        return tuple(windows), output

    def _aggregation(self, cols) -> tuple:
        """Bind group_by(), agg(), and having() to the selected names.

        Returns:
            The projected columns and the Aggregation.
        """
        columns = []

        def project(c) -> str:
            return self._project(columns, c)

        keys = tuple(project(c) for c in self._group)
        aggregates = []
        for name, spec in self._aggs.items():
            argument = None if spec.col is None else project(spec.col)
            aggregates.append(AggregateCall(spec.function, argument, name))
        tests = []
        for index, test in enumerate(self._having):
            argument = None if test.agg.col is None else project(test.agg.col)
            same = next((a for a in aggregates
                         if (a.function, a.argument)
                         == (test.agg.function, argument)), None)
            if same is None:
                same = AggregateCall(test.agg.function, argument,
                                     f"__having_{index}")
                aggregates.append(same)
            tests.append(HavingTest(same, test.comparison, test.value))
        output = []
        for c in cols:
            if isinstance(c, str) and c in self._aggs:
                output.append(c)
            else:
                output.append(project(c))
        return columns, Aggregation(keys, tuple(aggregates), tuple(output),
                                    tuple(tests))


def _sort_keys(keys) -> list:
    """Return (column, descending, nulls_first) for order_by() keys."""
    out = []
    for key in keys:
        if isinstance(key, str):
            key = col(key)
        if isinstance(key, ColSpec):
            key = SortSpec(key)
        if not isinstance(key, SortSpec):
            raise CompileError(
                "order_by takes a column name or col(...).asc() or "
                f".desc(), got {key!r}")
        nulls_first = (key.descending if key.nulls is None
                       else key.nulls == "first")
        out.append((key.col, key.descending, nulls_first))
    return out


def docs(catalog: Catalog, provider: str,
         turn: tuple[str, str] = ("", ""), layout: str = "ai-if") -> Query:
    return Query(catalog, provider, turn, layout)
