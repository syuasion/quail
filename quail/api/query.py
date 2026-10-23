"""A compiled query: its plan, its explain text, and its run."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

from quail.api.execute import execute_query
from quail.catalog import inputs
from quail.catalog.pairs import columns_key, estimate_pair_fraction
from quail.catalog.selection import selected_rows
from quail.catalog.tokens import first_document_columns
from quail.execution.metrics import NodeMetrics, scalar_node_metrics
from quail.execution.projection import projection_for
from quail.execution.reporting import collect_answers, execution_report
from quail.execution.result import QueryResult
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.types import PhysicalRequest, document_input
from quail.explain import explain
from quail.logical import (
    LogicalPlan,
    join_conditions,
    oriented_join_conditions,
)
from quail.logical.expressions import (
    column_references,
    evaluate_expression,
    source_columns,
)
from quail.physical import Aggregate as PhysicalAggregate
from quail.physical import HashJoin, Scan, encode_graph
from quail.planner import plan_query, refine_plan
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.tree_rewrites import logical_alternatives
from quail.planner.plan import Refusal, RefusalError
from quail.planner.search.statistics import filter_stop_keys

if TYPE_CHECKING:
    from quail.api.session import Session


@runtime_checkable
class QueryLike(Protocol):
    """What ``Session.sql`` returns, whether the query runs here or remotely.

    ``Query`` runs in this process; ``quail.remote.client.RemoteQuery``
    sends the SQL to Quail Server. Both offer these methods, so code
    written against one works with the other.
    """

    def run(self, plan=None) -> QueryResult: ...

    def submit(self, **options): ...

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader: ...

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table: ...

    def explain(self, **options) -> str: ...

    def plan(self): ...


class Query:
    """A compiled query that plans and executes in the current process."""

    def __init__(self, session: "Session", logical: LogicalPlan,
                 order: str | None = None):
        self.session = session
        self.logical = logical
        self._logical_input = logical
        self.order = order
        self._plan = None
        self._searched_plan = None
        self._document_tokens = None
        self._token_inputs = None
        self._token_futures = {}
        self._token_finished_at = {}
        self._estimated = ()
        self._planning_started_at = None
        self._planning_finished_at = None
        self._replanning_s = 0.0
        self._scan_selectivities = {}
        self._pair_selectivities = {}
        self.token_wait_s = 0.0

    def token_inputs(self) -> dict:
        """Return the token store of every scanned alias.

        Plans first, then waits for any background tokenization, so
        planning and the speed of light estimate share these stores.
        """
        self.plan()
        self.wait_for_tokens()
        return self._token_inputs

    def plan(self):
        """Plan the query once and return its physical plan or refusal.

        Document lengths are exact for a tokenized column. For any other
        column they are estimated from a sample while a background thread
        tokenizes it. The tree rewrites and the physical planner both read
        these lengths.

        Returns:
            The physical plan, or a Refusal when no backend can run it.
        """
        if self._plan is None:
            self._planning_started_at = time.perf_counter()
            config = self.session.config
            operators = self.logical.operators()
            self._document_tokens = {}
            self._token_inputs = {}
            estimated = []
            for s in operators.scans:
                exact = self.session.inputs.token_lengths(s.provider, s.column)
                if exact is not None:
                    self._document_tokens[s.alias] = exact
                    continue
                self._document_tokens[s.alias] = self.session.inputs.estimate_lengths(
                    s.provider, s.column)
                estimated.append(s.alias)
            self._estimated = tuple(estimated)
            # the planner estimates what the regular predicates and the join
            # equalities keep from samples; the executor applies them
            scan_fractions = self._scan_fractions(operators.scans)
            pair_fractions = self._pair_fractions(
                operators.scans, operators.joins)
            self._scan_selectivities = scan_fractions
            self._pair_selectivities = pair_fractions
            context = PlanningContext(
                catalog=self.session.catalog, engine_config=config,
                model=self.session.model, device=self.session.device,
                gpus=config.gpus, document_tokens=self._document_tokens,
                backend=config.backend, order=self.order,
                canvas_draws=config.canvas_draws,
                attention=config.attention,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions,
                scan_fractions=scan_fractions)
            candidates = logical_alternatives(
                self._logical_input,
                tuple(self.session.registry.logical_rules.values()),
                context)
            self.logical = candidates[0].plan
            scans = {}
            offset_aliases = set()
            for candidate in candidates:
                operators = candidate.plan.operators()
                offset_aliases.update(alias for extract in operators.extracts
                                      for alias in extract.call.aliases())
                for scan in operators.scans:
                    previous = scans.get(scan.alias, scan)
                    scans[scan.alias] = replace(scan, columns=tuple(dict.fromkeys(
                        (*previous.columns, *scan.columns))))
            backend = self.session.registry.backend(config.backend)
            # an extension backend may leave the capability flags unset
            reads_text = getattr(backend, "reads_document_text", False)
            locates_answers = getattr(backend, "locates_answers_in_text", False)
            for s in scans.values():
                columns = s.columns
                with_offsets = locates_answers and s.alias in offset_aliases
                if reads_text:
                    columns = tuple(dict.fromkeys(
                        (*columns, *source_columns(s.column))))
                if s.alias not in self._estimated:
                    self._token_inputs[s.alias] = self.session.inputs.tokenize(
                        s.provider, s.column, columns, with_offsets=with_offsets)
                    continue
                future = self.session.inputs.tokenize_async(
                    s.provider, s.column, columns, with_offsets=with_offsets)

                def record_finished(_future, alias=s.alias):
                    self._token_finished_at[alias] = time.perf_counter()

                future.add_done_callback(record_finished)
                self._token_futures[s.alias] = future
            self._plan = plan_query(
                self._logical_input, model=self.session.model,
                device=self.session.device,
                document_tokens=self._document_tokens,
                gpus=config.gpus,
                order=self.order,
                canvas_draws=config.canvas_draws,
                attention=config.attention,
                backend=config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions,
                scan_fractions=scan_fractions,
                memo=context.memo, catalog=self.session.catalog,
                engine_config=config, logical_candidates=candidates)
            if not isinstance(self._plan, Refusal):
                self.logical = self._plan.logical_plan
            extra = {}
            if config.gpu_timing:
                extra["gpu_timing"] = True
            if extra and not isinstance(self._plan, Refusal):
                self._plan = replace(self._plan, settings={
                    **self._plan.settings, **extra})
            self._planning_finished_at = time.perf_counter()
            self._searched_plan = self._plan
        return self._plan

    @property
    def planning_s(self) -> float:
        """Return wall time spent planning this query."""
        if self._planning_started_at is None:
            return 0.0
        finished = self._planning_finished_at or time.perf_counter()
        return finished - self._planning_started_at + self._replanning_s

    @property
    def input_ready_s(self) -> float:
        """Return planning and tokenization time, excluding backend startup."""
        if self._planning_started_at is None:
            return 0.0
        finished = [
            self._planning_finished_at or time.perf_counter(),
            *self._token_finished_at.values(),
        ]
        return max(finished) - self._planning_started_at + self._replanning_s

    def _pair_fractions(self, scans, joins) -> dict:
        """Estimated pairs over the cross product, per join with conditions."""
        providers = {scan.alias: scan.provider for scan in scans}
        samples = {}

        def keys(expressions):
            out = []
            for expression in expressions:
                refs = column_references(expression)
                for ref in refs:
                    if (ref.alias, ref.column) not in samples:
                        samples[(ref.alias, ref.column)] = (
                            self.session.inputs.sample_columns(
                                providers[ref.alias], [ref.column])[ref.column])
                out.append(evaluate_expression(expression, pa.table({
                    ref.column: samples[(ref.alias, ref.column)] for ref in refs})))
            return out

        fractions = {}
        for position, join in enumerate(joins):
            oriented = oriented_join_conditions(join)
            if oriented is None:
                continue
            _, _, conditions = oriented
            fractions[position] = estimate_pair_fraction(
                keys([left for left, _ in conditions]),
                keys([right for _, right in conditions]))
        return fractions

    def _scan_fractions(self, scans) -> dict:
        """Estimated fraction of each tested scan's documents its tests keep."""
        fractions = {}
        for scan in scans:
            if not scan.predicates:
                continue
            columns = pa.table(self.session.inputs.sample_columns(
                scan.provider,
                [predicate.column.column for predicate in scan.predicates]))
            sampled = columns.num_rows
            kept = len(selected_rows(columns, scan.predicates))
            # a sample that keeps nothing, or everything, says little
            # about the rest, so the estimate stays off 0 and 1
            fractions[scan.alias] = (kept + 1) / (sampled + 2)
        return fractions

    def explain(self, *, verbose: bool = False,
                analyze: bool = False) -> str:
        """Return the optimized plan, optionally measured by running it.

        Args:
            verbose: Include runtime settings and internal node fields.
            analyze: Run the query first, like EXPLAIN ANALYZE, and show
                each node's measured rows, time, and tokens beside the
                planner's estimates, then the measured totals.
        """
        # plan() first: it replaces self.logical with the optimized tree
        physical = self.plan()
        result = None
        if analyze:
            result = self.run()
            physical = self._plan
            # the root relation is lazy, so its row count is taken here
            rows = result.count()
            root = physical.graph.root.node_id
            result.node_metrics[root] = replace(
                result.node_metrics.get(root, NodeMetrics()),
                output_rows=rows)
            result.report.setdefault("node_metrics", {}).setdefault(
                root, {})["output_rows"] = rows
        registry = self.session.registry
        text = explain(self.logical, physical, verbose=verbose,
                       result=result,
                       rules=tuple(registry.rules.values()),
                       usd_per_hour=(
                           self.session.device.usd_per_hour or None))
        if self._estimated:
            text += ("\n\n  note: token counts for "
                     + ", ".join(repr(a) for a in self._estimated)
                     + f" are estimated from a {inputs.ESTIMATE_SAMPLE} document "
                     "sample")
        return text

    def wait_for_tokens(self) -> None:
        """Block until every background tokenization has finished.

        A plan made on estimated lengths is searched again with exact inputs,
        so prefix sharing can change which logical alternative wins.
        """
        started = time.perf_counter()
        refine = bool(self._token_futures)
        for alias, future in list(self._token_futures.items()):
            self._token_inputs[alias] = future.result()
            self._document_tokens[alias] = self._token_inputs[alias].lengths
            self._token_finished_at.setdefault(alias, time.perf_counter())
            del self._token_futures[alias]
        self.token_wait_s += time.perf_counter() - started
        if refine and self._plan is not None:
            started = time.perf_counter()
            options = dict(
                model=self.session.model,
                device=self.session.device, document_tokens=self._document_tokens,
                gpus=self.session.config.gpus, order=self.order,
                canvas_draws=self.session.config.canvas_draws,
                attention=self.session.config.attention,
                backend=self.session.config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer,
                pair_fractions=self._pair_selectivities)
            if self._plan is self._searched_plan:
                self._plan = plan_query(
                    self._logical_input, scan_fractions=self._scan_selectivities,
                    catalog=self.session.catalog, engine_config=self.session.config,
                    **options)
                if not isinstance(self._plan, Refusal):
                    self.logical = self._plan.logical_plan
            else:
                # Explicit physical edits must survive exact tokenization.
                self._plan = refine_plan(self._plan, **options)
            if self.session.config.gpu_timing and not isinstance(self._plan, Refusal):
                self._plan = replace(self._plan, settings={
                    **self._plan.settings, "gpu_timing": True})
            self._replanning_s += time.perf_counter() - started
            self._estimated = ()

    def run(self, plan=None) -> QueryResult:
        """Execute the query in the current process.

        Args:
            plan: An edited PhysicalPlan from plan().insert(),
                remove(), or move(); the planner's own plan when
                omitted.
        """
        return execute_query(self, plan=plan)

    def submit(self, **_options):
        """Submitting needs Quail Server; local queries use run()."""
        raise RuntimeError(
            "submit() needs a Session with an endpoint; this session runs "
            "queries in the current process, so call run() or collect()")

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Execute the query and stream Arrow record batches."""
        return self.run().execute_stream(
            batch_rows=batch_rows, limit=limit)

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        """Execute the query and explicitly collect one Arrow table."""
        return self.run().collect(
            limit=limit, batch_rows=batch_rows)

    def _prepare_physical(self):
        """Bind the query's tokenized inputs to its physical plan."""
        self.plan()
        self.wait_for_tokens()
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        plan.graph.validate(runtime_keys=set(self.session.registry.runtimes))
        plan.graph.validate_backend(plan.backend)
        inputs = {}
        logical_scans = {
            scan.alias: scan for scan in self.logical.operators().scans
        }
        for node in plan.nodes:
            if not isinstance(node, Scan):
                continue
            store = self._token_inputs[node.alias]
            logical_scan = logical_scans[node.alias]
            inputs[node.input_id] = document_input(
                store.tokens,
                (
                    evaluate_expression(logical_scan.column, store)
                    if set(source_columns(logical_scan.column))
                    <= set(store.projected_columns)
                    else None
                ),
            )
        envelope = plan.to_envelope(self.session.registry.codecs)
        return PhysicalRequest(envelope, inputs, self._column_tables())

    def _column_tables(self) -> dict:
        """One value table per alias whose columns the executor reads.

        A scan's regular predicates, a HashJoin, an apply(), and a stop key
        read values; the tables carry the columns they name.
        """
        needed = {}
        operators = self.logical.operators()
        for join in operators.joins:
            for condition in join_conditions(join):
                for ref in column_references(condition.left) + column_references(
                        condition.right):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for apply in operators.applies:
            for expression in apply.columns:
                for ref in column_references(expression):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for scan in operators.scans:
            for predicate in scan.predicates:
                needed.setdefault(scan.alias, {})[predicate.column.column] = None
        for alias, key in filter_stop_keys(self.logical).items():
            for column in key:
                needed.setdefault(alias, {})[column] = None
        # each document's key document comes from the tokens and the row
        # columns its name lists, not from a stored column
        for node in self.plan().nodes:
            if isinstance(node, PhysicalAggregate) and node.alias:
                for alias, key in zip((node.alias, node.partner), node.keys):
                    needed.setdefault(alias, {})[key] = None
            elif isinstance(node, HashJoin) and node.ports:
                for left, _ in node.on:
                    needed.setdefault(left.alias, {})[left.column] = None
        tables = {}
        for alias, columns in needed.items():
            store = self._token_inputs[alias]
            tables[columns_key(alias)] = pa.table({
                name: self._column_values(store, name) for name in columns})
        return tables

    @staticmethod
    def _column_values(store, name: str):
        """One alias's values of a stored column, or of a key document column."""
        grouped_by = first_document_columns(name)
        if grouped_by is None:
            return store.column(name)
        return store.tokens.first_documents_with(
            [store.column(column) for column in grouped_by])

    def finish(self, response, coordinator_wall: float = 0.0) -> QueryResult:
        """Finish the physical graph and attach execution details."""
        plan = self.plan()
        out = response.metrics
        report = execution_report(
            plan, out, self.session.inputs.notes, coordinator_wall)
        sources = {
            node.input_id: range(node.num_documents)
            for node in plan.nodes
            if isinstance(node, Scan)
        }
        observers = self.session.registry.new_observers()
        run = GenericRunner().run(
            plan.graph,
            ExecutionContext(
                runtimes=self.session.registry.runtimes,
                sources=sources,
                project=projection_for(
                    self.logical, self._token_inputs, self._document_tokens),
                observers=observers,
            ),
            initial_outputs=response.outputs,
            initial_metrics={
                node_id: NodeMetrics.from_scalars(metrics)
                for node_id, metrics in out.get("node_metrics", {}).items()
            },
        )
        if not isinstance(run.value, QueryResult):
            raise TypeError("physical graph root must return QueryResult")
        result = run.value
        result.plan = plan.graph
        result.node_metrics = {
            node_id: node_result.metrics
            for node_id, node_result in run.nodes.items()
        }
        # Saved reports need the graph and metrics without Python objects.
        report["executed_plan"] = encode_graph(
            plan.graph, self.session.registry.codecs)
        report["node_metrics"] = scalar_node_metrics(run.nodes)
        observer_reports = {
            observer.name: dict(observer.report())
            for observer in observers
        }
        if observer_reports:
            report["observers"] = observer_reports
        result.report = report

        (result.answer_tables, result.survivor_indices,
         result.true_join_tables, report["stages"]) = collect_answers(
             self.logical, plan, response.outputs, self._document_tokens)
        return result
