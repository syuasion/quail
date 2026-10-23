"""Render logical and physical operator trees."""

from collections import Counter
from collections.abc import Mapping

from quail import logical as logical_nodes
from quail.logical import (
    DEFAULT_SELECTIVITY,
    LogicalPlan,
    column_name,
    effective_selectivity,
)
from quail.physical import (
    Aggregate,
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    Barrier,
    Exchange,
    Filter,
    Foreign,
    HashJoin,
    Limit,
    PortRef,
    Project,
    Recombine,
    Scan,
    Sort,
    Window,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.optimizer.rules import LogicalRule, PhysicalRule
from quail.planner.plan import Refusal


def _number(value):
    if 0 < abs(value) < 1:
        return f"{value:.3g}"
    return f"{value:,.1f}".removesuffix(".0")


def _prompt(prompt):
    args = ", ".join(column_name(arg) for arg in prompt.args)
    return f"PROMPT({prompt.template!r}, {args})"


def _fields(fields, depth):
    lines = []
    pad = "  " * depth
    for key, value in fields.items():
        if isinstance(value, Mapping):
            lines.append(f"{pad}{key}:")
            lines.extend(_fields(value, depth + 1))
        elif isinstance(value, (list, tuple)) and value and isinstance(
                value[0], Mapping):
            for index, item in enumerate(value, 1):
                lines.append(f"{pad}{key} {index}:")
                lines.extend(_fields(item, depth + 1))
        else:
            lines.append(f"{pad}{key}={value}")
    return lines


def logical_tree(logical):
    """Return the logical operators and their expressions."""
    lines = []

    def visit(node, depth):
        details = []
        if isinstance(node, logical_nodes.Result):
            if node.limit is not None:
                lines.append(f"{'  ' * depth}Limit: {node.limit:,}")
                depth += 1
            if node.offset:
                lines.append(f"{'  ' * depth}Offset: {node.offset:,}")
                depth += 1
            if node.order:
                keys = ", ".join(str(key) for key in node.order)
                lines.append(f"{'  ' * depth}Sort: {keys}")
                depth += 1
            if node.distinct:
                lines.append(f"{'  ' * depth}Distinct")
                depth += 1
            visit(node.input, depth)
            return
        if isinstance(node, logical_nodes.Aggregate):
            title = f"Aggregate: {node.specification()}"
        elif isinstance(node, logical_nodes.Window):
            title = "Window: " + "; ".join(str(call) for call in node.calls)
        elif isinstance(node, logical_nodes.Project):
            fields = node.explain_fields()
            title = f"Project: {', '.join(fields['columns'])}"
            details = [f"{name} = struct({', '.join(members)})"
                       for name, members in fields.get("structs", {}).items()]
        elif isinstance(node, logical_nodes.Scan):
            title = f"Scan {node.provider} as {node.alias}"
            columns = ", ".join(dict.fromkeys(
                (column_name(node.column, qualified=False), *node.columns)))
            title += f" [{columns}]"
            if node.predicates:
                title += " where " + " and ".join(
                    str(predicate) for predicate in node.predicates)
        elif isinstance(node, logical_nodes.SemanticFilter):
            title = "SemanticFilter"
            details = [f"{_prompt(p.prompt)} "
                       f"(selectivity={_selectivity(p.selectivity)})"
                       for p in node.predicates]
            if node.stop_key:
                title += " stop_key=" + ",".join(node.stop_key)
        elif isinstance(node, logical_nodes.Filter):
            title = (f"Filter: {node.explain_fields()['condition']} "
                     f"(selectivity={_selectivity(node.selectivity)})")
        elif isinstance(node, logical_nodes.SemanticJoin):
            title = f"SemanticJoin ({node.semantics})"
            details = [f"{_prompt(node.prompt)} "
                       f"(selectivity={_selectivity(node.selectivity)})"]
            if node.anchor is not None:
                title += f" anchor={node.anchor}"
        elif isinstance(node, logical_nodes.SemanticClassify):
            title = f"SemanticClassify: {node.name}"
            if node.probabilities:
                title += f", {node.name}{logical_nodes.PROBABILITIES_SUFFIX}"
            details = [f"{_prompt(node.call.prompt)} "
                       f"labels={list(node.call.labels)}"]
        elif isinstance(node, logical_nodes.SemanticExtract):
            title = f"SemanticExtract: {node.name}, {node.span_name}"
            details = [f"question={node.call.question!r}"]
        elif isinstance(node, logical_nodes.Join):
            title = "Join"
            details = ([f"on {condition}" for condition in node.on]
                       or ["cross"])
        elif isinstance(node, logical_nodes.Apply):
            title = f"Apply {node.function} ({node.kind}, {node.ids})"
            details = ["columns: " + ", ".join(
                column_name(column) for column in node.columns)]
        else:
            title = node.type_name
            details = _fields(node.explain_fields(), 0)
        lines.append(f"{'  ' * depth}{title}")
        lines.extend(f"{'  ' * (depth + 1)}{detail}" for detail in details)
        for child in node.children():
            visit(child, depth + 1)

    visit(logical.root, 0)
    return "\n".join(lines)


DEFAULT_MARK = "*"
DEFAULT_NOTE = (f"{DEFAULT_MARK} no selectivity was given; the planner "
                f"assumes {DEFAULT_SELECTIVITY * 100:g}%")
PROMPT_CHARS = 32

# table columns in display order: (cell key, header)
_COLUMNS = (
    ("est_rows", "est. rows"),
    ("rows", "rows"),
    ("est_pass", "est. pass"),
    ("pass", "pass"),
    ("est_time", "est. time"),
    ("time", "time"),
    ("tokens", "fresh tokens"),
)


def _selectivity(value, mark=False):
    if value is None:
        default = f"{DEFAULT_SELECTIVITY * 100:g}%"
        return default + DEFAULT_MARK if mark else default + " (default)"
    return f"{value * 100:.3g}%"


def _seconds(value):
    if value is None:
        return ""
    if value < 0.001:
        return "<1 ms"
    if value < 1:
        return f"{value * 1000:.3g} ms"
    return f"{value:.3g} s"


def _short_prompt(prompt):
    template = prompt.template
    if len(template) > PROMPT_CHARS:
        template = template[:PROMPT_CHARS] + "…"
    return f"PROMPT({template!r})"


def _table(rows):
    """Lay out (text, cells) rows as a tree column and aligned columns."""
    columns = [(key, label) for key, label in _COLUMNS
               if any(cells.get(key) for _, cells in rows)]
    width = max(len(text) for text, _ in rows)
    widths = {
        key: max(len(label), *(len(cells.get(key, "")) for _, cells in rows))
        for key, label in columns
    }
    lines = []
    if columns:
        lines.append(" " * width + "".join(
            f"   {label:>{widths[key]}}" for key, label in columns))
    for text, cells in rows:
        line = text.ljust(width) + "".join(
            f"   {cells.get(key, ''):>{widths[key]}}" for key, _ in columns)
        lines.append(line.rstrip())
    return lines


def measured_stages(report) -> dict:
    """Index a report's stage measurements by the predicate they ran.

    Args:
        report: An execution report (QueryResult.report).

    Returns:
        ("filter", alias, written_position) and ("join", written_position) keys
        mapped to the stage dicts of report["stages"]. Stages saved
        before written_position was recorded are left out.
    """
    stages = {}
    for stage in (report or {}).get("stages", ()):
        written_position = stage.get("written_position")
        if written_position is None:
            continue
        if stage.get("op") == "filter":
            stages[("filter", stage["alias"], written_position)] = stage
        elif stage.get("op") == "join":
            stages[("join", written_position)] = stage
    return stages


def run_summary(report, graph, gpus, usd_per_hour=None) -> list:
    """Return the measured totals of one run as labeled lines.

    Args:
        report: The execution report (QueryResult.report).
        graph: The executed physical graph, for the input document count.
        gpus: GPUs the query ran on.
        usd_per_hour: Price of one GPU, for the cost per query.
    """
    wall = report.get("wall_s")
    if wall is None:
        return ["query time   not measured"]
    items = [("query time", f"{_seconds(wall)} (model startup excluded)")]
    if report.get("boot_s") is not None:
        kind = f" ({report['boot_kind']})" if report.get("boot_kind") else ""
        items.append(("startup", _seconds(report["boot_s"]) + kind))
    pairs = sum(stage.get("tuples", 0) for stage in report.get("stages", ())
                if stage.get("op") == "join")
    if not pairs:
        # a score over document pairs has no join stage records
        pairs = report.get("evaluated_document_pairs") or 0
    documents = sum(node.num_documents for node in graph.nodes
                    if isinstance(node, Scan))
    if wall > 0 and pairs:
        items.append(("throughput", f"{_number(pairs / wall)} document "
                      f"pairs/second over {pairs:,} evaluated pairs"))
    elif wall > 0:
        items.append(("throughput", f"{_number(documents / wall)} "
                      f"documents/second over {documents:,} input documents"))
    tokens = f"{report['fresh_tokens']:,} fresh"
    if report.get("cached_tokens") is not None:
        tokens += f", {report['cached_tokens']:,} read from KV"
    items.append(("tokens", tokens))
    if usd_per_hour is not None:
        cost = wall / 3600 * gpus * usd_per_hour
        items.append(("GPU cost", f"${cost:.4f} per query ({gpus} GPU at "
                      f"${usd_per_hour:.4f}/hour, startup excluded)"))
    width = max(len(label) for label, _ in items)
    return [f"{label.ljust(width)}   {value}" for label, value in items]


def _logical_context(logical):
    if logical is None:
        return {}, {}, []
    operators = logical.operators()
    scans = {scan.alias: scan for scan in operators.scans}
    return scans, operators.filters, list(operators.joins)


def _estimated_rows(graph):
    rows = {}
    for node in graph.topological_nodes():
        inputs = [rows.get(port.source) for port in node.inputs]
        value = None
        if isinstance(node, Scan):
            value = (node.num_documents if node.expected_rows is None
                     else node.expected_rows)
        elif isinstance(node, AiFilter):
            value = inputs[0] if inputs else None
            if value is not None:
                value *= effective_selectivity(node.selectivity)
        elif isinstance(node, (Project, Limit, Sort, Window, Exchange)) \
                and len(inputs) == 1:
            value = inputs[0]
            if isinstance(node, Limit) and value is not None:
                value = min(value, node.count)
            if isinstance(node, Sort) and value is not None:
                value = max(value - node.offset, 0)
                if node.fetch is not None:
                    value = min(value, node.fetch)
        elif isinstance(node, Foreign) and node.ids == "preserve":
            value = inputs[0] if inputs else None
        elif isinstance(node, Aggregate):
            # the group count is not estimated
            value = None
        # Join nodes expose both document ids and predicate answers.
        # Their evaluated tuple counts are not output row estimates.
        for output in node.outputs:
            if output.name.startswith("filter_answers:"):
                continue
            rows[PortRef(node.node_id, output.name)] = value
    return rows


def physical_tree(graph, *, logical=None, verbose=False, metrics=None,
                  estimates=None, stages=None):
    """Return a physical tree with estimated and measured columns.

    Args:
        graph: The physical graph to display.
        logical: Optional logical plan supplying source names and predicates.
        verbose: Include node ids, ports, settings, and stage details.
        metrics: Optional measured metrics indexed by node id.
        estimates: Optional per node estimates (PhysicalPlan.estimates).
        stages: Optional measured stages from measured_stages(report).
    """
    estimates_by_node = estimates or {}
    stages = stages or {}
    scans, filters, joins = _logical_context(logical)
    estimates = _estimated_rows(graph)
    uses = Counter(child for node in graph.nodes
                   for child in {port.source.node_id for port in node.inputs})
    shared = {node.node_id: index for index, node in enumerate(
        (node for node in graph.topological_nodes() if uses[node.node_id] > 1), 1)}
    visited = set()
    rows = []
    used_default = []

    def stage_cells(selectivity, expected, measured, evaluated_key):
        if selectivity is None:
            used_default.append(True)
        cells = {"est_rows": _number(expected),
                 "est_pass": _selectivity(selectivity, mark=True)}
        if measured is not None:
            cells["rows"] = f"{measured[evaluated_key]:,}"
            cells["pass"] = _selectivity(measured["observed_selectivity"])
        return cells

    def describe(node):
        """Return the node's title, detail lines, and predicate rows."""
        details = []
        predicates = []
        title = type(node).__name__
        if isinstance(node, Scan):
            source = scans.get(node.alias)
            title = (f"Scan {source.provider} as {node.alias}" if source else
                     f"Scan: {node.alias}")
            mean = node.total_tokens / node.num_documents if node.num_documents else 0
            details.append(f"tokens={node.total_tokens:,}, "
                           f"mean_document_tokens={_number(mean)}")
            if source is not None and source.predicates:
                details.append("where " + " and ".join(
                    str(predicate) for predicate in source.predicates))
        elif isinstance(node, Project):
            title += ": " + ", ".join(node.result_columns)
            details.extend(
                f"{name} = struct({', '.join(member for _, member in members)})"
                for name, members in node.structs)
        elif isinstance(node, Limit):
            title += f": {node.count:,}"
        elif isinstance(node, Aggregate) and node.partner:
            title += (f": one {node.alias}-{node.partner} pair per pair of texts, "
                      "on the executor")
        elif isinstance(node, Aggregate) and node.alias:
            title += f": one {node.alias} document per text, on the executor"
        elif isinstance(node, Aggregate):
            title += ": " + ", ".join(node.columns)
            if node.keys:
                details.append("group by " + ", ".join(node.keys))
            details.append(", ".join(
                f"{name} = {function}({argument or '*'})"
                for name, function, argument in node.aggregates))
            if node.having:
                details.append("having " + " and ".join(
                    f"{name} {comparison} {value!r}"
                    for name, comparison, value in node.having))
        elif isinstance(node, Window):
            title += ": " + ", ".join(node.columns)
            details.extend(str(call) for call in node.calls)
        elif isinstance(node, Sort):
            title += ": " + ", ".join(
                f"{column} {'DESC' if descending else 'ASC'} "
                f"NULLS {'FIRST' if nulls_first else 'LAST'}"
                for column, descending, nulls_first in node.keys)
            bounds = []
            if node.distinct:
                bounds.append("distinct")
            if node.offset:
                bounds.append(f"offset={node.offset:,}")
            if node.fetch is not None:
                bounds.append(f"fetch={node.fetch:,}")
            if bounds:
                details.append(", ".join(bounds))
        elif isinstance(node, AiFilter):
            title += f": {node.alias}"
            kv = []
            if node.arena_writes:
                kv.append("KV rewind=on")
            if node.keep_kv:
                kv.append("retain KV for later joins")
            details.append(", ".join(kv) if kv else
                           "KV: stored" if node.arena_writes else "KV: not stored")
            if node.stop_key:
                details.append(
                    f"stop per key: {', '.join(node.stop_key)} "
                    f"(one document at a time, the rest skipped once "
                    f"one survives)")
            written = filters.get(node.alias, ())
            label = f"predicate {node.written_position + 1}"
            if node.written_position < len(written):
                label += "  " + _short_prompt(
                    written[node.written_position].prompt)
            predicates.append((label, stage_cells(
                node.selectivity, node.expected_rows,
                stages.get(("filter", node.alias, node.written_position)),
                "evaluated")))
        elif isinstance(node, AiClassify):
            spec = node.spec
            title += f": {spec.name} over {spec.aliases[0]}"
            details.append(f"rule={spec.scoring}, "
                           f"labels={len(spec.labels)}, "
                           f"label tokens="
                           f"{sum(len(ids) for ids in spec.label_token_ids)}"
                           + (f", up to {spec.draws} noise draws"
                              if spec.draws > 1 else ""))
        elif isinstance(node, AiExtract):
            spec = node.spec
            title += f": {spec.name} over {spec.alias}"
            details.append(f"question={spec.question!r}, "
                           f"up to {spec.max_answer_tokens} answer tokens")
        elif isinstance(node, Filter):
            title += f": {node.predicate}"
        elif isinstance(node, AiJoin):
            title += f": anchor={node.anchor}"
            source = {"none": "not resident", "filter": "from filters",
                      "kept": "from an earlier join"}.get(
                          node.anchor_resident, node.anchor_resident)
            keep = "yes" if node.keep_kv else "no"
            details.append(f"KV: anchor={source}, retain after join={keep}")
            label = (f"join {node.written_position + 1} {node.semantics} "
                     f"({node.anchor}, {', '.join(node.partners)})")
            if node.pairs_from:
                label += f" over pairs from {node.pairs_from}"
            if node.written_position < len(joins):
                label += "  " + _short_prompt(
                    joins[node.written_position].prompt)
            predicates.append((label, stage_cells(
                node.selectivity, node.expected_rows,
                stages.get(("join", node.written_position)), "tuples")))
        elif isinstance(node, HashJoin) and node.ports:
            (answered,) = {port.source.node_id for port in node.inputs
                           if port.name.startswith("answers:")}
            covered = (f"{node.left} document with the same text"
                       if node.left == node.right
                       else f"{node.left}-{node.right} pair with the same texts")
            title += f": answers of {answered} to every {covered}"
        elif isinstance(node, HashJoin):
            title += ": " + " and ".join(
                f"{column_name(left)} = {column_name(right)}"
                for left, right in node.on)
            details.append(
                f"pairs kept: {node.pair_fraction:.4g} of the cross product")
        elif isinstance(node, Foreign):
            title += (f": {node.function} ({node.kind}, {node.ids}) on "
                      f"{', '.join(node.aliases)}")
            if node.columns:
                details.append("columns: " + ", ".join(
                    column_name(column) for column in node.columns))
        elif isinstance(node, Exchange):
            title += f": {node.anchor} to the GPU holding its KV"
        elif isinstance(node, Barrier):
            title += f": next_anchor={node.next_anchor}"
        elif isinstance(node, Recombine):
            title += ": " + ", ".join(node.alias_order)
        else:
            title = node.type_name
            if not verbose:
                details.extend(_fields(node.explain_fields(), 0))
        return title, details, predicates

    def visit(node, depth):
        pad = "  " * depth
        title, details, predicates = describe(node)
        reference = shared.get(node.node_id)
        if node.node_id in visited:
            rows.append((f"{pad}Reuse [{reference}]", {}))
            return
        visited.add(node.node_id)
        if reference:
            title += f" [{reference}]"
        values = [estimates.get(PortRef(node.node_id, output.name))
                  for output in node.outputs
                  if not output.name.startswith("filter_answers:")]
        value = values[0] if len(values) == 1 else None
        estimate = estimates_by_node.get(node.node_id, {})
        cells = {}
        if value is not None:
            cells["est_rows"] = _number(value)
        if "seconds" in estimate:
            cells["est_time"] = _seconds(estimate["seconds"])
        if metrics is not None:
            if node.node_id in metrics:
                measured = metrics[node.node_id]
                cells["rows"] = f"{measured.output_rows:,}"
                cells["time"] = _seconds(measured.wall_s)
                if measured.fresh_tokens:
                    cells["tokens"] = f"{measured.fresh_tokens:,}"
                if verbose:
                    details.extend(_fields(vars(measured), 0))
            else:
                title += " (metrics unavailable)"
        if "release_recompute_tokens" in estimate:
            details.append(
                "expected recompute at the join if the sequence's KV is "
                "released: "
                f"{estimate['release_recompute_tokens']:,} tokens, "
                f"{estimate['release_recompute_seconds']:.3f} s")
        rows.append((pad + title, cells))
        rows.extend((pad + "  " + detail, {}) for detail in details)
        rows.extend((pad + "  " + label, stage) for label, stage in predicates)
        if verbose:
            rows.extend((line, {}) for line in _fields(
                {"node_id": node.node_id, "type": node.type_name,
                 **node.explain_fields()}, depth + 1))
            rows.extend((f"{pad}  {port.name} <- "
                         f"{port.source.node_id}[{port.source.port}]", {})
                        for port in node.inputs)
        children = dict.fromkeys(port.source.node_id for port in node.inputs)
        for child in children:
            visit(graph.node(child), depth + 1)

    visit(graph.node(graph.root.node_id), 0)
    for node in graph.topological_nodes():
        if node.node_id not in visited:
            visit(node, 0)
    lines = _table(rows)
    seen = set()
    for member_id, pipeline in build_pipelines(graph).items():
        if id(pipeline) in seen or len(pipeline) < 2:
            continue
        seen.add(id(pipeline))
        lines.append(f"Pipeline on {pipeline[0].document_aliases[0]}: "
                     + " -> ".join(member.node_id for member in pipeline)
                     + " (each document runs every operator with its KV "
                     "resident)")
    if used_default:
        lines.append(DEFAULT_NOTE)
    return "\n".join(lines)


def rule_lines(rules) -> list[str]:
    """Return rule categories and the order used during search."""
    def names(selected):
        return ", ".join(rule.name for rule in selected) or "none"

    return [
        "optimizer rules:",
        "  logical: " + names(rule for rule in rules if isinstance(rule, LogicalRule)),
        "  physical: " + names(
            rule for rule in rules if isinstance(rule, PhysicalRule)),
        "  search: rewrite logical candidates, "
        "then lower each alternative",
        "  refine each graph in order: " + names(
            rule for rule in rules if isinstance(rule, PhysicalRule)),
        "  selection: lowest estimated cost after graph refinement",
    ]


def architecture_lines(model_name: str) -> list[str]:
    """Return the architecture, weight precision, and kernel path lines.

    The gated delta net recurrence kernel is read from QUAIL_GDN_PREFILL
    in this process, which is the worker's value when the worker was
    started from the same shell.

    Args:
        model_name: A registered model name.

    Returns:
        Two lines, or none for a name that is not registered.
    """
    import os
    from collections import Counter

    from quail.specs import MODELS

    spec = MODELS.get(model_name)
    if spec is None:
        return []
    kinds = Counter(spec.layer_kinds)
    layers = (", ".join(f"{count} {kind}" for kind, count in kinds.items())
              if kinds else f"{spec.layers} layers")
    gemm = ("fp8 GEMM (DeepGEMM)" if spec.weight_precision == "fp8"
            else "bf16 GEMM (cuBLAS)")
    if spec.arch == "qwen3_5":
        gemm = "vLLM per-layer GEMM"
        if spec.gdn_weight_precision is not None:
            gemm += f" (recurrent projections {spec.gdn_weight_precision})"
    paths = [f"linear layers: {gemm}"]
    if "full_attention" in kinds or "sliding_attention" in kinds:
        paths.append("attention: FlashAttention over paged KV")
    if "gated_delta_net" in kinds:
        requested = os.environ.get("QUAIL_GDN_PREFILL", "auto")
        paths.append("gated_delta_net: Triton convolution, "
                     f"recurrence={requested} (QUAIL_GDN_PREFILL)")
    if "short_conv" in kinds:
        paths.append("short_conv: Triton convolution")
    return [f"architecture={spec.arch} ({layers}), "
            f"weights={spec.weight_precision}",
            "kernels: " + "; ".join(paths)]


def explain(logical: LogicalPlan, physical, *, verbose: bool = False,
            result=None, usd_per_hour: float | None = None,
            rules=None) -> str:
    """Format the logical and physical operator trees.

    Args:
        logical: The optimized logical plan.
        physical: The physical plan or planning refusal.
        verbose: Include runtime settings, internal node fields, and
            the optimizer rules.
        result: The QueryResult of running the plan. When given, each
            node shows its measured rows, time, and tokens next to the
            estimates, and the measured totals follow the tree.
        usd_per_hour: Price of one GPU, for the measured cost per query.
        rules: Registered rules in order, listed when verbose.
    """
    lines = ["logical:"]
    lines.extend("  " + line for line in logical_tree(logical).splitlines())
    if isinstance(physical, Refusal):
        lines.append(f"refusal: {physical.constraint}: needed "
                     f"{physical.needed} {physical.unit}, available "
                     f"{physical.available}")
        lines.extend(f"  {reason}" for reason in physical.reasons)
        return "\n".join(lines)
    lines.append("")
    lines.append(f"physical: backend={physical.backend}, "
                 f"model={physical.model}, gpus={physical.gpus}")
    if physical.backend == "quail":
        chunk = physical.settings.get("chunk_tokens")
        admission = physical.settings.get("admission_tokens")
        budgets = ["KV=bf16"]
        if chunk is not None:
            budgets.append(f"chunk budget={chunk:,} tokens")
        if admission is not None:
            budgets.append(f"admission budget={admission:,} tokens")
        lines.append("  " + ", ".join(budgets))
        lines.extend("  " + line for line in architecture_lines(
            physical.model))
    lines.append("")
    lines.extend("  " + line for line in physical_tree(
        physical.graph, logical=logical, verbose=verbose,
        estimates=physical.estimates,
        metrics=None if result is None else result.node_metrics,
        stages=None if result is None else measured_stages(result.report),
    ).splitlines())
    if physical.estimates:
        lines.append("  est. time is each node's work alone; node times do "
                     "not add up to the plan estimate")
        lines.append("  because chunk packing shares forward passes across "
                     "nodes")
    if result is not None:
        lines.append("")
        lines.append("run:")
        lines.extend("  " + line for line in run_summary(
            result.report, physical.graph, physical.gpus, usd_per_hour))
    if verbose:
        lines.append("")
        lines.append("settings:")
        lines.extend(_fields({"model": physical.model,
                              "device": physical.device,
                              **physical.settings}, 1))
        if physical.backend == "quail":
            lines.append("  KV dtype=bf16")
        if rules is not None:
            lines.append("")
            lines.extend(rule_lines(rules))
    return "\n".join(lines)
