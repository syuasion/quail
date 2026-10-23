"""Build execution reports and collect model answer relations."""

import pyarrow as pa
from pyarrow import compute as pc

from quail.execution.ids import document_ids
from quail.execution.result import (
    answer_values,
    predicate_positions,
    true_answer_rows,
)
from quail.execution.rows import Rows, as_table
from quail.physical import AiJoin, Barrier, HashJoin, PortRef, ValueType


def execution_report(plan, metrics, notes, coordinator_wall):
    """Build a query report from its plan and backend measurements."""
    out = metrics

    expected_nodes = tuple(node for node in plan.nodes
                           if isinstance(node, (AiJoin, Barrier)))
    report = dict(
        backend=out.get("backend", plan.backend),
        wall_s=out["wall_s"], boot_s=out.get("boot_s"),
        estimated_seconds=plan.estimated_seconds,
        boot_kind=out.get("boot_kind"),
        boot=out.get("boot"),
        coordinator_wall_s=round(coordinator_wall, 2),
        fresh_tokens=out["fresh_tokens"],
        cached_tokens=out.get("cached_tokens"),
        stages=[],
        peak_gib=out.get("peak_gib"),
        order_rule=plan.settings.get("order_rule"),
        expected_join_plan=[
            {
                "type": node.type_name,
                "id": node.node_id,
                **node.explain_fields(),
            }
            for node in expected_nodes
        ],
        executed_join_plan=out.get("executed_join_plan", []),
        kv_manager=out.get("kv_manager"),
        node_metrics=out.get("node_metrics", {}),
        backend_metrics=out.get("backend_metrics"),
        remarks=list(notes))
    for key in ("gpu_s", "chunks", "memory_trace", "memory_trace_started",
                "memory_events", "memory_capacity", "memory_bytes", "memory_workers"):
        if key in out:
            report[key] = out[key]
    for key in (
        "evaluated_documents",
        "evaluated_document_pairs",
        "documents_per_second",
        "document_pairs_per_second",
        "usd_per_query",
    ):
        if key in out:
            report[key] = out[key]

    return report


def superseded_outputs(plan) -> set:
    """The model outputs a HashJoin joins back to every document, and so replaces."""
    return {port.source for node in plan.nodes
            if isinstance(node, HashJoin) and node.ports
            for port in node.inputs[1:]}


def collect_answers(logical, plan, outputs, documents):
    """Collect answer tables, surviving rows, and measured predicate selectivity."""
    # a HashJoin's answers for every document replace those it read
    superseded = superseded_outputs(plan)
    outputs = {ref: value for ref, value in outputs.items()
               if ref not in superseded}
    operators = logical.operators()
    scans, logical_filters, logical_joins = (
        operators.scans, operators.all_filters(), operators.joins)
    expected_nodes = tuple(node for node in plan.nodes
                           if isinstance(node, (AiJoin, Barrier)))
    report = {"stages": []}
    answer_tables = {kind: {} for kind in (
        "filters", "joins", "classifies", "scores", "extracts")}
    for node in plan.nodes:
        for port in node.outputs:
            answer = port.answer
            ref = PortRef(node.node_id, port.name)
            if answer is None or ref not in outputs:
                continue
            table = as_table(outputs[ref]).select(answer.columns)
            if answer.kind == "classifies":
                table = table.filter(pc.is_valid(table.column(answer.name)))
            answer_tables[answer.kind][answer.name] = table
    survivors = {
        scan.alias: pa.array(
            range(len(documents[scan.alias])), pa.int32())
        for scan in scans
    }

    output_types = {
        PortRef(node.node_id, output.name): output.value_type
        for node in plan.nodes
        for output in node.outputs
    }
    filter_relations = {}
    join_relations = {}
    for ref, table in outputs.items():
        value_type = output_types.get(ref)
        if isinstance(table, Rows):
            continue
        metadata = table.schema.metadata or {}
        if value_type is ValueType.FILTER_ANSWERS:
            alias = metadata.get(b"quail.alias")
            if alias is None:
                raise ValueError(
                    "a filter answer relation needs quail.alias metadata"
                )
            filter_relations.setdefault(
                alias.decode("utf-8"), []
            ).append(table)
        elif value_type is ValueType.JOIN_ANSWERS:
            written_position = metadata.get(b"quail.written_position")
            if written_position is None:
                raise ValueError(
                    "a join answer relation needs "
                    "quail.written_position metadata"
                )
            position = int(written_position.decode("ascii"))
            if position in join_relations:
                raise ValueError(
                    f"duplicate join answer relation {position}"
                )
            join_relations[position] = table

    missing_filter_relations = set(logical_filters) - set(
        filter_relations
    )
    if missing_filter_relations:
        raise ValueError(
            "execution response is missing filter answer relations for "
            f"{sorted(missing_filter_relations)}"
        )
    missing_join_relations = set(range(len(logical_joins))) - set(
        join_relations
    )
    if missing_join_relations:
        raise ValueError(
            "execution response is missing join answer relations for "
            f"{sorted(missing_join_relations)}"
        )

    for node in plan.graph.topological_nodes():
        for output in node.outputs:
            if not output.name.startswith("rows:"):
                continue
            ref = PortRef(node.node_id, output.name)
            if ref not in outputs:
                continue
            alias = output.name.split(":", 1)[1]
            survivors[alias] = document_ids(as_table(outputs[ref]).column(alias))

    for alias, relations in filter_relations.items():
        table = relations[0] if len(relations) == 1 else \
            pa.concat_tables(relations)
        # predicates in the order their first answers appear
        predicate_order = [
            int(position)
            for position in pc.unique(predicate_positions(table)).to_pylist()
        ]
        predicate_order.extend(
            position for position in range(len(logical_filters[alias]))
            if position not in predicate_order
        )
        for index, written_position in enumerate(predicate_order):
            mask = pc.equal(predicate_positions(table), written_position)
            stage_table = table.filter(mask)
            evaluated = stage_table.num_rows
            passed = pc.sum(answer_values(stage_table)).as_py() or 0
            answer_tables["filters"][(alias, written_position)] = stage_table
            report["stages"].append(dict(
                op="filter", alias=alias, stage=index,
                written_position=written_position,
                provided_selectivity=(
                    logical_filters[alias][written_position].selectivity
                ),
                observed_selectivity=round(
                    passed / max(1, evaluated), 4
                ),
                evaluated=evaluated,
            ))

    true_join_tables = {}
    expected_order = []
    for node in expected_nodes:
        for output in node.outputs:
            if not output.name.startswith("join_answers:"):
                continue
            expected_order.append(int(output.name.split(":", 1)[1]))
    expected_order.extend(
        position for position in sorted(join_relations)
        if position not in expected_order
    )
    for written_position in expected_order:
        if written_position not in join_relations:
            continue
        table = join_relations[written_position]
        metadata = table.schema.metadata or {}
        logical_join = logical_joins[written_position]
        logical_aliases = [
            argument.alias for argument in logical_join.prompt.args
        ]
        required_metadata = {b"quail.anchor", b"quail.semantics"}
        missing = required_metadata - set(metadata)
        if missing:
            raise ValueError(
                "a join answer relation is missing metadata "
                f"{sorted(key.decode('utf-8') for key in missing)}"
            )
        anchor = metadata[b"quail.anchor"].decode("utf-8")
        partners = [alias for alias in logical_aliases if alias != anchor]
        semantics = metadata[b"quail.semantics"].decode("utf-8")
        missing_columns = set(logical_aliases) - set(table.column_names)
        if missing_columns:
            raise ValueError(
                "a join answer relation is missing alias columns "
                f"{sorted(missing_columns)}"
            )
        answers = answer_values(table)
        answer_tables["joins"][written_position] = table
        report["stages"].append(dict(
            op="join", written_position=written_position, anchor=anchor,
            partners=partners,
            semantics=semantics,
            provided_selectivity=logical_join.selectivity,
            observed_selectivity=round(
                (pc.sum(answers).as_py() or 0) / max(1, len(answers)), 4
            ),
            tuples=len(answers),
        ))
        if semantics == "full":
            true_join_tables[written_position] = true_answer_rows(table)
    return answer_tables, survivors, true_join_tables, report["stages"]
