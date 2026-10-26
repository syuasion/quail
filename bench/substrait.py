"""Build a QUAIL-B query on a Quail session.

quail-b reads each query's Substrait plan into relations, operators,
and relational steps (`QuerySpec.info`). `build_query` maps that
reading to builder calls: a filter, join, classification, label test,
regular predicate, score, or extraction per operator, then the
aggregate, having, sort, and fetch steps, then the projection.
"""

from __future__ import annotations

import quail
from quail.frontend.expressions import AggSpec, HavingSpec, PredicateSpec
from quail_b.substrait import (
    Aggregate,
    Classify,
    ColumnTest,
    Extract,
    Fetch,
    Filter,
    Having,
    InList,
    PlanInfo,
    Score,
    Sort,
)


def build_query(session, info: PlanInfo, selectivity=None,
                order: str | None = None):
    """Build a Quail query from the benchmark's reading of a plan.

    Args:
        session: A session with every relation's table registered.
        info: The plan as `QuerySpec.info` reads it.
        selectivity: Prompt -> the fraction of documents or pairs
            expected to pass, given to the planner for ordering; a label
            filter's key is (classification prompt, frozenset of
            accepted labels).
        order: The filter order rule `select` takes.
    """
    selectivity = selectivity or {}
    per_alias = {}
    for item in info.operators:
        if isinstance(item, Classify) and item.partner is not None:
            continue    # labels a join's rows, so it follows the joins
        if isinstance(item, (Filter, Classify, InList, ColumnTest, Score,
                             Extract)):
            per_alias.setdefault(item.relation, []).append(item)
    prompts = {(item.relation, item.output): item.prompt
               for item in info.classifies}
    # a label, score, or answer struct is a named output of the builder
    named = {f"{item.relation}.{item.output}": item.output
             for item in (*info.classifies, *info.scores)}
    named.update({name: struct.name for struct in info.structs
                  for name in struct.fields})

    def field(name: str) -> str:
        return named.get(name, name)

    def document(alias: str):
        return quail.col(f"{alias}.{info.relation(alias).text_column}")

    def relation_query(alias):
        query = session.docs(info.relation(alias).table).alias(alias)
        for item in per_alias.get(alias, ()):
            if isinstance(item, ColumnTest):
                query = query.where(PredicateSpec(
                    quail.col(f"{alias}.{item.column}"), item.comparison,
                    item.value))
            elif isinstance(item, Score):
                query = query.ai_score(
                    quail.prompt(item.prompt, document(alias)),
                    name=item.output)
            elif isinstance(item, Extract):
                # a struct's fields are one call, made at its first field
                struct = info.struct_of(item)
                if struct.fields[0] != f"{alias}.{item.output}":
                    continue
                questions = {extract.output: extract.prompt
                             for extract in info.extracts
                             if info.struct_of(extract) == struct}
                query = query.ai_extract(
                    f"{alias}.{info.relation(alias).text_column}",
                    questions, name=struct.name)
            elif isinstance(item, Filter):
                query = query.ai_filter(
                    quail.prompt(item.prompt, document(alias)),
                    selectivity=selectivity.get(item.prompt))
            elif isinstance(item, Classify):
                query = query.ai_classify(
                    quail.prompt(item.prompt, document(alias)),
                    item.labels, name=item.output,
                    descriptions=item.descriptions)
            else:
                key = (prompts[(alias, item.output)], frozenset(item.accepted))
                query = query.label_in(item.output, item.accepted,
                                       selectivity=selectivity.get(key))
        return query

    joined = {info.base_alias}
    query = relation_query(info.base_alias)
    for join in info.joins:
        new = [alias for alias in join.relations if alias not in joined]
        if len(new) != 1:
            raise ValueError(f"join {join.id!r} must add one relation")
        left, right = join.relations
        query = query.join(relation_query(new[0]), on=[
            quail.col(f"{left}.{left_column}") == quail.col(f"{right}.{right_column}")
            for left_column, right_column in join.on
        ]).ai_filter(
            quail.prompt(join.prompt, document(left), document(right)),
            selectivity=selectivity.get(join.prompt))
        joined.add(new[0])
    for item in info.classifies:
        if item.partner is not None:
            query = query.ai_classify(
                quail.prompt(item.prompt, document(item.relation),
                             document(item.partner)),
                item.labels, name=item.output,
                descriptions=item.descriptions)
    for step in info.tail:
        if isinstance(step, Aggregate):
            if not step.measures and set(step.keys) == set(info.select):
                # an aggregate with no measures is a DISTINCT
                query = query.distinct()
                continue
            query = query.group_by(*(field(key) for key in step.keys))
            if step.measures:
                query = query.agg(**{
                    name: AggSpec(function, None if argument is None
                                  else field(argument))
                    for name, function, argument in step.measures})
        elif isinstance(step, Having):
            measures = {name: (function, argument)
                        for name, function, argument
                        in info.aggregate.measures}
            query = query.having(*(
                HavingSpec(AggSpec(measures[name][0],
                                   None if measures[name][1] is None
                                   else field(measures[name][1])),
                           comparison, value)
                for name, comparison, value in step.tests))
        elif isinstance(step, Sort):
            query = query.order_by(*(
                (quail.col(field(name)).desc() if descending
                 else quail.col(field(name)).asc()).nulls_last()
                for name, descending in step.keys))
        elif isinstance(step, Fetch):
            if step.offset:
                query = query.offset(step.offset)
            query = query.limit(step.count)
    return query.select(*dict.fromkeys(field(name) for name in info.select),
                        order=order)
