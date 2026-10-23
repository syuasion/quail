"""Model groups on several GPUs run the single-GPU executor on each worker's share."""

import pyarrow as pa
import pytest
from helpers.fakes import letter_tokens

import quail
from quail.api.execute import execute_query
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.executor.operators.filter import stop_groups
from quail.backends.quail.worker import quail_runtime_payload
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import filter_answer_table
from quail.execution.rows import Rows, extraction_rows
from quail.execution.types import PhysicalResponse
from quail.physical import AiExtract, AiFilter, decode_graph
from quail.planner.plan import EngineConfig

TEXTS = ["No date", "June", "No date", "August"]


@pytest.fixture(params=[2, 4])
def session(request):
    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=request.param),
        tokenizer=letter_tokens,
    ) as session:
        session.register("contracts", quail.DocumentProvider.from_table(
            pa.table({"id": range(len(TEXTS)), "text": TEXTS,
                      "group": ["lease", "sale", "lease", "sale"]}), id_col="id"))
        yield session


class FixedWorkers:
    """Answer each worker's group of nodes with fixed answers.

    A filter keeps every document it is given, an extraction answers
    from the text, and every other node runs its own runtime.
    """

    def __init__(self, session):
        self.session = session
        self.filter_limits = []
        self.filtered = []
        self.extracted = {}

    def execute(self, request):
        registry = self.session.registry
        graph = decode_graph(request.plan["graph"], registry.codecs)
        report = execute_distributed_graph(
            quail_runtime_payload(request, graph), graph, request.gpus,
            self.round, self.session.device, registry.runtimes, registry)
        return PhysicalResponse(report.pop("_outputs"), report)

    def round(self, kind, subs):
        assert kind == "execute"
        outputs = []
        for sub in subs:
            results = {}
            for node in sub["nodes"]:
                inputs = dict(sub["inputs"][node.node_id])
                for port in node.inputs:
                    if port.source.node_id in results:
                        inputs[port.name] = results[port.source.node_id].outputs[
                            port.source.port]
                if isinstance(node, (AiFilter, AiExtract)):
                    source = next(iter(inputs.values()))
                    ids = id_positions(source.ids["c"] if isinstance(source, Rows)
                                       else source).tolist()
                if isinstance(node, AiFilter):
                    self.filter_limits.append(sub["limit"])
                    groups = stop_groups(node, sub.get("query", {}).get(
                        "columns", {}).get("c"), ids)
                    if groups is not None:
                        firsts = {}
                        for document, group in zip(ids, groups):
                            firsts.setdefault(group, document)
                        ids = list(firsts.values())
                    if all(isinstance(member, AiFilter) for member in sub["nodes"]):
                        ids = ids[:sub["limit"]]
                    self.filtered.extend(ids)
                    result = NodeResult({
                        "rows:c": Rows({"c": pa.array(ids, pa.int32())}),
                        "filter_answers:c": filter_answer_table(
                            "c", ids, [0] * len(ids), [True] * len(ids)),
                    }, NodeMetrics(input_rows=len(ids), output_rows=len(ids)))
                elif isinstance(node, AiExtract):
                    self.extracted.setdefault(node.spec.name, []).extend(ids)
                    answers = [None if TEXTS[i] == "No date" else TEXTS[i] for i in ids]
                    result = NodeResult(
                        {"rows:c": extraction_rows(
                            "c", node.spec.name, node.spec.span_name, ids, answers,
                            [None if answer is None else (0, len(answer))
                             for answer in answers])},
                        NodeMetrics(input_rows=len(ids), output_rows=len(ids)))
                else:
                    result = self.session.registry.runtimes[node.runtime_key].execute(
                        node, inputs, None)
                results[node.node_id] = result
            outputs.append({"results": results, "retained": (),
                            "kv_manager": {"evicted_keys": 2},
                            "memory": {"peak_gib": 1.0}})
        return outputs


@pytest.mark.parametrize("condition,expected", [
    ("IS NOT NULL", [1, 3]), ("= 'absent'", []),
])
def test_an_extraction_after_a_test_receives_the_surviving_documents(
        session, condition, expected):
    query = session.sql(
        "SELECT c.id, AI.EXTRACT(c.text, "
        "{'ends': 'When does it end?', 'starts': 'When does it start?'}) AS lease "
        f"FROM contracts c WHERE lease.ends {condition}", order="as_written")
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute).collect()
    assert result.column("c.id").to_pylist() == expected
    assert result.column("lease").to_pylist() == [
        {"ends": TEXTS[i], "starts": TEXTS[i]} for i in expected]
    assert sorted(workers.extracted["lease.starts"]) == expected


@pytest.mark.parametrize("limit", [1, 3, 10])
def test_a_limit_waits_for_the_tests_after_the_filters(session, limit):
    query = session.sql(
        "SELECT c.id, AI.EXTRACT(c.text, {'ends': 'When does it end?'}) AS lease "
        "FROM contracts c WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text)) "
        f"AND lease.ends IS NOT NULL LIMIT {limit}", order="as_written")
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute).collect()
    assert result.column("c.id").to_pylist() == [1, 3][:limit]
    assert workers.filter_limits == [limit] * session.config.gpus


def test_a_filter_only_query_ends_at_the_limit_on_every_worker(session):
    query = session.sql(
        "SELECT c.id FROM contracts c "
        "WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text)) LIMIT 1")
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute).collect()
    assert result.num_rows == 1
    assert workers.filter_limits == [1] * session.config.gpus


def test_a_stop_key_keeps_its_group_on_one_worker_and_sums_the_counters(session):
    query = (session.docs("contracts").alias("c")
             .ai_filter(quail.prompt("Is {0} relevant?", quail.col("c.text")))
             .distinct().select("c.group"))
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute)
    assert result.to_rows() == [("lease",), ("sale",)]
    assert sorted(workers.filtered) == [0, 1]
    assert result.report["kv_manager"]["evicted_keys"] == 2 * session.config.gpus
    assert result.report["peak_gib"] == 1.0
