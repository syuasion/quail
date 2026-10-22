"""CPU checks for Quail's QUAIL-B runner: plans to queries, results to ids."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.oracle import Oracle, oracle_executor

import quail
from bench.quailb import (
    answer_oracle,
    build_query,
    join_anchors,
    prompt_pieces,
    queries,
    run_output,
)
from quail.api.execute import execute_query
from quail.baselines.nodes import RequestExecution
from quail.builtins import built_in_registry
from quail.catalog import DocumentProvider
from quail.execution.metrics import NodeMetrics, NodeResult, RunResult
from quail.execution.result import answer_table
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.logical.prompts import render_join_prompt_text
from quail.physical import decode_graph
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import QWEN3_4B_FP8
from quail_b.labels import GroundTruthCollection, PredicateLabels
from quail_b.minimum import DocumentTokens, token_metrics
from quail_b.predicates import PREDICATE_BY_KEY, PREDICATES, predicate_payload
from quail_b.prompts import DISCUSS_ASPECT, F1, F11, F13, REFUTE, SUPPORT
from quail_b.queries import get_query
from quail_b.rendering import render_filter_prompt, render_join_prompt
from quail_b.scoring import evaluate

# IMDB-3 is F1 on the reviews, then the DISCUSS_ASPECT join with the aspects
SPEC = get_query("IMDB-3")
CORPUS = {
    "reviews": pa.table({"id": ["r0", "r1"], "body": ["good film", "bad film"]}),
    "aspects": pa.table({"id": ["a0", "a1"], "aspect": ["acting", "ending"]}),
}


def _labels(key, template, answers, right_table=None):
    predicate = {
        "key": key, "template": template,
        "kind": "join" if right_table else "filter",
        "left_table": "reviews", "left_column": "body",
        "right_table": right_table, "right_column": right_table and "aspect",
    }
    return PredicateLabels(key, f"ls_{key}", predicate, answers,
                           {"qwen3-32b-fp8": len(answers)})


def _truth():
    predicates = (
        _labels("test.review.filter", F1, {("r0", None): True, ("r1", None): False}),
        _labels("test.review.aspect", DISCUSS_ASPECT, {
            ("r0", "a0"): True, ("r0", "a1"): False,
            ("r1", "a0"): False, ("r1", "a1"): True,
        }, "aspects"),
    )
    return GroundTruthCollection("gt_test", "c_test", 0.1, "qwen3-32b-fp8",
                                 {labels.key: labels for labels in predicates})


def fever_truth():
    """Labels for FEV-9 over a three claim, three evidence corpus."""
    corpus = {
        "claims": pa.table({"id": ["c0", "c1", "c2"],
                            "claim": ["person one", "person two", "a place"]}),
        "evidence": pa.table({"id": ["e0", "e1", "e2"],
                              "text": ["person one", "person two", "a place"]}),
    }
    true_pairs = {
        SUPPORT: {("c0", "e0"), ("c1", "e1"), ("c2", "e0"), ("c1", "e2")},
        REFUTE: {("c1", "e0"), ("c2", "e0"), ("c0", "e2")},
    }
    predicates = {}
    for spec in PREDICATES:
        if spec.template not in (F11, F13, SUPPORT, REFUTE):
            continue
        ids = corpus[spec.left_table]["id"].to_pylist()
        answers = (
            {(row_id, None): index < 2 for index, row_id in enumerate(ids)}
            if spec.kind == "filter" else {
                (left, right): (left, right) in true_pairs[spec.template]
                for left in ids
                for right in corpus[spec.right_table]["id"].to_pylist()
            }
        )
        predicates[spec.key] = PredicateLabels(
            spec.key, f"ls_{spec.key}", predicate_payload(spec), answers, {},
        )
    truth = GroundTruthCollection("gt_fev9", "c_fev9", 0.1, None, predicates)
    return corpus, truth


def _token_ids(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _config(backend):
    return EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                        device="h100-sxm")


def _session(tmp_path, backend="quail"):
    for name, table in CORPUS.items():
        pq.write_table(table, tmp_path / f"{name}.parquet")
    tokenizer = str.split if backend == "quail" else _token_ids
    sess = quail.Session(_config(backend), tokenizer=tokenizer)
    for name in CORPUS:
        sess.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col="id"))
    return sess


def _query(sess):
    # the same shape as SPEC, with the anchor pinned so the fake
    # executor below knows which alias the join keeps
    return (sess.docs("reviews").alias("r")
            .ai_filter(quail.prompt(F1, quail.col("r.body")))
            .ai_join(
                sess.docs("aspects").alias("a"),
                quail.prompt(DISCUSS_ASPECT, quail.col("r.body"),
                             quail.col("a.aspect")),
                anchor="r")
            .select("r.id", "a.id"))


_REPORT = {"wall_s": 2.0, "boot_s": 3.0, "boot_kind": "cold", "boot": {},
           "fresh_tokens": 2000, "peak_gib": 1.0}


def _execute(query, nodes, **metrics):
    """Run a query on a request backend whose node results nodes() plants."""
    def execute(request):
        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        outputs = export_physical_outputs(
            graph, RunResult(None, nodes(graph, request), NodeMetrics()))
        return PhysicalResponse(outputs, {**_REPORT, **metrics})

    return execute_query(query, physical_executor=execute)


def _run(query, join_answers):
    """Run the review query: r0 passes the filter; join_answers[a] is pair (r0, a)."""
    return execute_query(query, physical_executor=oracle_executor(Oracle(
        filters=lambda alias, document, position: document == 0,
        joins=lambda node, assignment: join_answers[assignment["a"]]),
        **_REPORT, store=None))


def _run_request_backend(query, join_answers):
    def nodes(graph, request):
        model = next(
            node for node in graph.nodes if isinstance(node, RequestExecution))
        filter_table = pa.table({
            "r": pa.array([0, 1], type=pa.int32()),
            "predicate": pa.array([0, 0], type=pa.int32()),
            "answer": pa.array([True, False], type=pa.bool_()),
        }).replace_schema_metadata({
            b"quail.kind": b"filter_answers",
            b"quail.alias": b"r",
        })
        join_table = answer_table(
            {"r": [0, 0], "a": [0, 1]},
            [bool(answer) for answer in join_answers],
            "join_answers",
            metadata={"anchor": "r", "partners": "a", "semantics": "full",
                      "written_position": 0},
        )
        true_aspects = [
            index for index, answer in enumerate(join_answers) if answer
        ]
        return {model.node_id: NodeResult({
            "rows:r": [0] if true_aspects else [],
            "rows:a": true_aspects,
            "filter_answers:r": filter_table,
            "join_answers:0": join_table,
        })}

    return _execute(query, nodes, cached_tokens=0)


TRACES = pa.table({
    "id": [f"s{i}" for i in range(6)],
    "trace": [f"trace {i} words" for i in range(6)],
    "trajectory_id": ["A", "A", "B", "B", "B", "C"],
    "turn_index": [5, 10, 10, 15, 20, 5],
    "token_count": [2000, 4000, 5000, 7000, 9000, 1000],
})


def test_relational_plans_build_and_run(tmp_path):
    pq.write_table(TRACES, tmp_path / "agent_traces.parquet")
    with quail.Session(_config("quail"), tokenizer=str.split) as sess:
        sess.register("agent_traces", DocumentProvider.from_parquet(
            str(tmp_path / "agent_traces.parquet"), id_col="id"))
        logical = build_query(sess, get_query("REL-AGENT-6")).logical
        root = logical.root
        assert root.input.keys == ("t.trajectory_id",)
        assert [str(a) for a in root.input.aggregates] == [
            "fixes = count(*)", "first_fix = min(t.turn_index)",
            "longest = max(t.token_count)"]
        assert [str(t) for t in root.input.having] == ["fixes >= 2"]
        assert [str(key) for key in root.order] == [
            "first_fix ASC NULLS LAST", "t.trajectory_id ASC NULLS LAST"]
        assert (root.limit, root.offset) == (50, 0)
        scan = next(node for node in logical.walk()
                    if type(node).__name__ == "Scan")
        tested = build_query(sess, get_query("REL-AGENT-1")).logical
        scan = next(node for node in tested.walk()
                    if type(node).__name__ == "Scan")
        assert [str(p) for p in scan.predicates] == [
            "t.turn_index >= 10", "t.token_count <= 6000"]
        paged = build_query(sess, get_query("REL-AGENT-2")).logical.root
        assert (paged.limit, paged.offset) == (10, 10)
        top = build_query(sess, get_query("REL-AGENT-4")).logical.root
        assert [column.name for column in top.input.columns
                if hasattr(column, "name")] == ["recovered_score"]
        assert top.order[0].name == "recovered_score" and top.limit == 20
        # an aggregate with no measures is a DISTINCT, which the filter
        # stops per trajectory
        query = build_query(sess, get_query("REL-AGENT-3"))
        distinct = query.logical.root
        assert distinct.distinct and [
            column.column for column in distinct.input.columns
        ] == ["trajectory_id"]
        assert "stop per key: trajectory_id" in query.explain()

        # the fixes: s0, s1 (A), s2, s4 (B), s5 (C): A and B twice
        oracle = Oracle(filters=lambda alias, document, position: document != 3)
        query = build_query(sess, get_query("REL-AGENT-6"))
        result = execute_query(query, physical_executor=oracle_executor(
            oracle, **_REPORT, store=None))
        output = run_output(result, get_query("REL-AGENT-6").info,
                            {"agent_traces": TRACES})
    assert output.rows.to_pydict() == {
        "trajectory_id": ["A", "B"], "fixes": [2, 2], "first_fix": [5, 10],
        "longest": [4000, 9000]}
    assert output.filter_answers["filter-1"].column("t").to_pylist() == [
        "s0", "s1", "s2", "s3", "s4", "s5"]


def test_a_score_query_reports_every_scored_document_and_its_pieces(tmp_path):

    spec = get_query("REL-AGENT-4")
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]

    oracle = Oracle(scores=lambda spec, ids: scores[ids[0]])

    with _session(tmp_path) as sess:
        sess.register("agent_traces", DocumentProvider.from_table(
            TRACES, id_col="id"))
        query = build_query(sess, spec)
        result = execute_query(query, physical_executor=oracle_executor(
            oracle, **_REPORT, store=None))
        output = run_output(result, spec.info, {"agent_traces": TRACES})
        output.prompt_pieces = prompt_pieces(query, spec.info, {})
    # the rows keep the fetch; the answers keep every scored document
    assert output.rows.num_rows == 6
    answers = output.score_answers["score-1"]
    assert answers.column("t").to_pylist() == [
        "s0", "s1", "s2", "s3", "s4", "s5"]
    assert answers.column("score").type == pa.float64()
    assert [round(score, 3) for score in answers.column("score").to_pylist()
            ] == scores
    # the word tokenizer keeps the pieces readable: the score is laid out
    # as the AI.IF prompt, so its head is the preamble
    pieces = output.prompt_pieces
    assert pieces["preamble"] == (sess.model.turn_prefix + quail.SHARED_PRE).split()
    assert [item["id"] for item in pieces["scores"]] == ["score-1"]
    assert pieces["scores"][0]["tail"][0] == "Evaluate"
    assert not pieces["filters"] and not pieces["classifies"]


CONTRACTS = pa.table({
    "id": [f"ct{i}" for i in range(3)],
    "title": ["A", "B", "C"],
    "text": ["audit yearly", "no audit", "audit at will"],
    "token_count": pa.array([2, 2, 3], pa.int32()),
})


def test_an_extract_query_reports_every_answer_and_its_pieces(tmp_path):

    spec = get_query("CUAD-7")

    oracle = Oracle(
        filters=lambda alias, document, position: document != 1,
        answers=lambda spec, document: "audit yearly" if document == 0 else None)

    with _session(tmp_path) as sess:
        sess.register("contracts", DocumentProvider.from_table(
            CONTRACTS, id_col="id"))
        query = build_query(sess, spec)
        result = execute_query(query, physical_executor=oracle_executor(
            oracle, **_REPORT, store=None))
        output = run_output(result, spec.info, {"contracts": CONTRACTS})
        output.prompt_pieces = prompt_pieces(query, spec.info, {})
        turn = sess.model.turn
    assert output.filter_answers["filter-1"].to_pydict() == {
        "c": ["ct0", "ct1", "ct2"], "answer": [True, False, True]}
    assert output.extract_answers["extract-1"].to_pydict() == {
        "c": ["ct0", "ct2"], "answer": ["audit yearly", None]}
    # the answer comes back as a field of the plan's answer struct
    assert output.rows.to_pydict() == {
        "c": ["ct0", "ct2"],
        "contract": [{"audit_rights": "audit yearly"}, {"audit_rights": None}]}
    assert turn[0].startswith("<|im_start|>system\n")
    # the estimate's oracle returns each extraction's reference spans
    (operator,) = spec.info.extracts
    (item,) = [item for item in PREDICATE_BY_KEY.values()
               if item.template == operator.prompt]
    truth = GroundTruthCollection("gt_extract", "c_extract", 0.1, None, {
        item.key: PredicateLabels(item.key, f"ls_{item.key}", predicate_payload(item),
                                  {("ct0", None): ["audit yearly"]}, {})})
    (node,) = query.logical.operators().extracts
    answer = answer_oracle(truth, {"contracts": CONTRACTS}, turn)
    assert answer(node.call.prompt, {"c": 0}) == ["audit yearly"]
    pieces = output.prompt_pieces
    assert pieces is not None
    assert [piece["id"] for piece in pieces["filters"]] == ["filter-1"]
    assert [piece["id"] for piece in pieces["extracts"]] == ["extract-1"]
    assert pieces["preamble"] == list(sess.tokenizer(turn[0] + "DOCUMENT:\n"))

    alone = get_query("CUAD-1")
    with _session(tmp_path) as sess:
        sess.register("contracts", DocumentProvider.from_table(
            CONTRACTS, id_col="id"))
        query = build_query(sess, alone)
        pieces = prompt_pieces(query, alone.info, {})
    assert pieces["preamble"] == (turn[0] + quail.SHARED_PRE).split()
    assert [item["id"] for item in pieces["extracts"]] == ["extract-1"]
    assert pieces["extracts"][0]["tail"][:2] == ["Question:", "Which"]
    assert pieces["extracts"][0]["tail"][-1] == "Phrase:"


def test_stock_vllm_reports_an_extraction_and_its_pieces(tmp_path):
    spec = get_query("CUAD-1")
    span_type = pa.struct([("start", pa.int64()), ("end", pa.int64())])

    def nodes(graph, request):
        model = next(
            node for node in graph.nodes if isinstance(node, RequestExecution))
        return {model.node_id: NodeResult({
            "rows:c": [0, 1, 2],
            "extract_answers:contract.governing_law": pa.table({
                "c": pa.array([0, 1, 2], pa.int32()),
                "contract.governing_law": pa.array(
                    ["audit yearly", None, "will"], pa.string()),
                "contract.governing_law_span": pa.array(
                    [{"start": 0, "end": 12}, None, {"start": 9, "end": 13}],
                    span_type),
            }),
        })}

    with _session(tmp_path, backend="stock_vllm") as sess:
        sess.register("contracts", DocumentProvider.from_table(
            CONTRACTS, id_col="id"))
        query = build_query(sess, spec)
        result = _execute(query, nodes, cached_tokens=0)
        output = run_output(result, spec.info, {"contracts": CONTRACTS})
        pieces = prompt_pieces(query, spec.info, {})
        turn = sess.model.turn
    assert output.extract_answers["extract-1"].to_pydict() == {
        "c": ["ct0", "ct1", "ct2"], "answer": ["audit yearly", None, "will"]}
    assert output.rows.column_names == ["c", "contract"]
    # the request node carries the extraction's own head and tail
    assert pieces["preamble"] == _token_ids(turn[0] + quail.SHARED_PRE)
    assert [item["id"] for item in pieces["extracts"]] == ["extract-1"]
    assert pieces["extracts"][0]["tail"][-1] == _token_ids(":")[-1]


def test_benchmark_results_and_scoring(tmp_path):
    with _session(tmp_path) as sess:
        quail_result = _run(_query(sess), [1, 0])
    output = run_output(quail_result, SPEC.info, CORPUS)

    assert output.filter_answers["filter-1"].to_pydict() == {
        "r": ["r0", "r1"], "answer": [True, False]}
    assert output.join_answers["join-1"].to_pydict() == {
        "r": ["r0", "r0"], "a": ["a0", "a1"], "answer": [True, False]}
    assert output.rows.to_pydict() == {"r": ["r0"], "a": ["a0"]}
    evaluation = evaluate(SPEC, output, _truth(), CORPUS)
    assert evaluation["answer_accuracy"]["accuracy"] == 1.0
    assert evaluation["output_accuracy"]["exact_match"] is True

    with _session(tmp_path, backend="stock_vllm") as sess:
        query = _query(sess)
        result = _run_request_backend(query, [0, 0])
        output = run_output(result, SPEC.info, CORPUS)
        output.prompt_pieces = prompt_pieces(query, SPEC.info,
                                             join_anchors(result))
        tokenizer_name = sess.model.hf_name

    evaluation = evaluate(SPEC, output, _truth(), CORPUS)
    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert [item["op"] for item in evaluation["per_predicate"]] == ["filter", "join"]
    assert output.rows.num_rows == 0
    pieces = output.prompt_pieces
    assert pieces["tokenizer"] == tokenizer_name
    # the model's user turn opens ahead of the document preamble
    assert pieces["preamble"] == _token_ids(
        sess.model.turn_prefix + quail.SHARED_PRE)
    assert [item["id"] for item in pieces["filters"]] == ["filter-1"]
    assert [(item["id"], item["anchor"]) for item in pieces["joins"]] == [
        ("join-1", "r")]
    stores = {tokenizer_name: DocumentTokens(
        CORPUS, lambda texts: [_token_ids(text) for text in texts])}
    measured = token_metrics(SPEC, output, CORPUS, stores)
    assert measured["minimum_tokens"] > 0
    assert measured["regret_tokens"] == 2000 - measured["minimum_tokens"]


def test_benchmark_query_prompts_labels_and_chat_rendering():
    for backend in ("quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        corpus, truth = fever_truth()
        spec = get_query("FEV-9")
        info = spec.info
        answer = answer_oracle(truth, corpus)
        with quail.Session(
            _config(backend), tokenizer=lambda text: list(text.encode("utf-8"))
        ) as session:
            for name, table in corpus.items():
                session.register(name, DocumentProvider.from_table(table, id_col="id"))
            query = build_query(session, spec)
            assert not isinstance(query.plan(), Refusal), backend
            operators = query.logical.operators()
            filters, joins = operators.filters, operators.joins
            tables = {relation.alias: relation.table for relation in info.relations}
            assert [scan.alias for scan in operators.scans] == list(tables), backend
            assert {alias: [p.prompt.template for p in sequence]
                    for alias, sequence in filters.items()} == {
                alias: [
                    quail.bind_prompt(item.prompt, (quail.ColumnRef(
                        alias, tables[alias],
                        info.relation(alias).text_column),)).template
                    for item in info.filters if item.relation == alias]
                for alias in tables if any(
                    item.relation == alias for item in info.filters)}, backend
            assert [tuple(arg.alias for arg in join.prompt.args)
                    for join in joins] == [join.relations for join in info.joins]
            # c0 is about a person and e0 supports it; c2 is not about a person
            assert answer(filters["c1"][0].prompt, {"c1": 0}) is True, backend
            assert answer(filters["c1"][0].prompt, {"c1": 2}) is False, backend
            assert answer(joins[0].prompt, {"c1": 0, "e1": 0}) is True, backend
            assert answer(joins[0].prompt, {"c1": 0, "e1": 1}) is False, backend
            # the same labels drive the speed of light estimate
            estimate = quail.speed_of_light_estimate(query, answer)
            assert estimate.post_filter_counts == {
                "c1": 2, "e1": 2, "c2": 2, "e2": 2}, backend
            assert len(estimate.join_stages) == 3, backend
            assert queries(session)["FEV-9"][0] == spec.description, backend
            # only the FEVER tables are registered
            assert all(query_id.startswith("FEV-")
                       for query_id in queries(session)), backend
    # Saved reference prompts omit the model turn.
    turn = QWEN3_4B_FP8.turn
    for spec in PREDICATES:
        left = quail.ColumnRef("left", spec.left_table, spec.left_column)
        if spec.kind == "filter":
            prompt = quail.bind_prompt(spec.template, (left,), turn=turn)
            assert prompt.tail.startswith("{0}")
            expected = (prompt.preamble + "doc one"
                        + prompt.tail.replace("{0}", "", 1))
            assert (turn[0] + render_filter_prompt(spec.template, "doc one")
                    + turn[1]) == expected
        else:
            right = quail.ColumnRef("right", spec.right_table, spec.right_column)
            prompt = quail.bind_join_prompt(spec.template, (left, right), turn=turn)
            documents = ("doc one", "doc two")
            for anchor in (0, 1):
                assert (turn[0] + render_join_prompt(spec.template, documents, anchor)
                        + turn[1]) == (
                    render_join_prompt_text(prompt, documents, anchor))


@pytest.mark.parametrize("case", ("IMDB-11", "IMDB-14", "IMDB-15"))
def test_classification_pieces_wrap_the_named_reference_prompt(tmp_path, case):
    from quail_b.minimum import validate_prompt_pieces
    from quail_b.rendering import render_classify_prompt

    # IMDB-11 only projects labels; IMDB-14 also filters them;
    # IMDB-15 includes a classification over joined rows.
    spec = get_query(case)
    info = spec.info
    with _session(tmp_path, backend="stock_vllm") as sess:
        # the byte tokenizer, so the pieces decode back to text
        query = build_query(sess, spec)
        pieces = prompt_pieces(query, info, {0: "r"})
    # the estimate's oracle returns each classification's reference label
    calls = [call for call, _ in query.logical.operators().labels.calls]
    predicates = {}
    for operator in info.classifies:
        (item,) = [item for item in PREDICATE_BY_KEY.values()
                   if item.template == operator.prompt]
        partners = (CORPUS[item.right_table]["id"].to_pylist()
                    if item.right_table else [None])
        predicates[item.key] = PredicateLabels(
            item.key, f"ls_{item.key}", predicate_payload(item),
            {(left, right): operator.labels[-1]
             for left in CORPUS["reviews"]["id"].to_pylist()
             for right in partners}, {})
    answer = answer_oracle(
        GroundTruthCollection("gt_classify", "c_classify", 0.1, None, predicates),
        CORPUS)
    assert len(calls) == len(info.classifies)
    for call, operator in zip(calls, info.classifies):
        assignment = {alias: 1 for alias in call.aliases()}
        assert answer(call.prompt, assignment) == operator.labels[-1]
    assert validate_prompt_pieces(spec, pieces)["classifies"] == pieces[
        "classifies"]

    def text(ids):
        return bytes(token - 1 for token in ids).decode("utf-8")

    assert len(pieces["classifies"]) == len(info.classifies)
    for operator, piece in zip(info.classifies, pieces["classifies"]):
        assert piece["id"] == operator.id
        rendered = text(pieces["preamble"]) + "good film"
        partner = None
        if operator.partner is not None:
            assert piece["anchor"] == operator.relation
            partner = "the acting"
            rendered += text(piece["frame"]) + text(piece["label"]) + partner
        expected = render_classify_prompt(
            operator.prompt, "good film", operator.labels,
            operator.descriptions, partner=partner)
        assert rendered + text(piece["tail"]) == (
            sess.model.turn_prefix + expected + sess.model.turn_suffix)
