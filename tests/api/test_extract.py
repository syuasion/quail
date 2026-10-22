"""AI.EXTRACT: the operator surface, its prompt, and its plan."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers import baselines as request_tests
from helpers.fakes import letter_tokens
from helpers.oracle import Oracle, finish, oracle_executor

import quail
from quail.baselines.nodes import RequestExecution, RequestExtractSpec
from quail.catalog import DocumentProvider
from quail.execution.rows import Rows, extraction_rows
from quail.execution.spans import locate_span, read_phrase
from quail.logical import (
    CompileError,
    ExtractStruct,
    extracted_above_joins,
)
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    ExtractSpec,
    Filter,
    PortRef,
    Project,
    decode_graph,
    encode_graph,
)
from quail.physical.codec import NodeCodec, decode_value, encode_value
from quail.physical.pipelines import build_pipelines
from quail.planner.plan import EngineConfig, Refusal

QUESTION = "When does the lease end?"
LEASE = "Lease ends on March 3, 2027.\nRent is $2,000."

@pytest.fixture()
def session(tmp_path):
    path = tmp_path / "contracts.parquet"
    pq.write_table(pa.table({
        "id": [1, 2], "text": [LEASE, "No dates here."],
    }), path)
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
    value.register("contracts",
                   DocumentProvider.from_parquet(str(path), id_col="id"))
    yield value
    value.close()


def _ends(session):
    return (session.docs("contracts").alias("c")
            .ai_extract("c.text", {"ends": QUESTION}, name="lease")
            .select("c.id", "lease"))


def test_extract_prefix_sharing_is_planned_from_exact_document_tokens():
    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens,
    ) as session:
        shared = "The following agreement is made between the parties. " * 4
        session.register("contracts", DocumentProvider.from_table(pa.table({
            "id": [1, 2, 3],
            "text": [shared + LEASE, shared + "No dates here.", "Unrelated."],
        }), id_col="id"))
        query = _ends(session)
        query.plan()
        query.wait_for_tokens()
        plan = query.plan()
        (node,) = [node for node in plan.nodes if isinstance(node, AiExtract)]
        assert node.spec.share_prefixes


def test_sql_and_builder_extract_the_same_answer_columns(session):
    built = _ends(session)
    for dialect in ('snowflake', 'bq'):
        query = session.sql(
            f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease "
            f"FROM contracts c", dialect=dialect)
        assert query.logical == built.logical


def test_planning_lowers_the_extraction_prompt_once(session, monkeypatch):
    query = _ends(session)
    tokenizer = session.tokenizer
    seen = []

    def count_prompt(text):
        if QUESTION in text:
            seen.append(text)
        return tokenizer(text)

    monkeypatch.setattr(session, "_tokenizer", count_prompt)
    assert not isinstance(query.plan(), Refusal)
    assert len(seen) == 1, seen


def test_extract_follows_the_filters_and_labels_of_its_table(session):
    query = (session.docs("contracts").alias("c")
             .ai_filter(quail.prompt("Is {0} a lease?", quail.col("c.text")))
             .ai_classify(quail.prompt("{0}", quail.col("c.text")),
                          ["lease", "deed"], name="kind")
             .label_in("kind", ["lease"], selectivity=0.5)
             .ai_extract("c.text", {"ends": QUESTION}, name="lease")
             .select("c.id", "kind", "lease"))
    plan = query.plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    (test,) = [n for n in plan.nodes if isinstance(n, Filter)]
    assert node.inputs[0].source.node_id == test.node_id
    # half the documents pass the label filter
    assert node.spec.expected_rows == pytest.approx(2 * 0.2 * 0.5)
    assert plan.nodes[-1].columns == (
        "c.id", "kind", "lease.ends", "lease.ends_span")


def test_extractions_of_a_table_form_one_pipeline_and_are_priced_as_resident(
        session):
    query = (session.docs("contracts").alias("c")
             .ai_extract("c.text", {"ends": QUESTION}, name="lease")
             .ai_extract("c.text", {"signer": "Who signs?"}, name="signing")
             .ai_extract("c.text", {"place": "Where?"}, name="venue")
             .select("c.id", "lease", "signing", "venue"))
    plan = query.plan()
    extracts = [n for n in plan.nodes if isinstance(n, AiExtract)]
    # the extractions show the documents under their prompt's head,
    # which the executor keys their KV by
    assert extracts[0].document_head == tuple(extracts[0].spec.prompt_token_parts[0])
    pipelines = build_pipelines(plan.graph)
    # the three run as one pipeline: the later ones read the KV the
    # first leaves and are priced without reading the document
    assert tuple(member.node_id for member in pipelines[extracts[0].node_id]) == tuple(
        extract.node_id for extract in extracts)
    assert extracts[1].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    assert extracts[2].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    different = replace(extracts[1], spec=replace(
        extracts[1].spec, draft_tokens=2))
    graph = replace(plan.graph, nodes=tuple(
        different if node.node_id == different.node_id else node
        for node in plan.nodes))
    separated = build_pipelines(graph)
    assert tuple(
        member.node_id for member in separated[extracts[0].node_id]) == (
            extracts[0].node_id,)
    assert tuple(
        member.node_id for member in separated[extracts[1].node_id]) == (
            extracts[1].node_id,)
    # a plain extraction shows the document under the query's shared
    # head, as the filter does, so it continues the filter's pipeline
    # and asks its question of the KV the filter wrote
    filtered = (session.docs("contracts").alias("c")
                .ai_filter(quail.prompt("Is {0} a lease?", quail.col("c.text")))
                .ai_extract("c.text", {"ends": QUESTION}, name="lease")
                .select("c.id", "lease"))
    plan = filtered.plan()
    (extract,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    (ai_filter,) = [n for n in plan.nodes if isinstance(n, AiFilter)]
    assert extract.document_head == ai_filter.document_head != ()
    pipelines = build_pipelines(plan.graph)
    assert tuple(member.node_id for member in pipelines[ai_filter.node_id]) == (
        ai_filter.node_id, extract.node_id)
    # a classification of the table writes the documents under the
    # same head, so the extraction continues its pipeline too
    classified = (session.docs("contracts").alias("c")
                  .ai_classify(quail.prompt("Which kind is {0}?", quail.col("c.text")),
                               ["lease", "license"], name="kind")
                  .ai_extract("c.text", {"ends": QUESTION}, name="lease")
                  .select("c.id", "kind", "lease"))
    plan = classified.plan()
    (extract,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert (tuple(classify.spec.prompt_token_parts[0])
            == tuple(extract.spec.prompt_token_parts[0]))
    pipelines = build_pipelines(plan.graph)
    assert tuple(member.node_id for member in pipelines[classify.node_id]) == (
        classify.node_id, extract.node_id)


def _candidates(session, query):
    """Return the planner's refined candidates for a planned query."""
    from quail.planner.optimize import search_candidates
    from quail.planner.optimizer.context import PlanningContext

    query.plan()
    context = PlanningContext(
        model=session.model, device=session.device, gpus=1,
        document_tokens=query._document_tokens, backend="quail",
        tokenizer=session.tokenizer)
    return search_candidates(query._logical_input, context, session.registry)


def test_extract_of_a_joined_table_is_offered_after_the_join(session, tmp_path):
    path = tmp_path / "questions.parquet"
    pq.write_table(pa.table({"id": [1], "ask": ["end dates"]}), path)
    session.register("questions",
                     DocumentProvider.from_parquet(str(path), id_col="id"))
    query = session.sql(
        f"SELECT c.id, q.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) "
        f"AS lease "
        f"FROM contracts c JOIN questions q "
        f"ON AI_FILTER(PROMPT('Does {{0}} answer {{1}}?', c.text, q.ask))")
    assert extracted_above_joins(query.logical.root) == frozenset()
    plan = query.plan()
    assert not isinstance(plan, Refusal)
    written, lifted = _candidates(session, query)
    # The logical rule offers extraction after the join, over the
    # documents the join matched: a fifth of them by default, so the
    # lifted plan is cheaper and wins
    assert extracted_above_joins(written.logical_plan.root) == frozenset()
    assert lifted.logical_plan == query.logical
    assert extracted_above_joins(lifted.logical_plan.root) == {"c"}
    assert lifted.estimated_seconds < written.estimated_seconds
    (join,) = [n for n in written.plan.nodes if isinstance(n, AiJoin)]
    assert ("ai-extract:0", "rows:c") in [
        (port.source.node_id, port.source.port) for port in join.inputs]
    (node,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    assert node.spec.expected_rows == pytest.approx(2 * 0.2)
    (before,) = [n for n in written.plan.nodes if isinstance(n, AiExtract)]
    assert before.spec.expected_rows == 2
    assert node.spec.estimated_seconds < before.spec.estimated_seconds
    assert plan.nodes[-1].columns == (
        "c.id", "q.id", "lease.ends", "lease.ends_span")


def test_extract_errors_and_refusals(session):
    sql = session.sql
    fields = f"{{'ends': '{QUESTION}'}}"
    for bad, message in (
            (f"SELECT c.id FROM contracts c WHERE AI.EXTRACT(c.text, "
             f"{fields}) = 'x'", "not a predicate"),
            (f"SELECT c.id, AI.EXTRACT('{QUESTION}', c.text) AS x "
             f"FROM contracts c", "document column"),
            (f"SELECT c.id, AI.EXTRACT(c.text, '{QUESTION}') AS x "
             f"FROM contracts c", "object of field questions"),
            ("SELECT c.id, AI.EXTRACT(c.text) AS x FROM contracts c",
             "object of field questions"),
            (f"SELECT c.id, AI.EXTRACT(c.text, {fields}, {{'trim': false}}) "
             f"AS x FROM contracts c", "takes a document column"),
            (f"SELECT c.id, UPPER(AI.EXTRACT(c.text, {fields})) AS x "
             f"FROM contracts c", "direct expressions")):
        with pytest.raises(CompileError, match=message):
            sql(bad)
    builder = session.docs("contracts").alias("c")
    with pytest.raises(CompileError, match="mapping from field names"):
        builder.ai_extract("c.text", QUESTION, name="ends")
    with pytest.raises(CompileError, match="without a dot"):
        builder.ai_extract("c.text", {"ends": QUESTION}, name="c.ends")
    with pytest.raises(CompileError, match="already used"):
        builder.ai_extract("c.text", {"ends": QUESTION}, name="c")
    with pytest.raises(CompileError, match="already used"):
        (builder.ai_extract("c.text", {"ends": QUESTION}, name="lease")
         .ai_classify(quail.prompt("{0}", quail.col("c.text")), ["a", "b"],
                      name="lease_span"))
    # the speed of light estimate prices an extraction as one question
    # per document, then its reference answer verified after the cue
    estimate = quail.speed_of_light_estimate(_ends(session), lambda *_: "March")
    (stage,) = estimate.filter_stages
    assert (stage["operator"], stage["alias"], stage["name"]) == (
        "AI.EXTRACT", "c", "lease.ends")
    assert (stage["evaluated"], stage["passed"]) == (2, 2)
    assert estimate.extraction_evaluations == 2
    assert estimate.seconds > 0 and "extraction" in estimate.assumptions()
    assert estimate.as_dict()["extraction_evaluations"] == 2
    # SGLang returns no decoded text, so it does not run it
    sglang = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                        backend="pipelined_sglang"),
                           tokenizer=letter_tokens)
    sglang.register("contracts", session.catalog.get("contracts"))
    refused = _ends(sglang).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "extract_needs_quail_backend"
    sglang.close()
    # a reranker gives no next-token probabilities over the document,
    # nor does a decision model or a diffusion model with a canvas
    reranker = quail.Session(
        EngineConfig(model="qwen3-reranker-0.6b-bf16", device="h100-sxm"),
        tokenizer=letter_tokens)
    reranker.register("contracts", session.catalog.get("contracts"))
    refused = _ends(reranker).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "reranker_only_scores"
    reranker.close()


def test_span_helpers_read_locate_and_widen_an_answer():
    assert read_phrase(' the "Premises" ') == 'the "Premises"'
    assert read_phrase("March 3, 2027") == "March 3, 2027"
    for empty in ("", " ", "\n"):
        assert read_phrase(empty) is None
    assert locate_span(LEASE, "March 3, 2027") == (14, 27)
    # case and runs of whitespace are forgiven; the first occurrence wins
    assert locate_span(LEASE, "RENT   is") == (29, 36)
    assert locate_span("a b a b", "a b") == (0, 3)
    assert locate_span(LEASE, "April") is None
    assert locate_span(LEASE, "   ") is None


@pytest.mark.parametrize("filtered", [False, True])
def test_score_and_extract_return_both_columns(session, monkeypatch, filtered):
    oracle = Oracle(answers=lambda spec, document: (
        "March 3, 2027" if document == 0 else None))
    query = session.sql(
        "SELECT c.id, AI.SCORE(PROMPT('Is this a lease? {0}', c.text)) AS score, "
        f"AI.EXTRACT(c.text, {{'ends': '{QUESTION}'}}) AS lease FROM contracts c "
        + ("WHERE AI.SCORE(PROMPT('Is this a lease? {0}', c.text)) > 0.15"
           if filtered else ""))
    result = finish(query, oracle)
    rows = result.collect()
    assert rows.column("score").type == pa.float64()
    expected = [
        {"c.id": 1, "score": 0.1, "lease": {"ends": "March 3, 2027"},
         "lease_span": {"ends": {"start": 14, "end": 27}}},
        {"c.id": 2, "score": 0.2, "lease": {"ends": None},
         "lease_span": {"ends": None}},
    ]
    assert rows.to_pylist() == (expected[1:] if filtered else expected)


class _PhraseClient(request_tests._Client):
    """Writes each document's answer from the text its prompt tokens spell."""

    def __init__(self):
        self.params = []

    def decode_params(self, max_tokens, stop=None):
        self.params.append((max_tokens, stop))
        return ("decode", max_tokens)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert sampling_params == ("decode", self.params[-1][0])
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            spelled = bytes(token - 1 for token in ids if 1 <= token <= 256)
            text = spelled.decode("utf-8", errors="ignore")
            if "March" in text:
                answer, tokens = "March 3, 2027", [1, 2, 3, 4]
            elif "five" in text:
                answer, tokens = "FIVE   years", [1, 2]
            elif "absent" in text:
                answer, tokens = "a phrase the document lacks", [1, 2, 3]
            else:
                # an empty phrase: the model ends its turn at once
                answer, tokens = "", [1]
            outputs.append(SimpleNamespace(
                prompt_token_ids=ids, num_cached_tokens=0,
                outputs=[SimpleNamespace(text=answer, token_ids=tokens)]))
        return outputs


class _TextPhraseClient(_PhraseClient):
    """Takes text prompts, as vLLM does, and answers from their words."""

    accepts_text = True

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.prompts = list(prompts)
        return super().generate(
            [{"prompt_token_ids": request_tests._tokens(prompt)}
             for prompt in prompts], sampling_params, use_tqdm)


TEXTS = [LEASE, "No dates here.", "Term: five years\nmore text",
         "the absent one"]


def test_request_backend_decodes_and_locates_one_answer_per_document():
    limit = 7
    documents = {"c": [request_tests._tokens(text) for text in TEXTS]}
    specs = (
        RequestExtractSpec(max_answer_tokens=limit,
                           alias="c", output="ends", question=QUESTION,
                           tail_token_ids=(90,)),
    )
    node = RequestExecution(
        node_id="request-model", backend_name="stock_vllm", aliases=("c",),
        preamble_token_ids=(3,), extracts=specs)
    client = _PhraseClient()
    execution = request_tests._execution(
        documents, client=client, document_texts={"c": TEXTS})
    result = execution.execute(node, {"input:0": [0, 1, 2, 3]})
    assert client.params == [(limit, None)]
    answers = result.outputs["extract_answers:ends"].to_pydict()
    assert answers == {
        "c": [0, 1, 2, 3],
        "ends": ["March 3, 2027", None, "five years",
                 "a phrase the document lacks"],
        "ends_span": [{"start": 14, "end": 27}, None, {"start": 6, "end": 16},
                      None]}
    # an extraction keeps every document
    assert result.outputs["rows:c"].ids["c"].to_pylist() == [0, 1, 2, 3]
    step, = result.metrics.extension["steps"]
    assert (step["kind"], step["output"], step["num_in"], step["num_out"]) == (
        "extract", "ends", 4, 3)
    assert (step["requests"], step["unanswered"], step["unaligned"]) == (4, 1, 1)
    assert step["generated_tokens"] == 4 + 1 + 2 + 3
    # the preamble, document, and tail of each prompt, and each
    # answer's tokens but the last, fed back
    assert step["fresh_tokens"] == sum(
        2 + len(ids) for ids in documents["c"]) + (3 + 0 + 1 + 2)
    prompt_tokens = sum(2 + len(ids) for ids in documents["c"])
    assert step["prompt_tokens"] == prompt_tokens
    assert result.metrics.extension["prompt_tokens"] == prompt_tokens
    assert result.metrics.evaluated_documents == 4
    # an engine that takes text gets the preamble, the document, and
    # the tail's text, tokenized by the engine itself
    texted = _TextPhraseClient()
    node = replace(node, preamble_text="DOCUMENT:\n",
                   extracts=(replace(specs[0], tail_text="\nQ?\nPhrase:"),))
    result = request_tests._execution(
        documents, client=texted, document_texts={"c": TEXTS}).execute(node, {})
    assert texted.prompts == ["DOCUMENT:\n" + text + "\nQ?\nPhrase:"
                              for text in TEXTS]
    assert result.outputs["extract_answers:ends"].to_pydict()["ends"] == [
        "March 3, 2027", None, "five years", "a phrase the document lacks"]


def test_stock_vllm_plans_and_runs_an_extraction(monkeypatch):
    session = request_tests._session(
        "stock_vllm", contracts={"id": ["a", "b", "c", "d"], "text": TEXTS})
    query = session.sql(
        f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}', "
        f"'term': 'How long is the term?'}}) AS lease FROM contracts c")
    boot = {"client": _PhraseClient(), "sampling_params": object(),
            "capacity": request_tests.CAPACITY}
    monkeypatch.setattr(
        "quail.baselines.vllm.VLLMEngine.boot",
        lambda self, spec, allowed_ids, *, resources: (
            boot, {"kind": "cold", "boot_s": 0.1}))
    result = request_tests._run_stock_vllm(session, query)
    rows = result.collect()
    assert rows.column_names == ["c.id", "lease", "lease_span"]
    assert rows.column("c.id").to_pylist() == ["a", "b", "c", "d"]
    # the fake client answers from the document, whatever the question
    answers = ["March 3, 2027", None, "five years", "a phrase the document lacks"]
    assert rows.column("lease").to_pylist() == [
        {"ends": answer, "term": answer} for answer in answers]
    spans = [{"start": 14, "end": 27}, None, {"start": 6, "end": 16}, None]
    assert rows.column("lease_span").to_pylist() == [
        {"ends": span, "term": span} for span in spans]
    answers = result.answer_tables["extracts"]["lease.ends"]
    assert answers.column_names == ["c", "lease.ends", "lease.ends_span"]
    assert answers.num_rows == 4
    steps = result.report["backend_metrics"]["steps"]
    assert [(step["kind"], step["requests"]) for step in steps] == [
        ("extract", 4), ("extract", 4)]
    session.close()


def test_extraction_settings_survive_plan_serialization(session):
    query = _ends(session)
    node = next(node for node in query.plan().nodes if isinstance(node, AiExtract))
    settings = dict(max_answer_tokens=7, draft_tokens=2, top_tokens=32,
                    first_window_chars=16, window_chars=96,
                    draft_policy="adaptive", expected_answer_tokens=9,
                    round_cost_per_token=0.25)
    spec = replace(node.spec, **settings)
    assert decode_value(ExtractSpec, encode_value(spec)) == spec
    # the planner set the round's KV read from the model and device
    assert node.spec.round_cost_per_token > 0
    request = RequestExtractSpec(alias="c", output="ends", question=QUESTION,
                                 tail_token_ids=(1, 2), max_answer_tokens=7)
    assert decode_value(RequestExtractSpec, encode_value(request)) == request


FIELDS = {"ends": QUESTION, "rent": "What is the rent?"}
STRUCT_SQL = ("SELECT c.id, AI.EXTRACT(c.text, {'ends': '" + QUESTION
              + "', 'rent': 'What is the rent?'}) AS lease FROM contracts c")


def _lease(session):
    return (session.docs("contracts").alias("c")
            .ai_extract("c.text", FIELDS, name="lease")
            .select("c.id", "lease"))


def test_extract_plans_one_extraction_per_field(session):
    built = _lease(session)
    for dialect in ("snowflake", "bq"):
        assert session.sql(STRUCT_SQL, dialect=dialect).logical == built.logical
    projection = built.logical.projection
    assert projection.structs == (ExtractStruct("lease", ("ends", "rent")),)
    assert projection.result_columns() == ("c.id", "lease")
    plan = built.plan()
    extracts = [n for n in plan.nodes if isinstance(n, AiExtract)]
    assert [n.spec.name for n in extracts] == ["lease.ends", "lease.rent"]
    assert [n.spec.question for n in extracts] == list(FIELDS.values())
    # the fields run as one pipeline; the second reads the document KV
    # the first wrote
    pipelines = build_pipelines(plan.graph)
    assert tuple(member.node_id for member in pipelines[extracts[0].node_id]) == (
        extracts[0].node_id, extracts[1].node_id)
    project = next(n for n in plan.nodes if isinstance(n, Project))
    assert project.columns == ("c.id", "lease.ends", "lease.ends_span",
                               "lease.rent", "lease.rent_span")
    assert project.structs == (
        ("lease", (("ends", "lease.ends"), ("rent", "lease.rent"))),
        ("lease_span", (("ends", "lease.ends_span"), ("rent", "lease.rent_span"))))
    assert project.outputs[0].schema == ("c.id", "lease", "lease_span")
    codec = NodeCodec(Project)
    assert codec.decode(codec.encode(project)) == project
    assert "lease = struct(lease.ends, lease.rent)" in built.explain()


def _struct_answers():
    answers = {"lease.ends": ["March 3, 2027", None],
               "lease.rent": ["$2,000", None]}
    return Oracle(answers=lambda spec, document: answers[spec.name][document])


@pytest.mark.parametrize("suffix", [
    "", " ORDER BY c.id DESC LIMIT 1",
    " WHERE AI.SCORE(PROMPT('Is this a lease? {0}', c.text)) > 0.05"])
def test_extract_returns_answer_and_span_structs(
        session, monkeypatch, suffix):
    result = finish(session.sql(STRUCT_SQL + suffix), _struct_answers())
    rows = result.collect()
    assert rows.column_names == ["c.id", "lease", "lease_span"]
    span = pa.struct([("start", pa.int32()), ("end", pa.int32())])
    assert rows.schema.field("lease").type == pa.struct(
        [("ends", pa.string()), ("rent", pa.string())])
    assert rows.schema.field("lease_span").type == pa.struct(
        [("ends", span), ("rent", span)])
    expected = [
        {"c.id": 1, "lease": {"ends": "March 3, 2027", "rent": "$2,000"},
         "lease_span": {"ends": {"start": 14, "end": 27},
                        "rent": {"start": 37, "end": 43}}},
        {"c.id": 2, "lease": {"ends": None, "rent": None},
         "lease_span": {"ends": None, "rent": None}},
    ]
    assert rows.to_pylist() == (expected[1:] if "DESC" in suffix else expected)


def test_extract_field_errors(session):
    for bad, message in (
            ("SELECT c.id, AI.EXTRACT(c.text, {'a': 'x?'}) FROM contracts c",
             "AS name"),
            ("SELECT c.id, AI.EXTRACT(c.text, {}) AS x FROM contracts c",
             "at least one field"),
            ("SELECT c.id, AI.EXTRACT(c.text, {'a': 'x?', 'a': 'y?'}) AS x "
             "FROM contracts c", "names 'a' twice"),
            ("SELECT c.id, AI.EXTRACT(c.text, {'a': 1}) AS x FROM contracts c",
             "question string"),
            ("SELECT c.id, AI.EXTRACT(c.text, {'a': ''}) AS x FROM contracts c",
             "at least one word"),
            ("SELECT c.id, AI.EXTRACT(c.text, {'a': 'x?'}) AS c FROM contracts c",
             "also a table alias"),
            ("SELECT c.id, AI.EXTRACT(c.text, {'a': 'x?'}) AS x, "
             "AI.EXTRACT(c.text, {'b': 'y?'}) AS x_span FROM contracts c", "unique"),
            ("SELECT c.id, AI.EXTRACT(c.text, {'a': 'x?'}) AS x "
             "FROM contracts c ORDER BY x", "ORDER BY does not compare"),
            ("SELECT DISTINCT AI.EXTRACT(c.text, {'a': 'x?'}) AS x "
             "FROM contracts c", "DISTINCT does not compare"),
            ("SELECT COUNT(*) AS n, AI.EXTRACT(c.text, {'a': 'x?'}) AS x "
             "FROM contracts c GROUP BY x", "GROUP BY does not read")):
        with pytest.raises(CompileError, match=message):
            session.sql(bad)
    builder = session.docs("contracts").alias("c")
    with pytest.raises(CompileError, match="field needs a name without a dot"):
        builder.ai_extract("c.text", {"a.b": "x?"}, name="x")
    with pytest.raises(CompileError, match="at least one field"):
        builder.ai_extract("c.text", {}, name="x")
    with pytest.raises(CompileError, match="ORDER BY does not compare"):
        (session.docs("contracts").alias("c")
         .ai_extract("c.text", FIELDS, name="lease")
         .order_by("lease").select("c.id", "lease"))


def _answers_executor(answers):
    """A physical executor whose extractions answer answers[document]."""
    return oracle_executor(Oracle(answers=answers), fresh_tokens=12)


def test_filters_on_answers_keep_the_documents_whose_answer_passes(session):
    from quail.api.execute import execute_query
    from quail.logical import Filter as LogicalFilter

    sql = session.sql
    built = (session.docs("contracts").alias("c")
             .ai_extract("c.text", {"ends": QUESTION, "rent": "How much?"},
                         name="lease")
             .where(quail.col("lease.ends").is_not_null().hint(selectivity=0.5)
                    & quail.col("lease.ends").ilike("%march%").hint(selectivity=0.5))
             .filter((quail.col("lease_span.ends").field("start") < 20)
                     .hint(selectivity=0.5))
             .where((quail.col("lease.rent").length() < 20).hint(selectivity=0.5))
             .select("c.id", "lease"))
    tests = " AND ".join(f"HINT({test}, {{'selectivity': 0.5}})" for test in (
        "lease.ends IS NOT NULL", "lease.ends ILIKE '%march%'",
        "lease_span.ends.start < 20", "LENGTH(lease.rent) < 20"))
    written = sql(
        f"SELECT c.id, AI.EXTRACT(c.text, {{'ends': '{QUESTION}', "
        f"'rent': 'How much?'}}) AS lease FROM contracts c WHERE {tests}")
    assert written.logical == built.logical
    # each test follows its field's extraction; the span and length
    # tests use the same Filter node
    kinds = [type(node).__name__ for node in built.logical.walk()]
    assert kinds == ["Scan", "SemanticExtract", "Filter", "Filter", "Filter",
                     "SemanticExtract", "Filter", "Project"]
    tests = [node for node in built.logical.walk()
             if isinstance(node, LogicalFilter)]
    assert [str(node.condition) for node in tests] == [
        "c.lease.ends IS NOT NULL", "c.lease.ends ilike '%march%'",
        "c.lease.ends_span.start < 20", "length(c.lease.rent) < 20"]
    assert [node.selectivity for node in tests] == [0.5, 0.5, 0.5, 0.5]
    operators = built.logical.operators()
    assert [(test.name, test.position)
            for test in operators.extract_filters["c"]] == [
        ("lease.ends", 0), ("lease.ends", 1), ("lease.ends", 2),
        ("lease.rent", 3)]
    plan = built.plan()
    assert not isinstance(plan, Refusal)
    extracts = [n for n in plan.nodes if isinstance(n, AiExtract)]
    filters = [n for n in plan.nodes if isinstance(n, Filter)]
    assert [str(node.predicate) for node in filters] == [
        "c.lease.ends IS NOT NULL", "c.lease.ends ilike '%march%'",
        "c.lease.ends_span.start < 20", "length(c.lease.rent) < 20"]
    assert filters[0].inputs[0].source.node_id == extracts[0].node_id
    assert extracts[1].inputs[0].source.node_id == filters[2].node_id
    # the three tests on the first field thin the second extraction,
    # which runs in the same pipeline over the resident KV
    assert extracts[1].spec.expected_rows == pytest.approx(2 * 0.5 ** 3)
    assert extracts[1].spec.estimated_seconds < extracts[0].spec.estimated_seconds
    pipelines = build_pipelines(plan.graph)
    assert [member.node_id for member in pipelines[extracts[0].node_id]] == [
        extracts[0].node_id, filters[0].node_id, filters[1].node_id,
        filters[2].node_id, extracts[1].node_id, filters[3].node_id]
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    assert "Filter: c.lease.ends ilike '%march%'" in built.explain()

    # one document answers; its answer passes the tests
    result = execute_query(built, physical_executor=_answers_executor(
        {0: "March 3, 2027", 1: None}))
    rows = result.collect()
    assert rows.column("c.id").to_pylist() == [1]
    assert rows.column("lease").to_pylist() == [
        {"ends": "March 3, 2027", "rent": "March 3, 2027"}]
    answers = result.answer_tables["extracts"]["lease.ends"]
    assert answers.column("lease.ends").to_pylist() == ["March 3, 2027", None]
    # the second question was asked of the survivor only
    assert result.answer_tables["extracts"]["lease.rent"].num_rows == 1
    passed = result.answer_tables["filters"]
    assert passed[("c", 0)].column("answer").to_pylist() == [True, False]
    assert passed[("c", 2)].column("answer").to_pylist() == [True]
    # a negated text test drops the null answer as SQL does
    negated = (session.docs("contracts").alias("c")
               .ai_extract("c.text", {"ends": QUESTION}, name="lease")
               .where(~quail.col("lease.ends").like("March%"))
               .select("c.id", "lease"))
    assert str(negated.logical.root.input.condition) == (
        "c.lease.ends not like 'March%'")
    result = execute_query(negated, physical_executor=_answers_executor(
        {0: "March 3, 2027", 1: None}))
    assert result.collect().num_rows == 0


def test_filters_on_answers_bind_in_both_dialects_and_refuse_bad_tests(session):
    sql = session.sql
    fields = f"{{'ends': '{QUESTION}'}}"
    head = f"SELECT c.id, AI.EXTRACT(c.text, {fields}) AS lease FROM contracts c "
    # the repeated call names the field; BigQuery's text functions
    # and Snowflake's parse to the same tests, but for case and anchors
    repeated = sql(head + f"WHERE AI.EXTRACT(c.text, {fields}).ends "
                   f"NOT LIKE 'x%'")
    assert str(repeated.logical.root.input.condition) == (
        "c.lease.ends not like 'x%'")
    bq = sql(head + "WHERE REGEXP_CONTAINS(lease.ends, '20..') AND "
             "CONTAINS_SUBSTR(lease.ends, 'MARCH') AND STARTS_WITH(lease.ends, 'M') "
             "AND NOT ENDS_WITH(lease.ends, '.') AND lease.ends NOT LIKE 'x%'",
             dialect="bq")
    snowflake = sql(head + "WHERE REGEXP_LIKE(lease.ends, '20..') AND "
                    "CONTAINS(lease.ends, 'MARCH') AND STARTSWITH(lease.ends, 'M') "
                    "AND NOT ENDSWITH(lease.ends, '.') AND lease.ends RLIKE 'M.*'")
    conditions = [str(test.condition)
                  for test in bq.logical.operators().extract_filters["c"]]
    assert conditions == [
        "c.lease.ends regex '20..'", "c.lease.ends icontains 'MARCH'",
        "c.lease.ends starts with 'M'", "c.lease.ends not ends with '.'",
        "c.lease.ends not like 'x%'"]
    conditions = [str(test.condition)
                  for test in snowflake.logical.operators().extract_filters["c"]]
    assert conditions == [
        "c.lease.ends regex '^(?:20..)$'", "c.lease.ends contains 'MARCH'",
        "c.lease.ends starts with 'M'", "c.lease.ends not ends with '.'",
        "c.lease.ends regex '^(?:M.*)$'"]
    # the answer tests of a joined table ride with its extraction when
    # the planner lifts it above the joins
    session.register("questions", DocumentProvider.from_table(
        pa.table({"id": [1], "ask": ["end dates"]}), id_col="id"))
    joined = sql(
        f"SELECT c.id, q.id, AI.EXTRACT(c.text, {fields}) AS lease "
        f"FROM contracts c JOIN questions q "
        f"ON AI_FILTER(PROMPT('Does {{0}} answer {{1}}?', c.text, q.ask)) "
        f"WHERE lease.ends IS NOT NULL")
    plan = joined.plan()
    assert not isinstance(plan, Refusal)
    assert extracted_above_joins(joined.logical.root) == {"c"}
    (test,) = [n for n in plan.nodes if isinstance(n, Filter)]
    (extract,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
    assert test.inputs[0].source.node_id == extract.node_id
    for bad, message in (
            (head + "WHERE lease.ends = 3", "compares it with a string"),
            (head + "WHERE lease.ends LIKE 3", "compares with a string"),
            (head + "WHERE lease.signer IS NOT NULL", "no field 'signer'"),
            (head + "WHERE lease_span.ends < 3", "test its start or end"),
            (head + "WHERE lease.ends.start < 3", "field under an answer"),
            (head + "WHERE AI.EXTRACT(c.text, {'other': 'q?'}).other IS NULL",
             "not returned by SELECT"),
            (f"SELECT c.id, AI.EXTRACT(c.text, {fields}) AS c FROM contracts c "
             f"WHERE c.ends IS NOT NULL", "both a table alias"),
            (head + "WHERE c.text BETWEEN 'a' AND 'b'", "not supported"),
            (head + "WHERE HINT(c.id = 1, {'selectivity': 0.5})",
             "tests a source column"),
            (head + "WHERE HINT(lease.ends IS NULL)", "takes a test"),
            (head + "WHERE HINT(lease.ends IS NULL, {'selectivity': 'x'})",
             "must be a number"),
            (head + "WHERE HINT(lease.ends IS NULL, {'anchor': 'c'})",
             "unknown option"),
            (head + "WHERE NOT c.text = 'a'", "NOT is supported as")):
        with pytest.raises(CompileError, match=message):
            sql(bad)
    builder = (session.docs("contracts").alias("c")
               .ai_extract("c.text", {"ends": QUESTION}, name="lease"))
    with pytest.raises(CompileError, match="no field 'signer'"):
        builder.where(quail.col("lease.signer").is_null())
    with pytest.raises(CompileError, match="has no fields"):
        builder.where(quail.col("c.text").field("x") == "a")
    with pytest.raises(CompileError, match="does not return"):
        builder.where(quail.col("lease.ends").is_null()).select("c.id")
    with pytest.raises(CompileError, match="source column"):
        builder.where((quail.col("c.id") == 1).hint(selectivity=0.5))
    with pytest.raises(TypeError, match="combines tests"):
        (quail.col("c.id") == 1) & 3
    # the speed of light estimate keeps the documents whose reference
    # answer passes, and does not price a span test
    tested = (session.docs("contracts").alias("c")
              .ai_extract("c.text", {"ends": QUESTION}, name="lease")
              .where(quail.col("lease.ends").is_not_null())
              .select("c.id", "lease"))
    reference = {0: "March 3, 2027", 1: None}
    estimate = quail.speed_of_light_estimate(
        tested, lambda prompt, assignment: reference[assignment["c"]])
    extract, test = estimate.filter_stages
    assert (test["operator"], test["evaluated"], test["passed"],
            test["condition"]) == ("ANSWER TEST", 2, 1, "c.lease.ends IS NOT NULL")
    spanned = (session.docs("contracts").alias("c")
               .ai_extract("c.text", {"ends": QUESTION}, name="lease")
               .where(quail.col("lease_span.ends").field("start") < 3)
               .select("c.id", "lease"))
    with pytest.raises(NotImplementedError, match="span"):
        quail.speed_of_light_estimate(spanned, lambda *_: None)
    # stock vLLM returns answers but not the filter
    stock = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                       backend="stock_vllm"),
                          tokenizer=letter_tokens)
    stock.register("contracts", session.catalog.get("contracts"))
    refused = (stock.docs("contracts").alias("c")
               .ai_extract("c.text", {"ends": QUESTION}, name="lease")
               .where(quail.col("lease.ends").is_not_null())
               .select("c.id", "lease")).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "extract_filter_needs_quail_backend"
    stock.close()


def test_a_tables_steps_are_ordered_by_cost_per_rejected_document(session):
    # the asks, the classifications with their label tests, and the
    # extractions with their answer tests are one ordering search
    def query(selectivity, order=None):
        return (session.docs("contracts").alias("c")
                .ai_filter(quail.prompt("Is {0} a lease?", quail.col("c.text")),
                           selectivity=0.9)
                .ai_classify(quail.prompt("Kind of {0}?", quail.col("c.text")),
                             ["lease", "deed"], name="kind")
                .where((quail.col("kind") == "lease").hint(selectivity=0.8))
                .ai_extract("c.text", {"ends": QUESTION}, name="lease")
                .where(quail.col("lease.ends").is_not_null().hint(selectivity=selectivity))
                .select("c.id", "kind", "lease", order=order))

    def order(plan):
        return [node.node_id for node in plan.nodes
                if type(node).__name__ in ("AiFilter", "AiClassify", "AiExtract")]

    # the steps run as one pipeline whatever their order: each reads
    # the KV the step before it left, rewound past its answer
    usual, selective = query(0.7).plan(), query(0.05).plan()
    for plan in (usual, selective):
        (extract,) = [n for n in plan.nodes if isinstance(n, AiExtract)]
        pipeline = build_pipelines(plan.graph)[extract.node_id]
        assert len(pipeline) == 5 and sorted(
            member.node_id for member in pipeline) == sorted(
            node.node_id for node in plan.nodes
            if type(node).__name__ not in ("Scan", "Project"))
    # the ask rejects a tenth, the label test a fifth; an extraction
    # test rejecting three tenths ranks the extraction before the ask,
    # and one rejecting nearly every document puts it first
    assert order(usual) == ["ai-classify:0", "ai-extract:0", "ai_filter:c"]
    assert order(selective) == ["ai-extract:0", "ai-classify:0", "ai_filter:c"]
    # the result reads the documents the last step keeps, whatever
    # kind of step it is
    for plan in (usual, selective):
        assert plan.nodes[-1].inputs[0].source == PortRef("ai_filter:c", "rows:c")
    (extract,) = [n for n in selective.nodes if isinstance(n, AiExtract)]
    (classify,) = [n for n in selective.nodes if isinstance(n, AiClassify)]
    (ask,) = [n for n in selective.nodes if isinstance(n, AiFilter)]
    assert extract.spec.expected_rows == 2
    assert classify.spec.expected_rows == pytest.approx(2 * 0.05)
    assert ask.expected_rows == pytest.approx(2 * 0.05 * 0.8, abs=0.05)
    # the later steps are priced over the resident KV
    assert classify.spec.estimated_seconds < extract.spec.estimated_seconds
    # the ask, the classification, and the extraction show the
    # documents under one head, so each step reads the KV the one
    # before it leaves
    from quail.planner.optimizer.context import PlanningContext
    from quail.planner.search.statistics import cached_statistics
    from quail.planner.search.table_steps import PlanPricing, table_steps

    priced = query(0.5)
    priced.plan()
    context = PlanningContext(
        model=session.model, device=session.device, gpus=1,
        document_tokens=priced._document_tokens, backend="quail",
        tokenizer=session.tokenizer)
    operators = priced.logical.operators()
    statistics = cached_statistics(
        priced.logical, {}, model=session.model, device=session.device,
        document_tokens=priced._document_tokens, context=context)
    steps = table_steps(operators, "c", range(1),
                        PlanPricing(context, statistics, "c"))
    assert [step.kind for step in steps] == ["ask", "classify", "extract"]
    assert len({step.document_head for step in steps}) == 1
    assert steps[0].document_head
    written = query(0.05, order="as_written").plan()
    assert order(written) == ["ai_filter:c", "ai-classify:0", "ai-extract:0"]
    # the speed of light estimate runs the same orders
    reference = {0: "March 3, 2027", 1: None}
    for selectivity, stages in ((0.7, [
            ("AI.CLASSIFY", 2, 2), ("IN", 2, 2), ("AI.EXTRACT", 2, 2),
            ("ANSWER TEST", 2, 1), ("AI.IF", 1, 1)]), (0.05, [
            ("AI.EXTRACT", 2, 2), ("ANSWER TEST", 2, 1), ("AI.CLASSIFY", 1, 1),
            ("IN", 1, 1), ("AI.IF", 1, 1)])):
        estimate = quail.speed_of_light_estimate(
            query(selectivity),
            lambda prompt, a: reference[a["c"]] if prompt.kind == "extract" else True,
            label=lambda prompt, a: "lease")
        assert [(stage["operator"], stage["evaluated"], stage["passed"])
                for stage in estimate.filter_stages] == stages


def test_rows_carry_their_ids_apart_from_their_values():
    # a classification after an extraction carries its answers as
    # value columns; the ids stay the ids
    answers = extraction_rows(
        "c", "lease.ends", "lease.ends_span", [0, 2],
        ["March 3, 2027", None], [(14, 27), None])
    labels = Rows({"c": [2, 0]}, pa.table({"kind": ["a", "b"]}))
    joined = labels.joined(answers, "c")
    assert joined.aliases == ("c",)
    assert joined.column_names == ["kind", "lease.ends", "lease.ends_span"]
    assert joined.column("lease.ends").to_pylist() == [None, "March 3, 2027"]
    assert joined.to_pydict()["c"] == [2, 0]
    for none in ([], (), np.array([])):
        assert joined.take(none).num_rows == 0
