"""AI.CLASSIFY: label scoring, prompt text, planning, execution, and results."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import (
    cpu_arena,
    fake_pack,
    fake_pipeline,
    fake_readout,
    fake_torch,
    letter_tokens,
)
from helpers.oracle import Oracle, finish

import quail
from bench.quailb import run_output
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.label_stages import label_requests
from quail.backends.quail.executor.operators.classify import QuailClassifier
from quail.backends.quail.executor.readout import AsyncLabelLogprobs, answer_rows
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.baselines.nodes import RequestExecution
from quail.catalog import DocumentProvider
from quail.execution.labels import (
    GreedyDecoder,
    best_label,
)
from quail.labels import label_trie
from quail.logical import (
    Alias,
    ColumnRef,
    CompileError,
    LogicalPlan,
    RegularPredicate,
    Scan,
    bind_classify_prompt,
)
from quail.logical.prompts import join_anchor_note
from quail.physical import (
    AiClassify,
    AiJoin,
    ClassifySpec,
    Filter,
    decode_graph,
    encode_graph,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.plan import EngineConfig, Refusal
from quail.planner.prompt_lowering import lower_prompt
from quail.planner.rules.tree_transforms import push_down_projection
from quail.specs import H100_SXM, QWEN3_4B_FP8
from quail_b.prompts import AGENT_PROGRESS, AGENT_PROGRESS_DESCRIPTIONS
from quail_b.prompts import AGENT_PROGRESS_LABELS as STAGES
from quail_b.queries import get_query

# one token per word: refund=1, request=2, status=3, shipping=4
LABELS = ("refund request", "refund status", "shipping")
IDS = ((1, 2), (1, 3), (4,))


def _bytes(text):
    return list(text.encode())


def test_greedy_decoder_follows_the_likeliest_child_to_a_label():
    targets = [1, 2, 3, 4]
    decoder = GreedyDecoder(IDS, targets, documents=2, cue=93)
    assert decoder.rounds == 2
    # the cue's request, then the run from token 1 to the next choice
    assert decoder.suffixes == [[93], [1]]
    # round 0 feeds the cue, request 0, for both documents
    assert [decoder.requests(document) for document in range(2)] == [[0], [0]]
    assert decoder.tokens == 2
    # document 0 follows token 1, document 1 takes the one-token label 4
    decoder.update(0, np.array([-1.0, -9.0, -9.0, -3.0]))
    decoder.update(1, np.array([-2.0, -9.0, -9.0, -0.5]))
    assert decoder.label.tolist() == [-1, 2]
    # round 1 feeds only token 1, request 1
    assert decoder.requests(1) is None and decoder.requests(0) == [1]
    assert decoder.tokens == 2 + 1
    # after token 1, token 3 beats token 2: "refund status"
    decoder.update(0, np.array([-9.0, -5.0, -1.0, -9.0]))
    assert decoder.label.tolist() == [1, 2]
    # a tie goes to the earlier label's token
    tied = GreedyDecoder(IDS, targets, documents=1, cue=93)
    tied.requests(0)
    tied.update(0, np.array([-1.0, -9.0, -9.0, -1.0]))
    assert tied.node[0] == (1,)
    with pytest.raises(ValueError, match="prefix"):
        GreedyDecoder(((1, 2), (1,)), [1, 2], documents=1, cue=93)


def test_greedy_decoder_feeds_forced_tokens_with_the_next_choice():
    from quail.labels import decode_round_tokens, decode_runs

    # "the cat sat" / "the cat ran": the one choice comes after two
    # forced tokens, fed with the cue; the chosen label's last token
    # is never fed
    labels = ((1, 2, 3), (1, 2, 4))
    decoder = GreedyDecoder(labels, [1, 2, 3, 4], documents=1, cue=93)
    assert decoder.suffixes == [[93, 1, 2]] and decoder.rounds == 1
    assert decoder.requests(0) == [0] and decoder.tokens == 3
    # a forced token is taken whatever its row says
    decoder.update(0, np.array([-1e9, -1e9, -0.5, -0.1]))
    assert decoder.label.tolist() == [1] and decoder.requests(0) is None
    # "hello my name is shreya" / "elephant in the room" decide at the
    # cue's row: one round, one token fed
    long = ((1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11))
    decoder = GreedyDecoder(long, list(range(1, 12)), documents=1, cue=93)
    assert decoder.suffixes == [[93]] and decoder.rounds == 1
    decoder.requests(0)
    decoder.update(0, np.array([-0.7] + [-9.0] * 6 + [-0.9] + [-9.0] * 3))
    assert decoder.label.tolist() == [0] and decoder.tokens == 1
    # "a b c d" / "a b c e" / "a f" / "g": a request per choice on the
    # path, each feeding the run to the next choice
    deep = ((1, 2, 3, 4), (1, 2, 3, 5), (1, 6), (7,))
    runs = decode_runs(deep)
    assert runs.runs == [[], [1], [2, 3]] and runs.rounds == 3
    assert decode_round_tokens(deep) == [[1, 1, 2], [1, 1, 2], [1, 1], [1]]
    # the only label is decided at the cue, with nothing fed
    decoder = GreedyDecoder(((1, 2, 3),), [1, 2, 3], documents=1, cue=93)
    assert decoder.suffixes == [[93]] and decoder.rounds == 1
    decoder.requests(0)
    decoder.update(0, np.array([-9.0, -9.0, -9.0]))
    assert decoder.label.tolist() == [0]


def test_label_readout_is_log_softmax_over_the_whole_vocabulary(monkeypatch):
    torch = pytest.importorskip("torch")
    head = torch.randn(300, 8, dtype=torch.bfloat16)
    hidden = torch.randn(70, 8, dtype=torch.bfloat16)
    readout = AsyncLabelLogprobs(torch, torch.nn.functional, head, [5, 17, 299])
    got = readout.logprobs(hidden)
    logits = torch.nn.functional.linear(hidden, head).float()
    expected = torch.log_softmax(logits, dim=1)[:, [5, 17, 299]]
    assert torch.allclose(got, expected, atol=1e-5)
    assert readout.dtype.shape == (3,)
    # a many-row answer's rows land at (answer, row within it)
    answers, rows = answer_rows([2, 1, 3])
    assert answers.tolist() == [0, 0, 1, 2, 2, 2]
    assert rows.tolist() == [0, 1, 0, 0, 1, 2]

    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: empty(
        *args, **{key: value for key, value in kwargs.items() if key != "pin_memory"}))
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(
        record=lambda: None, synchronize=lambda: None))
    first = readout.submit(hidden)
    second = readout.submit(hidden[:3])
    assert first[1].data_ptr() != second[1].data_ptr()
    smaller = readout.result(second)
    actual = readout.result(first)
    assert np.allclose(actual, expected.numpy(), atol=1e-5)
    again = readout.result(readout.submit(hidden[:1] * 2))
    assert again.shape == (1, 3)
    assert np.allclose(smaller, expected[:3].numpy(), atol=1e-5)
    assert np.allclose(actual, expected.numpy(), atol=1e-5)


def test_classifier_decodes_one_choice_per_round(monkeypatch):
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_rows=3, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS, scoring="trie_decode")
    requests = label_requests(spec)
    assert requests.rounds == 2 and requests.score is None
    assert requests.suffixes == [[93], [1]]
    assert not requests.read_all_rows
    documents = {"d": [[10, 11], [12], [13]]}
    targets = [1, 2, 3, 4]
    # document 0 wants "refund request", 1 "shipping" outright, 2
    # "refund status": the wanted label's next token gets -1, another
    # label's first token -2 for token 1 and -3 for token 4, else -5
    wanted = {0: IDS[0], 1: IDS[2], 2: IDS[1]}

    def logprob(document, seen, token):
        if seen + (token,) == tuple(wanted[document][:len(seen) + 1]):
            return -1.0
        if not seen and token in (1, 4):
            return -2.0 if token == 1 else -3.0
        return -5.0

    packed = []
    path = {}

    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            for suffix in entry["suffixes"]:
                packed.append((document, list(suffix), entry["f"],
                               entry.get("write_suffix_tokens", 0)))
                # the cue starts the path; a fed token extends it
                path[document] = (() if suffix[-1] == 93
                                  else path[document] + tuple(suffix))
                rows.append([logprob(document, path[document], token)
                             for token in targets])
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    readout = fake_readout(targets=np.asarray(targets), rows=1, normalize=True,
                           dtype=np.dtype((np.float32, (4,))))
    state = QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(64),
            pipeline=fake_pipeline(forward_chunk=forward),
            label_readout=readout,
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object(), model_spec=QWEN3_4B_FP8,
        ),
        torch=fake_torch(),
        chunk_tokens=64,
        answer_rows=object(),
        async_answers=fake_readout(),
    )
    arena = state.loaded_model.arena
    reserved = {}
    activate = arena.activate

    def recording_activate(key, tokens, capacity_tokens=None, **kw):
        reserved[key[2]] = capacity_tokens
        return activate(key, tokens, capacity_tokens=capacity_tokens, **kw)

    monkeypatch.setattr(arena, "activate", recording_activate)
    sent = []
    batch = QuailClassifier(state).classify(
        spec, [[0], [1], [2]], documents,
        on_answers=lambda rows, labels: sent.append((rows.tolist(), labels)))
    assert list(batch.scores) == ["refund request", "shipping", "refund status"]
    # each label streams once, as the chunk that decides it is read
    assert sent == [([1], ["shipping"]),
                    ([0, 2], ["refund request", "refund status"])]
    # each document reserves its prefix, the two-token frame, and one
    # kept token per round: the cue and the first label token
    assert reserved == {0: 3 + 2 + 2, 1: 2 + 2 + 2, 2: 2 + 2 + 2}
    # round 0 packs the frame and cue after each document's prefix (3
    # tokens for document 0, 2 for the others) and keeps all three in KV;
    # document 1 decodes its one-token label there. Round 1 feeds only
    # token 1, after the kept frame and cue, and keeps it
    assert sorted(packed) == sorted([
        (0, [91, 92, 93], 3, 3), (1, [91, 92, 93], 2, 3),
        (2, [91, 92, 93], 2, 3), (0, [1], 3 + 3, 1), (2, [1], 2 + 3, 1)])
    assert batch.suffix_tokens == 3 * 1 + 2 * 1


@pytest.fixture()
def session(tmp_path):
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({
        "id": [7, 9], "body": ["where is my refund", "love it"],
    }), path)
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
    value.register("documents",
                   DocumentProvider.from_parquet(str(path), id_col="id"))
    yield value
    value.close()


def _topic(session, *, accepted=("refund", "shipping")):
    query = session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic")
    if accepted:
        query = query.label_in("topic", accepted, selectivity=0.5)
    return query.select("d.id", "topic")


def _labels(labels):
    """An oracle labeling document i with labels[i]; a "kind" gets its first label."""
    return Oracle(labels=lambda spec, ids: (spec.labels[0] if spec.name == "kind"
                                            else labels[ids[0]]),
                  probabilities=True)


def test_classify_plans_filters_and_returns_labels(session):
    query = _topic(session)
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (test,) = [n for n in plan.nodes if isinstance(n, Filter)]
    assert classify.spec.labels == ("refund", "shipping", "praise")
    assert classify.spec.label_token_ids[0] == tuple(_bytes("refund"))
    # a tokenizer without decode marks no word start
    assert classify.spec.label_word_starts == tuple(
        (False,) * len(ids) for ids in classify.spec.label_token_ids)
    assert test.predicate == RegularPredicate(
        ColumnRef("d", "documents", "topic"), "in", ("refund", "shipping"))
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph

    result = finish(query, _labels(["refund", "praise"]))
    rows = result.collect()
    assert rows.column("topic").to_pylist() == ["refund"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["refund", "praise"]
    (answers,) = result.answer_tables["filters"].values()
    assert answers.column("answer").to_pylist() == [True, False]
    assert result.report["node_metrics"]["ai-classify:0"]["suffix_tokens"] == 6

    # a projected label with no filter classifies every document
    plain = _topic(session, accepted=())
    result = finish(plain, _labels(["praise", "shipping"]))
    assert result.collect().column("topic").to_pylist() == [
        "praise", "shipping"]

    # a document whose decoded answer names no label leaves the run
    result = finish(plain, _labels(["praise", None]))
    assert result.collect().column("topic").to_pylist() == ["praise"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["praise"]
    assert result.report["node_metrics"]["ai-classify:0"]["output_rows"] == 1

    result = finish(query, _labels([None, "refund"]))
    assert result.collect().column("topic").to_pylist() == ["refund"]
    (answers,) = result.answer_tables["filters"].values()
    assert answers.column("answer").to_pylist() == [True]
    assert result.answer_tables["classifies"]["topic"].column("d").to_pylist() == [1]

    # with probabilities, a map of each label's probability beside the
    # label, from a rule that scores every label
    probable = session.sql(
        "SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
        "ARRAY['refund', 'shipping', 'praise'], {'probabilities': true}) "
        "AS topic FROM documents d")
    built = (session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic", probabilities=True)
        .select("d.id", "topic"))
    assert probable.logical == built.logical
    plan = probable.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.probabilities
    assert classify.spec.scoring != "trie_decode"
    assert plan.nodes[-1].columns == ("d.id", "topic", "topic_probabilities")
    rows = finish(probable, _labels(["praise", "shipping"])).collect()
    assert rows.column("topic_probabilities").to_pylist() == [
        [("refund", 0.1), ("shipping", 0.1), ("praise", 0.8)],
        [("refund", 0.1), ("shipping", 0.8), ("praise", 0.1)]]


def _classify_then_filter(session, *, filters=1, later_filters=()):
    query = session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic")
    query = query.label_in("topic", ["refund", "shipping"], selectivity=0.5)
    if filters > 1:
        query = query.label_in("topic", ["refund"], selectivity=0.5)
    query = query.ai_classify(
        quail.prompt("What kind of {0}?", quail.col("d.body")),
        ["complaint", "question"], name="kind")
    for name, accepted in later_filters:
        query = query.label_in(name, accepted, selectivity=0.5)
    return query.select("d.id", "topic", "kind")


def test_sql_classifies_and_tests_labels(session):
    call = ("AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
            "ARRAY['refund', 'shipping', 'praise'])")
    query = session.sql(
        f"SELECT d.id, {call} AS topic FROM documents d "
        f"WHERE {call} IN ('refund', 'shipping')")
    built = (session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic")
        .label_in("topic", ["refund", "shipping"]).select("d.id", "topic"))
    assert query.logical == built.logical
    (test,) = [n for n in query.plan().nodes if isinstance(n, Filter)]
    assert test.predicate == RegularPredicate(
        ColumnRef("d", "documents", "topic"), "in", ("refund", "shipping"))
    # a filter on the label alone, with a selectivity option
    filtered = session.sql(
        f"SELECT d.id FROM documents d WHERE {call[:-1]}, "
        f"{{'selectivity': 0.5}}) IN ('praise')")
    (predicate,) = filtered.logical.operators().label_filters["d"]
    assert predicate.selectivity == 0.5
    assert predicate.condition.value == ("praise",)
    assert "d" not in filtered.logical.operators().filters
    # equality is membership in one label, on either side
    for form in (f"{call} = 'refund'", f"'refund' = {call}"):
        equal = session.sql(f"SELECT d.id FROM documents d WHERE {form}")
        (predicate,) = equal.logical.operators().label_filters["d"]
        assert predicate.condition.value == ("refund",)
    for bad in (f"SELECT d.id, {call} FROM documents d",
                f"SELECT d.id FROM documents d WHERE {call} < 'refund'",
                f"SELECT d.id FROM documents d WHERE {call} IN (1)"):
        with pytest.raises(CompileError):
            session.sql(bad)


def test_frontends_place_classifications_on_their_table(session):
    # a classification sits above its table's AI.IF filters, below the
    # first filter on its label; a later classification sits above
    # that filter
    query = _classify_then_filter(session, later_filters=(("kind", ("complaint",)),))
    query.plan()
    plan = query.logical
    kinds = [type(node).__name__ for node in plan.walk()]
    assert kinds == ["Scan", "SemanticClassify", "Filter",
                     "SemanticClassify", "Filter", "Project"]
    topic, kind = plan.operators().classifies
    assert (topic.name, topic.alias, kind.name) == ("topic", "d", "kind")
    assert plan.root.input.condition == quail.logical.RegularPredicate(
        ColumnRef("d", "documents", "kind"), "in", ("complaint",))
    assert plan.operators().labels.tests == {topic.call: [0], kind.call: [1]}
    # projection pushdown reaches the scan through the classifications
    assert push_down_projection(plan.root) is plan.root
    (scan,) = [node for node in plan.walk() if isinstance(node, Scan)]
    assert scan.columns == ("id", "body")
    mixed = (session.docs("documents").alias("d")
             .ai_filter(quail.prompt("Is {0} short?", quail.col("d.body")))
             .ai_classify(quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
                          name="x")
             .label_in("x", ["a"]).select("d.id"))
    assert [type(node).__name__ for node in mixed.logical.walk()] == [
        "Scan", "SemanticFilter", "SemanticClassify", "Filter", "Project"]
    operators = mixed.logical.operators()
    assert [type(p.expression).__name__
            for p in operators.filters["d"]] == ["ModelCall"]
    assert [test.position for test in operators.label_filters["d"]] == [1]
    # SQL names a classification only tested after its filter position
    hidden = session.sql(
        "SELECT d.id FROM documents d WHERE AI_FILTER(PROMPT('Is {0} short?', "
        "d.body)) AND AI.CLASSIFY(PROMPT('{0}', d.body), ARRAY['a', 'b']) "
        "IN ('a')")
    (node,) = hidden.logical.operators().classifies
    assert node.name == "__label_d_1"
    assert [type(node).__name__ for node in hidden.logical.walk()] == [
        "Scan", "SemanticFilter", "SemanticClassify", "Filter", "Project"]
    # a classification the query neither returns nor tests is not planned
    unused = (session.docs("documents").alias("d")
              .ai_filter(quail.prompt("Is {0} short?", quail.col("d.body")))
              .ai_classify(quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
                           name="x").select("d.id"))
    assert unused.logical.operators().classifies == ()
    text = _topic(session).explain()
    assert "SemanticClassify: topic" in text
    assert ("Filter: d.topic IN ('refund', 'shipping') (selectivity=50%)"
            in text.split("SemanticClassify")[0])
    assert "Project: d.id, topic" in text
    assert "AiClassify: topic over d" in text
    assert "rule=trie_decode, labels=3" in text
    assert "Filter: d.topic IN ('refund', 'shipping')" in text


def _corpus(session, tmp_path, name, column, rows, words):
    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table({
        "id": [f"{name}{i}" for i in range(rows)],
        column: [" ".join(f"w{i}x{j}" for j in range(words)) for i in range(rows)],
    }), path)
    session.register(name, DocumentProvider.from_parquet(str(path), id_col="id"))


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


def test_planner_places_and_scores_classifications(session, tmp_path):
    from quail.logical import classified_above_joins
    from quail.planner.rules import built_in_rules
    from quail.planner.rules.tree_transforms import (
        lift_maps,
    )

    assert [rule.name for rule in built_in_rules()] == [
        "should_eliminate_distinct", "should_push_down_distinct",
        "should_push_down_projection",
        "should_push_down_filters", "should_lift_maps",
        "should_eliminate_window_sort", "should_push_down_limit_through_window",
        "should_push_down_limit", "choose_classify_implementation", "should_retain_kv",
        "should_share_prefixes", "choose_attention_algorithm"]
    plan = _topic(session).plan()
    # choose_classify_implementation chose every classification's scoring rule
    assert all(node.spec.scoring for node in plan.nodes
               if isinstance(node, AiClassify))
    assert plan.settings["classify_placement"] == "before joins"

    # sixty long reviews and three aspects
    _corpus(session, tmp_path, "reviews", "body", 60, 120)
    _corpus(session, tmp_path, "aspects", "aspect", 3, 4)
    labels = [f"label number {i}" for i in range(12)]
    r, a = quail.col("r.body"), quail.col("a.aspect")
    join = quail.prompt("Does {0} mention {1}?", r, a)

    def classified(filtered=False):
        query = session.docs("reviews").alias("r")
        if filtered:
            query = query.ai_filter(quail.prompt("Is {0} long?", r),
                                    selectivity=0.5)
        return query.ai_classify(quail.prompt("Topic of {0}?", r), labels,
                                 name="topic")

    # a join keeping one pair in a hundred: classifying the matched
    # documents after it beats classifying all sixty before it
    after = (classified().label_in("topic", labels[:11], selectivity=0.95)
             .ai_join(session.docs("aspects").alias("a"), join, selectivity=0.01)
             .select("r.id", "a.id", "topic"))
    written = after.logical.root
    plan = after.plan()
    assert plan.settings["classify_placement"] == "after joins"
    assert [node.node_id for node in plan.nodes] == [
        "scan:r", "scan:a", "ai_join:r", "ai-classify:0", "filter:r:0",
        "recombine", "project"]
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.inputs[0].source.node_id == "ai_join:r"
    assert classify.spec.scoring == "trie_tree"
    assert classify.spec.expected_rows == pytest.approx(1.8)
    assert round(plan.estimated_seconds, 3) == 0.385
    (staged,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
    assert (staged.exec_index, staged.anchor) == (0, "r")
    assert classified_above_joins(after._logical_input.root) == frozenset()
    assert classified_above_joins(after.logical.root) == {"r"}
    written_candidate, lifted_candidate = _candidates(session, after)
    assert classified_above_joins(written_candidate.logical_plan.root) == frozenset()
    assert lifted_candidate.logical_plan == after.logical
    assert classified_above_joins(lifted_candidate.logical_plan.root) == {"r"}
    assert (lifted_candidate.estimated_seconds
            < written_candidate.estimated_seconds)
    # a cost-based tree rewrite prices a plan the same way the backend
    # prices its candidates
    from quail.planner import pricing
    from quail.planner.optimizer.context import PlanningContext

    context = PlanningContext(
        catalog=session.catalog, engine_config=session.config, model=session.model,
        device=session.device, document_tokens=after._document_tokens,
        tokenizer=session.tokenizer)
    assert pricing.estimated_seconds(
        lifted_candidate.logical_plan, context, session.registry) == (
        lifted_candidate.estimated_seconds)
    logical_text, physical_text = after.explain().split("physical:")
    assert "SemanticClassify: topic" in logical_text
    assert "AiClassify: topic over r" in physical_text
    lifted = lift_maps(written)
    assert [type(node).__name__ for node in LogicalPlan(lifted).walk()] == [
        "Scan", "Scan", "Join", "SemanticJoin", "SemanticClassify",
        "Filter", "Project"]
    assert classified_above_joins(lifted) == {"r"}
    assert classified_above_joins(written) == frozenset()
    assert lift_maps(lifted) is None
    # a query without a join gets one candidate
    assert lift_maps(_topic(session).logical.root) is None
    assert len(_candidates(session, _topic(session))) == 1

    # a filtered partner on a forced anchor stays classified before the
    # join, over the filter's survivors with their KV resident
    before = (classified(filtered=True)
              .label_in("topic", labels[:6], selectivity=0.5)
              .ai_join(session.docs("aspects").alias("a"), join,
                       selectivity=0.3, anchor="a")
              .select("r.id", "a.id", "topic"))
    plan = before.plan()
    assert plan.settings["classify_placement"] == "before joins"
    assert [node.node_id for node in plan.nodes] == [
        "scan:r", "scan:a", "ai_filter:r", "ai-classify:0", "filter:r:1",
        "ai_join:a", "project"]
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert (classify.spec.scoring, classify.spec.expected_rows) == (
        "trie_tree", 30.0)
    assert round(plan.estimated_seconds, 3) == 0.555
    written_candidate, lifted_candidate = _candidates(session, before)
    assert (written_candidate.estimated_seconds
            <= lifted_candidate.estimated_seconds)


def test_sql_classifies_the_rows_a_join_keeps(session, tmp_path):
    path = tmp_path / "aspects.parquet"
    pq.write_table(pa.table({"id": [1, 2], "aspect": ["price", "size"]}),
                   path)
    session.register("aspects",
                     DocumentProvider.from_parquet(str(path), id_col="id"))
    stance = ("AI.CLASSIFY(PROMPT('What does DOCUMENT {0} say about "
              "DOCUMENT {1}?', d.body, a.aspect), ARRAY['praise', 'complaint'])")
    query = session.sql(
        f"SELECT d.id, a.id, {stance} AS stance FROM documents d "
        f"JOIN aspects a ON AI_FILTER(PROMPT('Does {{0}} mention {{1}}?', "
        f"d.body, a.aspect))")
    (call,) = [column.expression for column in query.logical.root.columns
               if isinstance(column, Alias)]
    assert call.kind == "label" and call.aliases() == ("d", "a")
    assert call.prompt.frame == join_anchor_note(0)
    assert call.prompt.tail.startswith("\n\nAnswer with exactly one")
    # the classification sits above the join of the two tables
    assert [type(node).__name__ for node in query.logical.walk()] == [
        "Scan", "Scan", "Join", "SemanticJoin", "SemanticClassify", "Project"]
    assert query.logical.root.input.call is call
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (join,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
    assert classify.spec.partner is not None
    assert classify.spec.anchor == join.anchor
    assert classify.inputs[0].source.node_id == join.node_id
    # More than one independent label consumer ends the join pipeline.
    other = stance.replace("What does DOCUMENT", "How does DOCUMENT")
    refused = session.sql(
        f"SELECT d.id, a.id, {stance} AS stance, {other} AS other "
        "FROM documents d JOIN aspects a "
        "ON AI_FILTER(PROMPT('Does {0} mention {1}?', d.body, a.aspect))").plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "joined_classify_pipeline"
    assert "standalone pair classification" in refused.reasons[0]
    distributed = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=2),
        tokenizer=letter_tokens)
    for name in ("documents", "aspects"):
        distributed.register(name, session.catalog.get(name))
    # every worker runs the join and the classification as one pipeline
    spread = distributed.sql(
        f"SELECT d.id, a.id, {stance} AS stance FROM documents d "
        "JOIN aspects a ON AI_FILTER(PROMPT('Does {0} mention {1}?', "
        "d.body, a.aspect))").plan()
    assert not isinstance(spread, Refusal)
    (classify,) = [n for n in spread.nodes if isinstance(n, AiClassify)]
    assert classify.node_id in {member.node_id for member in build_pipelines(
        spread.graph)[classify.inputs[0].source.node_id]}
    distributed.close()
    # a joined row's label is not tested in WHERE
    with pytest.raises(CompileError, match="one table|filter on a label"):
        session.sql(
            f"SELECT d.id, a.id FROM documents d JOIN aspects a "
            f"ON AI_FILTER(PROMPT('Does {{0}} mention {{1}}?', d.body, "
            f"a.aspect)) "
            f"WHERE {stance} = 'praise'")


def test_planner_prices_the_rules_and_takes_the_cheapest():
    from quail.cost import budgets
    from quail.planner.operators.classify import ClassifyScoring
    from quail.planner.validation import ClassifyRefusedError

    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    capacity = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, chunk)
    long_labels = tuple(tuple(range(100 + 5 * i, 104 + 5 * i))
                        for i in range(30))
    letters = tuple((200 + i,) for i in range(30))
    table = ClassifyScoring(alias="d", mean=200.0, longest=300, budget=chunk,
                   chunk=chunk, model=QWEN3_4B_FP8, device=H100_SXM, tokenizer=_bytes,
                   capacity=capacity, lengths=(200,) * 1000, tree=True)
    # thirty four-token labels sharing nothing, on resident documents:
    # the greedy decode asks the cue with the frame and decides at its
    # row; the letters read one row after a frame 30 tokens longer;
    # the packed trie streams 91 tokens a document
    scoring, chosen = table.choose(1000, 20, 30, long_labels, True,
                                   lettered=(20, 60, letters))
    assert scoring == "trie_decode" and chosen.suffix_tokens == 1000
    lettered = table.estimate("letters", 1000, 20, 60, letters, True)
    packed = table.estimate("trie_tree", 1000, 20, 30, long_labels, True)
    assert packed.suffix_tokens == 1000 * len(label_trie(long_labels))
    assert chosen.seconds < lettered.seconds < packed.seconds
    # one-token labels: the decode and the trie read the cue row after
    # the shorter prompt that names the labels; the decode asks it with
    # the frame instead of streaming it after
    ones = tuple((300 + i,) for i in range(3))
    scoring, _ = table.choose(1000, 20, 30, ones, True,
                              lettered=(20, 60, letters[:3]))
    assert scoring == "trie_decode"
    # fresh documents: a greedy decode feeds the cue and decides at its
    # row, as no two labels share a first token, and costs least
    scoring, decoded = table.choose(1000, 20, 30, long_labels, False,
                                    lettered=(20, 60, letters))
    assert scoring == "trie_decode"
    assert decoded.suffix_tokens == 1000 * 1
    fresh_trie = table.estimate("trie_tree", 1000, 20, 30, long_labels, False)
    assert decoded.rounds == 1 and decoded.seconds < fresh_trie.seconds
    # labels that share a first token decide at a second choice; the
    # decode feeds a run of their shared tokens with the cue
    shared = tuple((100, 101) + ids[2:] for ids in long_labels)
    decoded = table.estimate("trie_decode", 1000, 20, 30, shared, False)
    assert decoded.rounds == 1 and decoded.suffix_tokens == 1000 * 3
    # word starts let the packed trie stop at the word that tells a
    # label apart: here its first, so only the cue row is fed and read
    worded = tuple((False, True, True, True) for _ in long_labels)
    decided = table.estimate("trie_tree", 1000, 20, 30, long_labels, True,
                             worded)
    assert decided.suffix_tokens == 1000
    assert decided.seconds < lettered.seconds < packed.seconds
    # a decode cannot end at a label that is another label's prefix
    prefixed = long_labels[:-1] + (long_labels[0][:2],)
    assert table.choose(1000, 20, 30, prefixed, False,
                        lettered=(20, 60, letters))[0] == "letters"
    assert table.choose(1000, 20, 30, prefixed, False)[0] == "trie_tree"
    # without tree attention or a lettered prompt, the decode alone;
    # with neither and no decode, nothing can run
    unified = ClassifyScoring(**{**table.__dict__, "tree": False})
    assert unified.choose(1000, 20, 30, long_labels, False)[0] == "trie_decode"
    with pytest.raises(ClassifyRefusedError, match="no label scoring rule"):
        unified.choose(1000, 20, 30, prefixed, True)


def test_streamed_classifications_share_one_pipeline(session):
    query = _classify_then_filter(session)
    plan = query.plan()
    classifies = [n for n in plan.nodes if isinstance(n, AiClassify)]
    topic, kind = classifies
    assert [node.spec.name for node in classifies] == ["topic", "kind"]
    assert tuple(
        member.node_id for member in build_pipelines(plan.graph)[topic.node_id]) == (
        topic.node_id, "filter:d:0", kind.node_id)
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    assert kind.spec.estimated_seconds < topic.spec.estimated_seconds

    scorer = _labels(["refund", "praise"])
    result = finish(query, scorer)
    assert result.collect().to_pydict() == {
        "d.id": [7], "topic": ["refund"], "kind": ["complaint"]}
    labels = result.answer_tables["classifies"]
    assert labels["topic"].column("topic").to_pylist() == ["refund", "praise"]
    assert labels["kind"].select(["d", "kind"]).to_pydict() == {
        "d": [0], "kind": ["complaint"]}
    assert scorer.calls == [("topic", [[0], [1]]), ("kind", [[0]])]

    split = _classify_then_filter(session, filters=2)
    assert tuple(
        member.node_id for member in build_pipelines(
            split.plan().graph)[topic.node_id]) == (
        topic.node_id, "filter:d:0", "filter:d:1", kind.node_id)

    # a later test on a label runs right after its classification, so
    # the next classification reads fewer documents
    interleaved = _classify_then_filter(session, later_filters=(
        ("kind", ("complaint",)), ("topic", ("refund",))))
    assert tuple(
        member.node_id for member in build_pipelines(
            interleaved.plan().graph)[topic.node_id]) == (
        topic.node_id, "filter:d:0", "filter:d:2", kind.node_id, "filter:d:1")
    scorer = _labels(["shipping", "praise"])
    result = finish(interleaved, scorer)
    assert result.collect().num_rows == 0
    # the second test on topic drops the last document before kind
    assert scorer.calls == [("topic", [[0], [1]])]


def test_classify_refusals_and_builder_errors(session):
    query = _topic(session)
    # a label the classification cannot return would match no row
    with pytest.raises(CompileError, match="not among its labels"):
        (session.docs("documents").alias("d").ai_classify(
            quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
            name="x").label_in("x", ["c"]).select("d.id"))
    with pytest.raises(CompileError, match="differ ignoring case"):
        session.docs("documents").alias("d").ai_classify(
            quail.prompt("{0}", quail.col("d.body")), ["a", "A"], name="x")
    mixed = session.docs("documents").alias("d").ai_filter(
        quail.prompt("Is {0} short?", quail.col("d.body"))).ai_classify(
        quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
        name="x").select("d.id", "x")
    # beside AI.IF the classification is a step of the general plan
    assert [type(node).__name__ for node in mixed.plan().nodes] == [
        "Scan", "AiFilter", "AiClassify", "Project"]
    assert not isinstance(query.plan(), Refusal)

    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.scoring in ("letters", "trie_tree", "trie_decode")
    # Qwen3 32B keeps its separate output head, so it classifies too
    big = quail.Session(EngineConfig(model="qwen3-32b-fp8", device="h100-sxm"),
                        tokenizer=_bytes)
    big.register("documents", session.catalog.get("documents"))
    assert not isinstance(_topic(big).plan(), Refusal)
    big.close()

    # stock vLLM classifies too, with one decode request per document
    vllm = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                      backend="stock_vllm"), tokenizer=_bytes)
    vllm.register("documents", session.catalog.get("documents"))
    request = next(node for node in _topic(vllm).plan().nodes
                   if isinstance(node, RequestExecution))
    assert [spec.output for spec in request.classifies] == ["topic"]
    # it decodes the label as text, so it has no label probabilities
    probable = vllm.docs("documents").alias("d").ai_classify(
        quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
        name="x", probabilities=True).select("d.id", "x")
    assert probable.plan().constraint == "classify_probabilities_need_quail_backend"
    vllm.close()


def test_bench_reads_classify_plans_and_reports_labels_by_operator():
    plan = get_query("IMDB-14").info
    (sentiment, complaint) = plan.classifies
    (critical,) = plan.in_lists
    assert (sentiment.output, complaint.output) == ("sentiment", "complaint")
    assert critical.accepted == ("negative", "mixed")
    assert plan.select == ("r.id", "r.sentiment", "r.complaint")

    tables = {"reviews": pa.table({"id": ["a", "b", "c"]})}
    result = SimpleNamespace(
        answer_tables={
            "filters": {("r", 0): pa.table({
                "r": [0, 1, 2], "answer": [True, False, True]})},
            "joins": {},
            "classifies": {
                "sentiment": pa.table({
                    "r": pa.array([0, 1, 2], pa.int32()),
                    "sentiment": ["negative", "positive", "mixed"]}),
                "complaint": pa.table({
                    "r": pa.array([0, 2], pa.int32()),
                    "complaint": ["poor acting", "too long"]})},
        },
        report={"wall_s": 1.0},
        collect=lambda: pa.table({
            "d": ["a", "c"], "s": ["negative", "mixed"],
            "c": ["poor acting", "too long"]}))
    output = run_output(result, plan, tables)
    assert output.filter_answers == {}
    assert output.rows.column_names == ["r", "sentiment", "complaint"]
    assert output.classify_answers[complaint.id].to_pydict() == {
        "r": ["a", "c"], "label": ["poor acting", "too long"]}
    assert output.classify_answers[sentiment.id].column("r").to_pylist() == [
        "a", "b", "c"]


def test_planner_reads_letters_on_a_diffusion_model(tmp_path):
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({"id": [1, 2], "body": ["ab", "cd"]}), path)
    session = quail.Session(
        EngineConfig(model=DIFFUSION_GEMMA_26B_FP8.name, device="h100-sxm"),
        tokenizer=_bytes)
    session.register("documents", DocumentProvider.from_parquet(path, id_col="id"))
    # the reply opens after the turn suffix, so a letter is scored
    # without a leading space; one read beats 48 denoising steps
    plan = _topic(session).plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.scoring == "letters"
    assert [list(ids) for ids in classify.spec.label_token_ids] == [
        _bytes(letter) for letter in "ABC"[:len(classify.spec.labels)]]
    assert classify.spec.estimated_seconds > 0
    streamed = _classify_then_filter(session).plan()
    classifies = [n for n in streamed.nodes if isinstance(n, AiClassify)]
    assert len(classifies) == 2
    assert tuple(
        member.node_id for member in build_pipelines(
            streamed.graph)[classifies[0].node_id]) == (
        classifies[0].node_id, "filter:d:0", classifies[1].node_id)
    session.close()


def test_trie_paths_cover_every_node_once_and_gather_ancestors():
    from quail.execution.labels import tree_scores, trie_paths

    # labels a=[1,2,3], b=[1,2,4], c=[1,5,6], d=[7,8]: the rows are the
    # nodes with children, (), (1,), (1,2), (1,5), and (7,)
    labels = ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8))
    paths = trie_paths(labels)
    assert [nodes for nodes, _, _ in paths] == [
        [(), (1,), (1, 2)], [(1, 5)], [(7,)]]
    assert [start for _, start, _ in paths] == [0, 2, 1]
    # (1, 5) reads the cue row and the (1,) row; (7,) reads the cue row
    assert [gathers for _, _, gathers in paths] == [[], [(0, 2)], [(0, 1)]]
    nodes = [node for path, _, _ in paths for node in path]
    assert sorted(nodes) == sorted(label_trie(labels))
    # a label whose node is not on the first path reads the right row
    targets = [1, 2, 3, 4, 5, 6, 7, 8]
    logprobs = np.full((len(nodes), len(targets)), -9.0)
    logprobs[0, 0] = -1.0     # () -> 1
    logprobs[0, 6] = -2.0     # () -> 7
    logprobs[1, 1] = -0.5     # (1,) -> 2
    logprobs[1, 4] = -0.25    # (1,) -> 5
    logprobs[2, 2] = -3.0     # (1, 2) -> 3
    logprobs[2, 3] = -0.1     # (1, 2) -> 4
    logprobs[3, 5] = -0.5     # (1, 5) -> 6
    logprobs[4, 7] = -0.3     # (7,) -> 8
    scores = tree_scores(labels, paths, targets, logprobs)
    assert scores.tolist() == [-4.5, -1.6, -1.75, -2.3]
    assert best_label(scores) == 1
    assert best_label([-1.0, -2.0, -1.0]) == 0
    with pytest.raises(ValueError, match="no tokens"):
        label_trie([(1,), ()])
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_rows=1, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 93)), labels=("a", "b", "c", "d"),
        label_token_ids=labels, scoring="trie_tree")
    requests = label_requests(spec)
    # the cue, then each node's last token in path order
    assert requests.suffixes == [[93, 1, 2, 5, 7]]
    assert requests.paths == paths


def test_tree_scores_give_a_prefix_label_only_its_longest_matches():
    from quail.execution.labels import tree_scores, trie_paths

    # yes=[1], "yes, partly"=[1,2,3], no=[4]; rows (), (1,), (1, 2)
    labels = ((1,), (1, 2, 3), (4,))
    paths = trie_paths(labels)
    targets = [1, 2, 3, 4]
    logprobs = np.full((3, 4), -9.0)
    logprobs[0, 0] = np.log(0.7)     # () -> yes
    logprobs[0, 3] = np.log(0.3)     # () -> no
    logprobs[1, 1] = np.log(0.8)     # yes -> ,
    logprobs[2, 2] = np.log(0.9)     # yes, -> partly
    scores = tree_scores(labels, paths, targets, logprobs)
    assert np.exp(scores) == pytest.approx([0.196, 0.504, 0.3])
    assert best_label(scores) == 1

    # a=[1], ab=[1,2], abc=[1,2,3]: a gives up only ab's share, which
    # already holds abc's
    labels = ((1,), (1, 2), (1, 2, 3))
    paths = trie_paths(labels)
    logprobs = np.full((3, 4), -9.0)
    logprobs[0, 0] = np.log(0.5)
    logprobs[1, 1] = np.log(0.6)
    logprobs[2, 2] = np.log(0.5)
    scores = tree_scores(labels, paths, targets, logprobs)
    assert np.exp(scores) == pytest.approx([0.2, 0.15, 0.15])

    # an extension that holds all of a label's probability leaves it none
    logprobs[1, 1] = 0.0
    scores = tree_scores(labels, paths, targets, logprobs)
    assert scores[0] == -np.inf
    assert np.exp(scores[1:]) == pytest.approx([0.25, 0.25])


def test_trie_tree_reads_through_the_deciding_word():
    from quail.execution.labels import path_reads, tree_scores, trie_paths
    from quail.labels import fed_nodes, read_nodes, word_starts

    pieces = {1: b" hello", 2: b" my", 3: b" name", 4: b" is", 5: b" sh",
              6: b"re", 7: b"ya", 8: b" elephant", 9: b" in", 10: b" the",
              11: b" room"}
    hello = ((1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11))
    starts = word_starts(hello, lambda ids: pieces[ids[0]])
    assert starts == ((False, True, True, True, True, False, False),
                      (False, True, True, True))
    # the labels differ at their first word: only the cue row is read,
    # and no label token is fed
    read = read_nodes(hello, starts)
    assert read == {()} and fed_nodes(read) == {()}
    paths = trie_paths(hello, read)
    assert paths == [([()], 0, [])] and path_reads(paths, read) == [True]
    scores = tree_scores(hello, paths, [1, 8], np.log([[0.5, 0.4]]), read)
    assert np.exp(scores) == pytest.approx([0.5, 0.4])
    # " sp" + "am" against " ham": the deciding token is part of a
    # word, so the word is read to its end, and " special" does not
    # count as spam
    spam = ((1, 2), (3,))
    starts = word_starts(spam, lambda ids: {1: b" sp", 2: b"am", 3: b" ham"}[ids[0]])
    read = read_nodes(spam, starts)
    assert read == {(), (1,)}
    paths = trie_paths(spam, read)
    logprobs = np.log([[0.45, 1e-9, 0.40], [1e-9, 0.6, 1e-9]])
    scores = tree_scores(spam, paths, [1, 2, 3], logprobs, read)
    assert np.exp(scores) == pytest.approx([0.27, 0.40])
    # " category" ":" " alpha" / " beta" / " other": the punctuation
    # joins the word after it; " alpha" and " beta" are whole words
    pieces = {1: b" category", 2: b":", 3: b" alpha", 4: b" beta", 5: b" other"}
    labels = ((1, 2, 3), (1, 2, 4), (5,))
    starts = word_starts(labels, lambda ids: pieces[ids[0]])
    assert starts == ((False, False, False), (False, False, False), (False,))
    assert read_nodes(labels, starts) == {(), (1,), (1, 2)}
    # a word after the deciding one is fed only when a later read needs
    # it: " the cat sat" / " the cat ran" feed " the cat" and read once
    pieces = {1: b" the", 2: b" cat", 3: b" sat", 4: b" ran"}
    cat = ((1, 2, 3), (1, 2, 4))
    read = read_nodes(cat, word_starts(cat, lambda ids: pieces[ids[0]]))
    assert read == {(), (1, 2)} and fed_nodes(read) == {(), (1,), (1, 2)}
    paths = trie_paths(cat, read)
    assert paths == [([(), (1,), (1, 2)], 0, [])]
    assert path_reads(paths, read) == [True, False, True]
    logprobs = np.log([[0.9, 1e-9, 1e-9, 1e-9], [1e-9, 1e-9, 0.3, 0.6]])
    scores = tree_scores(cat, paths, [1, 2, 3, 4], logprobs, read)
    assert np.exp(scores) == pytest.approx([0.27, 0.54])
    # digits after a lone space token continue its word
    year = ((1, 2, 3), (1, 2, 4))
    starts = word_starts(year, lambda ids: {1: b" ", 2: b"2", 3: b"4", 4: b"5"}[ids[0]])
    assert starts == ((False, False, False), (False, False, False))
    # without word starts every node is read, as before
    assert read_nodes(hello) == set(label_trie(hello))
    assert word_starts(hello, None) == tuple((False,) * len(ids) for ids in hello)


def test_pack_chunk_packs_paths_as_segments_reading_ancestors(monkeypatch):
    from helpers.fakes import cpu_staging, plain_arena

    from quail.execution.labels import trie_paths

    torch = cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 0)
    arena.activate(key, 4, capacity_tokens=16, base_tokens=4)
    paths = trie_paths(((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8)))
    group = dict(key=key, prefix=None, f=4, suffixes=[[93, 1, 2, 5, 7]],
                 read_all_rows=True, paths=paths)
    chunk = chunk_mod.pack_chunk(torch, arena, [group], attention_mode="tree")
    # rows: cue, (1,), (1,2) | (1,5) | (7,): positions past the frame
    assert chunk.positions.tolist() == [4, 5, 6, 6, 5]
    assert chunk.meta["cu_a"].tolist() == [0, 3, 4, 5]
    reads = chunk.meta["reads"]
    assert reads["rows"].tolist() == [0, 1, 2, 3, 4]
    nodes = reads["nodes"]
    assert nodes["rows"].tolist() == [3, 4]
    assert nodes["key_rows"].tolist() == [0, 1, 0]
    assert nodes["cu_q"].tolist() == [0, 1, 2]
    assert nodes["cu_k"].tolist() == [0, 2, 3]
    assert nodes["b_index"].tolist() == [3, 4]
    assert chunk.final_indices.tolist() == [0, 1, 2, 3, 4]
    assert chunk.rows_per_answer.tolist() == [5]
    with pytest.raises(ValueError, match="tree attention"):
        chunk_mod.pack_chunk(torch, arena, [group], attention_mode="unified")
    # a read mask keeps every row in the pass and reads the marked ones
    masked = dict(group, read_mask=[True, False, True, True, False])
    chunk = chunk_mod.pack_chunk(torch, arena, [masked], attention_mode="tree")
    assert chunk.positions.tolist() == [4, 5, 6, 6, 5]
    assert chunk.final_indices.tolist() == [0, 2, 3]
    assert chunk.rows_per_answer.tolist() == [3]
    with pytest.raises(ValueError, match="read_mask"):
        chunk_mod.pack_chunk(torch, arena, [dict(group, read_mask=[True])],
                             attention_mode="tree")


def test_merge_partial_equals_attention_over_the_union_of_keys():
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.attention import merge_partial

    torch.manual_seed(0)
    heads, dim, keys = 2, 4, 6
    q = torch.randn(3, heads, dim)
    k = torch.randn(keys, heads, dim)
    v = torch.randn(keys, heads, dim)

    def attend(rows, key_slice):
        scores = torch.einsum("rhd,khd->hrk", q[rows], k[key_slice])
        weights = scores.softmax(-1)
        out = torch.einsum("hrk,khd->rhd", weights, v[key_slice])
        return out, scores.logsumexp(-1)

    out_b, lse_b = attend(slice(0, 3), slice(0, 4))
    out_c, lse_c = attend(slice(1, 3), slice(4, 6))
    whole, _ = attend(slice(1, 3), slice(0, 6))
    merge_partial(out_b, lse_b, torch.tensor([1, 2]), out_c, lse_c)
    assert torch.allclose(out_b[1:], whole, atol=1e-5)
    assert torch.allclose(out_b[0], attend(slice(0, 1), slice(0, 4))[0][0])


@pytest.mark.parametrize("worded", [False, True])
def test_classifier_scores_the_packed_trie(monkeypatch, worded):
    from quail.execution.labels import trie_paths
    from quail.labels import read_nodes

    # "a b c" / "a b d" / "a e f" / "g h": with word starts the shared
    # " a" is fed unread, and nothing after a label's deciding word
    labels = ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8))
    starts = tuple((False,) + (True,) * (len(ids) - 1) for ids in labels)
    read = read_nodes(labels, starts if worded else ())
    paths = trie_paths(labels, read)
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_rows=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 93)), labels=("a", "b", "c", "d"),
        label_token_ids=labels, scoring="trie_tree",
        label_word_starts=starts if worded else ())
    requests = label_requests(spec)
    targets = requests.targets
    nodes = [node for path, _, _ in paths for node in path]
    # "a e" and "g" are decided at their first word, so " e" and " g"
    # are never fed
    assert requests.suffixes == ([[93, 1, 2]] if worded else [[93, 1, 2, 5, 7]])
    assert requests.read_mask == [True] * (3 if worded else 5)
    read_count = sum(requests.read_mask)

    # document 0 prefers label b, document 1 label d
    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            wanted = labels[1 if document == 0 else 3]
            if entry.get("paths") is None:
                rows.append([0.0] * len(targets))     # the frame entry
                continue
            assert entry["paths"] == paths
            for node, is_read in zip(nodes, requests.read_mask):
                if is_read:
                    rows.append([
                        -1.0 if node + (token,) == wanted[:len(node) + 1]
                        else -5.0 for token in targets])
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    def submit(rows, rows_per_answer=None):
        # one record per answer: the frame entry's one row, a suffix's
        # read rows
        out = np.full((len(rows_per_answer), read_count, len(targets)), np.nan)
        row = 0
        for index, count in enumerate(rows_per_answer):
            out[index, :count] = rows[row:row + count]
            row += count
        return out

    readout = fake_readout(
        targets=np.asarray(targets), rows=read_count, normalize=True,
        dtype=np.dtype((np.float32, (read_count, len(targets)))), submit=submit)
    state = QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(64),
            pipeline=fake_pipeline(forward_chunk=forward),
            label_readout=readout,
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object(), model_spec=QWEN3_4B_FP8,
        ),
        torch=fake_torch(),
        chunk_tokens=64,
        answer_rows=object(),
        async_answers=fake_readout(),
    )
    batch = QuailClassifier(state).classify(
        spec, [[0], [1]], {"d": [[10, 11], [12]]})
    assert list(batch.scores) == ["b", "d"]
    # the trie's fed rows once per document
    assert batch.suffix_tokens == 2 * (3 if worded else 5)


def test_sql_category_forms_options_and_label_tables(session):
    from quail.logical.prompts import CLASSIFY_INSTRUCTION

    def call_of(sql):
        query = session.sql(sql)
        (column,) = [c for c in query.logical.root.columns if isinstance(c, Alias)]
        return column.expression

    plain = call_of("SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', "
                    "d.body), ARRAY['refund', 'praise']) AS topic "
                    "FROM documents d")
    named = call_of("SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', "
                    "d.body), categories => ARRAY['refund', 'praise']) AS topic "
                    "FROM documents d")
    assert named == plain
    pairs = call_of("SELECT d.id, AI.CLASSIFY(d.body, ARRAY[('refund', 'money "
                    "back'), ('praise', NULL)]) AS topic FROM documents d")
    objects = call_of("SELECT d.id, AI.CLASSIFY(d.body, ARRAY[{'label': 'refund',"
                      " 'description': 'money back'}, {'label': 'praise'}]) "
                      "AS topic FROM documents d")
    assert pairs == objects
    assert pairs.labels == ("refund", "praise")
    assert pairs.descriptions == ("money back", "")
    assert "- refund: money back" in pairs.prompt.tail
    # a bare column asks the default instruction; a task description
    # follows it
    assert pairs.prompt.tail.startswith("{0}\n\n" + CLASSIFY_INSTRUCTION)
    tasked = call_of("SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                     "{'task_description': 'Pick the request kind.'}) AS topic "
                     "FROM documents d")
    assert "Pick the request kind." in tasked.prompt.tail
    # a registered label table, in ordinal order, then by label text
    session.register("kinds", DocumentProvider.from_table(pa.table({
        "label": ["praise", "refund"], "description": [None, "money back"],
        "ordinal": [2, 1]}), id_col="label"))
    table = call_of("SELECT d.id, AI.CLASSIFY(d.body, kinds) AS topic "
                    "FROM documents d")
    assert table.labels == ("refund", "praise")
    assert table.descriptions == ("money back", "")
    session.register("plain_kinds", DocumentProvider.from_table(pa.table({
        "label": ["refund", "praise"]}), id_col="label"))
    assert call_of("SELECT d.id, AI.CLASSIFY(d.body, plain_kinds) AS topic "
                   "FROM documents d").labels == ("praise", "refund")
    built = (session.docs("documents").alias("d")
             .ai_classify(quail.prompt("{0}", quail.col("d.body")), "kinds",
                          name="topic").select("d.id", "topic"))
    (column,) = [c for c in built.logical.root.columns if isinstance(c, Alias)]
    assert column.expression == table
    long_text = " ".join(["word"] * 26)
    for bad in (f"SELECT d.id, AI.CLASSIFY(d.body, ARRAY[('refund', '{long_text}'),"
                f" ('praise', NULL)]) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'examples': 'x'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'layout': 'labels_first'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'probabilities': 'yes'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY[1, 2]) AS topic "
                "FROM documents d"):
        with pytest.raises(CompileError):
            session.sql(bad)


def test_choice_letters_are_one_token_each_and_capped():
    from quail.logical.prompts import MAX_LETTERS
    from quail.planner.prompt_lowering import choice_letters

    assert choice_letters(30) == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcd")
    assert choice_letters(60)[52:56] == ("AA", "AB", "AC", "AD")
    # every single letter maps to one shared id, and a two-letter
    # answer is three bytes
    same = lambda text: [1] if len(text) == 2 else _bytes(text)  # noqa: E731
    assert choice_letters(60, same) == ("A",)    # one token id for all
    assert len(choice_letters(60, letter_tokens)) == 60
    assert choice_letters(60, _bytes) == ()
    # without a prefix a single letter is one byte
    assert len(choice_letters(60, _bytes, prefix="")) == 52
    with pytest.raises(CompileError, match="at most"):
        choice_letters(MAX_LETTERS + 1)


def test_classify_prompt_letters_its_categories():
    from quail.logical.prompts import LETTERS_INSTRUCTION, answer_prefix

    ref = ColumnRef("t", "agent_traces", "trace")
    bound = bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES,
                                 AGENT_PROGRESS_DESCRIPTIONS)
    lettered = lower_prompt(bound, letter_tokens).lettered
    assert lettered.letters == tuple("ABCDE") and bound.letters == ()
    assert lettered.preamble == bound.preamble
    assert LETTERS_INSTRUCTION in lettered.tail
    for letter, label, description in zip(
            "ABCDE", STAGES, AGENT_PROGRESS_DESCRIPTIONS):
        assert f"\n- {letter}: {label} ({description})" in lettered.tail
        assert f"\n- {label}: {description}" in bound.tail
    assert lettered.lettered is None
    # a label follows the answer cue after a space; a reply that opens
    # a model turn starts without one
    assert bound.label_prefix == " " and answer_prefix(("", "")) == " "
    turned = bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES,
                                  turn=("<user>", "<model>"))
    assert turned.label_prefix == ""
    assert lower_prompt(turned, letter_tokens).lettered.tail.endswith("<model>")
    # no lettered prompt when the tokenizer has no one-token letters
    assert lower_prompt(bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES),
                        _bytes).lettered is None
    tokenized = lower_prompt(bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES),
                             letter_tokens)
    assert list(tokenized.lettered.tail_token_ids) == _bytes(
        tokenized.lettered.tail.replace("{0}", ""))


def test_classifier_reads_the_letter_at_the_first_canvas_row(monkeypatch):
    from quail.specs.base import CANVAS_SEED, AnswerCanvas

    settings = AnswerCanvas(rows=4, turn_close_id=6, pad_id=0)
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_rows=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((1,), (2,)), scoring="letters")
    vocab = 8
    wanted = {0: 1, 1: 2}      # document index -> the letter its first row favors
    packed = {}

    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            row = entry["key"][2]
            packed[row] = list(entry["canvas"])
            # the frame and the cue pack as one entry before the canvas
            assert [list(s) for s in entry["suffixes"]] == [[91, 92, 93]]
            assert entry["read_all_rows"]
            for position in range(settings.rows):
                rows.append([-1.0 if position == 0 and token == wanted[row]
                             else -5.0 for token in (1, 2)])
        return np.asarray(rows, dtype=np.float32)

    def submit(rows, rows_per_answer=None):
        padded = np.full((len(rows_per_answer), 4, 2), np.nan, np.float32)
        start = 0
        for answer, count in enumerate(rows_per_answer):
            padded[answer, :count] = rows[start:start + count]
            start += count
        return padded

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    readout = fake_readout(targets=np.asarray([1, 2]), rows=4,
                              dtype=np.dtype((np.float32, (4, 2))),
                              submit=submit, result=lambda rows: rows)
    state = QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(64),
            pipeline=fake_pipeline(forward_chunk=forward, canvas_ids=(7,),
                                   tree_attention=False),
            model_spec=SimpleNamespace(name="tiny", vocab=vocab,
                                       answer_canvas=settings),
            label_readout=readout,
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object(),
        ),
        torch=fake_torch(),
        chunk_tokens=64,
        answer_rows=object(),
        async_answers=fake_readout(),
    )
    documents = {"d": [[9]] * 4 + [[10, 11], [12]]}
    batch = QuailClassifier(state).classify(spec, [[4], [5]], documents)
    assert list(batch.scores) == ["a", "b"]
    # each canvas: a random token from the document's row, the turn
    # close, then padding
    for index, row in enumerate((4, 5)):
        noise = np.random.default_rng((CANVAS_SEED, row, 0)).integers(0, vocab)
        assert packed[index] == [noise, 6, 0, 0]
    # the cue and the canvas after the prompt and the frame, once
    assert batch.suffix_tokens == 2 * (1 + 4)
    assert batch.fresh_tokens + batch.cached_tokens == (3 + 2) + 2 * (2 + 5)


def test_a_label_in_where_names_the_classification_by_its_as_name(session):
    call = ("AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
            "ARRAY['refund', 'shipping', 'praise'])")
    named = session.sql(
        f"SELECT d.id, {call} AS topic FROM documents d "
        f"WHERE topic IN ('refund', 'shipping')")
    repeated = session.sql(
        f"SELECT d.id, {call} AS topic FROM documents d "
        f"WHERE {call} IN ('refund', 'shipping')")
    assert named.logical == repeated.logical
    equal = session.sql(
        f"SELECT d.id, {call} AS topic FROM documents d "
        f"WHERE HINT(topic = 'refund', {{'selectivity': 0.3}})")
    assert str(equal.logical.root.input.condition) == "d.topic IN ('refund')"
    assert equal.logical.root.input.selectivity == 0.3
    builder = (session.docs("documents").alias("d")
               .ai_classify(quail.prompt("What is {0} about?", quail.col("d.body")),
                            ["refund", "shipping", "praise"], name="topic")
               .where((quail.col("topic") == "refund").hint(selectivity=0.3)))
    assert builder.select("d.id", "topic").logical == equal.logical
    with pytest.raises(CompileError, match="== or isin"):
        builder.where(quail.col("topic").like("re%"))
    # a name that is also a column of a table in scope is ambiguous
    with pytest.raises(CompileError, match="both a column"):
        session.sql(f"SELECT d.id, {call} AS body FROM documents d "
                    f"WHERE body = 'refund'")
