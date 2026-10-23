"""Model operators over copied documents are asked once per distinct text.

The planner groups the table by text hash before the operators and joins
their answers back to every document, or pair, after them. Each query here
runs with and without that rule and must return the same rows.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.catalog.tokens import first_document_column
from quail.physical import Aggregate, AiFilter, AiJoin, HashJoin
from quail.physical.pipelines import build_pipelines
from quail.planner.rules.physical import ShouldAskOncePerDocument

BODIES = ["noisy room", "cold room", "noisy room", "slow lift", "noisy room",
          "cold room"]
# a copy of a review can sit in another hotel than its first
HOTELS = ["h1", "h2", "h2", "h1", "h3", "h1"]
NOTES = ["noise", "cold", "noise", "lift"]
NOTE_HOTELS = ["h1", "h2", "h2", "h3"]
TOPIC = {"noisy room": "noise", "cold room": "cold", "slow lift": "lift"}


def _session():
    session = quail.Session(CONFIG, tokenizer=fake_tok)
    session.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": [f"r{index}" for index in range(len(BODIES))], "body": BODIES,
        "hotel": HOTELS,
    }), id_col="id"))
    session.register("notes", quail.DocumentProvider.from_table(pa.table({
        "id": [f"n{index}" for index in range(len(NOTES))], "body": NOTES,
        "hotel": NOTE_HOTELS,
    }), id_col="id"))
    return session


def _matches(node, assignment):
    return TOPIC[BODIES[assignment["r"]]] == NOTES[assignment["n"]]


def _sorted(table):
    return sorted(tuple(row.values()) for row in table.to_pylist())


def _run(build, oracle, rule=True, monkeypatch=None, session=_session):
    """The query's rows, its answer tables, its plan, and the oracle's calls."""
    with session() as session, monkeypatch.context() as patch:
        if not rule:
            patch.setattr(ShouldAskOncePerDocument, "rewrite",
                          lambda self, graph, context: None)
        query = build(session)
        # tokenizing starts with the first plan; the rule needs every token
        query.plan()
        query.wait_for_tokens()
        plan = query.plan()
        result = finish(query, oracle)
        answers = {(kind, name): _sorted(table)
                   for kind, tables in result.answer_tables.items()
                   for name, table in tables.items()}
        return sorted(result.to_rows()), answers, plan


def _both(build, grouped_oracle, plain_oracle, monkeypatch, session=_session):
    """Run a query with the rule and without it; both give the same answers."""
    grouped = _run(build, grouped_oracle, True, monkeypatch, session)
    plain = _run(build, plain_oracle, False, monkeypatch, session)
    assert grouped[:2] == plain[:2]
    return grouped


class _Asked:
    """An oracle that records the documents and pairs it is asked about."""

    def __init__(self):
        self.documents, self.pairs = [], []

    def oracle(self, labels=None):
        def filters(alias, document, written_position):
            self.documents.append(document)
            if written_position == 0:
                return not BODIES[document].startswith("slow")
            return BODIES[document].startswith("noisy")

        def joins(node, assignment):
            self.pairs.append((assignment["r"], assignment["n"]))
            return _matches(node, assignment)

        return Oracle(filters=filters, joins=joins, labels=labels)


def test_a_filter_over_copies_asks_once_per_text_and_returns_every_copy(
        monkeypatch):
    grouped, plain = _Asked(), _Asked()

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} about a room?", quail.col("r.body")),
                           selectivity=0.5)
                .select("r.id"))

    rows, _, plan = _both(build, grouped.oracle(), plain.oracle(), monkeypatch)
    # a text hash GROUP BY before the filter, a JOIN back after it
    assert [type(node).__name__ for node in plan.nodes
            if isinstance(node, (Aggregate, AiFilter, HashJoin))] == [
                "Aggregate", "AiFilter", "HashJoin"]
    (aggregate,) = [node for node in plan.nodes if isinstance(node, Aggregate)]
    assert aggregate.alias == "r"
    assert rows == [("r0",), ("r1",), ("r2",), ("r4",), ("r5",)]
    assert (grouped.documents, plain.documents) == ([0, 1, 3], list(range(6)))


def test_a_classification_over_copies_labels_every_copy_from_one_answer(
        monkeypatch):
    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_classify(quail.prompt("What is {0} about?", quail.col("r.body")),
                             ["noise", "cold", "lift"], name="topic")
                .select("r.id", "topic"))

    def oracle():
        return Oracle(labels=lambda spec, ids: TOPIC[BODIES[ids[0]]])

    grouped, plain = oracle(), oracle()
    rows, _, _ = _both(build, grouped, plain, monkeypatch)
    assert rows == [("r0", "noise"), ("r1", "cold"), ("r2", "noise"),
                    ("r3", "lift"), ("r4", "noise"), ("r5", "cold")]
    assert sorted(row for _, rows in grouped.calls for (row,) in rows) == [0, 1, 3]
    assert sorted(row for _, rows in plain.calls for (row,) in rows) == list(range(6))


def test_a_chain_of_filters_is_asked_once_per_text_and_stays_one_unit(
        monkeypatch):
    grouped, plain = _Asked(), _Asked()

    def build(session):
        body = quail.col("r.body")
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} about a room?", body),
                           selectivity=0.5)
                .ai_filter(quail.prompt("Is {0} about noise?", body),
                           selectivity=0.5)
                .select("r.id"))

    rows, _, plan = _both(build, grouped.oracle(), plain.oracle(), monkeypatch)
    assert rows == [("r0",), ("r2",), ("r4",)]
    # one Aggregate before the first filter; the second reads the first
    (aggregate,) = [node for node in plan.nodes if isinstance(node, Aggregate)]
    first, second = [node for node in plan.nodes if isinstance(node, AiFilter)]
    assert first.inputs[0].source.node_id == aggregate.node_id
    assert second.inputs[0].source.node_id == first.node_id
    # each filter asks a text once: three texts, then the one that is left
    assert len(grouped.documents) < len(plain.documents)
    assert sorted(set(grouped.documents)) == sorted(grouped.documents[:3])


def test_a_filter_with_a_stop_key_keeps_a_text_apart_under_each_key(
        monkeypatch):
    # hotel h1 stops at "fine", so its "ok" is never asked; h2's "ok" is
    # a copy of it that must still be asked, as its own key document
    bodies = ["fine", "ok", "ok", "ok", "meh", "meh"]
    hotels = ["h1", "h1", "h2", "h2", "h3", "h3"]
    asked = {True: [], False: []}

    def session():
        session = quail.Session(CONFIG, tokenizer=fake_tok)
        session.register("reviews", quail.DocumentProvider.from_table(pa.table({
            "id": [f"r{index}" for index in range(len(bodies))],
            "body": bodies, "hotel": hotels}), id_col="id"))
        return session

    def oracle(rule):
        def filters(alias, document, written_position):
            asked[rule].append(document)
            return bodies[document] != "meh"
        return Oracle(filters=filters)

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} short?", quail.col("r.body")))
                .distinct().select("r.hotel"))

    rows, answers, plan = _run(build, oracle(True), True, monkeypatch, session)
    plain_rows, plain_answers, _ = _run(build, oracle(False), False, monkeypatch,
                                        session)
    assert rows == plain_rows == [("h1",), ("h2",)]
    (aggregate,) = [node for node in plan.nodes if isinstance(node, Aggregate)]
    assert aggregate.keys == (first_document_column(("hotel",)),)
    assert asked == {True: [0, 2, 4], False: [0, 2, 4, 5]}
    # a copy of an asked document gets its answer even where its stop key
    # group had stopped, so the plain run's answers are a subset
    (grouped_answers,), (plain,) = answers.values(), plain_answers.values()
    assert set(plain) < set(grouped_answers)
    assert (3, 0, True) in grouped_answers


def test_a_join_over_key_pairs_asks_each_pair_of_texts_once(monkeypatch):
    grouped, plain = _Asked(), _Asked()

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_filter(quail.prompt("Is {0} about a room?",
                                        quail.col("r.body")), selectivity=0.5)
                .join(session.docs("notes").alias("n"),
                      on=quail.col("r.hotel") == quail.col("n.hotel"))
                .ai_filter(quail.prompt("Does {0} match {1}?", quail.col("r.body"),
                                        quail.col("n.body")))
                .select("r.id", "n.id"))

    rows, _, plan = _both(build, grouped.oracle(), plain.oracle(), monkeypatch)
    assert rows == [("r0", "n0"), ("r1", "n1"), ("r2", "n2")]
    assert sorted(node.partner for node in plan.nodes
                  if isinstance(node, Aggregate)) == ["", "", "n"]
    # the hotel pairs of the rooms: r0-n0, r1-n1, r1-n2, r2-n1, r2-n2,
    # r4-n3, r5-n0; their key documents' pairs: 0-0, 1-1, 1-0, 0-1, 0-3
    assert sorted(plain.pairs) == [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2),
                                   (4, 3), (5, 0)]
    assert sorted(grouped.pairs) == [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1)]
    # the filter and the join it feeds still run as one pipeline
    (join,) = [node for node in plan.nodes if isinstance(node, AiJoin)]
    (first,) = [node for node in plan.nodes if isinstance(node, AiFilter)]
    pipeline = build_pipelines(plan.graph).get(join.node_id, (join,))
    assert first.node_id in {node.node_id for node in pipeline}


@pytest.mark.parametrize("semantics", ["full", "exists", "anti"])
def test_a_join_of_every_pair_asks_each_pair_of_texts_once(monkeypatch,
                                                           semantics):
    grouped, plain = _Asked(), _Asked()

    def build(session):
        return (session.docs("reviews").alias("r")
                .ai_join(session.docs("notes").alias("n").where(
                             quail.col("n.hotel") != "h3"),
                         quail.prompt("Does {0} match {1}?", quail.col("r.body"),
                                      quail.col("n.body")),
                         anchor="r", semantics=semantics)
                .select(*(("r.id", "n.id") if semantics == "full" else ("r.id",))))

    rows, _, plan = _both(build, grouped.oracle(), plain.oracle(), monkeypatch)
    assert rows == {
        "full": [("r0", "n0"), ("r0", "n2"), ("r1", "n1"), ("r2", "n0"),
                 ("r2", "n2"), ("r4", "n0"), ("r4", "n2"), ("r5", "n1")],
        "exists": [("r0",), ("r1",), ("r2",), ("r4",), ("r5",)],
        # the lift note is in hotel h3, so no note matches r3
        "anti": [("r3",)],
    }[semantics]
    assert set(grouped.pairs) <= {(0, 0), (0, 1), (1, 0), (1, 1), (3, 0), (3, 1)}
    assert len(grouped.pairs) < len(plain.pairs)
    assert any(isinstance(node, HashJoin) and node.semantics == semantics
               for node in plan.nodes)


def test_a_classification_of_joined_pairs_labels_every_pair_from_one_answer(
        monkeypatch):
    def build(session):
        return session.sql(
            "SELECT r.id, n.id, AI.CLASSIFY(PROMPT('How does {0} relate to {1}?', "
            "r.body, n.body), ARRAY['same', 'other']) AS relation "
            "FROM reviews r JOIN notes n ON AI_FILTER(PROMPT('Does {0} match "
            "{1}?', r.body, n.body))")

    grouped, plain = [], []

    def oracle(labeled):
        def label(spec, ids):
            labeled.append(tuple(ids))
            return "same"
        return Oracle(joins=_matches, labels=label)

    rows, _, plan = _both(build, oracle(grouped), oracle(plain), monkeypatch)
    assert len(rows) == 9
    assert {relation for *_, relation in rows} == {"same"}
    # the true pairs of three review texts and three note texts
    assert sorted(grouped) == [(0, 0), (1, 1), (3, 3)]
    assert len(plain) == 9
    assert sum(isinstance(node, HashJoin) for node in plan.nodes) == 2


def test_a_self_join_asks_each_pair_of_texts_once(monkeypatch):
    grouped, plain = [], []

    def oracle(asked):
        def joins(node, assignment):
            asked.append((assignment["a"], assignment["b"]))
            return all("room" in BODIES[assignment[alias]] for alias in "ab")
        return Oracle(joins=joins)

    def build(session):
        return session.sql(
            "SELECT a.id, b.id FROM reviews a JOIN reviews b "
            "ON a.hotel = b.hotel "
            "AND AI_FILTER(PROMPT('Do {0} and {1} agree?', a.body, b.body))")

    rows, _, _ = _both(build, oracle(grouped), oracle(plain), monkeypatch)
    # same-hotel pairs in both orders, each review with itself too: the
    # rooms of h1 are r0 and r5, of h2 r1 and r2, of h3 r4
    assert rows == [("r0", "r0"), ("r0", "r5"), ("r1", "r1"), ("r1", "r2"),
                    ("r2", "r1"), ("r2", "r2"), ("r4", "r4"), ("r5", "r0"),
                    ("r5", "r5")]
    assert len(plain) == 14
    # the pairs of their key documents 0, 1, and 3, on either side
    assert sorted(grouped) == [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3),
                               (3, 0), (3, 1), (3, 3)]
