"""A model operator over copied documents is asked once per distinct text."""

import pyarrow as pa
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.physical import Expand, Representatives

BODIES = ["noisy room", "cold room", "noisy room", "slow lift", "noisy room",
          "cold room"]


def _session():
    session = quail.Session(CONFIG, tokenizer=fake_tok)
    session.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": [f"r{index}" for index in range(len(BODIES))], "body": BODIES,
    }), id_col="id"))
    return session


def test_a_filter_over_copies_asks_once_per_text_and_returns_every_copy():
    asked = []

    def noisy(alias, document, written_position):
        asked.append(document)
        return BODIES[document].startswith("noisy")

    with _session() as session:
        query = (session.docs("reviews").alias("r")
                 .ai_filter(quail.prompt("Is {0} about noise?", quail.col("r.body")))
                 .select("r.id"))
        query.plan()
        # the planner groups copies once it has every document's tokens
        query.wait_for_tokens()
        plan = query.plan()
        assert [type(node) for node in plan.nodes
                if isinstance(node, (Representatives, Expand))] == [
                    Representatives, Expand]
        assert "should_ask_once_per_document" in query.explain(verbose=True)
        result = finish(query, Oracle(filters=noisy))
        assert sorted(result.to_rows()) == [("r0",), ("r2",), ("r4",)]
        assert sorted(asked) == [0, 1, 3]
        answers = result.answer_tables["filters"]
        (table,) = answers.values()
        assert table.num_rows == len(BODIES)


def test_a_classification_over_copies_labels_every_copy_from_one_answer():
    with _session() as session:
        query = (session.docs("reviews").alias("r")
                 .ai_classify(quail.prompt("What is {0} about?", quail.col("r.body")),
                              ["noise", "temperature", "lift"], name="topic")
                 .select("r.id", "topic"))
        oracle = Oracle(labels=lambda spec, ids: {
            "noisy room": "noise", "cold room": "temperature",
            "slow lift": "lift"}[BODIES[ids[0]]])
        result = finish(query, oracle)
        assert sorted(result.to_rows()) == [
            ("r0", "noise"), ("r1", "temperature"), ("r2", "noise"),
            ("r3", "lift"), ("r4", "noise"), ("r5", "temperature")]
        asked = sorted(row for _, rows in oracle.calls for (row,) in rows)
        assert asked == [0, 1, 3]
