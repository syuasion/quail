"""Sharding a group's inputs across workers and merging their outputs."""

import pyarrow as pa

from quail.backends.quail.coordinator import (
    merge_join_answers,
    merge_results,
    shard_input,
)
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import filter_answer_table
from quail.execution.rows import Rows
from quail.physical import InputPort, PortRef, ValueType
from quail.physical.base import input_ports


def test_a_shard_keeps_its_own_rows_and_pairs_and_every_partner():
    ids = pa.array([4, 1, 3], pa.int32())
    rows = Rows({"r": ids}, pa.table({"label": ["a", "b", "c"]}))
    own = InputPort("rows", ValueType.ROWS, PortRef("classify:r", "rows:r"))
    partner = InputPort("partner", ValueType.ROWS, PortRef("scan:p", "rows:p"))
    assert shard_input(own, rows, "r", [1, 4]).table().to_pylist() == [
        {"r": 4, "label": "a"}, {"r": 1, "label": "b"}]
    assert shard_input(partner, Rows({"p": ids}), "r", [1, 4]).ids["p"] is ids
    pairs = pa.table({"r": [4, 1, 1, 3], "p": [0, 1, 2, 0]})
    port = InputPort("pairs", ValueType.PAIRS, PortRef("hash_join:r-p", "pairs:0"))
    assert shard_input(port, pairs, "r", [1]).to_pylist() == [
        {"r": 1, "p": 1}, {"r": 1, "p": 2}]


def test_merged_rows_follow_the_input_order_and_sum_the_metrics():
    from quail.physical import AiFilter

    node = AiFilter(node_id="ai_filter:r", alias="r", question_token_ids=(7,),
                    inputs=input_ports((PortRef("scan:r", "rows:r"),)))
    outputs = [{"results": {node.node_id: NodeResult(
        {"rows:r": Rows({"r": pa.array(kept, pa.int32())}),
         "filter_answers:r": filter_answer_table(
             "r", kept, [0] * len(kept), [True] * len(kept))},
        NodeMetrics(fresh_tokens=len(kept)))}} for kept in ([4, 0], [3])]
    (merged,) = merge_results([node], outputs, "r", [3, 4, 1, 0], 2.0).values()
    assert merged.outputs["rows:r"].ids["r"].to_pylist() == [3, 4, 0]
    assert merged.outputs["filter_answers:r"].column("r").to_pylist() == [3, 4, 0]
    assert (merged.metrics.fresh_tokens, merged.metrics.wall_s) == (3, 2.0)


def test_join_merge_and_partner_lists():
    # the merge keeps stages aligned though stage 2 skips gated anchors
    outs = [
        dict(joins=[dict(rows={0: [1, 0], 1: [0, 0]}, anchor_index=[0, 4],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [0, 1, 1]}, anchor_index=[0, 4],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 1]}, anchor_index=[3],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [1, 0, 0]}, anchor_index=[3],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
    ]
    merged = [merge_join_answers([out["joins"][i] for out in outs])
              for i in range(len(outs[0]["joins"]))]
    assert len(merged) == 2
    assert merged[0]["anchor_index"] == [0, 4, 3]
    assert merged[0]["rows"] == {0: [1, 0], 1: [0, 0], 2: [1, 1]}
    assert merged[1]["anchor_index"] == [0, 4, 3]
    assert merged[1]["rows"] == {0: [0, 1, 1], 2: [1, 0, 0]}
    assert merged[1]["partner_index"] == [[0], [1], [2]]

    outs = [
        dict(joins=[dict(rows={0: [1], 1: [0, 1]}, anchor_index=[0, 4],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [2], 1: [0, 1]})],
             fresh_tokens=1, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 0]}, anchor_index=[3],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [1, 2]})],
             fresh_tokens=1, wall_s=1.0),
    ]
    merged = merge_join_answers([out["joins"][0] for out in outs])
    assert merged["anchor_partners"] == {0: [2], 1: [0, 1], 2: [1, 2]}
