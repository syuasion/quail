"""AI.EXTRACT: its spec and cost."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.cost.retention import coefficients
from quail.logical import effective_selectivity
from quail.physical import (
    AiExtract,
    ExtractSpec,
    Filter,
    PortRef,
)
from quail.physical.base import input_ports
from quail.planner.operators.common import (
    bind_inference,
    inference_node,
)


def extract_spec(context, call, name, expected_rows, resident=False) -> ExtractSpec:
    """Build an extraction's specification and price it.

    The specification carries the token ids of the prompt around the
    document. The documents' KV is priced as computed afresh unless
    the extraction shares a pipeline with the one before it, which
    read the documents.

    Args:
        context: The planning context, with the model and tokenizer.
        call: Logical AI.EXTRACT call.
        name: Result column name.
        expected_rows: Expected number of documents to extract from.
        resident: Whether the documents' KV is already read, by a
            step of the same pipeline before this one.
    """
    prompt = context.lowered(call.prompt)
    (ref,) = call.prompt.args
    lengths = context.document_tokens[ref.alias]
    model, device = context.model, context.device
    # one document token's KV read per round, in tokens of compute: the
    # pass is compute-bound, so a fed token costs its dense FLOPs
    token_seconds = coefficients(model, device)["linear_seconds"]
    round_cost_per_token = model.kappa / device.hbm_bw / token_seconds
    spec = bind_inference(
        ExtractSpec, call.prompt, name=name,
        expected_rows=expected_rows, estimated_seconds=0.0,
        prompt_token_parts=(tuple(prompt.preamble_token_ids),
                            tuple(prompt.tail_token_ids)),
        question=call.question,
        expected_answer_tokens=extract_cost.ExtractAssumptions().answer_tokens,
        round_cost_per_token=round_cost_per_token)
    cost = extract_cost.estimate(
        expected_rows,
        lengths,
        len(prompt.preamble_token_ids),
        len(prompt.tail_token_ids),
        chunk=budgets.chunk_budget(context.model, context.device),
        model=context.model,
        device=context.device,
        resident=resident,
        max_answer_tokens=spec.max_answer_tokens,
        draft_tokens=spec.draft_tokens,
        draft_policy=spec.draft_policy,
        round_cost_per_token=spec.round_cost_per_token,
        gpus=context.gpus,
        memo=context.memo,
    )
    return replace(spec, estimated_seconds=cost.seconds)


def emit_extract(builder, alias, expected_rows, names, port, resident=False) -> tuple:
    """Copy the answers of the alias's documents.

    A Filter on an extraction's answer or span follows that extraction
    and drops the documents the next extraction would read, as a label
    filter does after its classification.

    Args:
        builder: The graph builder.
        alias: The table.
        expected_rows: The documents the extractions read.
        names: The extractions to emit, in run order.
        port: The port the documents stand at before the extractions.
        resident: Whether the step before the run left the documents'
            KV for the first extraction to read.

    Returns:
        Where the documents stand after them, and the rows ports of
        the answers SELECT returns.
    """
    extracts = {node.name: node for node in builder.operators.extracts
                if node.alias == alias}
    calls = [extracts[name] for name in names]
    tests = builder.operators.extract_filters.get(alias, ())
    row_ports = []
    for index, logical_node in enumerate(calls):
        # the extractions of a table and the Filters on their answers
        # run in one pipeline (quail.physical.pipelines): a document
        # is read once for all of them, and a Filter drops it before
        # the next question
        spec = extract_spec(
            builder.context,
            logical_node.call,
            logical_node.name,
            expected_rows,
            resident=resident or index > 0,
        )
        node = inference_node(
            AiExtract, spec, (port,),
            index=sum(isinstance(n, AiExtract) for n in builder.nodes),
            backend="quail", model=builder.model.name)
        builder.nodes.append(node)
        port = PortRef(node.node_id, node.rows_port)
        for test in tests:
            if test.name != logical_node.name:
                continue
            filter_id = f"filter:{alias}:{test.position}"
            builder.nodes.append(Filter(
                node_id=filter_id, inputs=input_ports((port,)),
                predicate=test.condition, aliases=(alias,),
                selectivity=test.selectivity, written_position=test.position))
            port = PortRef(filter_id, f"rows:{alias}")
            expected_rows *= effective_selectivity(test.selectivity)
        row_ports.append(port)
    return port, row_ports

