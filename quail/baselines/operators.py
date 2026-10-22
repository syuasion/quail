"""The per-operator request runs of the request backends and their answers."""

from __future__ import annotations

import time

import pyarrow as pa

from quail.catalog.pairs import (
    allowed_members,
    members_by_partner,
    pair_partner,
    partner_map,
)
from quail.execution.labels import match_label
from quail.execution.result import (
    answer_table,
    filter_answer_table,
)
from quail.execution.rows import Rows
from quail.execution.spans import locate_span, read_phrase


def token_list(values) -> list[int]:
    return [int(token) for token in values]


def text_value(values, index: int) -> str:
    value = values[index]
    return value.as_py() if hasattr(value, "as_py") else str(value)

def operator_at_a_time_filter(client, sampling_params, bodies, questions,
                               read_answer, *, body_texts=None,
                               question_texts=None):
    active = list(range(len(bodies)))
    answers = {}
    requests = prompt_tokens = cached_tokens = 0
    started = time.perf_counter()
    stages = []
    for stage_index, question in enumerate(questions):
        evaluated = list(active)
        if body_texts is not None and question_texts is not None:
            prompts = [
                body_texts[index] + question_texts[stage_index]
                for index in evaluated
            ]
        else:
            prompts = [
                {"prompt_token_ids": bodies[index] + list(question)}
                for index in evaluated
            ]
        outputs = (
            client.generate(prompts, sampling_params, use_tqdm=False)
            if prompts else []
        )
        next_active = []
        for document, output in zip(evaluated, outputs):
            answer = bool(read_answer(output))
            answers[(document, stage_index)] = answer
            requests += 1
            prompt_tokens += len(output.prompt_token_ids)
            cached_tokens += int(getattr(output, "num_cached_tokens", 0) or 0)
            if answer:
                next_active.append(document)
        active = next_active
        stages.append({
            "stage": stage_index,
            "num_in": len(evaluated),
            "num_out": len(active),
        })
    return {
        "wall_s": time.perf_counter() - started,
        "survivors": active,
        "answers": answers,
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": prompt_tokens - cached_tokens,
        "stages": stages,
        "doc_cap": None,
    }


def pipelined_filter(client, sampling_params, bodies, questions, read_answer,
                      tag, *, body_texts=None, question_texts=None):
    if not bodies:
        return {
            "wall_s": 0.0,
            "survivors": [],
            "answers": {},
            "requests": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "fresh_tokens": 0,
            "stages": [],
            "doc_cap": 0,
        }
    result = client.run_filter_sequence(
        sampling_params,
        bodies,
        [list(question) for question in questions],
        read_answer,
        tag=tag,
        body_texts=body_texts,
        question_texts=question_texts,
    )
    stages = []
    for stage in range(1, len(questions) + 1):
        evaluated = [
            document for document in range(len(bodies))
            if (document, stage) in result["answers"]
        ]
        stages.append({
            "stage": stage - 1,
            "num_in": len(evaluated),
            "num_out": sum(
                bool(result["answers"][(document, stage)])
                for document in evaluated
            ),
        })
    return {
        "wall_s": result["wall"],
        "survivors": result["survivors"],
        "answers": {
            (document, stage - 1): bool(answer)
            for (document, stage), answer in result["answers"].items()
        },
        "requests": result["requests"],
        "prompt_tokens": result["prompt_tokens"],
        "cached_tokens": result["cached_tokens"],
        "fresh_tokens": result["prompt_tokens"] - result["cached_tokens"],
        "stages": stages,
        "doc_cap": result["doc_cap"],
    }

def filter_answers(alias, written_positions, answers) -> pa.Table:
    documents = []
    predicates = []
    values = []
    for (document, stage_index), answer in sorted(answers.items()):
        documents.append(int(document))
        predicates.append(int(written_positions[stage_index]))
        values.append(bool(answer))
    return filter_answer_table(alias, documents, predicates, values)


def _generation_counts(outputs) -> dict:
    """Count prompt reuse and generated tokens fed into later decode steps."""
    prompt_tokens = cached_tokens = generated_tokens = fed_back = 0
    for output in outputs:
        prompt_tokens += len(output.prompt_token_ids)
        cached_tokens += int(getattr(output, "num_cached_tokens", 0) or 0)
        generated = len(output.outputs[0].token_ids)
        generated_tokens += generated
        fed_back += max(generated - 1, 0)
    return {
        "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens,
        "fresh_tokens": prompt_tokens - cached_tokens + fed_back,
        "generated_tokens": generated_tokens,
    }


def classify_documents(client, spec, bodies) -> dict:
    """Decode one label per prompt body with greedy generation.

    Generation stops after one more token than the longest label has.

    Args:
        client: The engine client.
        spec: The RequestClassifySpec with the labels and the prompt tail.
        bodies: Token ids for each prompt before the tail, preamble included.

    Returns:
        A dict with one label per body (None when the answer names no label),
        the unmatched count, the request and token counts, and the generation
        seconds. Fresh tokens count the uncached prompt tokens and each
        generated token fed back to decode the next one.
    """
    tail = token_list(spec.tail_token_ids)
    longest = max(len(ids) for ids in spec.label_token_ids)
    params = client.decode_params(longest + 1)
    prompts = [{"prompt_token_ids": body + tail} for body in bodies]
    started = time.perf_counter()
    outputs = client.generate(prompts, params, use_tqdm=False) if prompts else []
    wall_s = time.perf_counter() - started
    labels = []
    unmatched = 0
    for output in outputs:
        text = output.outputs[0].text or ""
        label = match_label(text, spec.labels)
        unmatched += label is None
        labels.append(label)
    return {
        "labels": labels,
        "wall_s": wall_s,
        "requests": len(prompts),
        **_generation_counts(outputs),
        "unmatched": unmatched,
    }


def choose_documents(client, spec, bodies) -> dict:
    """Score one option block per label per prompt body; take the best label.

    Args:
        client: The decision engine client.
        spec: The RequestClassifySpec with the labels, the prompt tail,
            and the option offsets.
        bodies: Token ids for each prompt before the tail, preamble included.

    Returns:
        The same fields as classify_documents; no answer is unmatched
        and none generates tokens.
    """
    from quail.baselines.vllm_decision import choice_index, pooling_params

    tail = token_list(spec.tail_token_ids)
    prompts = [{"prompt_token_ids": body + tail} for body in bodies]
    started = time.perf_counter()
    outputs = (client.generate(prompts, pooling_params(spec.option_offsets))
               if prompts else [])
    wall_s = time.perf_counter() - started
    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    cached_tokens = sum(int(output.num_cached_tokens or 0) for output in outputs)
    return {
        "labels": [spec.labels[choice_index(output)] for output in outputs],
        "wall_s": wall_s,
        "requests": len(prompts),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": prompt_tokens - cached_tokens,
        "generated_tokens": 0,
        "unmatched": 0,
    }


def classification_rows(spec, columns, labels) -> Rows:
    """Build classified rows and remove rows without a matching label."""
    rows = Rows(columns, pa.table({spec.output: pa.array(labels, pa.string())}))
    return rows.filter(rows.column(spec.output).is_valid())


def extract_documents(client, spec, bodies, texts, body_texts=None) -> dict:
    """Decode one answer per prompt body and locate it in the document's text.

    Generation stops where the model ends its turn. An answer
    the document holds is returned as the document's own text at its
    span; "none" gives nulls;
    an answer the document does not hold is returned as written, with
    a null span.

    Args:
        client: The engine client.
        spec: The RequestExtractSpec with the prompt tail.
        bodies: Token ids for each prompt before the tail, preamble included.
        texts: The document text of each body.
        body_texts: The text of each prompt before the tail, for an
            engine that tokenizes the prompt itself; None sends token ids.

    Returns:
        A dict with one answer and one span per body, the counts of
        unanswered and unaligned documents, the request and token
        counts, and the generation seconds.
    """
    tail = token_list(spec.tail_token_ids)
    params = client.decode_params(spec.max_answer_tokens)
    if body_texts is not None:
        prompts = [body_text + spec.tail_text for body_text in body_texts]
    else:
        prompts = [{"prompt_token_ids": body + tail} for body in bodies]
    started = time.perf_counter()
    outputs = client.generate(prompts, params, use_tqdm=False) if prompts else []
    wall_s = time.perf_counter() - started
    answers, spans = [], []
    unanswered = unaligned = 0
    for output, document in zip(outputs, texts):
        phrase = read_phrase(output.outputs[0].text or "")
        span = None if phrase is None else locate_span(document, phrase)
        if phrase is None:
            unanswered += 1
        elif span is None:
            unaligned += 1
        answers.append(phrase if span is None else document[span[0]:span[1]])
        spans.append(span)
    return {
        "answers": answers,
        "spans": spans,
        "wall_s": wall_s,
        "requests": len(prompts),
        **_generation_counts(outputs),
        "unanswered": unanswered,
        "unaligned": unaligned,
    }


def allowed_members_by_anchor(anchor, partners, anchor_ids, members, pairs) -> list:
    """Per anchor, the member indices its equality conditions allow."""
    position = partners.index(pair_partner(anchor, partners))
    by_partner = members_by_partner(members, position)
    rows = partner_map(pairs, anchor, partners[position])
    return [allowed_members(rows, by_partner, anchor_id)
            for anchor_id in anchor_ids]


def join_answers(spec, rows, answers, anchor, partners) -> pa.Table:
    columns = {
        alias: [int(row[index]) for row in rows]
        for index, alias in enumerate(spec.aliases)
    }
    return answer_table(
        columns,
        answers,
        "join_answers",
        metadata={
            "anchor": anchor,
            "partners": ",".join(partners),
            "semantics": spec.semantics,
            "written_position": spec.written_position,
            **(
                {"selectivity": spec.selectivity}
                if spec.selectivity is not None else {}
            ),
        },
    )


def request_step(result, **details) -> dict:
    """Attach common request measurements to an operator's execution details."""
    return {
        **details,
        "wall_s": result.get("wall_s", result.get("wall", 0.0)),
        "requests": result.get("requests", details.get("pairs", 0)),
        **{name: result[name] for name in (
            "prompt_tokens", "fresh_tokens", "cached_tokens", "generated_tokens")
           if name in result},
    }
