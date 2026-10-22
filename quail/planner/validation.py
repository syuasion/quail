"""Execution constraints checked during planning and graph construction."""

from quail.cost import budgets
from quail.cost.joins import PricedJoin, longest_join_input
from quail.logical import (
    Alias,
    Apply,
    CompileError,
    SemanticClassify,
    SemanticExtract,
    classified_above_joins,
    has_score,
    is_score,
)
from quail.logical.expressions import aliased_call
from quail.logical.prompts import prompt_aliases
from quail.physical import AiClassify, AiJoin
from quail.physical.pipelines import build_pipelines
from quail.planner.plan import Refusal


def _refusal(reason: str, constraint: str = "unsupported_score_query"):
    return Refusal(
        reasons=(reason,),
        constraint=constraint,
        needed=1,
        available=0,
        unit="queries",
    )


class ScoreRefusedError(Exception):
    """Carries a Refusal out of the plan builder."""

    def __init__(self, refusal: Refusal):
        super().__init__(refusal.reasons[0])
        self.refusal = refusal


def score_projections(logical) -> tuple[Alias, ...]:
    return tuple(
        expression
        for expression in logical.projection.columns
        if aliased_call(expression, "score") is not None
    )


def score_refusal(logical, context) -> Refusal | None:
    """Check the execution constraints of score operators in a plan."""
    operators = logical.operators()
    projected = score_projections(logical)
    predicates = [p.expression for values in operators.filters.values()
                  for p in values] + [j.predicate for j in operators.joins]
    if context is None or context.tokenizer is None:
        return _refusal("AI.SCORE planning needs the model's tokenizer")
    prompts = [p.prompt for values in operators.filters.values() for p in values]
    prompts.extend(j.prompt for j in operators.joins)
    prompts.extend(item.expression.prompt for item in projected)
    if any(len(prompt_aliases(prompt)) > 2 for prompt in prompts):
        return _refusal("AI.SCORE supports one document or one document pair")
    if any(isinstance(node, Apply) for node in logical.walk()):
        return _refusal("AI.SCORE cannot be mixed with apply()")
    if not all(is_score(expression) for expression in predicates):
        return _refusal("AI.SCORE cannot be mixed with generative AI predicates")
    if any(join.semantics != "full" for join in operators.joins):
        return _refusal("AI.SCORE supports full joins only")
    if any(len(call.aliases()) != 1 for call, _ in operators.labels.calls):
        return _refusal(
            "AI.CLASSIFY over a document pair needs its AI join pipeline "
            "on one GPU; AI.SCORE plans cannot execute that classification",
            "joined_classify_pipeline")
    if classified_above_joins(logical.root) and operators.joins:
        return _refusal("classification after a score join is unsupported")
    pairs = [j.prompt for j in operators.joins] + [
        item.expression.prompt for item in projected
        if len(item.expression.aliases()) == 2]
    if len(set(pairs)) > 1:
        return _refusal("AI.SCORE currently supports one document pair expression")
    if any(join.anchor not in (None, prompt_aliases(join.prompt)[0])
           for join in operators.joins):
        return _refusal("AI.SCORE anchors on its first prompt document")
    return None


def classification_refusal(context) -> Refusal | None:
    """Check whether the model supports classification planning.

    Args:
        context: Planning context with the tokenizer and model specification.

    Returns:
        A Refusal describing missing support, or None if planning can proceed.
    """
    model = context.model
    if context.tokenizer is None:
        reason = "AI.CLASSIFY planning needs the model's tokenizer"
    elif model.canvas_tokens and model.answer_canvas is None:
        reason = f"{model.name!r} names no answer canvas to read labels on"
    else:
        return None
    return Refusal(reasons=(reason,), constraint="unsupported_classify_query",
                   needed=1, available=0, unit="queries")


def joined_classification_refusal(graph) -> Refusal | None:
    """Check that each pair classification can run in its join's pipeline.

    Args:
        graph: Physical operator graph.

    Returns:
        A Refusal if a pair classification cannot share its source join's
        pipeline, or None if every pair classification can run.
    """
    pipelines = build_pipelines(graph)
    for node in graph.nodes:
        if not isinstance(node, AiClassify) or node.spec.partner is None:
            continue
        pipeline = pipelines.get(node.node_id)
        if pipeline is not None and any(
                isinstance(member, AiJoin)
                and any(port.source.node_id == member.node_id for port in node.inputs)
                for member in pipeline):
            continue
        return Refusal(
            reasons=(f"AI.CLASSIFY {node.spec.name!r} over a document pair "
                     "must run in its join's pipeline; this plan requires "
                     "unsupported standalone pair classification",),
            constraint="joined_classify_pipeline", needed=1, available=0,
            unit="supported join pipelines")
    return None


def has_label(logical) -> bool:
    """Return whether a logical plan classifies documents or joined rows."""
    return any(isinstance(node, SemanticClassify) for node in logical.walk())


def has_extract(logical) -> bool:
    """Return whether a logical plan copies answers from documents."""
    return any(isinstance(node, SemanticExtract) for node in logical.walk())


def extraction_refusal(context) -> Refusal | None:
    """Check whether the model supports extraction planning.

    An extraction reads the model's next-token probabilities over the
    document's own tokens, which a reranker, a decision model, and a
    diffusion model with an answer canvas do not give.

    Args:
        context: Planning context with the tokenizer and model specification.

    Returns:
        A Refusal describing missing support, or None if planning can proceed.
    """
    model = context.model
    if context.tokenizer is None:
        reason = "AI.EXTRACT planning needs the model's tokenizer"
    elif model.role in ("reranker", "decision") or model.canvas_tokens:
        reason = (f"AI.EXTRACT reads next-token probabilities over the "
                  f"document's tokens, which {model.name!r} does not give")
    else:
        return None
    return Refusal(reasons=(reason,), constraint="unsupported_extract_query",
                   needed=1, available=0, unit="queries")


class ClassifyRefusedError(Exception):
    """A classification cannot run within the selected execution constraints."""

    def __init__(self, reason, needed, available):
        super().__init__(reason)
        self.reason, self.needed, self.available = reason, needed, available

    @classmethod
    def over_budget(cls, what: str, need: int,
                    budget: int) -> "ClassifyRefusedError":
        """The error for a prompt longer than the forward pass budget."""
        return cls(
            f"{what} needs {need} tokens with its classification prompt, "
            f"but the forward pass budget is {budget} tokens", need, budget)

    def refusal(self) -> Refusal:
        """Convert this exception to a planning Refusal."""
        return Refusal(reasons=(self.reason,), constraint="suffix_over_chunk",
                       needed=self.needed, available=self.available, unit="tokens")


def anchor_candidates(spec: PricedJoin, honor_forced: bool = True) -> list:
    """Candidate anchors for one join.

    Gates keep their outer table, a forced full anchor is honored, a
    free full join offers every table.
    """
    if spec.semantics != "full":
        return [spec.anchor]
    if honor_forced and not spec.anchor_free:
        return [spec.anchor]
    return list(spec.aliases)


def join_input_tokens(spec: PricedJoin, anchor: str, maximum: dict,
                      pre: int) -> int:
    """Count tokens for the longest input under one join orientation."""
    return longest_join_input(
        pre, maximum[anchor], spec.frame_tokens[anchor], spec.tail_tokens,
        [(spec.label_tokens[partner], maximum[partner])
         for partner in spec.aliases if partner != anchor])


def score_input_tokens(spec, context) -> tuple[int, int]:
    """Return the prefix and suffix token counts of the longest input.

    The prefix is what stays resident while suffixes stream past it:
    the fixed prompt head, plus the first document for a pair.
    """
    parts = spec.prompt_token_parts
    canvas = context.model.canvas_tokens
    longest = [
        max(context.document_tokens[alias], default=0) for alias in spec.aliases
    ]
    if len(spec.aliases) == 1:
        return len(parts[0]), longest[0] + len(parts[1]) + canvas
    prefix = len(parts[0]) + longest[0] + len(parts[1])
    return prefix, longest[1] + len(parts[2]) + canvas


def validate_query(plan, operators, context, model, device, gpus):
    """Return a refusal for unsupported inference or hardware requirements."""
    labels, applies = operators.labels, operators.applies
    scored = has_score(plan)
    if scored:
        refusal = score_refusal(plan, context)
        if refusal is not None:
            return refusal
    if labels.calls:
        if context is None:
            return Refusal(
                reasons=("AI.CLASSIFY planning needs a planning context",),
                constraint="unsupported_classify_query",
                needed=1, available=0, unit="queries")
        if any(len(call.aliases()) not in (1, 2) for call, _ in labels.calls):
            return Refusal(
                reasons=("AI.CLASSIFY reads one document, or one from each "
                         "side of a join",),
                constraint="unsupported_classify_query",
                needed=1, available=0, unit="queries")
        refusal = classification_refusal(context)
        if refusal is not None:
            return refusal
    extracts = operators.extracts
    if extracts:
        if context is None:
            return Refusal(
                reasons=("AI.EXTRACT planning needs a planning context",),
                constraint="unsupported_extract_query",
                needed=1, available=0, unit="queries")
        refusal = extraction_refusal(context)
        if refusal is not None:
            return refusal
    join_applies = {}
    for apply in applies:
        if apply.ids == "pairs":
            join_applies.setdefault(apply.written_position, []).append(apply)
    names = [apply.function for apply in applies]
    if len(set(names)) != len(names):
        raise CompileError(
            f"each apply() needs its own name; {names} repeat one")
    if any(len(group) > 1 for group in join_applies.values()):
        raise CompileError("a join takes one apply() returning pairs")
    weight_gpus = budgets.minimum_weight_gpus(model, device)
    if weight_gpus > 1:
        return Refusal(
            reasons=(
                f"the model needs {weight_gpus} GPUs of memory, "
                f"but Quail loads one complete copy per GPU",
            ),
            constraint="weights_need_more_cards",
            needed=weight_gpus, available=1, unit="cards")
    needed = budgets.chunk_memory_bytes(model, 1)
    available = device.mem_bytes * budgets.POOL_FRACTION
    if needed > available:
        return Refusal(
            reasons=(
                f"{model.name} does not fit {device.name}: even a one-token "
                "chunk exceeds the memory available for weights, activations, "
                "and checkpoint reservations",
            ),
            constraint="execution_needs_more_memory",
            needed=needed, available=available, unit="bytes")
    return None
