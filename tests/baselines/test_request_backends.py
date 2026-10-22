"""CPU tests for vLLM and SGLang backend interfaces."""

import asyncio
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.baselines import (
    CAPACITY,
    _Client,
    _execution,
    _run_stock_vllm,
    _session,
)

import quail
from quail.baselines.nodes import (
    RequestClassifySpec,
    RequestExecution,
    RequestFilterSpec,
    RequestJoinSpec,
)
from quail.baselines.sglang import SGLangClient, pipelined_sglang_backend
from quail.baselines.vllm import VLLMEngine, sampling_kwargs, stock_vllm_backend
from quail.builtins import built_in_registry
from quail.physical import (
    Project,
    decode_graph,
)
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8

DOCS = {"id": ["a", "b"], "body": ["one", "two"]}


def _filter_query(session):
    return (session.docs("documents").alias("d")
            .ai_filter(quail.prompt("useful {0}", quail.col("d.body")))
            .select("d.id"))


def test_request_backends_plan_validate_and_execute(monkeypatch):
    registry = built_in_registry()
    assert set(registry.backends) == {
        "quail", "stock_vllm", "pipelined_vllm", "dumb_vllm", "pipelined_sglang",
    }

    for backend_name in ("stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        session = _session(backend_name, documents=DOCS)
        plan = _filter_query(session).plan()
        request_node = next(
            node for node in plan.nodes if isinstance(node, RequestExecution))
        assert plan.backend == request_node.backend == backend_name
        assert request_node.filters[0].alias == "d"
        plan.graph.validate_backend(backend_name)
        envelope = plan.to_envelope(registry.codecs)
        decoded = decode_graph(envelope["graph"], registry.codecs)
        assert decoded.node("request-model") == request_node
        session.close()

    for backend in (stock_vllm_backend(), pipelined_sglang_backend()):
        support = backend.supports(QWEN3_4B_FP8, H100_SXM, 2)
        assert not support.supported
        assert "one model copy on one GPU" in support.reason

    boot = {"client": _Client(), "sampling_params": object(), "capacity": CAPACITY}
    monkeypatch.setattr(
        "quail.baselines.vllm.VLLMEngine.boot",
        lambda self, spec, allowed_ids, *, resources: (
            boot, {"kind": "cold", "boot_s": 0.1}))
    session = _session(
        "stock_vllm",
        documents={**DOCS, "key": ["k1", "k2"]},
        notes={"id": ["x", "y", "z"], "text": ["p", "q", "r"],
               "key": ["k2", "k1", "k3"]},
    )
    result = _run_stock_vllm(session, _filter_query(session))
    assert result.report["backend"] == "stock_vllm"
    assert result.report["backend_metrics"]["requests"] == 2
    assert result.count() == 0
    query = (
        session.docs("documents").alias("d")
        .join(session.docs("notes").alias("n"),
              on=[quail.col("d.key") == quail.col("n.key")])
        .ai_filter(quail.prompt("{0} matches {1}", quail.col("d.body"),
                                quail.col("n.text")))
        .select("d.id", "n.id")
    )
    result = _run_stock_vllm(session, query)
    assert result.report["backend_metrics"]["requests"] == 2
    joins = result.answer_tables["joins"][0].to_pydict()
    assert sorted(zip(*joins.values())) == [(0, 1, False), (1, 0, False)]

    # a classification of the join's pairs: its label table names both
    # aliases before the label column; the join kept no pair here
    boot["client"] = _JoinedClient()
    query = (
        session.docs("documents").alias("d")
        .join(session.docs("notes").alias("n"),
              on=[quail.col("d.key") == quail.col("n.key")])
        .ai_filter(quail.prompt("{0} matches {1}", quail.col("d.body"),
                                quail.col("n.text")))
        .ai_classify(quail.prompt("{0} about {1}", quail.col("d.body"),
                                  quail.col("n.text")), ["a", "b"], name="kind")
        .select("d.id", "n.id", "kind")
    )
    result = _run_stock_vllm(session, query)
    labels = result.answer_tables["classifies"]["kind"]
    assert labels.column_names == ["d", "n", "kind"]
    assert labels.num_rows == 0 and result.count() == 0
    session.close()


class _ScoredClient(_Client):
    """Samples the wrong token; only the logprobs hold the answer."""

    def generate(self, prompts, sampling_params, use_tqdm=False):
        outputs = super().generate(prompts, sampling_params, use_tqdm)
        for output in outputs:
            expected = output.outputs[0].token_ids[0] == 1
            output.outputs[0].token_ids = [int(not expected)]
            output.outputs[0].logprobs = [{
                1: SimpleNamespace(logprob=-1 if expected else -2),
                0: SimpleNamespace(logprob=-2 if expected else -1),
            }]
        return outputs


FILTER_NODE = RequestExecution(
    node_id="request-model", backend_name="stock_vllm", aliases=("d",),
    preamble_token_ids=(3,),
    filters=(RequestFilterSpec(alias="d", written_positions=(0,),
                               question_token_ids=((90,),)),),
)


def _join_node(backend_name="stock_vllm"):
    return RequestExecution(
        node_id="request-model", backend_name=backend_name, aliases=("r", "p"),
        preamble_token_ids=(3,),
        joins=(RequestJoinSpec(
            written_position=0, aliases=("r", "p"), outer_aliases=("r",),
            anchor="r", semantics="full", selectivity=0.5,
            label_token_ids=(("r", (40,)), ("p", (41,))),
            frame_token_ids=(("r", (30,)), ("p", (31,))),
            tail_token_ids=(50,),
        ),),
    )


def test_filter_and_join_answers_and_suffix_major_fallback():
    for model in (QWEN3_4B_FP8, DIFFUSION_GEMMA_26B_FP8):
        settings = {}
        if model is DIFFUSION_GEMMA_26B_FP8:
            settings = {"client": _ScoredClient(), "false_ids": [0],
                        "sampling_params": SimpleNamespace(logprobs=500)}
        execution = _execution(
            {"d": [[np.int32(10)], [11]], "r": [[10], [11]], "p": [[20], [21]]},
            model=model, **settings)

        filtered = execution.execute(FILTER_NODE, {})
        assert filtered.outputs["filter_answers:d"].to_pydict() == {
            "d": [0, 1], "predicate": [0, 0], "answer": [True, False]}, model.name
        assert filtered.outputs["rows:d"].ids["d"].to_pylist() == [0], model.name
        assert filtered.metrics.evaluated_documents == 2, model.name

        joined = execution.execute(_join_node(), {})
        assert joined.outputs["join_answers:0"].to_pydict() == {
            "r": [0, 0, 1, 1], "p": [0, 1, 0, 1],
            "answer": [True, True, False, False]}, model.name
        assert joined.outputs["rows:r"].ids["r"].to_pylist() == [0], model.name
        assert joined.outputs["rows:p"].ids["p"].to_pylist() == [0, 1], model.name
        assert joined.metrics.evaluated_document_pairs == 4, model.name

    class RecordingClient(_Client):
        def generate(self, prompts, sampling_params, use_tqdm=False):
            self.prompts = [tuple(prompt["prompt_token_ids"]) for prompt in prompts]
            return super().generate(prompts, sampling_params, use_tqdm=use_tqdm)

    # two anchor prefixes of three tokens each: six tokens held at once
    orders = {}
    for kv_tokens in (5, 6):
        execution = _execution(
            {"r": [[10], [11]], "p": [[20], [21]]}, join_submission="suffix-major",
            capacity={**CAPACITY, "kv_cache_size_tokens": kv_tokens})
        execution.client = RecordingClient()
        execution.execute(_join_node("pipelined_sglang"), {})
        orders[kv_tokens] = [(prompt[1], prompt[4])
                             for prompt in execution.client.prompts]
    assert orders[6] == [(10, 20), (11, 20), (10, 21), (11, 21)]
    assert orders[5] == [(10, 20), (10, 21), (11, 20), (11, 21)]


def test_sglang_submission_and_cancellation():
    class Engine:
        def __init__(self):
            self.calls = []

        def generate(self, *, input_ids, sampling_params):
            self.calls.append(input_ids)
            return [
                {"output_ids": [prompt[0] % 2], "meta_info": {"cached_tokens": 1}}
                for prompt in input_ids
            ]

    engine = Engine()
    client = SGLangClient(engine, {})
    prompts = [{"prompt_token_ids": [index, 9]} for index in range(17_000)]
    result = client.generate(prompts, {})
    assert engine.calls == [[prompt["prompt_token_ids"] for prompt in prompts]]
    assert [output.outputs[0].token_ids[0] for output in result] == [
        index % 2 for index in range(17_000)]
    assert all(output.num_cached_tokens == 1 for output in result)
    assert client.generate([], {}) == []
    assert len(engine.calls) == 1

    async def run():
        started = asyncio.Event()
        submitted = []
        aborted = []

        async def generate(**kwargs):
            submitted.append(kwargs["rid"])
            started.set()
            await asyncio.Event().wait()

        engine = SimpleNamespace(
            async_generate=generate,
            tokenizer_manager=SimpleNamespace(abort_request=aborted.append),
        )
        task = asyncio.create_task(SGLangClient(engine, {})._generate_one([1, 2], {}))
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert aborted == submitted
        assert len(aborted) == 1

    asyncio.run(run())


def test_vllm_engine_settings_tokenizer_and_canvas_follow_the_model(monkeypatch):
    import os

    from quail.backends.quail.executor.models.diffusion_gemma import canvas_token_ids
    from quail.baselines.vllm import (
        GigatokenVLLMTokenizer,
        diffusion_canvas,
        engine_environment,
    )

    engine = VLLMEngine()
    key = "VLLM_ENABLE_V1_MULTIPROCESSING"
    # the environment cache is cleared on entry and exit
    cleared = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        envs=SimpleNamespace(disable_envs_cache=lambda: cleared.append(1))))
    for previous in (None, "1"):
        if previous is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, previous)
        for spec in (replace(QWEN3_4B_FP8, vision_tower=True),
                     DIFFUSION_GEMMA_26B_FP8):
            with pytest.raises(RuntimeError, match="boot failed"):
                with engine_environment(spec):
                    assert os.environ[key] == "0"
                    raise RuntimeError("boot failed")
            assert os.environ.get(key) == previous
            with engine_environment(QWEN3_4B_FP8):
                assert os.environ.get(key) == previous
    assert len(cleared) == 16
    qwen = engine.llm_kwargs(QWEN3_4B_FP8)
    assert qwen["max_num_batched_tokens"] == 25_305
    assert qwen["tokenizer_mode"] == "gigatoken"
    assert "diffusion_config" not in qwen
    # the mixture-of-experts model batches its own chunk cap; its
    # one-row canvas commits after one denoising step, and stays under
    # vLLM's eight-sequence cap trigger
    gemma = engine.llm_kwargs(DIFFUSION_GEMMA_26B_FP8)
    assert gemma["max_num_batched_tokens"] == 65_536
    assert gemma["diffusion_config"] == {"canvas_length": 1,
                                         "max_denoising_steps": 1}
    assert gemma["max_num_seqs"] == 127
    assert gemma["max_logprobs"] == -1
    wide = engine.llm_kwargs(replace(DIFFUSION_GEMMA_26B_FP8, canvas_tokens=256))
    assert wide["diffusion_config"] == {"canvas_length": 256}
    assert wide["max_num_seqs"] == qwen["max_num_seqs"]
    assert sampling_kwargs([1, 2]) == {
        "temperature": 0.0, "max_tokens": 1, "min_tokens": 1,
        "allowed_token_ids": [1, 2], "detokenize": False}
    # the diffusion sampler rejects everything but the length; a
    # one-row canvas returns its logprobs, a longer one free text
    assert sampling_kwargs([1, 2], canvas_tokens=1) == {
        "max_tokens": 1, "logprobs": -1, "detokenize": False}
    assert sampling_kwargs([1, 2], canvas_tokens=256) == {"max_tokens": 16}

    adapter = SimpleNamespace(get_vocab=lambda: {"a": 0, "abc": 1})
    sources = []

    class Tokenizer:
        def __init__(self, source):
            sources.append(source)

        def as_hf(self):
            return adapter

    monkeypatch.setitem(
        sys.modules, "gigatoken", SimpleNamespace(Tokenizer=Tokenizer)
    )
    monkeypatch.setattr(
        "quail.baselines.vllm.resolve_model_path",
        lambda name, revision, **kwargs: f"{name}@{revision}")
    loaded = GigatokenVLLMTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-FP8", truncation_side="right", revision="pinned"
    )
    assert loaded is adapter
    assert sources == ["Qwen/Qwen3-4B-FP8@pinned"]
    assert loaded.truncation_side == "right"
    assert loaded.max_token_id == 1
    assert loaded.max_chars_per_token == 3

    spec = DIFFUSION_GEMMA_26B_FP8

    class States:
        canvas_length = 1
        vocab_size = spec.vocab

        def __init__(self):
            self.canvas = np.full((4, 1), -1, dtype=np.int64)

    module = SimpleNamespace(DiffusionGemmaRequestStates=States)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models",
                        SimpleNamespace(diffusion_gemma=module))
    with diffusion_canvas(spec) as tokens:
        states = module.DiffusionGemmaRequestStates()
    assert module.DiffusionGemmaRequestStates is States
    assert tokens == canvas_token_ids(spec.vocab, 1)
    states.init_canvas(np.array([3, 1]))
    assert states.canvas[:, 0].tolist() == [-1, tokens[0], -1, tokens[0]]
    states.canvas[1] = 42
    states.init_canvas(np.array([1]))
    assert states.canvas[1, 0] == tokens[0]
    states.vocab_size = 100
    with pytest.raises(ValueError, match="canvas geometry"):
        states.init_canvas(np.array([0]))
    with pytest.raises(RuntimeError, match="boot failed"), diffusion_canvas(spec):
        raise RuntimeError("boot failed")
    assert module.DiffusionGemmaRequestStates is States
    with diffusion_canvas(QWEN3_4B_FP8) as tokens:
        assert tokens == ()
        assert module.DiffusionGemmaRequestStates is States


class _LabelClient(_Client):
    """Decodes answers: document 10 says a, document 11 says B with a remark."""

    def decode_params(self, max_tokens):
        return ("decode", max_tokens)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert sampling_params == ("decode", 3)
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            text = " a\nbecause" if 10 in ids else " B, clearly"
            outputs.append(SimpleNamespace(
                prompt_token_ids=ids, num_cached_tokens=0,
                outputs=[SimpleNamespace(text=text, token_ids=[1, 2])]))
        return outputs


def test_request_backends_classify_with_one_decode_request_per_document():
    # labels of at most two tokens, so each document's request decodes
    # up to three; document 0 decodes a and document 1 decodes B
    spec = RequestClassifySpec(
        alias="d", output="kind", tail_token_ids=(90,),
        labels=("a", "b", "c"), label_token_ids=((60, 62), (61, 62), (60, 63)),
        tests=((0, ("b",)),))
    node = RequestExecution(
        node_id="request-model", backend_name="stock_vllm", aliases=("d",),
        preamble_token_ids=(3,), classifies=(spec,))
    execution = _execution({"d": [[10], [11]]}, client=_LabelClient())
    result = execution.execute(node, {})
    assert result.outputs["label_answers:kind"].to_pydict() == {
        "d": [0, 1], "kind": ["a", "b"]}
    assert result.outputs["rows:d"].ids["d"].to_pylist() == [1]
    assert result.outputs["label_in_answers:kind"].to_pydict() == {
        "d": [0, 1], "predicate": [0, 0], "answer": [False, True]}
    assert result.metrics.extension["requests"] == 2
    (step,) = result.metrics.extension["steps"]
    assert (step["kind"], step["num_in"], step["num_out"]) == ("classify", 2, 1)
    assert (step["generated_tokens"], step["unmatched"]) == (4, 0)
    # two 3-token prompts, and each answer's first token fed back
    assert result.metrics.fresh_tokens == 2 * 3 + 2 * (2 - 1)
    assert step["prompt_tokens"] == 6
    assert result.metrics.extension["prompt_tokens"] == 6

    session = _session("stock_vllm", documents=DOCS)
    query = (session.docs("documents").alias("d")
             .ai_classify(quail.prompt("kind of {0}", quail.col("d.body")),
                          ["a", "b", "c"], name="kind")
             .label_in("kind", ["b"])
             .select("d.id", "kind"))
    plan = query.plan()
    request_node = next(
        node for node in plan.nodes if isinstance(node, RequestExecution))
    (planned,) = request_node.classifies
    assert planned.labels == ("a", "b", "c")
    assert planned.tests == ((0, ("b",)),)
    assert not request_node.filters
    project = next(node for node in plan.nodes if isinstance(node, Project))
    assert [port.source.port for port in project.inputs] == [
        "rows:d", "label_answers:kind"]
    assert project.columns == ("d.id", "kind")
    session.close()


class _JoinedClient(_Client):
    """Joins by token, and decodes a joined row's label from its partner."""

    def decode_params(self, max_tokens):
        return ("decode", max_tokens)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        outputs = super().generate(prompts, sampling_params, use_tqdm)
        if sampling_params != ("decode", 3):
            return outputs
        self.prompts = [tuple(prompt["prompt_token_ids"]) for prompt in prompts]
        for output in outputs:
            output.outputs[0].text = " a" if 20 in output.prompt_token_ids else " b"
            output.outputs[0].token_ids = [1, 2]
        return outputs


def test_request_backends_decode_one_request_per_row_the_join_keeps():
    # anchor r0 matches both partners; r1 matches none
    spec = RequestClassifySpec(
        alias="r", output="kind", tail_token_ids=(90,),
        labels=("a", "b", "c"), label_token_ids=((60, 62), (61, 62), (60, 63)),
        partner="p", join_layout_token_ids=((70,), (71,)), join_written_position=0)
    node = replace(_join_node(), classifies=(spec,))
    client = _JoinedClient()
    execution = _execution({"r": [[10], [11]], "p": [[20], [21]]}, client=client)
    result = execution.execute(node, {})
    assert result.outputs["join_answers:0"].to_pydict() == {
        "r": [0, 0, 1, 1], "p": [0, 1, 0, 1],
        "answer": [True, True, False, False]}
    # each kept pair: preamble, anchor, its note, the partner's label and
    # document, then the question
    assert client.prompts == [(3, 10, 70, 71, 20, 90), (3, 10, 70, 71, 21, 90)]
    assert result.outputs["label_answers:kind"].to_pydict() == {
        "r": [0, 0], "p": [0, 1], "kind": ["a", "b"]}
    assert result.outputs["rows:r"].ids["r"].to_pylist() == [0]
    join_step, classify_step = result.metrics.extension["steps"]
    assert join_step["kind"] == "join"
    assert (classify_step["alias"], classify_step["partner"]) == ("r", "p")
    assert (classify_step["num_in"], classify_step["num_out"]) == (2, 2)
    assert result.metrics.evaluated_document_pairs == 4 + 2


def test_match_label_takes_the_longest_label_the_answer_starts_with():
    from quail.execution.labels import match_label

    labels = ("cardiac", "cardiac disorders", "vascular disorders")
    assert match_label(" Cardiac disorders\nbecause", labels) == "cardiac disorders"
    assert match_label("cardiac.", labels) == "cardiac"
    assert match_label("  vascular disorders", labels) == "vascular disorders"
    assert match_label("none of these", labels) is None
    assert match_label("", labels) is None
    # a leading "thought" line, then a leading "ANSWER:", are skipped
    assert match_label("thought\nCardiac\n", labels) == "cardiac"
    assert match_label("ANSWER: vascular disorders", labels) == \
        "vascular disorders"
    assert match_label("thought\n ANSWER:cardiac", labels) == "cardiac"
    assert match_label("thought\nnone", labels) is None
    assert match_label("thoughtful cardiac", labels) is None
    assert match_label("cardiac\nthought", labels) == "cardiac"


def test_filter_clients_accept_the_pipelined_filter_call():
    """The request planner passes text prompts to every client's filter run."""
    import inspect

    from quail.baselines.vllm import VLLMClient

    for client in (SGLangClient, VLLMClient):
        signature = inspect.signature(client.run_filter_sequence)
        signature.bind(None, {}, [[1]], [[2]], None, tag="q",
                       body_texts=None, question_texts=None)


@pytest.mark.parametrize("backend_name", ["stock_vllm", "pipelined_vllm"])
@pytest.mark.parametrize("operation", ["classify", "extract"])
def test_request_planner_checks_map_placement(backend_name, operation):
    from quail.logical import LogicalPlan
    from quail.planner.optimizer.candidates import ModelRegion
    from quail.planner.optimizer.context import PlanningContext
    from quail.planner.plan import Refusal
    from quail.planner.rules.logical import ShouldLiftMaps

    with _session(backend_name, documents=DOCS, partners=DOCS) as session:
        documents = session.docs("documents").alias("d")
        if operation == "classify":
            documents = documents.ai_classify(
                quail.prompt("kind of {0}", quail.col("d.body")),
                ["a", "b"], name="kind")
            output = "kind"
        else:
            documents = documents.ai_extract(
                "d.body", {"kind": "What kind is it?"}, name="fields")
            output = "fields"
        query = documents.ai_join(
            session.docs("partners").alias("p"),
            quail.prompt("Does {0} mention {1}?", quail.col("d.body"),
                         quail.col("p.body")), selectivity=0.1,
        ).select("d.id", "p.id", output)
        root = query.logical.root
        original, lifted = ShouldLiftMaps().rewrites(root, None)
        assert original is root
        assert lifted != original
        plan = query.plan()
        assert not isinstance(plan, Refusal)
        written, moved = plan.settings["candidates"]
        assert written["selected"] and written["refusal"] is None
        context = PlanningContext(
            model=session.model, device=session.device,
            document_tokens=query._document_tokens, tokenizer=session.tokenizer,
            backend=backend_name)
        candidate, = session.registry.backend(backend_name).plan(
            ModelRegion(LogicalPlan(lifted)), context)
        if operation == "classify":
            assert isinstance(candidate.plan, Refusal)
            assert candidate.plan.constraint == "classify_placement_not_supported"
            assert moved["refusal"] == candidate.plan.constraint
            assert moved["estimated_seconds"] is None
        else:
            assert not isinstance(candidate.plan, Refusal)
            assert moved["refusal"] is None
            assert candidate.plan.graph == plan.graph


class _Adapter:
    """An engine adapter whose boot fails until told otherwise."""

    kind = "fake"

    def __init__(self):
        self.live = []
        self.fail = True

    def boot(self, spec, allowed_ids, *, resources):
        client = SimpleNamespace(close=lambda: self.live.remove(client))
        self.live.append(client)
        resources.callback(client.close)
        if self.fail:
            raise RuntimeError("warmup failed")
        return {"client": client}, {"kind": "cold"}


def test_a_request_backend_owns_its_engine_from_boot_to_release(monkeypatch):
    monkeypatch.setattr("quail.backends.resources._release_vllm_parallel_state",
                        lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    adapter = _Adapter()
    backend = stock_vllm_backend().__class__("test", adapter, "pipelined")
    context = SimpleNamespace(
        registry=built_in_registry(), runtime_state={},
        request=SimpleNamespace(plan={
            "model": "qwen3-4b-fp8",
            "settings": {"true_ids": [1], "false_ids": [2]}}))
    with pytest.raises(RuntimeError, match="warmup failed"):
        backend.prepare_request(context)
    assert adapter.live == [] and context.runtime_state == {}

    adapter.fail = False
    backend.prepare_request(context)
    engine = context.runtime_state["engine"]
    assert len(adapter.live) == 1
    assert engine.prepared_boot["kind"] == "cold"
    monkeypatch.setattr("quail.baselines.request.execute_request_graph",
                        lambda context, backend, settings, boot: boot["kind"])
    assert backend.execute_request(context) == "cold"
    assert backend.execute_request(context) == "warm"
    backend.release_request(context.runtime_state)
    assert adapter.live == [] and context.runtime_state == {}


def test_an_experiment_can_turn_on_vllm_ngram_drafts(monkeypatch):
    from quail.baselines.vllm import VLLMEngine, speculative_config

    assert "speculative_config" not in VLLMEngine().llm_kwargs(QWEN3_4B_FP8)
    monkeypatch.setenv("QUAIL_VLLM_SPECULATIVE", "ngram:5")
    assert VLLMEngine().llm_kwargs(QWEN3_4B_FP8)["speculative_config"] == {
        "method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}
    for value in ("ngram", "ngram:0", "eagle:5"):
        with pytest.raises(ValueError):
            speculative_config(value)
