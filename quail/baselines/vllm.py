"""vLLM engine adapter and the vLLM request backends."""

from __future__ import annotations

import os
import time
from contextlib import ExitStack, closing, contextmanager

from quail.baselines.request import RequestBackend
from quail.baselines.request_scheduling import (
    MAX_BATCHED_TOKENS,
    MAX_SEQUENCES,
    run_filter_sequence,
)
from quail.specs.files import resolve_model_path

GPU_MEMORY_UTILIZATION = 0.91
CUDA_GRAPH_CAPTURE_SIZE = 8_192
# vLLM caps a diffusion model at 8 sequences per step whenever the
# setting is 128 or more, sized for its 256-row canvas; one row needs
# no cap, so stay just under the trigger.
DIFFUSION_SEQUENCES = 127
# DiffusionGemma ignores logprob_token_ids in vLLM 0.26. Request the full
# vocabulary so TRUE/FALSE scores are returned regardless of their ranks.
DIFFUSION_LOGPROBS = -1
# generated tokens a longer canvas gets; the answer word is read
# from the text
DIFFUSION_TEXT_TOKENS = 16


class GigatokenVLLMTokenizer:
    """Load Gigatoken through vLLM's tokenizer registry."""

    @classmethod
    def from_pretrained(cls, path_or_repo_id, *_args, **kwargs):
        """Return Gigatoken with the Hugging Face tokenizer interface."""
        from gigatoken import Tokenizer

        truncation_side = kwargs.pop("truncation_side", "left")
        path = resolve_model_path(path_or_repo_id, kwargs.get("revision"),
                                  include_weights=False)
        tokenizer = Tokenizer(path).as_hf()
        tokenizer.truncation_side = truncation_side
        vocab = tokenizer.get_vocab()
        tokenizer.max_token_id = max(vocab.values())
        tokenizer.max_chars_per_token = max(len(token) for token in vocab)
        return tokenizer


@contextmanager
def engine_environment(spec):
    """Scope vLLM's process selection to construction of one engine."""
    from vllm import envs

    key = "VLLM_ENABLE_V1_MULTIPROCESSING"
    previous = os.environ.get(key)
    # vLLM 0.26 reads process selection from the environment in LLM's
    # constructor, and caches what it read; its public LLM API has no
    # per-instance override
    envs.disable_envs_cache()
    if spec.vision_tower or spec.canvas_tokens == 1:
        os.environ[key] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
        envs.disable_envs_cache()


@contextmanager
def diffusion_canvas(spec):
    """Initialize a one-token vLLM canvas with Quail's fixed token.

    Yields:
        The initial canvas token IDs, or an empty tuple for other models.
    """
    if spec.canvas_tokens != 1:
        yield ()
        return
    from vllm.model_executor.models import diffusion_gemma

    from quail.backends.quail.executor.models.diffusion_gemma import (
        canvas_token_ids,
    )

    tokens = canvas_token_ids(spec.vocab, spec.canvas_tokens)
    original = diffusion_gemma.DiffusionGemmaRequestStates

    class FixedCanvasStates(original):
        def init_canvas(self, slots):
            if self.canvas_length != 1 or self.vocab_size != spec.vocab:
                raise ValueError("vLLM canvas geometry differs from the model spec")
            self.canvas[slots] = tokens[0]

    # vLLM 0.26 has no public canvas-input setting. Instances retain this
    # subclass after construction; later engines see the original class.
    diffusion_gemma.DiffusionGemmaRequestStates = FixedCanvasStates
    try:
        yield tokens
    finally:
        diffusion_gemma.DiffusionGemmaRequestStates = original



def _capacity(llm) -> dict:
    config = llm.llm_engine.vllm_config
    cache = config.cache_config
    tokens = getattr(cache, "kv_cache_size_tokens", None)
    blocks = getattr(cache, "num_gpu_blocks", None)
    block_size = getattr(cache, "block_size", None)
    if tokens is None and blocks is not None and block_size is not None:
        tokens = blocks * block_size
    if tokens is None or block_size is None:
        raise RuntimeError("vLLM did not report its KV capacity")
    scheduler = config.scheduler_config
    return {
        "kv_cache_size_tokens": int(tokens),
        "num_gpu_blocks": None if blocks is None else int(blocks),
        "block_size": int(block_size),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "max_model_len": int(config.model_config.max_model_len),
        "gpu_memory_utilization": float(cache.gpu_memory_utilization),
        "kv_cache_dtype": str(cache.cache_dtype),
    }


def diffusion_kwargs(spec) -> dict:
    """Set the canvas length and denoising limit through the public API."""
    if not spec.canvas_tokens:
        return {}
    diffusion = {"canvas_length": spec.canvas_tokens}
    kwargs = {"diffusion_config": diffusion}
    if spec.canvas_tokens == 1:
        diffusion["max_denoising_steps"] = 1
        # vLLM refuses a request for more logprobs than this
        kwargs["max_logprobs"] = DIFFUSION_LOGPROBS
    return kwargs


def text_only_kwargs(spec) -> dict:
    """Turn off image and video inputs of a checkpoint with a vision tower."""
    if not spec.vision_tower:
        return {}
    return {"limit_mm_per_prompt": {"image": 0, "video": 0}}


def sampling_kwargs(allowed_ids: list[int], canvas_tokens: int = 0) -> dict:
    """Set public sampling options for the model's answer format.

    Diffusion sampling rejects temperature and allowed_token_ids.
    One canvas token uses returned scores; longer canvases use text.
    """
    if canvas_tokens == 1:
        return {"max_tokens": 1, "logprobs": DIFFUSION_LOGPROBS,
                "detokenize": False}
    if canvas_tokens:
        return {"max_tokens": DIFFUSION_TEXT_TOKENS}
    return {"temperature": 0.0, "max_tokens": 1, "min_tokens": 1,
            "allowed_token_ids": allowed_ids, "detokenize": False}


class VLLMClient:
    """The request operations the backends need from one vLLM LLM."""

    accepts_text = True

    def __init__(self, llm, capacity: dict):
        self.llm = llm
        self.capacity = capacity

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    def decode_params(self, max_tokens: int, stop=None):
        """Build sampling parameters for greedy text generation.

        Args:
            max_tokens: The most tokens to generate.
            stop: Optional strings that end the generation; the text
                returned excludes the string that stopped it.
        """
        from vllm import SamplingParams

        return SamplingParams(max_tokens=max_tokens, temperature=0.0,
                              stop=stop)

    def reset_prefix_cache(self):
        return self.llm.reset_prefix_cache()

    def close(self):
        """Stop the engine and its workers."""
        self.llm.llm_engine.engine_core.shutdown()
        self.llm = None

    def run_filter_sequence(self, sampling_params, body_ids, question_ids,
                         read_answer, *, tag="q", body_texts=None,
                         question_texts=None):
        """Pipeline filter stages through the engine's step loop."""
        return run_filter_sequence(
            self.llm.llm_engine,
            sampling_params,
            body_ids,
            question_ids,
            self.capacity["kv_cache_size_tokens"],
            tag=tag,
            read_answer=read_answer,
            block_size=self.capacity["block_size"],
            max_num_seqs=self.capacity["max_num_seqs"],
            body_texts=body_texts,
            question_texts=question_texts,
            render_prompt=self.llm._preprocess_cmpl_one,
        )


class VLLMEngine:
    """Boot vLLM for a request backend, with Quail's tuned engine settings."""

    kind = "vllm"
    label = "vLLM"
    # prompts are rendered as text and tokenized by vLLM
    reads_document_text = True
    runtime_package = "vllm==0.26.0"
    # a request can decode an answer as text
    scores_labels = True
    # a request returns the logprobs a canvas model's answer is read from
    reads_logprobs = True
    # a decision model runs on the pooling runner
    roles = frozenset({"generative", "decision"})

    def llm_kwargs(self, spec) -> dict:
        """Return the LLM constructor arguments beyond the model name.

        A spec with its own chunk cap (a mixture-of-experts model)
        batches at least that many tokens per step. A one-row-canvas
        diffusion model runs more sequences per step than vLLM's
        default for its 256-row canvas.
        """
        batched = max(MAX_BATCHED_TOKENS, spec.chunk_cap_tokens)
        kwargs = {
            "max_num_batched_tokens": batched,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "tokenizer_mode": "gigatoken",
            "compilation_config": {
                "cudagraph_capture_sizes": [CUDA_GRAPH_CAPTURE_SIZE]
            },
            "max_num_seqs": (DIFFUSION_SEQUENCES if spec.canvas_tokens == 1
                             else min(MAX_SEQUENCES,
                                      spec.max_num_seqs or MAX_SEQUENCES)),
        }
        if spec.moe_backend is not None:
            kwargs["moe_backend"] = spec.moe_backend
        speculative = os.environ.get("QUAIL_VLLM_SPECULATIVE")
        if speculative:
            kwargs["speculative_config"] = speculative_config(speculative)
        kwargs.update(diffusion_kwargs(spec))
        kwargs.update(text_only_kwargs(spec))
        return kwargs

    def boot(self, spec, allowed_ids: list[int], *,
             resources: ExitStack) -> tuple[dict, dict]:
        """Load the spec's model and return its settings and boot record.

        Args:
            spec: The model to load.
            allowed_ids: The token ids a filter answer may be.
            resources: Owns the engine; closing it stops the engine.
        """
        # a session tokenizes on background threads before the engine
        # starts, and a forked worker would inherit their locks
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if spec.role == "decision":
            from quail.baselines.vllm_decision import boot_decision

            return boot_decision(spec, self.llm_kwargs(spec), resources=resources)
        from vllm import LLM, SamplingParams
        from vllm.renderers.registry import RENDERER_REGISTRY
        from vllm.tokenizers import TokenizerRegistry

        registered = (
            "quail.baselines.vllm", "GigatokenVLLMTokenizer"
        )
        if TokenizerRegistry.tokenizers.get("gigatoken") != registered:
            TokenizerRegistry.register("gigatoken", *registered)
        renderer = ("vllm.renderers.hf", "HfRenderer")
        if RENDERER_REGISTRY.renderers.get("gigatoken") != renderer:
            RENDERER_REGISTRY.register("gigatoken", *renderer)

        path = resolve_model_path(spec.hf_name, spec.revision)
        started = time.perf_counter()
        with engine_environment(spec), diffusion_canvas(spec) as canvas_ids:
            llm = LLM(model=path, **self.llm_kwargs(spec))
            client = resources.enter_context(closing(VLLMClient(llm, {})))
        boot_s = time.perf_counter() - started
        sampling_params = SamplingParams(
            **sampling_kwargs(allowed_ids, spec.canvas_tokens))
        capacity = client.capacity
        capacity.update(_capacity(llm))
        cache = llm.llm_engine.vllm_config.cache_config
        capacity.update(
            enable_prefix_caching=cache.enable_prefix_caching,
            input_tokenizer_mode=(
                llm.llm_engine.vllm_config.model_config.tokenizer_mode
            ),
            canvas_length=spec.canvas_tokens,
            initial_canvas_token_ids=list(canvas_ids),
            max_denoising_steps=1 if spec.canvas_tokens == 1 else None,
            logprobs=sampling_params.logprobs,
        )
        client.generate(["TRUE"], sampling_params)
        return (
            {
                "client": client,
                "sampling_params": sampling_params,
                "capacity": capacity,
            },
            {
                "kind": "cold",
                "llm_init_s": round(boot_s, 2),
                "weight_load_s": None,
                "kv_profile_s": None,
                "boot_s": round(boot_s, 2),
            },
        )


class DefaultVLLMEngine(VLLMEngine):
    """Boot vLLM with every engine setting at its default.

    Only the model name is passed, as `vllm serve Qwen/Qwen3-4B-FP8`
    would, plus the canvas a diffusion model's readout needs. The
    engine kind differs from `VLLMEngine` so this engine never shares
    a loaded model with the tuned vLLM backends.
    """

    kind = "dumb_vllm"
    label = "vLLM with default settings"

    def llm_kwargs(self, spec) -> dict:
        return {**diffusion_kwargs(spec), **text_only_kwargs(spec)}


def speculative_config(value: str) -> dict:
    """Return the vLLM speculation an experiment turns on.

    Args:
        value: "ngram:<tokens>", vLLM's n-gram drafts from the prompt
            with that many draft tokens, from QUAIL_VLLM_SPECULATIVE.

    Raises:
        ValueError: The value is not ngram:<tokens>.
    """
    method, _, tokens = value.partition(":")
    if method != "ngram" or not tokens.isdigit() or int(tokens) < 1:
        raise ValueError(
            f"QUAIL_VLLM_SPECULATIVE={value!r} is not ngram:<tokens>")
    # the lookup length is vLLM's documented example
    return {"method": "ngram", "num_speculative_tokens": int(tokens),
            "prompt_lookup_max": 4}


def stock_vllm_backend() -> RequestBackend:
    """Return stock vLLM with operator-at-a-time filter execution."""
    return RequestBackend(
        name="stock_vllm",
        engine=VLLMEngine(),
        filter_submission="operator-at-a-time",
    )


def pipelined_vllm_backend() -> RequestBackend:
    """Return vLLM with per document filter pipelining."""
    return RequestBackend(
        name="pipelined_vllm",
        engine=VLLMEngine(),
        filter_submission="pipelined",
    )


def dumb_vllm_backend() -> RequestBackend:
    """Return vLLM at its default settings, operator-at-a-time filters."""
    return RequestBackend(
        name="dumb_vllm",
        engine=DefaultVLLMEngine(),
        filter_submission="operator-at-a-time",
    )
