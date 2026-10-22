"""Model and device spec structs for the planner."""

from dataclasses import dataclass
from typing import Literal

Precision = Literal["fp8", "bf16", "fp32"]
# What a model answers with: a generative model scores TRUE against
# FALSE for AI_FILTER and AI_JOIN; a reranker scores yes against no
# for AI.SCORE; a decision model scores its yes and no options with
# its own head for AI_FILTER and AI_JOIN.
Role = Literal["generative", "reranker", "decision"]
# What a layer keeps for a sequence. Full and sliding attention keep KV
# (every token, or the last sliding_window tokens); Gated DeltaNet and
# short convolution keep a fixed-size state that does not grow with
# the tokens read.
LayerKind = Literal[
    "full_attention", "sliding_attention", "gated_delta_net", "short_conv"]

# The recurrent state is kept in fp32 and the convolution columns in
# bf16.
GDN_STATE_ELEMENT_BYTES = 4
CONV_STATE_ELEMENT_BYTES = 2

# Peak per-token activation bytes per hidden dim.
ACT_BYTES_PER_HIDDEN = 32


# Every random canvas token comes from this seed: a filter's fixed
# canvas, and a classified document's canvas with the document's row.
CANVAS_SEED = 0
# A canvas answer whose first draw is less certain than this, in nats,
# takes the rest of its noise draws and averages them.
CANVAS_ENTROPY_NATS = 0.1


@dataclass(frozen=True)
class AnswerCanvas:
    """Token layout for a diffusion model's classification answer.

    Attributes:
        rows: Rows of the canvas packed after the classification cue.
        turn_close_id: The token that closes the model's turn, seeded
            after the answer's row.
        pad_id: The padding token, which fills the rows after it.
    """

    rows: int
    turn_close_id: int
    pad_id: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: float        # active projection parameters per token
    layers: int          # L
    hidden: int          # h
    num_q_heads: int             # query heads
    num_kv_heads: int            # KV heads
    d_head: int          # head dim
    ffn_width: int       # dense or shared gate_up output columns
    #                      (gate_up: 2 x intermediate at Qwen3)
    w_bytes: float       # weight bytes per param (fp8 = 1)
    hf_name: str = ""    # the checkpoint (weights + tokenizer)
    revision: str = ""   # pinned hub commit hash; resolves from the
    #                      HF cache without the API round trips a
    #                      branch name pays. "" tracks the default.
    kv_bytes: float = 2.0    # KV bytes per element (bf16 default)
    w_mem_bytes: float = 0.0    # measured weight footprint as loaded;
    #                             0 falls back to params * w_bytes.
    #                             Embeddings and quant scales sit
    #                             outside the dense param count, so the
    #                             measured number is larger.
    vocab: int = 0       # vocabulary rows in the embedding and lm_head
    tied_head: bool = False    # lm_head shares the embedding tensor
    weight_precision: Precision = "fp8"
    attention_precision: Precision = "bf16"
    arch: str = "qwen3"    # forward pass in executor/models/<arch>.py
    role: Role = "generative"
    decision_head_dim: int = 0    # width of a decision model's option
    #                               head (its key, query, and MLP rows);
    #                               the head's FLOPs and bytes follow
    # KV geometry of the full_attention layers, when it differs from the
    # sliding layers' num_kv_heads x d_head; 0 means the same as theirs.
    full_num_kv_heads: int = 0         # KV heads of a full-attention layer
    full_d_head: int = 0       # head dim of a full-attention layer
    sliding_window: int = 0    # tokens the other layers see behind each
    #                            row; 0 means no layer slides
    # Experts beside the dense MLP, for a mixture-of-experts model;
    # 0 means the dense MLP alone.
    experts: int = 0
    experts_active: int = 0    # experts one token multiplies
    expert_intermediate: int = 0    # one expert's MLP width
    canvas_tokens: int = 0    # rows a diffusion model appends after
    #                           the answer cue. 0 for an autoregressive
    #                           model, which answers at the last
    #                           prompt row.
    canvas_answer_row: int = 0    # canvas row the answer is read at;
    #                               later than 0 when the model opens
    #                               its turn with fixed tokens, such as
    #                               an empty thinking channel
    answer_canvas: "AnswerCanvas | None" = None    # a diffusion model's
    #                                        answer canvas, which the
    #                                        letters rule reads
    turn_prefix: str = ""     # chat-turn text before every prompt
    turn_suffix: str = ""     # chat-turn text after the answer cue
    bos_text: str = ""    # the beginning-of-text token at the start of
    #                       turn_prefix, for a model whose tokenizer adds it
    #                       to every prompt: Quail tokenizes without special
    #                       tokens, so the prefix carries it, and the text
    #                       sent to vLLM drops it because vLLM adds its own
    prompt_format: str = "raw-v1"    # names the turn layout in run records
    prompt_layout: str = "ai-if"     # the PROMPT_LAYOUTS entry AI.IF
    #                                  prompts are written in
    chunk_cap_tokens: int = 0    # upper bound on tokens per chunk; 0
    #                              leaves the memory and kernel bounds
    moe_backend: str | None = None    # vLLM's moe_backend setting, one
    #                                   of its MoEBackend names ("triton",
    #                                   "cutlass", "deep_gemm", ...);
    #                                   None lets vLLM pick
    layer_kinds: tuple[LayerKind, ...] = ()    # one kind per layer; required
    #                                            when sliding_window is set;
    #                                            empty means every layer is
    #                                            full_attention
    vision_tower: bool = False    # the checkpoint ships a vision tower;
    #                               vLLM must be told to take no images
    #                               or video, or it builds an image
    #                               processor that cannot copy Quail's
    #                               tokenizer
    max_num_seqs: int = 0    # sequences vLLM may run at once; 0 keeps the
    #                          engine setting. A hybrid model holds one
    #                          recurrent-state block per running sequence,
    #                          and vLLM refuses a larger setting.
    attn_output_gate: bool = False    # q_proj also emits a gate the width
    #                                   of the queries, which scales the
    #                                   attention output
    # Gated DeltaNet layers, for a model whose layer_kinds name them.
    # Each layer keeps a (value heads x value dim x key dim) state and
    # the last conv_width - 1 columns of its convolution input.
    gdn_key_heads: int = 0
    gdn_value_heads: int = 0
    gdn_key_dim: int = 0
    gdn_value_dim: int = 0
    gdn_weight_precision: Precision | None = None
    conv_width: int = 0

    def __post_init__(self):
        if self.layer_kinds and len(self.layer_kinds) != self.layers:
            raise ValueError(
                f"{self.name}: layer_kinds has {len(self.layer_kinds)} "
                f"entries for {self.layers} layers")
        if self.sliding_window and not self.layer_kinds:
            raise ValueError(
                f"{self.name}: sliding_window needs layer_kinds to say which "
                f"layers slide")

    def layer_kind(self, layer: int) -> LayerKind:
        """Return the kind of one layer."""
        return self.layer_kinds[layer] if self.layer_kinds else "full_attention"

    def keeps_kv(self, layer: int) -> bool:
        """Whether the layer stores KV; Gated DeltaNet and convolution layers do not."""
        return self.layer_kind(layer) in ("full_attention", "sliding_attention")

    @property
    def attention_layers(self) -> tuple[int, ...]:
        """The layers that store KV."""
        return tuple(i for i in range(self.layers) if self.keeps_kv(i))

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        """The layers that keep a Gated DeltaNet state."""
        return tuple(i for i in range(self.layers)
                     if self.layer_kind(i) == "gated_delta_net")

    @property
    def short_conv_layers(self) -> tuple[int, ...]:
        """The layers that keep a short convolution's input columns."""
        return tuple(i for i in range(self.layers)
                     if self.layer_kind(i) == "short_conv")

    @property
    def state_layers(self) -> tuple[int, ...]:
        """The layers that keep state between a document's tokens."""
        return tuple(i for i in range(self.layers)
                     if self.layer_kind(i) in ("gated_delta_net", "short_conv"))

    @property
    def conv_state_channels(self) -> int:
        """Channels of the convolution input a state layer keeps."""
        return self.conv_channels if self.gdn_layers else self.hidden

    @property
    def conv_channels(self) -> int:
        """Channels of a Gated DeltaNet layer's convolution: q, k, and v joined."""
        return (2 * self.gdn_key_heads * self.gdn_key_dim
                + self.gdn_value_heads * self.gdn_value_dim)

    @property
    def gdn_state_bytes(self) -> int:
        """Bytes of one layer's recurrent state, a matrix per value head."""
        return (self.gdn_value_heads * self.gdn_value_dim * self.gdn_key_dim
                * GDN_STATE_ELEMENT_BYTES)

    @property
    def conv_state_bytes(self) -> int:
        """Bytes of one layer's convolution input columns kept between tokens."""
        return (self.conv_channels * (self.conv_width - 1)
                * CONV_STATE_ELEMENT_BYTES)

    @property
    def checkpoint_bytes(self) -> int:
        """Bytes of the state a sequence leaves in every state layer.

        A Gated DeltaNet layer keeps its state matrix and convolution
        columns; a short convolution layer keeps its input columns.
        Zero for a model without either.
        """
        return (len(self.gdn_layers) * (self.gdn_state_bytes
                                        + self.conv_state_bytes)
                + len(self.short_conv_layers) * self.hidden
                * (self.conv_width - 1) * CONV_STATE_ELEMENT_BYTES)

    def is_full_layer(self, layer: int) -> bool:
        """Whether the layer keeps every token with the full KV geometry."""
        return self.layer_kind(layer) == "full_attention"

    @property
    def kv_shapes(self) -> tuple:
        """Per layer, the (KV heads, head dim) its KV stores."""
        full = (self.full_num_kv_heads or self.num_kv_heads,
                self.full_d_head or self.d_head)
        return tuple(
            (0, 0) if not self.keeps_kv(i)
            else full if self.is_full_layer(i) else (self.num_kv_heads, self.d_head)
            for i in range(self.layers))

    @property
    def widest_projection(self) -> int:
        """Output columns of the widest dense projection in any layer."""
        query_width = 2 if self.attn_output_gate else 1
        attention = max(
            (query_width * self.num_q_heads + 2 * num_kv_heads) * head
            for num_kv_heads, head in self.kv_shapes)
        gdn = (2 * self.gdn_key_heads * self.gdn_key_dim
               + 2 * self.gdn_value_heads * self.gdn_value_dim)
        return max(self.ffn_width, attention, gdn)

    @property
    def kappa(self) -> float:
        """KV bytes per cached token, summed over the layers."""
        return self.kv_elements_per_token * self.kv_bytes

    @property
    def kv_elements_per_token(self) -> int:
        """KV elements per token, dtype-free: 2 * sum of num_kv_heads * d_head."""
        return 2 * sum(num_kv_heads * d_head for num_kv_heads, d_head in self.kv_shapes)

    @property
    def sliding_layer_set(self) -> frozenset:
        """The layers that keep only the last sliding_window tokens of KV."""
        if not self.sliding_window:
            return frozenset()
        return frozenset(i for i in range(self.layers)
                         if self.layer_kind(i) == "sliding_attention")

    @property
    def kappa_sliding(self) -> float:
        """KV bytes per token on the sliding layers alone."""
        sliding = self.sliding_layer_set
        return 2 * self.kv_bytes * sum(
            num_kv_heads * d_head
            for i, (num_kv_heads, d_head) in enumerate(self.kv_shapes)
            if i in sliding)

    @property
    def kappa_full(self) -> float:
        """KV bytes per token on the layers that keep every token."""
        return self.kappa - self.kappa_sliding

    @property
    def turn(self) -> tuple[str, str]:
        """The chat-turn text wrapped around every prompt."""
        return self.turn_prefix, self.turn_suffix

    # W_mem and W_resident are the weight-memory names used throughout
    # the code and the engine wiki.
    @property
    def W_mem(self) -> float:  # noqa: N802
        """Weight bytes as loaded, before the full untied head is discarded."""
        return self.w_mem_bytes or self.params * self.w_bytes

    @property
    def head_mem_bytes(self) -> float:
        """Return untied output-head memory in bytes, or zero for tied weights.

        The executor keeps this matrix for AI.CLASSIFY, so it stays
        resident beside the other weights. Both Qwen3 checkpoints store
        the head in bf16, hence the 2 bytes per element.
        """
        if self.tied_head:
            return 0.0
        return self.vocab * self.hidden * 2.0

    @property
    def W_resident(self) -> float:  # noqa: N802
        """Return model weight bytes resident on the GPU, including the output head."""
        return self.W_mem

    @property
    def act_per_token(self) -> float:
        """Reserved activation bytes per batched token."""
        dense = ACT_BYTES_PER_HIDDEN * self.hidden
        # Triton experts use two BF16 workspaces, plus activation quantization
        # and routing buffers. They are reused across layers.
        expert = 2 * self.experts_active * (
            max(self.hidden, self.expert_intermediate)
            + max(self.hidden, 2 * self.expert_intermediate)
            + self.expert_intermediate)
        routing = 4 * self.experts + 16 * self.experts_active
        return dense + expert + routing

    @property
    def intermediate(self) -> int:
        """MLP intermediate width; gate_up packs two of them."""
        return self.ffn_width // 2

@dataclass(frozen=True)
class DeviceSpec:
    name: str
    mem_bytes: float     # M: physical memory
    hbm_bw: float        # BW: memory bandwidth, bytes/s
    peak_flops: float    # R_D at the executor's compute dtype
    bf16_flops: float = 0.0    # dense bf16 peak. Attention runs in
    #                            bf16 (FlashAttention-3 over bf16 KV),
    #                            so the pair FLOPs price against this,
    #                            not against the fp8 ceiling. 0 falls
    #                            back to half of peak_flops, the fp8-
    #                            to-bf16 ratio on every tensor core
    #                            generation we run on.
    fp32_flops: float = 0.0    # fp32 peak on the CUDA cores, not the
    #                            tensor cores: the decision head runs
    #                            there. 0 means no rate is known.
    usd_per_hour: float = 0.0    # rental price of one device; 0 means
    #                              no price is known
    price_source: str = ""       # where usd_per_hour was read from
    mamf_fp8_flops: float = 0.0    # 0 uses the specification rate
    mamf_bf16_flops: float = 0.0

    def gemm_flops(self, precision: Precision) -> float:
        """Return measured GEMM throughput, or the specification rate."""
        measured = {"fp8": self.mamf_fp8_flops, "bf16": self.mamf_bf16_flops}
        return measured.get(precision, 0.0) or self.arithmetic_bandwidth(precision)

    @property
    def attn_flops(self) -> float:
        """Dense peak for the attention kernels, FLOP/s."""
        return self.bf16_flops or self.peak_flops / 2

    def arithmetic_bandwidth(self, precision: Precision) -> float:
        """Return arithmetic throughput for one component precision."""
        if precision == "fp8":
            return self.peak_flops
        if precision == "bf16":
            return self.attn_flops
        if precision == "fp32" and self.fp32_flops:
            return self.fp32_flops
        raise ValueError(f"unsupported precision: {precision}")
