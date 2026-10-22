"""Decoder components used by the roofline calculation.

Any pre-norm decoder with grouped-query attention and a gated MLP
is priced from its ModelSpec shape; nothing here is Qwen3-specific.
A mixture-of-experts model names its experts on the spec: FLOPs
follow the params one token multiplies, weight bytes follow the
params a full chunk reads. A hybrid model adds its Gated DeltaNet
projections to the attention projections, and a short convolution
model its gate and output projections. The Gated DeltaNet chunked
recurrence is priced as its own component at the bf16 tensor-core
rate, and each convolution at the fp32 CUDA-core rate. The state's
movement is priced like KV movement: one checkpoint of
ModelSpec.checkpoint_bytes for every write and every read in Work.
"""

from __future__ import annotations

from quail.cost.roofline import CostComponent
from quail.cost.work import Work
from quail.specs import ModelSpec

# vLLM's FLA_CHUNK_SIZE: tokens per chunk of the chunked gated delta rule
GDN_CHUNK_TOKENS = 64


def attention_projection_params(model: ModelSpec) -> int:
    """Return Q, K, V, and output projection parameters over the layers."""
    h = model.hidden
    query_width = 2 if model.attn_output_gate else 1
    return sum(h * query_width * model.num_q_heads * head + 2 * h * num_kv_heads * head
               + model.num_q_heads * head * h
               for num_kv_heads, head in model.kv_shapes)


def gdn_projection_params(model: ModelSpec) -> int:
    """Return Gated DeltaNet input and output projection parameters.

    Per layer: the query, key, and value projection, the z gate, the
    two per-head gates, and the output projection. The state update, the
    convolution, and the state's movement are not priced here.
    """
    key_width = model.gdn_key_heads * model.gdn_key_dim
    value_width = model.gdn_value_heads * model.gdn_value_dim
    per_layer = model.hidden * (2 * key_width + 2 * value_width
                                + 2 * model.gdn_value_heads) \
        + value_width * model.hidden
    return per_layer * len(model.gdn_layers)


def short_conv_projection_params(model: ModelSpec) -> int:
    """Return short convolution input and output projection parameters.

    Per layer: the projection to the three gates (B, C, and x, each
    the model's width) and the output projection. The convolution
    itself is not priced here.
    """
    return 4 * model.hidden * model.hidden * len(model.short_conv_layers)


def gdn_core_flops_per_token(model: ModelSpec) -> float:
    """Return FLOPs per token of the chunked gated delta rule, over all layers.

    The kernel (vLLM's chunk_gated_delta_rule) works on chunks of C
    tokens. Per chunk and value head, with key dim K and value dim V:
    three C x C x K products (K K^T, the w update, Q K^T), two
    C x C x V products (the u update and the A times new-value
    product), three C x K x V products (w times state, K^T times new
    value, Q times state), and a C x C triangular inverse of
    about 2 C^3 / 3. Each product costs 2 FLOPs per multiply-add. Divided
    by C this is 6 C K + 4 C V + 6 K V + 2 C^2 / 3 per token and head.
    """
    c = GDN_CHUNK_TOKENS
    k, v = model.gdn_key_dim, model.gdn_value_dim
    per_head = 6 * c * k + 4 * c * v + 6 * k * v + 2 * c * c / 3
    return per_head * model.gdn_value_heads * len(model.gdn_layers)


def conv_flops_per_token(model: ModelSpec) -> float:
    """Return FLOPs per token of the depthwise convolutions, over all layers.

    Each channel takes one multiply and one add per tap. A Gated
    DeltaNet layer convolves its q, k, and v channels; a short
    convolution layer convolves the model's width.
    """
    gdn = len(model.gdn_layers) * model.conv_channels
    short = len(model.short_conv_layers) * model.hidden
    return 2.0 * model.conv_width * (gdn + short)


def mlp_params(model: ModelSpec) -> int:
    """Return the MLP parameters one token multiplies."""
    return _mlp_params(model, model.experts_active)


def mlp_weight_params(model: ModelSpec) -> int:
    """Return the MLP parameters a full chunk reads: every expert."""
    return _mlp_params(model, model.experts)


def _mlp_params(model: ModelSpec, experts: int) -> int:
    """The dense MLP plus this many experts, over the layers."""
    per_layer = 3 * model.hidden * (
        model.intermediate + experts * model.expert_intermediate)
    return per_layer * model.layers


def dense_params(model: ModelSpec) -> int:
    """Return parameters used by the modeled dense components."""
    return (attention_projection_params(model) + gdn_projection_params(model)
            + short_conv_projection_params(model) + mlp_params(model))


def attention_pair_flops(model: ModelSpec) -> tuple[int, int]:
    """Return FLOPs per pair across full layers and sliding layers."""
    sliding = model.sliding_layer_set
    full_flops = sliding_flops = 0
    for layer, (_, head) in enumerate(model.kv_shapes):
        if layer in sliding:
            sliding_flops += 4 * model.num_q_heads * head
        else:
            full_flops += 4 * model.num_q_heads * head
    return full_flops, sliding_flops


def dense_decoder_components(work: Work, model: ModelSpec, passes: float,
                             rows: float = 0.0, head_rows: int = 0,
                             ) -> tuple[CostComponent, ...]:
    """Build the modeled decoder components, with a readout when rows is set."""
    if passes < 0:
        raise ValueError("passes must be nonnegative")
    attn_proj = (attention_projection_params(model)
                 + gdn_projection_params(model)
                 + short_conv_projection_params(model))
    separate_gdn = (gdn_projection_params(model)
                    if model.gdn_weight_precision is not None else 0)
    attn_proj -= separate_gdn
    mlp = mlp_params(model)
    full_flops, sliding_flops = attention_pair_flops(model)
    components = (
        CostComponent(
            name="attn_proj",
            flops=2.0 * attn_proj * work.tokens,
            bytes_moved=attn_proj * model.w_bytes * passes,
            precision=model.weight_precision, gemm=True,
        ),
        CostComponent(
            name="mlp",
            flops=2.0 * mlp * work.tokens,
            bytes_moved=mlp_weight_params(model) * model.w_bytes * passes,
            precision=model.weight_precision, gemm=True,
        ),
        CostComponent(
            name="attention",
            flops=full_flops * work.pairs + sliding_flops * work.sliding_pairs,
            bytes_moved=(
                model.kappa_full * (work.kv_written + work.kv_read)
                + model.kappa_sliding * (work.kv_written + work.sliding_kv_read)),
            precision=model.attention_precision,
        ),
    )
    if model.experts:
        router = model.layers * model.hidden * model.experts
        components += (CostComponent(
            name="router", flops=2.0 * router * work.tokens,
            bytes_moved=2 * router * passes, precision="bf16", gemm=True),)
    if separate_gdn:
        precision = model.gdn_weight_precision
        components += (CostComponent(
            name="gdn_proj",
            flops=2.0 * separate_gdn * work.tokens,
            bytes_moved=separate_gdn * {"fp8": 1, "bf16": 2, "fp32": 4}[
                precision] * passes,
            precision=precision, gemm=True,
        ),)
    if model.gdn_layers:
        components += (CostComponent(
            name="gdn_core",
            flops=gdn_core_flops_per_token(model) * work.tokens,
            bytes_moved=0.0,
            precision="bf16",
        ),)
    if model.checkpoint_bytes:
        components += (CostComponent(
            name="state",
            flops=0.0,
            bytes_moved=model.checkpoint_bytes * (
                work.state_written + work.state_read),
            precision="bf16",
        ),)
    if model.gdn_layers or model.short_conv_layers:
        components += (CostComponent(
            name="conv",
            flops=conv_flops_per_token(model) * work.tokens,
            bytes_moved=0.0,
            precision="fp32",
        ),)
    if rows:
        components += (_readout(model, rows, passes, head_rows),)
    return components


def _readout(model, rows: float, passes: float, head_rows: int) -> CostComponent:
    """Price the answer readout: the decision head, or output-head rows.

    A decision model passes each option row and the prompt's last row
    through its fp32 head: four projections of the hidden size to the
    head width per row, and the head's weights once per pass. Any other
    model multiplies each answer row by head_rows rows of the bf16
    output head, the whole vocabulary when head_rows is 0, read once per
    pass.
    """
    if model.role == "decision":
        width, hidden = model.decision_head_dim, model.hidden
        return CostComponent(
            name="readout", flops=4.0 * width * hidden * rows,
            bytes_moved=4.0 * (4 * width * hidden + 2 * width + 4 * hidden) * passes,
            precision="fp32")
    head = 2.0 * model.hidden * (head_rows or model.vocab)
    return CostComponent(name="readout", flops=head * rows,
                         bytes_moved=head * passes, precision="bf16", gemm=True)
