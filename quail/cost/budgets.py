"""Budget quantities derived from model and device specs.

Chunk budget, admission budget, and roofline arithmetic.
"""

from math import ceil

from quail.specs import DeviceSpec, ModelSpec

POOL_FRACTION = 0.95    # fraction of device memory the executor claims
PAGE_TOKENS = 16        # KV arena page size, tokens
INT32_MAX = 2**31 - 1
ACT_BYTES = 2           # bf16 activations, bytes per element
ACT_RESERVE_CHUNKS = 2  # chunks of activation memory reserved outside
#                         the arena for overlapped chunk construction


def minimum_weight_gpus(model: ModelSpec, device: DeviceSpec) -> int:
    """Return how many pooled GPU memories would hold the loaded weights.

    Quail does not split weights across GPUs. A result above one causes a
    planning refusal.
    """
    gpus = 1
    while model.W_mem > device.mem_bytes * POOL_FRACTION * gpus:
        gpus *= 2
    return gpus


def kernel_index_cap(model: ModelSpec) -> int:
    """Max tokens per chunk from the int32 element-offset limit.

    Constraint: rows x widest projection < 2^31.
    """
    width = max(model.widest_projection, model.experts_active
                * max(model.hidden, 2 * model.expert_intermediate))
    return INT32_MAX // width


def chunk_memory_bound(model: ModelSpec, device: DeviceSpec) -> int:
    """Return the largest chunk that fits its memory reservations, or zero."""
    available = device.mem_bytes * POOL_FRACTION
    free = available - model.W_resident
    lower = 0
    upper = max(0, int(free // (ACT_RESERVE_CHUNKS * model.act_per_token)))
    while lower < upper:
        chunk = (lower + upper + 1) // 2
        if chunk_memory_bytes(model, chunk) <= available:
            lower = chunk
        else:
            upper = chunk - 1
    return lower


def chunk_memory_bytes(model: ModelSpec, chunk_tokens: int) -> float:
    """Return resident weights, activation memory, and checkpoint reservations."""
    return (model.W_resident
            + ACT_RESERVE_CHUNKS * chunk_tokens * model.act_per_token
            + checkpoint_memory_bytes(model, chunk_tokens))


def pages(tokens: int) -> int:
    """Return the KV pages that hold tokens."""
    return -(-tokens // PAGE_TOKENS)


def chunk_budget(model: ModelSpec, device: DeviceSpec) -> int:
    """Return the chunk budget allowed by the model and device.

    Memory, kernel indexing, and the model's configured cap are hard limits.
    Return zero when even one token cannot fit the memory reservations.
    """
    b = min(chunk_memory_bound(model, device), kernel_index_cap(model))
    if model.chunk_cap_tokens:
        b = min(b, model.chunk_cap_tokens)
    return b


def arena_bytes(model: ModelSpec, device: DeviceSpec,
                chunk_tokens: int) -> float:
    """Bytes left for KV after resident weights and the activation reserve."""
    return (device.mem_bytes * POOL_FRACTION - model.W_resident
            - ACT_RESERVE_CHUNKS * chunk_tokens * model.act_per_token)


# Questions one pass may start from checkpoints at once; each needs a
# temporary copy of the checkpoint state.
CHECKPOINT_PASS_QUESTIONS = 1_024
# Document length a hybrid model's retained pool is split for when the
# mean is unknown.
CHECKPOINT_DEFAULT_DOC_TOKENS = 1_000
MIB = 2**20
INTERMEDIATE_BLOCK_TOKENS = 64


# Rows past a document that its pages also cover: the shared question
# preamble and a stage tail, taken as a round number for the split.
SPLIT_EXTRA_TOKENS = 64


def transient_sliding_pages(chunk_tokens: int, window: int) -> int:
    """Sliding-layer pages one chunk's fresh prefixes take before trimming.

    The chunk's rows, plus one page of rounding per document longer
    than the window, of which a chunk holds at most chunk / window.
    """
    return -(-chunk_tokens // PAGE_TOKENS) + -(-chunk_tokens // window)


def checkpoint_pages(model: ModelSpec) -> float:
    """KV pages of memory one document checkpoint takes."""
    return model.checkpoint_bytes / (model.kappa * PAGE_TOKENS)


def checkpoint_pass_bytes(model: ModelSpec, questions: int,
                          chunk_tokens: int) -> int:
    """Temporary bytes a pass of `questions` starts from checkpoints.

    Each question copies one layer's state and its convolution columns.
    The pass also holds a 1 MiB intermediate buffer per question and
    one per 64 tokens of the chunk.
    """
    buffers = questions + -(-chunk_tokens // INTERMEDIATE_BLOCK_TOKENS)
    return (questions * (model.gdn_state_bytes + model.conv_state_bytes)
            + buffers * MIB)


def execution_pages(chunk_tokens: int) -> int:
    """KV pages kept free for two chunks of execution."""
    return 2 * -(-chunk_tokens // PAGE_TOKENS)


def checkpoint_memory_bytes(model: ModelSpec, chunk_tokens: int) -> float:
    """Return workspace, execution KV, and one retained document's KV and state.

    Models without checkpoints reserve zero bytes. A retained document can
    be as long as a chunk; startup reserves at least the default document
    length used by the checkpoint pool.
    """
    if not model.checkpoint_bytes:
        return 0
    document_tokens = max(chunk_tokens, CHECKPOINT_DEFAULT_DOC_TOKENS)
    reserved_pages = (execution_pages(chunk_tokens) + pages(document_tokens)
                      + ceil(checkpoint_pages(model)))
    return (checkpoint_pass_bytes(model, CHECKPOINT_PASS_QUESTIONS, chunk_tokens)
            + reserved_pages * model.kappa * PAGE_TOKENS)


def hybrid_pool_pages(model: ModelSpec, device: DeviceSpec,
                      chunk_tokens: int) -> int:
    """Pages a hybrid model shares between retained KV and checkpoints.

    The arena after weights and two chunks of activations, less the
    temporary copies of one pass of questions and two chunks of KV
    kept free for execution.
    """
    free = (arena_bytes(model, device, chunk_tokens)
            - checkpoint_pass_bytes(model, CHECKPOINT_PASS_QUESTIONS,
                                  chunk_tokens))
    page_bytes = model.kappa * PAGE_TOKENS
    return int(free // page_bytes) - execution_pages(chunk_tokens)


def checkpoint_slots(model: ModelSpec, device: DeviceSpec, chunk_tokens: int,
                     kv_pages: int) -> int:
    """Checkpoints the pool holds beside an arena of `kv_pages`; 0 without them."""
    pages = checkpoint_pages(model)
    if not pages:
        return 0
    retained = kv_pages - execution_pages(chunk_tokens)
    return max(0, int((hybrid_pool_pages(model, device, chunk_tokens)
                       - retained) // pages))


def hybrid_arena_pages(model: ModelSpec, device: DeviceSpec, chunk_tokens: int,
                       mean_document_tokens: float | None) -> int:
    """Arena KV pages: two chunks of execution plus the retained KV.

    A retained document of `l` tokens costs ceil(l / 16) KV pages plus
    its checkpoint, so the pool holds floor(pool / that) documents, and
    the KV the checkpoints leave is retained KV.
    """
    mean = CHECKPOINT_DEFAULT_DOC_TOKENS if mean_document_tokens is None else max(
        1.0, mean_document_tokens)
    pool = hybrid_pool_pages(model, device, chunk_tokens)
    ckpt = checkpoint_pages(model)
    slots = int(pool // (-(-mean // PAGE_TOKENS) + ckpt))
    kv_pages = int(pool - slots * ckpt)
    if kv_pages <= 0:
        raise ValueError("the pool cannot hold one chunk's KV beside a checkpoint")
    if slots < 1:
        raise ValueError(
            f"the pool holds no checkpoint of {ckpt:.1f} pages beside a "
            f"document of {mean:.0f} tokens")
    return kv_pages + execution_pages(chunk_tokens)


def arena_pages(model: ModelSpec, device: DeviceSpec,
                chunk_tokens: int | None = None,
                mean_document_tokens: float | None = None) -> tuple[int, int]:
    """Pages of the two KV pools: (every-token pool, sliding-layer pool).

    A model with Gated DeltaNet layers gets the KV pages left beside its
    checkpoints (see hybrid_arena_pages) and 0 sliding pages.

    A model without sliding layers gets one pool and 0 sliding pages.
    With sliding layers, a document holds its whole prefix on the
    full-attention layers and only the last window on the sliding
    layers, so the sliding pool is sized to the fraction of a mean
    document that the window keeps. It never drops below the pages one
    chunk's fresh prefixes take before they are trimmed. An unknown
    mean sizes it as if every document fit in the window, the largest
    fraction.
    """
    if chunk_tokens is None:
        chunk_tokens = chunk_budget(model, device)
    if model.state_layers:
        return hybrid_arena_pages(model, device, chunk_tokens,
                                  mean_document_tokens), 0
    free = arena_bytes(model, device, chunk_tokens)
    page_bytes_full = model.kappa_full * PAGE_TOKENS
    page_bytes_sliding = model.kappa_sliding * PAGE_TOKENS
    if not page_bytes_sliding:
        return int(free // page_bytes_full), 0
    window = model.sliding_window
    mean = window if mean_document_tokens is None else max(1.0, mean_document_tokens)
    ratio = min(1.0, (min(mean, window) + SPLIT_EXTRA_TOKENS)
                / (mean + SPLIT_EXTRA_TOKENS))
    full = int(free // (page_bytes_full + ratio * page_bytes_sliding))
    sliding = int(full * ratio)
    floor = transient_sliding_pages(chunk_tokens, window)
    if sliding < floor:
        sliding = floor
        full = int((free - sliding * page_bytes_sliding) // page_bytes_full)
    if full <= 0:
        raise ValueError("the KV arena cannot hold one chunk's sliding KV")
    return full, sliding


def arena_tokens(model: ModelSpec, device: DeviceSpec,
                 chunk_tokens: int | None = None,
                 mean_document_tokens: float | None = None) -> int:
    """Admission budget: tokens of document KV that can be resident at once.

    Computed from the memory left after resident weights and the
    activation reservation. With sliding layers this is the
    every-token pool; see arena_pages.
    """
    full, _ = arena_pages(model, device, chunk_tokens, mean_document_tokens)
    return full * PAGE_TOKENS


# ---- roofline arithmetic

def _projection_shapes(model: ModelSpec):
    """Return (in_dim, out_dim, layers, precision) of each dense projection.

    `layers` counts the layers that have the projection: the attention
    projections sit in the layers that store KV, the Gated DeltaNet
    projections in its layers, and the MLP in every layer.
    """
    query_width = 2 if model.attn_output_gate else 1
    qkv_out = (query_width * model.num_q_heads + 2 * model.num_kv_heads) * model.d_head
    inter = model.intermediate
    attention = len(model.attention_layers)
    shapes = [(model.hidden, qkv_out, attention, model.weight_precision),
              (model.num_q_heads * model.d_head, model.hidden, attention,
               model.weight_precision),
              (model.hidden, model.ffn_width, model.layers, model.weight_precision),
              (inter, model.hidden, model.layers, model.weight_precision)]
    if model.gdn_layers:
        key_width = model.gdn_key_heads * model.gdn_key_dim
        value_width = model.gdn_value_heads * model.gdn_value_dim
        gdn = len(model.gdn_layers)
        precision = model.gdn_weight_precision or model.weight_precision
        shapes += [(model.hidden,
                    2 * key_width + 2 * value_width + 2 * model.gdn_value_heads,
                    gdn, precision),
                   (value_width, model.hidden, gdn, precision)]
    if model.short_conv_layers:
        short = len(model.short_conv_layers)
        shapes += [(model.hidden, 3 * model.hidden, short, model.weight_precision),
                   (model.hidden, model.hidden, short, model.weight_precision)]
    return tuple(shapes)


def compute_knee(model: ModelSpec, device: DeviceSpec) -> float:
    """Chunk size (tokens) where the dense projections become compute-bound.

    That is where they cross the roofline ridge.
    """
    compute_seconds = weight_bytes = activation_bytes = 0.0
    for din, dout, layers, precision in _projection_shapes(model):
        share = layers / model.layers
        params = share * din * dout
        compute_seconds += 2 * params / device.gemm_flops(precision)
        weight_bytes += params * {"fp8": 1, "bf16": 2, "fp32": 4}[precision]
        activation_bytes += share * (din + dout) * ACT_BYTES
    denom = compute_seconds * device.hbm_bw - activation_bytes
    if denom <= 0:
        raise ValueError("projections never cross the ridge")
    return weight_bytes / denom


def _attention_time(model: ModelSpec, device: DeviceSpec,
                    chunk: int, context: int) -> float:
    """Ideal seconds for the attention kernels in one chunk, all layers."""
    layers = len(model.attention_layers)
    flops = 4.0 * chunk * context * model.num_q_heads * model.d_head
    moved = (context * model.kappa / layers
             + 2.0 * chunk * model.num_q_heads * model.d_head * ACT_BYTES)
    return max(flops / device.arithmetic_bandwidth(model.attention_precision),
               moved / device.hbm_bw) * layers


# Below this many dense parameters a model runs unified attention only.
# On Decision-2.0-Kai-0.6B (0.44e9) the tree path ran 4 to 7 percent
# slower than unified on the QUAIL-B IMDB-2 and IMDB-8 joins, where the
# roofline below picked tree.
TREE_ATTENTION_MIN_PARAMS = 2e9


def tree_attention_allowed(model: ModelSpec) -> bool:
    """Whether the planner and the executor may run the model's tree path.

    A diffusion model packs canvas rows, which the tree path does not.
    A hybrid model's state layers have no tree path.
    """
    return (not model.canvas_tokens and not model.state_layers
            and model.params >= TREE_ATTENTION_MIN_PARAMS)


def choose_attention_path(model: ModelSpec, device: DeviceSpec, *,
                          readers: float, reader_rows: float,
                          node_tokens: float) -> str:
    """Pick "unified" or "tree" attention for readers of one shared node.

    The node is a join's anchor or a filter's parent document; its
    readers are the partners or the children borrowing its pages.

    Under unified attention every reader reads the node's KV itself.
    Under tree attention the readers' rows are stacked into one read of
    the node, then merged with each reader's attention over its own
    rows. Both times take the larger of compute and memory, so tree
    wins only for readers that are memory-bound on the node: few rows
    each, many of them.

    Args:
        model: The model whose attention runs.
        device: The device it runs on.
        readers: Readers sharing the node in one forward pass.
        reader_rows: Query rows per reader.
        node_tokens: KV tokens of the shared node.
    """
    if readers <= 1 or reader_rows <= 0 or node_tokens <= 0:
        return "unified"
    unified = readers * _attention_time(model, device, reader_rows,
                                        node_tokens)
    rows = readers * reader_rows
    tree = _attention_time(model, device, rows, node_tokens)
    # the merge kernel reads two partial outputs and writes one, in
    # bf16, once per layer
    merge = 3 * rows * model.num_q_heads * model.d_head * ACT_BYTES / device.hbm_bw
    tree += merge * len(model.attention_layers)
    return "tree" if tree < unified else "unified"


def parallel_copies(gpus: int, num_documents: float) -> float:
    """Return the model copies that share work spread evenly over documents."""
    return max(1.0, min(gpus, num_documents))
