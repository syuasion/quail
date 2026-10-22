# Measure GEMM MAMF

The cost model uses **MAMF**, the maximum achievable matrix multiplication
rate measured in short bursts. Measure FP8 and BF16 separately on each GPU
type. Attention and memory bandwidth retain their specification rates.

We run [mamf-finder at commit 30ddd970](https://github.com/stas00/ml-engineering/blob/30ddd97054eb01035c0a5079a58f011df603b525/compute/accelerator/benchmarks/mamf-finder.py)
unchanged and verify its SHA-256 before every run. FP8 uses `torch._scaled_mm`
with tensor scales and FP8 output. BF16 uses `torch.mm` with BF16 output.
Both use the vendor library, cuBLAS. These measurements do not time Quail's
DeepGEMM kernels or its model shapes.

The upstream script also measures MSMF, its sustained rate under full load.
We keep that output in the raw log but use only MAMF in the specification.
A burst rate is more appropriate for our optimistic SoL estimate. The highest
observed rate is a calibration, not a proven upper limit on every kernel or GPU.

## Run on Modal

For example, measure both precisions on one H100 and one RTX PRO 6000:

```bash
log="/tmp/quail-calibrate-$(date -u +%Y%m%dT%H%M%SZ).log"
uv run modal run --detach -m calibrate.modal_app::gemm \
  --device h100-sxm,rtx-pro-6000-blackwell-server \
  --precision fp8,bf16 --gpus 1 \
  --prediction 'H100 FP8 1500 to 1630 TFLOPS; BF16 800 to 900 TFLOPS. RTX rates below its specification.' \
  2>&1 | tee "$log"
```

Each GPU measures its precisions in sequence. `--gpus` repeats the measurement
on that many GPU allocations per device type. The CLI prints each function
call ID and volume path. Retrieve a detached result with
`modal.FunctionCall.from_id("fc-...").get()`.

H100 uses the published MAMF shape for each precision from the
[upstream table at 8d028c29](https://github.com/stas00/ml-engineering/blob/8d028c29/compute/accelerator/README.md).
A device without a published shape uses upstream's automatic shape search.
The environment is `nvcr.io/nvidia/pytorch:26.04-py3` for both devices.

Results are on `quail-results` under
`/results/calibrate/<device>/gemm/<stamp>/`. Each directory contains a log per
precision and `summary.json` with GPU identity, script hash, arguments, and
validated MAMF rates. Missing headlines, incorrect dtypes, and measurements
without a validated boost clock fail the run. Raw logs are saved on failure.

## Update a device specification

Download the selected run directories, then pass them to the converter:

```bash
mkdir -p ./calibration
modal volume get quail-results calibrate/<device>/gemm/<stamp>/ ./calibration/
uv run python -m calibrate.to_spec ./calibration/
```

Copy the printed `mamf_fp8_flops` and `mamf_bf16_flops` into the matching
`quail/specs/` file and cite the volume paths there. The converter takes the
highest MAMF for each precision across the supplied runs. It rejects mixed
device types and missing or invalid measurements. Earlier summaries without
validation metadata must be revalidated from their raw logs before use.

To add a device, add its Modal GPU type to `targets.py`. Add published MAMF
shapes there only when the upstream table provides them. Keep each device's
results separate. This package is not included in the Quail wheel.
