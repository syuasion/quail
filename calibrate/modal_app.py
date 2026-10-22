"""Measure GEMM MAMF on Modal and save results to quail-results."""

from datetime import datetime, timezone
from pathlib import Path

import modal

from calibrate import gemm as gemm_run
from calibrate.targets import DTYPES, TARGETS

NGC_IMAGE = "nvcr.io/nvidia/pytorch:26.04-py3"
MAMF_DIR = Path("/opt/mamf-finder")
ROOT = Path("/results/calibrate")

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
image = (
    modal.Image.from_registry(NGC_IMAGE)
    .pip_install("nvidia-ml-py")
    .run_commands(
        f"mkdir -p {MAMF_DIR}",
        f"curl -fsSL {gemm_run.MAMF_URL} -o {MAMF_DIR / gemm_run.MAMF_NAME}")
    .add_local_python_source("calibrate")
)


@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/results": results_vol})
def measure_gemm(device: str, stamp: str, precisions: tuple[str, ...],
                 prediction: str) -> dict:
    """Measure each precision on one GPU and commit the results and logs."""
    return gemm_run.run(ROOT, device, stamp, precisions, MAMF_DIR, prediction,
                        on_save=results_vol.commit)


@app.local_entrypoint()
def gemm(device: str = "h100-sxm", precision: str = "fp8,bf16", gpus: int = 1,
         prediction: str = ""):
    """Measure comma-separated devices and precisions on gpus copies each."""
    devices = tuple(device.split(","))
    precisions = tuple(precision.split(","))
    if (gpus < 1 or not set(devices) <= TARGETS.keys()
            or not set(precisions) <= DTYPES.keys()):
        raise ValueError("choose supported devices, precisions, and a positive gpus")
    if not prediction.strip():
        raise ValueError("state --prediction before measuring")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print(f"prediction: {prediction}", flush=True)
    calls = []
    for name in dict.fromkeys(devices):
        for index in range(gpus):
            run_stamp = f"{stamp}-g{index}"
            call = measure_gemm.with_options(gpu=TARGETS[name]).spawn(
                name, run_stamp, tuple(dict.fromkeys(precisions)), prediction)
            calls.append(call)
            print(f"{name}: function call id {call.object_id}; "
                  f"summary {ROOT / name / 'gemm' / run_stamp / 'summary.json'}",
                  flush=True)
    for call in calls:
        summary = call.get()
        for precision, run in summary["runs"].items():
            print(f"{summary['device']} {summary['gpu']['uuid']} {precision}: "
                  f"MAMF {run['mamf']['tflops']} TFLOPS", flush=True)
