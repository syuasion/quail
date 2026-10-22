"""Measure FP8 and BF16 GEMM MAMF with the unchanged upstream mamf-finder."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

from calibrate.targets import DTYPES, SHAPES, TARGETS

MAMF_COMMIT = "30ddd97054eb01035c0a5079a58f011df603b525"
MAMF_URL = ("https://raw.githubusercontent.com/stas00/ml-engineering/"
            f"{MAMF_COMMIT}/compute/accelerator/benchmarks/mamf-finder.py")
MAMF_SHA256 = "24239a3cba82746cabb1cf756afbd4467ed1a1b884eafd842b67cbf948ef7256"
MAMF_NAME = f"mamf-finder-{MAMF_COMMIT[:8]}.py"

HEADLINE = re.compile(
    r"^MAMF \(max achievable,\s+boost burst\):\s+(\d+) TFLOPS @ "
    r"(\d+x\d+x\d+) \(MxNxK\)(?:\s+\d+W)?\s+(\d+)MHz(?: \d+C)?$",
    re.MULTILINE)


def parse_log(text: str, precision: str) -> dict:
    """Return a positive MAMF measured at the GPU's boost clock.

    Raises:
        ValueError: If the dtype, headline, or clock validation is missing or bad.
    """
    if f"** Dtype: {DTYPES[precision]}\n" not in text:
        raise ValueError(f"missing or incorrect dtype for {precision}")
    invalid = ("WARNING: no SM clock readings", "WARNING: no MAMF candidate",
               "telemetry: off", "note: highest SM clock this run")
    if any(message in text for message in invalid):
        raise ValueError("MAMF did not reach a validated boost clock")
    matches = HEADLINE.findall(text)
    rated = re.search(r"rated max SM clock (\d+) MHz", text)
    reference = re.search(r"boost reference: (\d+)MHz", text)
    if len(matches) != 1 or rated is None or reference is None:
        raise ValueError("missing MAMF headline or boost clock validation")
    tflops, shape, mhz = matches[0]
    if (int(tflops) <= 0 or int(rated[1]) <= 0
            or int(reference[1]) < 0.97 * int(rated[1])
            or int(mhz) < 0.97 * int(reference[1])):
        raise ValueError("MAMF must be positive and measured at the boost clock")
    return {"tflops": int(tflops), "shape": shape, "mhz": int(mhz)}


def gpu_info() -> dict:
    """Return nvidia-smi's identity, power limits, clocks, and driver for GPU 0."""
    fields = ("name", "uuid", "power.limit", "power.max_limit", "clocks.max.sm",
              "driver_version")
    row = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}",
         "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True, check=True).stdout.strip()
    return dict(zip(fields, (value.strip() for value in row.split(",")), strict=True))


def fetch_mamf_finder(directory: Path) -> Path:
    """Download the pinned mamf-finder and verify its SHA-256.

    Raises:
        ValueError: If the downloaded or cached file has a different SHA-256.
    """
    path = directory / MAMF_NAME
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MAMF_URL) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MAMF_SHA256:
        raise ValueError(f"{path} has sha256 {digest}, expected {MAMF_SHA256}")
    return path


def run(root: Path, device: str, stamp: str, precisions: tuple[str, ...],
        mamf_directory: Path, prediction: str, on_save: Callable[[], None]) -> dict:
    """Measure each precision on GPU 0 and save validated rates and raw logs."""
    mamf_finder = fetch_mamf_finder(mamf_directory)
    run_dir = root / device / "gemm" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    gpu = gpu_info()
    expected = TARGETS[device].rstrip("!").replace("-", " ")
    if expected not in gpu["name"]:
        raise ValueError(f"expected {device}, got {gpu['name']}")
    summary = {"device": device, "gpu": gpu, "mamf_finder": MAMF_URL,
               "sha256": MAMF_SHA256, "prediction": prediction, "runs": {}}
    for precision in precisions:
        log_path = run_dir / f"{precision}.txt"
        shape = SHAPES.get(device, {}).get(precision)
        shape_args = (["--m", str(shape[0]), "--n", str(shape[1]),
                       "--k", str(shape[2])] if shape else ["--search", "auto"])
        argv = [sys.executable, str(mamf_finder), "--dtype", DTYPES[precision],
                *shape_args, "--output_file", str(log_path), "--no-verbose",
                "--notes", f"quail calibrate {stamp} {device} {precision}"]
        try:
            subprocess.run(argv, check=True)
            mamf = parse_log(log_path.read_text(), precision)
            summary["runs"][precision] = {
                "dtype": DTYPES[precision], "argv": argv[1:],
                "log": str(log_path), "mamf": mamf}
            print(f"{device} {precision}: MAMF {mamf['tflops']} TFLOPS", flush=True)
        finally:
            # Preserve the raw log even when measurement or validation fails.
            on_save()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    on_save()
    return summary
