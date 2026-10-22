"""Print GEMM MAMF fields from one device's validated calibration summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from calibrate.gemm import MAMF_SHA256
from calibrate.targets import DTYPES, TARGETS


def load(paths) -> list[dict]:
    """Read summary files or directories containing summary.json."""
    summaries = []
    for path in map(Path, paths):
        files = sorted(path.rglob("summary.json")) if path.is_dir() else [path]
        summaries.extend(json.loads(file.read_text()) for file in files)
    if not summaries:
        raise ValueError("no calibration summaries found")
    return summaries


def spec_text(summaries: list[dict]) -> str:
    """Return the highest validated MAMF for each precision of one device.

    Raises:
        ValueError: If summaries are empty, invalid, or mix device types.
    """
    devices = {summary.get("device") for summary in summaries}
    if len(devices) != 1 or not devices <= TARGETS.keys():
        raise ValueError("select summaries for exactly one supported device")
    rates = {}
    for summary in summaries:
        if summary.get("sha256") != MAMF_SHA256 or not summary.get("runs"):
            raise ValueError("summary lacks validation from the pinned script")
        for precision, run in summary["runs"].items():
            if precision not in DTYPES or run.get("dtype") != DTYPES[precision]:
                raise ValueError("summary has an unknown or mismatched precision")
            mamf = run.get("mamf") or {}
            rate = mamf.get("tflops", 0)
            if not math.isfinite(rate) or rate <= 0 or not mamf.get("mhz", 0) > 0:
                raise ValueError("summary has a missing or invalid MAMF")
            rates[precision] = max(rates.get(precision, 0), rate * 1e12)
    return "\n".join(f"mamf_{precision}_flops={rates[precision]:.6g},"
                     for precision in DTYPES if precision in rates)


def main(argv=None) -> None:
    """Print device specification fields for the supplied summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(spec_text(load(args.paths)))


if __name__ == "__main__":
    main()
