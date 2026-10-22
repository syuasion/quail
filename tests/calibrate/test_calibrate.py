"""CPU checks for calibration inputs, validation, and saved specifications."""

import json
from copy import deepcopy

import pytest

from calibrate import gemm
from calibrate.targets import DTYPES
from calibrate.to_spec import load, spec_text

MAMF_LOG_TEXT = """\
** Dtype: float8_e4m3fn
  rated max SM clock 1980 MHz
  boost reference: 1980MHz (highest clock seen this run), need ≥1921MHz
MAMF (max achievable,  boost burst): 1629 TFLOPS @ 1280x16896x16384 (MxNxK) 159W 1980MHz
MSMF (max sustainable, saturated): 1429 TFLOPS @ 1536x2816x16384 (MxNxK) 1603MHz
"""


@pytest.mark.parametrize("precision", DTYPES)
def test_only_the_validated_mamf_is_used(precision):
    text = MAMF_LOG_TEXT.replace("float8_e4m3fn", DTYPES[precision])
    assert gemm.parse_log(text, precision) == {
        "tflops": 1629, "shape": "1280x16896x16384", "mhz": 1980}


@pytest.mark.parametrize("text", [
    "",
    MAMF_LOG_TEXT.replace("float8_e4m3fn", "bfloat16"),
    MAMF_LOG_TEXT.replace("1629 TFLOPS", "0 TFLOPS"),
    MAMF_LOG_TEXT.replace("159W 1980MHz", "159W 1500MHz"),
    MAMF_LOG_TEXT.replace("boost reference: 1980MHz", "boost reference: 1500MHz"),
    MAMF_LOG_TEXT.replace("rated max SM clock 1980 MHz", ""),
    MAMF_LOG_TEXT + "WARNING: no MAMF candidate reached the boost clock",
    MAMF_LOG_TEXT + "WARNING: no SM clock readings",
    MAMF_LOG_TEXT + "note: highest SM clock this run 1500MHz is below the rated max",
])
def test_missing_or_unvalidated_measurements_are_rejected(text):
    with pytest.raises(ValueError):
        gemm.parse_log(text, "fp8")


def test_gemm_saves_each_precision_and_preserves_failed_logs(tmp_path, monkeypatch):
    fake = tmp_path / "mamf.py"
    script_text = (
        "import sys\nfrom pathlib import Path\n"
        "dtype = sys.argv[sys.argv.index('--dtype') + 1]\n"
        "out = Path(sys.argv[sys.argv.index('--output_file') + 1])\n"
        f"text = {MAMF_LOG_TEXT!r}\n"
        "out.write_text(text.replace('float8_e4m3fn', dtype))\n")
    fake.write_text(script_text)
    monkeypatch.setattr(gemm, "fetch_mamf_finder", lambda _: fake)
    monkeypatch.setattr(gemm, "gpu_info", lambda: {"name": "NVIDIA H100"})
    committed = {}

    def on_save():
        committed.update({str(path.relative_to(tmp_path)): path.read_text()
                          for path in tmp_path.rglob("*.txt")})
    summary = gemm.run(tmp_path, "h100-sxm", "good", tuple(DTYPES), tmp_path,
                        "both precisions validate", on_save)
    directory = tmp_path / "h100-sxm/gemm/good"
    assert json.loads((directory / "summary.json").read_text()) == summary
    assert set(summary["runs"]) == set(DTYPES)
    assert all((directory / f"{precision}.txt").exists() for precision in DTYPES)
    for precision, expected in [("fp8", "1280"), ("bf16", "2816")]:
        argv = summary["runs"][precision]["argv"]
        assert argv[argv.index("--dtype") + 1] == DTYPES[precision]
        assert argv[argv.index("--m") + 1] == expected

    fake.write_text(script_text.replace(
        "out.write_text", "if dtype == 'bfloat16': text = text.replace('1629', '0')\n"
        "out.write_text"))
    with pytest.raises(ValueError, match="positive"):
        gemm.run(tmp_path, "h100-sxm", "bad", tuple(DTYPES), tmp_path,
                  "invalid measurement fails", on_save)
    assert (tmp_path / "h100-sxm/gemm/bad/bf16.txt").exists()
    assert not (tmp_path / "h100-sxm/gemm/bad/summary.json").exists()
    assert "h100-sxm/gemm/bad/bf16.txt" in committed


def test_a_wrong_script_is_rejected_before_any_measurement(tmp_path):
    (tmp_path / gemm.MAMF_NAME).write_text("print('not mamf-finder')\n")
    with pytest.raises(ValueError, match="sha256"):
        gemm.run(tmp_path, "h100-sxm", "bad", ("fp8",), tmp_path, "", lambda: None)
    assert not (tmp_path / "h100-sxm").exists()


def _summary():
    return {"device": "h100-sxm", "sha256": gemm.MAMF_SHA256, "runs": {
        precision: {"dtype": dtype, "mamf": {
            "tflops": rate, "mhz": 1980, "shape": "1280x16896x16384"}}
        for (precision, dtype), rate in zip(DTYPES.items(), (1600, 850), strict=True)}}


def test_the_spec_uses_the_highest_mamf_per_precision(tmp_path):
    first = _summary()
    second = deepcopy(first)
    second["runs"]["fp8"]["mamf"]["tflops"] = 1629
    second["runs"]["bf16"]["mamf"]["tflops"] = 840
    for index, summary in enumerate((first, second)):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps(summary))
    assert spec_text(load([tmp_path])) == (
        "mamf_fp8_flops=1.629e+15,\nmamf_bf16_flops=8.5e+14,")


def test_the_spec_rejects_mixed_devices_and_invalid_results(tmp_path):
    summary = _summary()
    with pytest.raises(ValueError, match="exactly one"):
        spec_text([summary, {**summary, "device": "rtx-pro-6000-blackwell-server"}])
    with pytest.raises(ValueError, match="no calibration"):
        load([tmp_path])
    for rate in (0, -1, float("nan"), float("inf")):
        invalid = deepcopy(summary)
        invalid["runs"]["bf16"]["mamf"]["tflops"] = rate
        with pytest.raises(ValueError, match="invalid MAMF"):
            spec_text([invalid])
    for invalid in ({**summary, "runs": {}}, {**summary, "sha256": "wrong"}):
        with pytest.raises(ValueError, match="validation"):
            spec_text([invalid])
