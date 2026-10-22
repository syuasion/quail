"""Modal GPUs and published MAMF shapes, in M, N, K order."""

TARGETS = {
    "h100-sxm": "H100!",
    "rtx-pro-6000-blackwell-server": "RTX-PRO-6000",
}

DTYPES = {"fp8": "float8_e4m3fn", "bf16": "bfloat16"}

# stas00/ml-engineering, compute/accelerator/README.md at 8d028c29.
SHAPES = {
    "h100-sxm": {
        "fp8": (1280, 16896, 16384),
        "bf16": (2816, 1536, 40960),
    },
}
