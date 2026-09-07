"""Source-pinned Qwen3/Hopper FA3 forward and backward extension."""

import torch  # Load libtorch/libc10 before the linked native extension.

from . import _C

EXPECTED_BUILD = {
    "sgl_attn_commit": "f89bc2306632d1ec5f97b014dded4254f5b4a907",
    "cutlass_commit": "7127592069c2fe01b041e174ba4345ef9b279671",
    "nvcc_version": "12.6.85",
    "ptxas_version": "12.8.93",
    "architecture": "sm_90a",
    "dtype": "bfloat16",
    "head_dimension": 128,
    "native_backward": True,
}


def validate_build():
    """Reject a different extension instead of silently using another kernel."""
    observed = dict(_C.build_contract())
    if observed != EXPECTED_BUILD:
        raise RuntimeError(f"opd-fa3 build contract differs: {observed}")
    return observed


validate_build()

from .interface import (  # noqa: E402
    flash_attn_varlen_backward,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

__version__ = "0.1.0"
