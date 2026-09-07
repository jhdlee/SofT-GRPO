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
    "host_backward_repair": "scheduler_semaphore_and_explicit_gradients_v1",
    "upstream_flash_api_sha256": "35f2f6f5db472886219c7391a8c4d73ef619db4c8ef97ba23a914463ccf27f15",
    "patched_flash_api_sha256": "e989b32ee79bb31429c45be2900050e8ce45a5985601360fbc25e9cc418a5767",
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
