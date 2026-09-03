"""Request-local deterministic random numbers for rollout sampling.

The upstream sampler draws from PyTorch's process-global CUDA generator.  That
makes a resumed rollout depend on how many unrelated random numbers were drawn
before the checkpoint.  This module instead derives each draw from an immutable
request seed, a per-request token counter, a named stream, and a lane index.

The tensor mixer is deliberately small and device agnostic.  It is not intended
for cryptography; it maps uniformly distributed 63-bit request seeds to the
uniform variates needed by categorical and Gumbel sampling without touching any
global RNG state.
"""

from __future__ import annotations

import hashlib
import json
from copy import copy
from typing import Any, Mapping, Sequence

import torch

MAX_SEED = (1 << 63) - 1
_FLOAT_MANTISSA_BITS = 24
_FLOAT_BUCKETS = 1 << _FLOAT_MANTISSA_BITS


def _stable_component(value: Any) -> bytes:
    """Serialize a seed component without relying on Python's salted hash."""

    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError:
        rendered = repr(value)
    return rendered.encode("utf-8")


def derive_seed(root_seed: int, *components: Any) -> int:
    """Derive a stable positive int64 seed from a root and identifiers."""

    root_seed = int(root_seed)
    if not 0 <= root_seed <= MAX_SEED:
        raise ValueError(f"root_seed must be in [0, {MAX_SEED}], got {root_seed}")

    digest = hashlib.blake2b(digest_size=8, person=b"softgrpo")
    digest.update(root_seed.to_bytes(8, byteorder="little", signed=False))
    for component in components:
        payload = _stable_component(component)
        digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
        digest.update(payload)
    return int.from_bytes(digest.digest(), byteorder="little", signed=False) & MAX_SEED


def derive_parallel_seed(request_seed: int, sample_index: int) -> int:
    """Derive the stream seed for one in-engine parallel sample."""

    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    return derive_seed(int(request_seed), "parallel_sample", int(sample_index))


def expand_parallel_sampling_params(
    sampling_params: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    batch_size: int,
    parallel_sample_num: int,
) -> list[dict[str, Any]]:
    """Expand request parameters without dictionary aliasing.

    SGLang's tokenizer manager consumes the first ``batch_size`` entries and
    then creates prompt-major parallel children itself.  Child seeds therefore
    must be derived at that later boundary; this helper only preserves the base
    request seeds while matching the existing sample-major argument expansion.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if parallel_sample_num < 1:
        raise ValueError(
            f"parallel_sample_num must be positive, got {parallel_sample_num}"
        )

    if sampling_params is None:
        bases: list[Mapping[str, Any]] = [{} for _ in range(batch_size)]
    elif isinstance(sampling_params, Mapping):
        bases = [sampling_params for _ in range(batch_size)]
    else:
        bases = list(sampling_params)
        if len(bases) != batch_size:
            raise ValueError(
                "sampling_params list length must match the unexpanded batch "
                f"size ({len(bases)} != {batch_size})"
            )

    expanded: list[dict[str, Any]] = []
    for _sample_index in range(parallel_sample_num):
        for base in bases:
            item = copy(dict(base))
            expanded.append(item)
    return expanded


def _mix63(values: torch.Tensor) -> torch.Tensor:
    """Avalanche signed int64 values while retaining only 63 positive bits."""

    if values.dtype != torch.int64:
        raise TypeError(f"values must be torch.int64, got {values.dtype}")
    values = values & MAX_SEED
    values = (values ^ (values >> 30)) * 6364136223846793005
    values = values & MAX_SEED
    values = (values ^ (values >> 27)) * 1442695040888963407
    values = values & MAX_SEED
    values = values ^ (values >> 31)
    return values & MAX_SEED


def stateless_uniform(
    seeds: torch.Tensor,
    counters: torch.Tensor,
    *,
    width: int,
    stream: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``[batch, width]`` uniforms without consuming global RNG state."""

    if seeds.dtype != torch.int64 or counters.dtype != torch.int64:
        raise TypeError("seeds and counters must use torch.int64")
    if seeds.ndim != 1 or counters.shape != seeds.shape:
        raise ValueError("seeds and counters must be equally shaped rank-one tensors")
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    if stream < 0:
        raise ValueError(f"stream must be nonnegative, got {stream}")

    lane = torch.arange(width, dtype=torch.int64, device=seeds.device).view(1, -1)
    values = seeds.view(-1, 1)
    values = values + counters.view(-1, 1) * 2862933555777941757
    values = values + int(stream) * 3202034522624059733
    values = values + lane * 3935559000370003845
    mixed = _mix63(values)

    # Taking the high 24 bits yields exactly representable float32 buckets.  A
    # half-bucket offset keeps log transforms away from both zero and one.
    buckets = mixed >> (63 - _FLOAT_MANTISSA_BITS)
    return (buckets.to(dtype) + 0.5) / float(_FLOAT_BUCKETS)


def stateless_gumbel(
    seeds: torch.Tensor,
    counters: torch.Tensor,
    *,
    width: int,
    stream: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return standard Gumbel samples from the request-local stream."""

    uniform = stateless_uniform(
        seeds,
        counters,
        width=width,
        stream=stream,
        dtype=dtype,
    )
    return -torch.log(-torch.log(uniform))


def stateless_categorical(
    probs: torch.Tensor,
    seeds: torch.Tensor,
    counters: torch.Tensor,
    *,
    stream: int = 1,
) -> torch.Tensor:
    """Sample one categorical index per row from a request-local stream."""

    if probs.ndim != 2 or probs.shape[0] != seeds.numel():
        raise ValueError("probs must have shape [len(seeds), vocabulary]")
    uniform = stateless_uniform(
        seeds,
        counters,
        width=1,
        stream=stream,
        dtype=torch.float32,
    )
    cumulative = probs.float().cumsum(dim=-1)
    indices = torch.searchsorted(cumulative, uniform, right=False)
    return indices.clamp_max(probs.shape[-1] - 1).to(torch.int64)
