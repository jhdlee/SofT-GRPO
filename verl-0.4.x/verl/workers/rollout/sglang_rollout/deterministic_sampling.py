"""Stable rollout-seed construction at the VERL/SGLang boundary."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from sglang.srt.sampling.stateless_random import derive_parallel_seed, derive_seed


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def prompt_fingerprint(prompt_token_ids: Sequence[int]) -> str:
    """Hash prompt IDs without architecture-dependent binary packing."""

    payload = json.dumps(
        [int(token_id) for token_id in prompt_token_ids],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def derive_request_seed(
    *,
    root_seed: int,
    rollout_iteration: int,
    example_identity: Any,
    prompt_token_ids: Sequence[int],
    external_sample_index: int = 0,
) -> int:
    """Derive a stable base seed for one prompt request.

    ``external_sample_index`` distinguishes replicas made by VERL before the
    request reaches SGLang (validation does this).  SGLang separately derives
    the in-engine parallel sample index for ``sampling_params.n``.
    """

    if rollout_iteration < 0:
        raise ValueError("rollout_iteration must be nonnegative")
    if external_sample_index < 0:
        raise ValueError("external_sample_index must be nonnegative")
    identity = {
        "example": _python_scalar(example_identity),
        "prompt_sha256": prompt_fingerprint(prompt_token_ids),
        "external_sample_index": int(external_sample_index),
    }
    return derive_seed(int(root_seed), "rollout", int(rollout_iteration), identity)


def build_request_sampling_params(
    base_sampling_params: Mapping[str, Any],
    *,
    root_seed: int,
    rollout_iteration: int,
    example_identities: Sequence[Any],
    prompt_token_ids: Sequence[Sequence[int]],
    external_sample_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Return one independent seeded parameter dictionary per input prompt."""

    count = len(prompt_token_ids)
    if len(example_identities) != count:
        raise ValueError("example_identities and prompt_token_ids must align")
    if external_sample_indices is None:
        external_sample_indices = [0] * count
    if len(external_sample_indices) != count:
        raise ValueError("external_sample_indices and prompt_token_ids must align")

    params: list[dict[str, Any]] = []
    base_seeds: list[int] = []
    for identity, token_ids, sample_index in zip(
        example_identities, prompt_token_ids, external_sample_indices
    ):
        seed = derive_request_seed(
            root_seed=int(root_seed),
            rollout_iteration=int(rollout_iteration),
            example_identity=identity,
            prompt_token_ids=token_ids,
            external_sample_index=int(sample_index),
        )
        item = dict(base_sampling_params)
        item["seed"] = seed
        params.append(item)
        base_seeds.append(seed)
    return params, base_seeds


def expand_parallel_seeds(base_seeds: Sequence[int], parallel_sample_num: int) -> list[int]:
    """Mirror SGLang tokenizer manager's prompt-major output ordering."""

    if parallel_sample_num < 1:
        raise ValueError("parallel_sample_num must be positive")
    if parallel_sample_num == 1:
        return [int(seed) for seed in base_seeds]
    return [
        derive_parallel_seed(int(seed), sample_index)
        for seed in base_seeds
        for sample_index in range(parallel_sample_num)
    ]
