"""GPU runtime contracts for the native-soft ICL experiment.

This module deliberately delegates sampling to the SGLang fork shipped by the
pinned SofT-GRPO checkout.  It only validates the returned continuous action
metadata and persists the minimal state needed to replay a trajectory later.

The persisted action for a latent step is the released sampler's fixed top-k
support and its *perturbed logits* (``output_topk_gumbel_list``).  The compact
clipped Gumbel draw is retained alongside it solely to compare the generating
actor with HF replay; it is never fed back into generation.  The action
embedding is reconstructed as::

    softmax(perturbed_logits / gumbel_temperature) @ input_embeddings[support]

No full-vocabulary logits are written to disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .icl import (
    ICL_PROTOCOL,
    INFERENCE_MODES,
    request_seed,
    validate_matrix_cell,
)


RUNTIME_SCHEMA_VERSION = 2
RUNTIME_PROTOCOL = ICL_PROTOCOL
UPSTREAM_SOFTGRPO_COMMIT = "8d3c61380b15c3400818da5ce41c62c293a1bfb4"
# The released evaluator gives rendered chat-template text to ``Engine.generate``.
# SGLang's TokenizerManager then calls ``tokenizer.encode(input_text)`` without
# overriding ``add_special_tokens``.  DeepSeek's rendered template already
# contains BOS, so the released path intentionally has two leading BOS IDs.
# Replay must preserve that observable upstream behavior to align query logits.
UPSTREAM_TEXT_PROMPT_TOKENIZATION = "tokenizer.encode-default-specials-v1"
SAMPLER_IMPLEMENTATION_FILES = (
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/layers/sampler.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/layers/logits_processor.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/schedule_batch.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/scheduler_output_processor_mixin.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/io_struct.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/detokenizer_manager.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/multi_tokenizer_mixin.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/managers/tokenizer_manager.py",
    "Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python/sglang/srt/model_executor/model_runner.py",
)
ICL_IMPLEMENTATION_FILES = (
    "opd_tools/constants.py",
    "opd_tools/data.py",
    "opd_tools/graders.py",
    "opd_tools/manifest.py",
    "opd_tools/records.py",
    "opd_tools/icl.py",
    "opd_tools/icl_assets.py",
    "opd_tools/icl_runtime.py",
    "opd_tools/icl_replay.py",
    "opd_tools/icl_eval.py",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            value = stream.read(chunk_bytes)
            if not value:
                break
            digest.update(value)
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    """Seal the upstream pin, fork commit, and exact sampler implementation."""

    repository = Path(__file__).resolve().parents[1]
    try:
        fork_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot resolve the SofT-GRPO fork commit") from error
    if not re.fullmatch(r"[0-9a-f]{40}", fork_commit):
        raise RuntimeError("SofT-GRPO fork commit is not a full SHA")
    expected_values = {
        value
        for value in (
            os.environ.get("OPD_EXPECTED_SUBMODULE_COMMIT"),
            os.environ.get("OPD_SUBMODULE_COMMIT"),
        )
        if value
    }
    if len(expected_values) > 1:
        raise RuntimeError("submodule commit environment variables disagree")
    if expected_values and expected_values != {fork_commit}:
        raise RuntimeError(
            "declared submodule commit differs from the running checkout"
        )

    try:
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not superproject:
            raise RuntimeError("SofT-GRPO must run as the pinned parent submodule")
        parent_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=superproject,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot resolve the parent repository commit") from error
    if not re.fullmatch(r"[0-9a-f]{40}", parent_commit):
        raise RuntimeError("parent repository commit is not a full SHA")
    expected_parent = os.environ.get("OPD_PARENT_COMMIT")
    if expected_parent and expected_parent != parent_commit:
        raise RuntimeError(
            "OPD_PARENT_COMMIT differs from the running parent checkout"
        )

    def inventory_for(paths: Sequence[str]) -> list[dict[str, Any]]:
        result = []
        for relative in paths:
            path = repository / relative
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("implementation file is missing: %s" % relative)
            result.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return result

    sampler_inventory = inventory_for(SAMPLER_IMPLEMENTATION_FILES)
    implementation_inventory = inventory_for(ICL_IMPLEMENTATION_FILES)
    return {
        "upstream_softgrpo_commit": UPSTREAM_SOFTGRPO_COMMIT,
        "parent_commit": parent_commit,
        "fork_commit": fork_commit,
        "sampler_identifier": (
            "released-sglang-top5-gumbel-continuous+noise-observation-v1"
        ),
        "sampler_files": sampler_inventory,
        "sampler_sha256": hashlib.sha256(
            canonical_json_bytes(sampler_inventory)
        ).hexdigest(),
        "implementation_files": implementation_inventory,
        "implementation_sha256": hashlib.sha256(
            canonical_json_bytes(implementation_inventory)
        ).hexdigest(),
    }


def stable_request_seed(
    *, benchmark: str, example_id: str, sample_index: int, study_seed: int = 11
) -> int:
    """Delegate to the CPU contract's sole common-random-number derivation."""

    if study_seed != 11:
        raise ValueError("the registered evaluation uses study seed 11")
    return request_seed(benchmark, example_id, sample_index)


def validate_generation_cell(
    model_label: str,
    mode: str,
    condition: str,
    benchmark: str = "math500",
) -> None:
    validate_matrix_cell(model_label, mode, condition, benchmark)


@dataclass(frozen=True)
class SamplingSettings:
    """The training-matched released SofT-GRPO sampling protocol."""

    top_p: float = 0.95
    top_k: int = 5
    temperature: float = 1.0
    gumbel_temperature: float = 0.1
    max_new_tokens: int = 8192
    think_end: str = "</think>"

    def __post_init__(self) -> None:
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(self.top_k, bool) or int(self.top_k) != self.top_k or self.top_k < 2:
            raise ValueError("top_k must be an integer >= 2")
        for name in ("temperature", "gumbel_temperature"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and positive" % name)
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if self.think_end != "</think>":
            raise ValueError("the released mode switch is locked to </think>")

    def request_params(self, request_seed: int) -> dict[str, Any]:
        if isinstance(request_seed, bool) or not isinstance(request_seed, int) or request_seed < 0:
            raise ValueError("request_seed must be a nonnegative integer")
        return {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "after_thinking_temperature": float(self.temperature),
            "after_thinking_top_p": float(self.top_p),
            "after_thinking_top_k": int(self.top_k),
            "after_thinking_min_p": 0.0,
            "n": 1,
            "max_new_tokens": int(self.max_new_tokens),
            "think_end_str": self.think_end,
            "gumbel_softmax_temperature": float(self.gumbel_temperature),
            "early_stopping_entropy_threshold": 0.0,
            "early_stopping_length_threshold": 256,
            "noise_factor": 1.0,
            "noise_gaussian": False,
            "noise_gumbel": True,
            "noise_on_logits": True,
            "noise_on_inputs": False,
            "seed": request_seed,
        }


@dataclass(frozen=True)
class CompletionRecord:
    """Compact, JSON-serializable completion metadata."""

    model_label: str
    inference_mode: str
    benchmark: str
    condition: str
    example_id: str
    sample_index: int
    request_seed: int
    response: str
    response_token_count: int
    finish_reason: str
    capped: bool
    latent_token_count: int
    hard_token_count: int
    close_tag: bool
    soft_to_hard: bool
    all_soft: bool
    boxed_answer: bool
    boundary_valid: bool
    mixture_entropy_mean: float | None
    top1_weight_mean: float | None
    soft_hard_agreement: float | None
    stored_mixture_reconstruction_abs_error_max: float | None
    replay_row: int
    schema_version: int = RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_generation_cell(
            self.model_label, self.inference_mode, self.condition, self.benchmark
        )
        if self.schema_version != RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported runtime record schema")
        if not self.benchmark or not self.example_id:
            raise ValueError("benchmark and example_id must be nonempty")
        if not 0 <= self.sample_index < 8:
            raise ValueError("sample_index must be in [0, 8)")
        if self.request_seed != stable_request_seed(
            benchmark=self.benchmark,
            example_id=self.example_id,
            sample_index=self.sample_index,
        ):
            raise ValueError("request_seed does not match the common-seed derivation")
        if not isinstance(self.response, str):
            raise TypeError("response must be text")
        if not self.finish_reason:
            raise ValueError("finish_reason must be nonempty")
        counts = (
            self.response_token_count,
            self.latent_token_count,
            self.hard_token_count,
            self.replay_row,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("token counts and replay_row must be nonnegative integers")
        if self.latent_token_count + self.hard_token_count != self.response_token_count:
            raise ValueError("latent and categorical counts must partition the response")
        for name in (
            "capped",
            "close_tag",
            "soft_to_hard",
            "all_soft",
            "boxed_answer",
            "boundary_valid",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("%s must be bool" % name)
        diagnostics = (
            self.mixture_entropy_mean,
            self.top1_weight_mean,
            self.soft_hard_agreement,
            self.stored_mixture_reconstruction_abs_error_max,
        )
        if self.inference_mode == "hard_token":
            if self.latent_token_count or self.soft_to_hard or self.all_soft:
                raise ValueError("hard-token records cannot contain latent state")
            if any(value is not None for value in diagnostics):
                raise ValueError("hard-token records cannot contain mixture diagnostics")
        elif self.latent_token_count:
            if any(value is None or not math.isfinite(float(value)) for value in diagnostics):
                raise ValueError("native-soft mixture diagnostics must be finite")
            if self.mixture_entropy_mean is not None and self.mixture_entropy_mean < 0:
                raise ValueError("mixture entropy must be nonnegative")
            for name in ("top1_weight_mean", "soft_hard_agreement"):
                value = float(getattr(self, name))
                if not 0.0 <= value <= 1.0:
                    raise ValueError("%s must be in [0, 1]" % name)
        elif any(value is not None for value in diagnostics):
            raise ValueError("a trajectory without latent actions has no mixture diagnostics")
        expected_all_soft = self.response_token_count > 0 and self.hard_token_count == 0
        if self.inference_mode == "native_soft" and self.all_soft != expected_all_soft:
            raise ValueError("all_soft disagrees with the token partition")
        if self.boundary_valid and not (
            self.close_tag
            and self.soft_to_hard
            and self.boxed_answer
            and self.latent_token_count > 0
            and self.hard_token_count > 1
        ):
            raise ValueError("boundary_valid lacks a real soft-to-hard boxed response")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompletionRecord":
        if not isinstance(value, Mapping):
            raise TypeError("completion record must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError(
                "completion record fields differ: missing=%s unknown=%s"
                % (sorted(expected - set(value)), sorted(set(value) - expected))
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class TrajectoryMetadata:
    """One trajectory's sufficient replay state."""

    response_token_ids: tuple[int, ...]
    latent_support_ids: tuple[tuple[int, ...], ...]
    latent_perturbed_logits: tuple[tuple[float, ...], ...]
    latent_gumbel_noise: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.response_token_ids, dtype=np.int64)
        supports = np.asarray(self.latent_support_ids, dtype=np.int64)
        logits = np.asarray(self.latent_perturbed_logits, dtype=np.float64)
        noise = np.asarray(self.latent_gumbel_noise, dtype=np.float64)
        if token_ids.ndim != 1 or np.any(token_ids < 0):
            raise ValueError("response token IDs must be a nonnegative vector")
        if len(self.latent_support_ids) == 0:
            if self.latent_perturbed_logits or self.latent_gumbel_noise:
                raise ValueError("latent support/logit/noise row counts differ")
            return
        if supports.ndim != 2 or supports.shape[1] != 5:
            raise ValueError("every latent action must retain exactly five support IDs")
        if logits.shape != supports.shape or not np.isfinite(logits).all():
            raise ValueError("latent perturbed logits must align and be finite")
        if noise.shape != supports.shape or not np.isfinite(noise).all():
            raise ValueError("latent Gumbel noise must align and be finite")
        if np.any(supports < 0):
            raise ValueError("latent support IDs must be nonnegative")
        if supports.shape[0] > token_ids.shape[0]:
            raise ValueError("latent actions cannot outnumber response tokens")
        if any(len(set(row)) != 5 for row in supports.tolist()):
            raise ValueError("a latent top-five support must contain unique IDs")
        if not np.array_equal(token_ids[: supports.shape[0]], supports[:, 0]):
            raise ValueError("released hard shadows must equal support argmax IDs")


def _finish_reason(value: Any) -> str:
    if isinstance(value, Mapping):
        result = value.get("type", value.get("matched", value.get("reason")))
    else:
        result = getattr(value, "type", value)
    if result is None:
        raise ValueError("SGLang did not return a finish reason")
    return str(result)


def _output_token_ids(meta: Mapping[str, Any]) -> list[int]:
    values = meta.get("output_token_logprobs")
    if not isinstance(values, list):
        raise ValueError("SGLang must return output-token log probabilities")
    result = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("malformed SGLang output-token logprob entry")
        result.append(int(value[1]))
    return result


def _row_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values.astype(np.float64) / float(temperature)
    scaled -= scaled.max(axis=-1, keepdims=True)
    numerator = np.exp(scaled)
    return numerator / numerator.sum(axis=-1, keepdims=True)


def _balanced_nonempty_final_box_span(response: str) -> tuple[int, int] | None:
    matches = list(re.finditer(r"\\boxed\s*\{", response))
    if not matches:
        return None
    match = matches[-1]
    opening = response.find("{", match.start(), match.end())
    if opening < 0:
        return None
    depth = 0
    for cursor in range(opening, len(response)):
        character = response[cursor]
        if character == "{" and (cursor == 0 or response[cursor - 1] != "\\"):
            depth += 1
        elif character == "}" and (cursor == 0 or response[cursor - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return (
                    (match.start(), cursor + 1)
                    if response[opening + 1 : cursor].strip()
                    else None
                )
            if depth < 0:
                return None
    return None


def parse_sglang_completion(
    *,
    output: Mapping[str, Any],
    model_label: str,
    mode: str,
    benchmark: str,
    condition: str,
    example_id: str,
    sample_index: int,
    replay_row: int,
    settings: SamplingSettings,
    think_end_id: int,
    probability_tolerance: float = 5e-3,
) -> tuple[CompletionRecord, TrajectoryMetadata]:
    """Validate one exact upstream result and retain only sufficient replay state."""

    validate_generation_cell(model_label, mode, condition, benchmark)
    if not isinstance(output, Mapping):
        raise TypeError("SGLang output must be a mapping")
    meta = output.get("meta_info")
    if not isinstance(meta, Mapping):
        raise ValueError("SGLang output has no meta_info mapping")
    response = output.get("text")
    if not isinstance(response, str):
        raise ValueError("SGLang response must be text")
    token_ids = _output_token_ids(meta)
    finish_reason = _finish_reason(meta.get("finish_reason"))
    capped = finish_reason.lower() in {"length", "max_tokens", "max_new_tokens"}
    final_box_span = _balanced_nonempty_final_box_span(response)
    boxed = final_box_span is not None
    request_seed = stable_request_seed(
        benchmark=benchmark, example_id=example_id, sample_index=sample_index
    )

    if mode == "hard_token":
        trajectory = TrajectoryMetadata(tuple(token_ids), (), (), ())
        record = CompletionRecord(
            model_label=model_label,
            inference_mode=mode,
            benchmark=benchmark,
            condition=condition,
            example_id=example_id,
            sample_index=sample_index,
            request_seed=request_seed,
            response=response,
            response_token_count=len(token_ids),
            finish_reason=finish_reason,
            capped=capped,
            latent_token_count=0,
            hard_token_count=len(token_ids),
            close_tag="</think>" in response,
            soft_to_hard=False,
            all_soft=False,
            boxed_answer=boxed,
            boundary_valid=False,
            mixture_entropy_mean=None,
            top1_weight_mean=None,
            soft_hard_agreement=None,
            stored_mixture_reconstruction_abs_error_max=None,
            replay_row=replay_row,
        )
        return record, trajectory

    raw_supports = meta.get("output_topk_idx_list")
    raw_logits = meta.get("output_topk_gumbel_list")
    raw_noise = meta.get("output_topk_gumbel_noise_list")
    raw_probabilities = meta.get("output_topk_prob_list")
    supports = np.asarray(raw_supports, dtype=np.int64)
    perturbed = np.asarray(raw_logits, dtype=np.float64)
    gumbel_noise = np.asarray(raw_noise, dtype=np.float64)
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if (
        supports.ndim != 2
        or supports.shape != (len(token_ids), settings.top_k)
        or perturbed.shape != supports.shape
        or gumbel_noise.shape != supports.shape
        or probabilities.shape != supports.shape
    ):
        raise ValueError("released continuous metadata shape does not match output")
    if (
        not np.isfinite(perturbed).all()
        or not np.isfinite(gumbel_noise).all()
        or not np.isfinite(probabilities).all()
    ):
        raise FloatingPointError("released continuous metadata contains NaN/Inf")

    categorical = np.all(supports[:, 1:] == 0, axis=-1)
    latent = ~categorical
    if categorical.any():
        categorical_supports = supports[categorical]
        categorical_probabilities = probabilities[categorical]
        categorical_tokens = np.asarray(token_ids, dtype=np.int64)[categorical]
        if not np.array_equal(categorical_supports[:, 0], categorical_tokens):
            raise RuntimeError(
                "categorical metadata head does not equal its emitted token"
            )
        expected_one_hot = np.zeros_like(categorical_probabilities)
        expected_one_hot[:, 0] = 1.0
        categorical_error = float(
            np.max(np.abs(categorical_probabilities - expected_one_hot))
        )
        categorical_tolerance = min(probability_tolerance, 1e-6)
        if categorical_error > categorical_tolerance:
            raise RuntimeError(
                "categorical metadata probabilities are not one-hot: "
                "max error %.6g > %.6g"
                % (categorical_error, categorical_tolerance)
            )
    if np.any(np.diff(latent.astype(np.int8)) > 0):
        raise RuntimeError("released trajectory re-entered soft mode after </think>")
    latent_count = int(latent.sum())
    if latent_count and not bool(np.all(latent[:latent_count])):
        raise RuntimeError("latent actions are not one contiguous prefix")
    latent_supports = supports[:latent_count]
    latent_perturbed = perturbed[:latent_count]
    latent_noise = gumbel_noise[:latent_count]
    if latent_count:
        if any(len(set(row)) != settings.top_k for row in latent_supports.tolist()):
            raise RuntimeError("released top-five support contains repeated token IDs")
        reconstructed = _row_softmax(
            latent_perturbed, settings.gumbel_temperature
        )
        probability_error = float(
            np.max(np.abs(reconstructed - probabilities[:latent_count]))
        )
        if probability_error > probability_tolerance:
            raise RuntimeError(
                "stored perturbed logits do not reconstruct the released mixture: "
                "max error %.6g > %.6g" % (probability_error, probability_tolerance)
            )
        entropy = -(reconstructed * np.log(np.maximum(reconstructed, 1e-300))).sum(
            axis=-1
        )
        top1 = reconstructed.max(axis=-1)
        hard_shadows = latent_supports[:, 0]
        agreement = hard_shadows == np.asarray(token_ids[:latent_count])
        if not bool(agreement.all()):
            raise RuntimeError("SGLang output IDs disagree with released soft argmaxes")
        entropy_mean: float | None = float(entropy.mean())
        top1_mean: float | None = float(top1.mean())
        agreement_mean: float | None = float(agreement.mean())
        error_value: float | None = probability_error
    else:
        entropy_mean = top1_mean = agreement_mean = error_value = None

    hard_count = len(token_ids) - latent_count
    close_tag_start = response.find("</think>")
    close_tag = close_tag_start >= 0
    transitioned = latent_count > 0 and hard_count > 0
    delimiter_at_boundary = transitioned and token_ids[latent_count] == int(think_end_id)
    categorical_box = bool(
        final_box_span is not None
        and close_tag
        and final_box_span[0] >= close_tag_start + len("</think>")
    )
    boundary_valid = bool(
        delimiter_at_boundary
        and categorical_box
        and hard_count > 1
        and not capped
    )
    trajectory = TrajectoryMetadata(
        tuple(token_ids),
        tuple(tuple(int(value) for value in row) for row in latent_supports.tolist()),
        tuple(tuple(float(value) for value in row) for row in latent_perturbed.tolist()),
        tuple(tuple(float(value) for value in row) for row in latent_noise.tolist()),
    )
    record = CompletionRecord(
        model_label=model_label,
        inference_mode=mode,
        benchmark=benchmark,
        condition=condition,
        example_id=example_id,
        sample_index=sample_index,
        request_seed=request_seed,
        response=response,
        response_token_count=len(token_ids),
        finish_reason=finish_reason,
        capped=capped,
        latent_token_count=latent_count,
        hard_token_count=hard_count,
        close_tag=close_tag,
        soft_to_hard=transitioned,
        all_soft=bool(len(token_ids) > 0 and hard_count == 0),
        boxed_answer=boxed,
        boundary_valid=boundary_valid,
        mixture_entropy_mean=entropy_mean,
        top1_weight_mean=top1_mean,
        soft_hard_agreement=agreement_mean,
        stored_mixture_reconstruction_abs_error_max=error_value,
        replay_row=replay_row,
    )
    return record, trajectory


def boundary_gate(records: Sequence[CompletionRecord], *, max_failure_rate: float = 0.05) -> dict[str, Any]:
    """Apply the preregistered capped/all-soft native-soft cell gate.

    A correctly transitioned but malformed answer is scored wrong; it does not
    count against the distinct 5% trajectory-mechanism threshold.  GPU smoke
    separately requires at least one fully valid categorical boxed response.
    """

    if not records:
        raise ValueError("boundary gate requires at least one record")
    if any(record.inference_mode != "native_soft" for record in records):
        raise ValueError("boundary gate accepts native-soft records only")
    if not 0.0 <= max_failure_rate < 1.0:
        raise ValueError("max_failure_rate must be in [0, 1)")
    failures = sum(record.capped or record.all_soft for record in records)
    failure_rate = failures / len(records)
    return {
        "valid": failure_rate <= max_failure_rate,
        "failure_count": failures,
        "record_count": len(records),
        "failure_rate": failure_rate,
        "max_failure_rate": max_failure_rate,
        "demonstrated_boundary_count": sum(record.boundary_valid for record in records),
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".%s." % path.name, suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".%s." % path.name, suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def pack_trajectories(trajectories: Sequence[TrajectoryMetadata]) -> dict[str, np.ndarray]:
    response_offsets = [0]
    latent_offsets = [0]
    response_ids: list[int] = []
    supports: list[tuple[int, ...]] = []
    perturbed: list[tuple[float, ...]] = []
    noise: list[tuple[float, ...]] = []
    for trajectory in trajectories:
        response_ids.extend(trajectory.response_token_ids)
        supports.extend(trajectory.latent_support_ids)
        perturbed.extend(trajectory.latent_perturbed_logits)
        noise.extend(trajectory.latent_gumbel_noise)
        response_offsets.append(len(response_ids))
        latent_offsets.append(len(supports))
    return {
        "response_offsets": np.asarray(response_offsets, dtype=np.int64),
        "latent_offsets": np.asarray(latent_offsets, dtype=np.int64),
        "response_token_ids": np.asarray(response_ids, dtype=np.int32),
        "latent_support_ids": np.asarray(supports, dtype=np.int32).reshape(-1, 5),
        "latent_perturbed_logits": np.asarray(perturbed, dtype=np.float32).reshape(-1, 5),
        "latent_gumbel_noise": np.asarray(noise, dtype=np.float32).reshape(-1, 5),
    }


def unpack_trajectory(arrays: Mapping[str, np.ndarray], row: int) -> TrajectoryMetadata:
    response_offsets = arrays["response_offsets"]
    latent_offsets = arrays["latent_offsets"]
    if row < 0 or row + 1 >= len(response_offsets) or len(response_offsets) != len(latent_offsets):
        raise IndexError("replay row is outside the packed trajectory inventory")
    response = arrays["response_token_ids"][response_offsets[row] : response_offsets[row + 1]]
    supports = arrays["latent_support_ids"][latent_offsets[row] : latent_offsets[row + 1]]
    perturbed = arrays["latent_perturbed_logits"][latent_offsets[row] : latent_offsets[row + 1]]
    noise = arrays["latent_gumbel_noise"][latent_offsets[row] : latent_offsets[row + 1]]
    return TrajectoryMetadata(
        tuple(int(value) for value in response.tolist()),
        tuple(tuple(int(value) for value in values) for values in supports.tolist()),
        tuple(tuple(float(value) for value in values) for values in perturbed.tolist()),
        tuple(tuple(float(value) for value in values) for values in noise.tolist()),
    )


class AtomicChunkStore:
    """Commit and authenticate one resumable generation chunk.

    The manifest is written last and is the only completion marker.  Existing
    chunks are never overwritten: resume verifies and skips them.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()

    def paths(self, chunk_key: str) -> tuple[Path, Path, Path]:
        if not chunk_key or any(part in {"", ".", ".."} for part in Path(chunk_key).parts):
            raise ValueError("chunk key must be a nonempty relative path")
        relative = Path(chunk_key)
        if relative.is_absolute():
            raise ValueError("chunk key must be relative")
        base = self.root / relative
        return (
            base.with_suffix(".jsonl"),
            base.with_suffix(".replay.npz"),
            base.with_suffix(".manifest.json"),
        )

    def resume_state(
        self,
        chunk_key: str,
        *,
        expected_identity: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Verify a commit or remove only its uncommitted exact-path remnants."""

        records_path, replay_path, manifest_path = self.paths(chunk_key)
        if manifest_path.exists():
            return self.verify(chunk_key, expected_identity=expected_identity)
        for path in (records_path, replay_path):
            if path.exists():
                if not path.is_file() or path.is_symlink():
                    raise RuntimeError("uncommitted chunk path is not a regular file")
                path.unlink()
            if path.parent.exists():
                prefix = ".%s." % path.name
                for temporary in path.parent.iterdir():
                    if temporary.name.startswith(prefix) and temporary.name.endswith(".tmp"):
                        if not temporary.is_file() or temporary.is_symlink():
                            raise RuntimeError("unexpected atomic temporary path")
                        temporary.unlink()
        return None

    def verify(self, chunk_key: str, *, expected_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        records_path, replay_path, manifest_path = self.paths(chunk_key)
        existence = [path.is_file() for path in (records_path, replay_path, manifest_path)]
        if not all(existence):
            if any(existence):
                raise RuntimeError("generation chunk is only partially committed")
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise RuntimeError("generation chunk schema changed")
        if expected_identity is not None and manifest.get("identity") != dict(expected_identity):
            raise RuntimeError("committed chunk identity differs from this invocation")
        for label, path in (("records", records_path), ("replay", replay_path)):
            entry = manifest.get("files", {}).get(label, {})
            if entry.get("size") != path.stat().st_size or entry.get("sha256") != sha256_file(path):
                raise RuntimeError("generation chunk authentication failed: %s" % path)
        records = []
        with records_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                records.append(CompletionRecord.from_mapping(json.loads(line)))
        if len(records) != manifest.get("row_count"):
            raise RuntimeError("generation chunk record count changed")
        with np.load(replay_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if set(arrays) != {
            "response_offsets",
            "latent_offsets",
            "response_token_ids",
            "latent_support_ids",
            "latent_perturbed_logits",
            "latent_gumbel_noise",
        }:
            raise RuntimeError("packed replay field inventory changed")
        if len(arrays["response_offsets"]) != len(records) + 1:
            raise RuntimeError("packed replay row count changed")
        for index, record in enumerate(records):
            if record.replay_row != index:
                raise RuntimeError("JSON record does not align with its replay row")
            trajectory = unpack_trajectory(arrays, index)
            if len(trajectory.response_token_ids) != record.response_token_count:
                raise RuntimeError("response token count differs from replay metadata")
            if len(trajectory.latent_support_ids) != record.latent_token_count:
                raise RuntimeError("latent token count differs from replay metadata")
        return manifest

    def commit(
        self,
        chunk_key: str,
        records: Sequence[CompletionRecord],
        trajectories: Sequence[TrajectoryMetadata],
        *,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not records or len(records) != len(trajectories):
            raise ValueError("a chunk needs aligned, nonempty records and trajectories")
        if [record.replay_row for record in records] != list(range(len(records))):
            raise ValueError("replay_row must be the zero-based row within its chunk")
        records_path, replay_path, manifest_path = self.paths(chunk_key)
        if any(path.exists() for path in (records_path, replay_path, manifest_path)):
            return self.verify(chunk_key, expected_identity=identity)
        _atomic_bytes(
            records_path,
            b"".join(canonical_json_bytes(record.to_dict()) for record in records),
        )
        _atomic_npz(replay_path, pack_trajectories(trajectories))
        manifest = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "protocol": RUNTIME_PROTOCOL,
            "identity": dict(identity),
            "row_count": len(records),
            "files": {
                "records": {
                    "size": records_path.stat().st_size,
                    "sha256": sha256_file(records_path),
                },
                "replay": {
                    "size": replay_path.stat().st_size,
                    "sha256": sha256_file(replay_path),
                },
            },
        }
        _atomic_bytes(manifest_path, canonical_json_bytes(manifest))
        return self.verify(chunk_key, expected_identity=identity)

    def load(self, chunk_key: str) -> tuple[list[CompletionRecord], dict[str, np.ndarray]]:
        self.verify(chunk_key)
        records_path, replay_path, _ = self.paths(chunk_key)
        with records_path.open("r", encoding="utf-8") as stream:
            records = [CompletionRecord.from_mapping(json.loads(line)) for line in stream]
        with np.load(replay_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return records, arrays


def validate_atomic_reasoning_tokens(tokenizer: Any) -> tuple[int, int]:
    result = []
    for tag in ("<think>", "</think>"):
        ids = tokenizer.encode(tag, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError("%s must be one atomic tokenizer ID; got %r" % (tag, ids))
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != tag:
            raise RuntimeError("tokenizer does not round-trip %s atomically" % tag)
        result.append(int(ids[0]))
    if result[0] == result[1]:
        raise RuntimeError("<think> and </think> share a token ID")
    return result[0], result[1]


def required_context_length(
    tokenizer: Any, rendered_prompts: Sequence[str], settings: SamplingSettings
) -> int:
    if not rendered_prompts:
        raise ValueError("at least one rendered prompt is required")
    maximum = max(
        # Match TokenizerManager._tokenize_one_request in the bundled SGLang
        # path, which calls tokenizer.encode(text) with tokenizer defaults.
        len(tokenizer.encode(prompt))
        for prompt in rendered_prompts
    )
    return maximum + settings.max_new_tokens


class ReleasedSofTGRPOEngine:
    """Thin owner for the exact SGLang sampler bundled by upstream."""

    def __init__(
        self,
        *,
        model_path: str,
        mode: str,
        num_gpus: int,
        context_length: int,
        settings: SamplingSettings,
        max_running_requests: int = 32,
        gpu_memory_utilization: float = 0.8,
    ) -> None:
        if mode not in INFERENCE_MODES:
            raise ValueError("unsupported inference mode")
        if num_gpus <= 0 or context_length <= settings.max_new_tokens:
            raise ValueError("GPU count/context length is invalid")
        if not 0.0 < gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        try:
            import sglang as sgl
        except ImportError as error:
            raise RuntimeError(
                "native-soft generation requires the pinned editable SGLang fork"
            ) from error
        repository = Path(__file__).resolve().parents[1]
        expected_package = (
            repository
            / "Soft-Thinking+noise+loss-main"
            / "sglang_soft_thinking_pkg"
            / "python"
            / "sglang"
        ).resolve()
        observed_package = Path(sgl.__file__).resolve()
        if not observed_package.is_relative_to(expected_package):
            raise RuntimeError(
                "imported sglang is not the bundled SofT-GRPO sampler: %s"
                % observed_package
            )
        native_soft = mode == "native_soft"
        self.mode = mode
        self.settings = settings
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=num_gpus,
            trust_remote_code=True,
            random_seed=11,
            context_length=context_length,
            max_running_requests=max_running_requests,
            mem_fraction_static=gpu_memory_utilization,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_soft_thinking=native_soft,
            add_noise_gumbel_softmax=native_soft,
            max_topk=settings.top_k,
            sampling_backend="flashinfer",
        )

    def generate(self, prompts: Sequence[str], seeds: Sequence[int]) -> list[Mapping[str, Any]]:
        if not prompts or len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must be aligned and nonempty")
        params = [self.settings.request_params(int(seed)) for seed in seeds]
        outputs = self.engine.generate(
            list(prompts), sampling_params=params, return_logprob=True
        )
        if isinstance(outputs, Mapping):
            outputs = [outputs]
        if len(outputs) != len(prompts):
            raise RuntimeError("SGLang changed the request/output cardinality")
        return list(outputs)

    def shutdown(self) -> None:
        self.engine.shutdown()


def generation_chunk_metrics(
    records: Sequence[CompletionRecord], *, elapsed_seconds: float
) -> dict[str, float | int]:
    if not records or elapsed_seconds <= 0:
        raise ValueError("metrics require records and positive elapsed time")
    tokens = sum(record.response_token_count for record in records)
    native = records[0].inference_mode == "native_soft"
    if any((record.inference_mode == "native_soft") != native for record in records):
        raise ValueError("a generation metric chunk may not mix inference modes")
    result: dict[str, float | int] = {
        "generation/records_committed": len(records),
        "generation/response_tokens": tokens,
        "generation/tokens_per_second": tokens / elapsed_seconds,
        "generation/cap_rate": sum(record.capped for record in records) / len(records),
        "generation/boxed_answer_rate": sum(record.boxed_answer for record in records) / len(records),
    }
    if native:
        gate = boundary_gate(records)
        latent_slots = sum(record.latent_token_count for record in records)
        result.update(
            {
                "generation/capped_or_all_soft_rate": float(gate["failure_rate"]),
                "generation/boundary_valid_rate": sum(
                    record.boundary_valid for record in records
                )
                / len(records),
                "generation/close_tag_rate": sum(record.close_tag for record in records) / len(records),
                "generation/soft_to_hard_rate": sum(record.soft_to_hard for record in records) / len(records),
                "generation/all_soft_rate": sum(record.all_soft for record in records) / len(records),
                "generation/latent_length_mean": sum(record.latent_token_count for record in records) / len(records),
                "generation/hard_answer_length_mean": sum(
                    max(
                        record.hard_token_count
                        - int(record.close_tag and record.soft_to_hard),
                        0,
                    )
                    for record in records
                )
                / len(records),
                "generation/mixture_entropy_slot_mean": (
                    0.0
                    if not latent_slots
                    else sum(
                        (record.mixture_entropy_mean or 0.0) * record.latent_token_count
                        for record in records
                    )
                    / latent_slots
                ),
                "generation/top1_weight_slot_mean": (
                    0.0
                    if not latent_slots
                    else sum(
                        (record.top1_weight_mean or 0.0) * record.latent_token_count
                        for record in records
                    )
                    / latent_slots
                ),
                "generation/soft_hard_agreement_slot_mean": (
                    0.0
                    if not latent_slots
                    else sum(
                        (record.soft_hard_agreement or 0.0) * record.latent_token_count
                        for record in records
                    )
                    / latent_slots
                ),
                "generation/stored_mixture_reconstruction_abs_error_max": max(
                    (
                        record.stored_mixture_reconstruction_abs_error_max
                        or 0.0
                    )
                    for record in records
                ),
            }
        )
    return result


def init_online_wandb(*, run_id: str, config: Mapping[str, Any], job_type: str):
    """Initialize the mandatory online run without importing W&B on CPU tests."""

    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("ICL evaluation requires WANDB_MODE=online")
    if os.environ.get("WANDB_RESUME", "allow") != "allow":
        raise RuntimeError("ICL evaluation requires WANDB_RESUME=allow")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("ICL evaluation requires wandb") from error
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "opd-softgrpo-icl"),
        group=os.environ.get("WANDB_GROUP", "native-soft-icl-seed11"),
        id=run_id,
        resume="allow",
        job_type=job_type,
        config=dict(config),
    )


def stable_wandb_run_id(config: Mapping[str, Any], *, prefix: str) -> str:
    digest = hashlib.sha256(canonical_json_bytes(dict(config))).hexdigest()[:20]
    return "%s-%s" % (prefix, digest)
