"""Pure-CPU calibration design for choosing Qwen3 ICL data parallelism.

The calibration workload is shared verbatim between DP1 and DP2.  Each
model/mode observes 20 representative MATH-500 requests and two AIME 2024
requests under each of the three registered prompt conditions, for 66 requests
in total.  Runtime inference bootstraps the six ``(condition, benchmark)``
cells and expands them with the exact production counts before adding one
engine load and finalization for each of the two inference modes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .icl import (
    CORE_CONDITIONS,
    QWEN3_STUDY_ID,
    request_seed,
)


THROUGHPUT_PROTOCOL = "opd-qwen3-icl-throughput-calibration-v1"
SELECTION_SEED = 42
BOOTSTRAP_SEED = 11
BOOTSTRAP_RESAMPLES = 10_000
MATH_CALIBRATION_COUNT = 20
AIME_CALIBRATION_COUNT = 2
REQUESTS_PER_MODEL_MODE = 66
RUNTIME_LIMIT_HOURS = 18.0
RUNTIME_LIMIT_SECONDS = int(RUNTIME_LIMIT_HOURS * 3600)
LENGTH_QUANTILES = 4
INFERENCE_MODES = ("native_soft", "hard_token")
PRODUCTION_COUNTS = MappingProxyType(
    {
        condition: MappingProxyType({"math500": 500, "aime2024": 30})
        for condition in CORE_CONDITIONS
    }
)
PRODUCTION_REQUESTS_PER_MODE = sum(
    count for values in PRODUCTION_COUNTS.values() for count in values.values()
)
PRODUCTION_REQUESTS_PER_MODEL = len(INFERENCE_MODES) * PRODUCTION_REQUESTS_PER_MODE


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_rank(namespace: str, *parts: Any) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            [THROUGHPUT_PROTOCOL, SELECTION_SEED, namespace, *parts]
        )
    ).hexdigest()


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _freeze_prompt_tokens(
    value: Mapping[str, int],
) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(CORE_CONDITIONS):
        raise ValueError("prompt token counts must cover the three core conditions")
    return MappingProxyType(
        {
            condition: _require_positive_integer(
                value[condition], "prompt token count"
            )
            for condition in CORE_CONDITIONS
        }
    )


@dataclass(frozen=True)
class ThroughputExample:
    """One benchmark example plus rendered prompt lengths for all conditions."""

    example_id: str
    benchmark: str
    subject: str
    level: str
    prompt_tokens: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be nonempty")
        if self.benchmark not in ("math500", "aime2024"):
            raise ValueError("throughput examples must be MATH-500 or AIME 2024")
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("subject must be nonempty")
        if not isinstance(self.level, str) or not self.level.strip():
            raise ValueError("level must be nonempty")
        object.__setattr__(self, "prompt_tokens", _freeze_prompt_tokens(self.prompt_tokens))

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "benchmark": self.benchmark,
            "subject": self.subject,
            "level": self.level,
            "prompt_tokens": {
                condition: self.prompt_tokens[condition]
                for condition in CORE_CONDITIONS
            },
        }


@dataclass(frozen=True)
class ThroughputRequest:
    """One immutable request reused at every candidate DP size."""

    condition: str
    benchmark: str
    example_id: str
    subject: str
    level: str
    prompt_tokens: int
    request_seed: int
    length_quantile: int

    def __post_init__(self) -> None:
        if self.condition not in CORE_CONDITIONS:
            raise ValueError("throughput request condition is not registered")
        if self.benchmark not in ("math500", "aime2024"):
            raise ValueError("throughput request benchmark is not registered")
        for name in ("example_id", "subject", "level"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("throughput request %s must be nonempty" % name)
        _require_positive_integer(self.prompt_tokens, "prompt_tokens")
        if (
            isinstance(self.request_seed, bool)
            or not isinstance(self.request_seed, int)
            or self.request_seed < 0
        ):
            raise ValueError("request_seed must be a nonnegative integer")
        if (
            isinstance(self.length_quantile, bool)
            or not isinstance(self.length_quantile, int)
            or not 0 <= self.length_quantile < LENGTH_QUANTILES
        ):
            raise ValueError("length_quantile is outside the registered range")

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.condition, self.benchmark, self.example_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "benchmark": self.benchmark,
            "example_id": self.example_id,
            "subject": self.subject,
            "level": self.level,
            "prompt_tokens": self.prompt_tokens,
            "request_seed": self.request_seed,
            "length_quantile": self.length_quantile,
        }


@dataclass(frozen=True)
class ThroughputPlan:
    """Authenticated 66-request calibration workload for one model/mode."""

    requests: Tuple[ThroughputRequest, ...]
    protocol: str = THROUGHPUT_PROTOCOL
    selection_seed: int = SELECTION_SEED

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        if self.protocol != THROUGHPUT_PROTOCOL or self.selection_seed != SELECTION_SEED:
            raise ValueError("throughput plan protocol or selection seed changed")
        if len(self.requests) != REQUESTS_PER_MODEL_MODE:
            raise ValueError("throughput plan must contain exactly 66 requests")
        for selected in self.requests:
            if not isinstance(selected, ThroughputRequest):
                raise TypeError("throughput plan rows must be ThroughputRequest values")
            expected_seed = request_seed(
                selected.benchmark,
                selected.example_id,
                0,
                study=QWEN3_STUDY_ID,
            )
            if selected.request_seed != expected_seed:
                raise ValueError("throughput plan request seed changed")
        if len({request.key for request in self.requests}) != len(self.requests):
            raise ValueError("throughput plan request identities must be unique")
        for condition in CORE_CONDITIONS:
            counts = {
                benchmark: sum(
                    request.condition == condition
                    and request.benchmark == benchmark
                    for request in self.requests
                )
                for benchmark in ("math500", "aime2024")
            }
            if counts != {
                "math500": MATH_CALIBRATION_COUNT,
                "aime2024": AIME_CALIBRATION_COUNT,
            }:
                raise ValueError("each condition must contain 20 MATH and two AIME requests")

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "selection_seed": self.selection_seed,
            "study_id": QWEN3_STUDY_ID,
            "requests_per_model_mode": REQUESTS_PER_MODEL_MODE,
            "production_requests_per_mode": PRODUCTION_REQUESTS_PER_MODE,
            "production_requests_per_model": PRODUCTION_REQUESTS_PER_MODEL,
            "production_counts": {
                condition: dict(PRODUCTION_COUNTS[condition])
                for condition in CORE_CONDITIONS
            },
            "requests": [request.to_dict() for request in self.requests],
        }


def _length_quantiles(
    examples: Sequence[ThroughputExample], condition: str
) -> dict[str, int]:
    ordered = sorted(
        examples,
        key=lambda example: (
            example.prompt_tokens[condition],
            _stable_rank("length-tie", condition, example.example_id),
        ),
    )
    size = len(ordered)
    return {
        example.example_id: min(LENGTH_QUANTILES - 1, rank * LENGTH_QUANTILES // size)
        for rank, example in enumerate(ordered)
    }


def _proportional_allocations(
    groups: Mapping[Tuple[str, str, int], Sequence[ThroughputExample]],
    target: int,
    condition: str,
) -> dict[Tuple[str, str, int], int]:
    total = sum(len(values) for values in groups.values())
    allocations = {
        key: math.floor(target * len(values) / total)
        for key, values in groups.items()
    }
    remaining = target - sum(allocations.values())
    order = sorted(
        groups,
        key=lambda key: (
            -(target * len(groups[key]) / total - allocations[key]),
            _stable_rank("allocation-tie", condition, *key),
        ),
    )
    for key in order:
        if remaining == 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("could not allocate the complete stratified sample")
    return allocations


def _select_math(
    examples: Sequence[ThroughputExample], condition: str
) -> Tuple[ThroughputRequest, ...]:
    quantiles = _length_quantiles(examples, condition)
    groups: dict[Tuple[str, str, int], list[ThroughputExample]] = {}
    for example in examples:
        key = (example.subject, example.level, quantiles[example.example_id])
        groups.setdefault(key, []).append(example)
    allocations = _proportional_allocations(
        groups, MATH_CALIBRATION_COUNT, condition
    )
    selected = []
    for stratum in sorted(groups):
        ordered = sorted(
            groups[stratum],
            key=lambda example: _stable_rank(
                "within-stratum", condition, *stratum, example.example_id
            ),
        )
        selected.extend(
            (example, stratum[2])
            for example in ordered[: allocations[stratum]]
        )
    selected.sort(
        key=lambda value: (
            value[0].subject,
            value[0].level,
            value[1],
            _stable_rank("selected-order", condition, value[0].example_id),
        )
    )
    return tuple(
        ThroughputRequest(
            condition=condition,
            benchmark="math500",
            example_id=example.example_id,
            subject=example.subject,
            level=example.level,
            prompt_tokens=example.prompt_tokens[condition],
            request_seed=request_seed(
                "math500", example.example_id, 0, study=QWEN3_STUDY_ID
            ),
            length_quantile=quantile,
        )
        for example, quantile in selected
    )


def _select_aime(
    examples: Sequence[ThroughputExample], condition: str
) -> Tuple[ThroughputRequest, ...]:
    ordered = sorted(
        examples,
        key=lambda example: (
            example.prompt_tokens[condition],
            _stable_rank("aime-length-tie", condition, example.example_id),
        ),
    )
    midpoint = len(ordered) // 2
    halves = (ordered[:midpoint], ordered[midpoint:])
    selected = [
        min(
            half,
            key=lambda example: _stable_rank(
                "aime-half", condition, half_index, example.example_id
            ),
        )
        for half_index, half in enumerate(halves)
    ]
    return tuple(
        ThroughputRequest(
            condition=condition,
            benchmark="aime2024",
            example_id=example.example_id,
            subject=example.subject,
            level=example.level,
            prompt_tokens=example.prompt_tokens[condition],
            request_seed=request_seed(
                "aime2024", example.example_id, 0, study=QWEN3_STUDY_ID
            ),
            length_quantile=half_index * 2,
        )
        for half_index, example in enumerate(selected)
    )


def build_throughput_plan(
    examples: Sequence[ThroughputExample],
) -> ThroughputPlan:
    """Select the deterministic workload from full MATH-500 and AIME 2024."""

    values = tuple(examples)
    if len({(example.benchmark, example.example_id) for example in values}) != len(values):
        raise ValueError("throughput example identities must be unique")
    math_examples = tuple(value for value in values if value.benchmark == "math500")
    aime_examples = tuple(value for value in values if value.benchmark == "aime2024")
    if len(math_examples) != 500 or len(aime_examples) != 30:
        raise ValueError("throughput selection requires full MATH-500 and AIME 2024")
    requests = []
    for condition in CORE_CONDITIONS:
        requests.extend(_select_math(math_examples, condition))
        requests.extend(_select_aime(aime_examples, condition))
    return ThroughputPlan(requests=tuple(requests))


@dataclass(frozen=True)
class RequestTiming:
    condition: str
    benchmark: str
    example_id: str
    request_seed: int
    elapsed_seconds: float
    generated_tokens: int

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.condition, self.benchmark, self.example_id)

    def __post_init__(self) -> None:
        if self.condition not in CORE_CONDITIONS:
            raise ValueError("timing condition is not registered")
        if self.benchmark not in ("math500", "aime2024"):
            raise ValueError("timing benchmark is not registered")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("timing example_id must be nonempty")
        if (
            isinstance(self.request_seed, bool)
            or not isinstance(self.request_seed, int)
            or self.request_seed < 0
        ):
            raise ValueError("timing request_seed must be a nonnegative integer")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds <= 0
        ):
            raise ValueError("timing elapsed_seconds must be finite and positive")
        if (
            isinstance(self.generated_tokens, bool)
            or not isinstance(self.generated_tokens, int)
            or self.generated_tokens < 0
        ):
            raise ValueError("generated_tokens must be a nonnegative integer")


@dataclass(frozen=True)
class BenchmarkObservation:
    """One model/mode/DP execution of the immutable 66-request plan."""

    model_label: str
    inference_mode: str
    data_parallel_size: int
    plan_sha256: str
    timings: Tuple[RequestTiming, ...]
    generation_wall_seconds: float
    engine_load_seconds: float
    finalization_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timings", tuple(self.timings))
        if not isinstance(self.model_label, str) or not self.model_label:
            raise ValueError("model_label must be nonempty")
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError("inference_mode is not registered")
        if self.data_parallel_size not in (1, 2):
            raise ValueError("calibration observations must use DP1 or DP2")
        if not isinstance(self.plan_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.plan_sha256
        ):
            raise ValueError("plan_sha256 must be a full SHA-256 digest")
        if len(self.timings) != REQUESTS_PER_MODEL_MODE:
            raise ValueError("observation must contain exactly 66 request timings")
        for name in (
            "generation_wall_seconds",
            "engine_load_seconds",
            "finalization_seconds",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError("%s must be finite and nonnegative" % name)
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)
        if self.generation_wall_seconds <= 0:
            raise ValueError("generation_wall_seconds must be positive")
        if self.timings and max(
            timing.elapsed_seconds for timing in self.timings
        ) > self.generation_wall_seconds:
            raise ValueError("request elapsed time exceeds generation wall time")


def validate_observation(
    plan: ThroughputPlan, observation: BenchmarkObservation
) -> None:
    """Bind one timing run to exactly the selected request IDs and seeds."""

    if observation.plan_sha256 != plan.content_sha256:
        raise ValueError("observation is bound to a different throughput plan")
    expected = {
        request.key: request.request_seed
        for request in plan.requests
    }
    observed = {timing.key: timing.request_seed for timing in observation.timings}
    if len(observed) != len(observation.timings) or observed != expected:
        raise ValueError("observation request identities or seeds differ from the plan")


@dataclass(frozen=True)
class RuntimeEstimate:
    model_label: str
    data_parallel_size: int
    total_seconds: float
    ci_low_seconds: float
    ci_high_seconds: float
    generation_seconds: float
    engine_load_seconds: float
    finalization_seconds: float
    overhead_seconds: float
    effective_concurrency: Mapping[str, float]
    observed_tokens_per_second: Mapping[str, float]
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED

    @property
    def upper_hours(self) -> float:
        return self.ci_high_seconds / 3600.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_label": self.model_label,
            "data_parallel_size": self.data_parallel_size,
            "total_seconds": self.total_seconds,
            "ci_low_seconds": self.ci_low_seconds,
            "ci_high_seconds": self.ci_high_seconds,
            "generation_seconds": self.generation_seconds,
            "engine_load_seconds": self.engine_load_seconds,
            "finalization_seconds": self.finalization_seconds,
            "overhead_seconds": self.overhead_seconds,
            "effective_concurrency": dict(self.effective_concurrency),
            "observed_tokens_per_second": dict(self.observed_tokens_per_second),
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "upper_hours": self.upper_hours,
            "uncertainty_interpretation": (
                "conditional_workload_composition_at_observed_throughput"
            ),
            "between_job_system_variability_included": False,
        }


def _mode_projection(
    plan: ThroughputPlan,
    observation: BenchmarkObservation,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, np.ndarray, float, float]:
    validate_observation(plan, observation)
    timing_by_key = {timing.key: timing for timing in observation.timings}
    observed_service = sum(timing.elapsed_seconds for timing in observation.timings)
    effective_concurrency = observed_service / observation.generation_wall_seconds
    if not math.isfinite(effective_concurrency) or effective_concurrency <= 0:
        raise ValueError("observation has invalid effective concurrency")

    observed_tokens = sum(timing.generated_tokens for timing in observation.timings)
    tokens_per_second = observed_tokens / observation.generation_wall_seconds
    if not math.isfinite(tokens_per_second) or tokens_per_second <= 0:
        raise ValueError("observation has no positive generated-token throughput")
    # Project stratified output-token demand using the directly observed
    # aggregate decode throughput. Frontend request latency includes queue
    # waiting when the outer queue is intentionally twice server capacity, so
    # it is retained as a diagnostic but is not treated as independent service
    # time in the allocation estimate. Consequently, this bootstrap is a
    # conditional workload-composition interval, not an estimate of
    # between-job cluster variability; the 18h-in-24h decision margin covers
    # the latter operationally.
    point_tokens = 0.0
    bootstrap_tokens = np.zeros(resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for condition in CORE_CONDITIONS:
        for benchmark in ("math500", "aime2024"):
            generated_tokens = np.asarray(
                [
                    timing_by_key[request.key].generated_tokens
                    for request in plan.requests
                    if request.condition == condition
                    and request.benchmark == benchmark
                ],
                dtype=np.float64,
            )
            production_count = PRODUCTION_COUNTS[condition][benchmark]
            point_tokens += float(generated_tokens.mean()) * production_count
            indices = rng.integers(
                0,
                len(generated_tokens),
                size=(resamples, len(generated_tokens)),
            )
            bootstrap_tokens += (
                generated_tokens[indices].mean(axis=1) * production_count
            )
    generation = point_tokens / tokens_per_second
    bootstrap_generation = bootstrap_tokens / tokens_per_second
    return generation, bootstrap_generation, effective_concurrency, tokens_per_second


def estimate_model_runtime(
    plan: ThroughputPlan,
    observations: Sequence[BenchmarkObservation],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> RuntimeEstimate:
    """Estimate one model job containing soft and discrete engine executions."""

    if resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("runtime calibration requires exactly 10,000 resamples")
    if bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError("runtime calibration requires bootstrap seed 11")
    values = tuple(observations)
    if len(values) != 2 or {value.inference_mode for value in values} != set(INFERENCE_MODES):
        raise ValueError("runtime estimate requires native-soft and hard-token observations")
    if len({value.model_label for value in values}) != 1:
        raise ValueError("runtime observations must describe one model")
    if len({value.data_parallel_size for value in values}) != 1:
        raise ValueError("runtime observations must use one DP size")
    by_mode = {value.inference_mode: value for value in values}
    generation = 0.0
    bootstrap_generation = np.zeros(resamples, dtype=np.float64)
    concurrency = {}
    throughput = {}
    engine_load = 0.0
    finalization = 0.0
    for mode in INFERENCE_MODES:
        observation = by_mode[mode]
        projected, samples, effective, tokens_per_second = _mode_projection(
            plan,
            observation,
            resamples=resamples,
            seed=bootstrap_seed,
        )
        generation += projected
        bootstrap_generation += samples
        concurrency[mode] = effective
        throughput[mode] = tokens_per_second
        engine_load += observation.engine_load_seconds
        finalization += observation.finalization_seconds
    overhead = engine_load + finalization
    total_samples = bootstrap_generation + overhead
    ci_low, ci_high = np.quantile(total_samples, [0.025, 0.975], method="linear")
    return RuntimeEstimate(
        model_label=values[0].model_label,
        data_parallel_size=values[0].data_parallel_size,
        total_seconds=generation + overhead,
        ci_low_seconds=float(ci_low),
        ci_high_seconds=float(ci_high),
        generation_seconds=generation,
        engine_load_seconds=engine_load,
        finalization_seconds=finalization,
        overhead_seconds=overhead,
        effective_concurrency=MappingProxyType(concurrency),
        observed_tokens_per_second=MappingProxyType(throughput),
    )


@dataclass(frozen=True)
class ParallelismDecision:
    status: str
    selected_data_parallel_size: Optional[int]
    next_benchmark_data_parallel_size: Optional[int]
    runtime_limit_seconds: int
    estimates: Mapping[int, RuntimeEstimate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_data_parallel_size": self.selected_data_parallel_size,
            "next_benchmark_data_parallel_size": self.next_benchmark_data_parallel_size,
            "runtime_limit_seconds": self.runtime_limit_seconds,
            "estimates": {
                str(dp): estimate.to_dict()
                for dp, estimate in sorted(self.estimates.items())
            },
        }


def choose_smallest_data_parallel_size(
    plan: ThroughputPlan,
    observations_by_dp: Mapping[int, Sequence[BenchmarkObservation]],
    *,
    runtime_limit_hours: float = RUNTIME_LIMIT_HOURS,
) -> ParallelismDecision:
    """Choose DP1, otherwise request DP2 evidence, otherwise fall back to DP8."""

    if runtime_limit_hours != RUNTIME_LIMIT_HOURS:
        raise ValueError("the production runtime limit is sealed at 18 hours")
    unknown = set(observations_by_dp) - {1, 2}
    if unknown:
        raise ValueError("only DP1 and DP2 are calibration candidates")
    estimates = {
        dp: estimate_model_runtime(plan, observations_by_dp[dp])
        for dp in sorted(observations_by_dp)
    }
    if 1 not in estimates:
        return ParallelismDecision(
            status="benchmark_required",
            selected_data_parallel_size=None,
            next_benchmark_data_parallel_size=1,
            runtime_limit_seconds=RUNTIME_LIMIT_SECONDS,
            estimates=MappingProxyType(estimates),
        )
    if estimates[1].ci_high_seconds <= RUNTIME_LIMIT_SECONDS:
        return ParallelismDecision(
            status="selected",
            selected_data_parallel_size=1,
            next_benchmark_data_parallel_size=None,
            runtime_limit_seconds=RUNTIME_LIMIT_SECONDS,
            estimates=MappingProxyType(estimates),
        )
    if 2 not in estimates:
        return ParallelismDecision(
            status="benchmark_required",
            selected_data_parallel_size=None,
            next_benchmark_data_parallel_size=2,
            runtime_limit_seconds=RUNTIME_LIMIT_SECONDS,
            estimates=MappingProxyType(estimates),
        )
    selected = 2 if estimates[2].ci_high_seconds <= RUNTIME_LIMIT_SECONDS else 8
    return ParallelismDecision(
        status="selected",
        selected_data_parallel_size=selected,
        next_benchmark_data_parallel_size=None,
        runtime_limit_seconds=RUNTIME_LIMIT_SECONDS,
        estimates=MappingProxyType(estimates),
    )


__all__ = [
    "AIME_CALIBRATION_COUNT",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BenchmarkObservation",
    "INFERENCE_MODES",
    "LENGTH_QUANTILES",
    "MATH_CALIBRATION_COUNT",
    "PRODUCTION_COUNTS",
    "PRODUCTION_REQUESTS_PER_MODE",
    "PRODUCTION_REQUESTS_PER_MODEL",
    "ParallelismDecision",
    "REQUESTS_PER_MODEL_MODE",
    "RUNTIME_LIMIT_HOURS",
    "RUNTIME_LIMIT_SECONDS",
    "RequestTiming",
    "SELECTION_SEED",
    "THROUGHPUT_PROTOCOL",
    "ThroughputExample",
    "ThroughputPlan",
    "ThroughputRequest",
    "build_throughput_plan",
    "choose_smallest_data_parallel_size",
    "estimate_model_runtime",
    "validate_observation",
]
