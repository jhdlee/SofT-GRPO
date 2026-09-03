"""Detached continuous-prefix replay for the native-soft ICL experiment.

Replay is evaluation-only.  A no-demo trajectory is reconstructed from the
released sampler's stored top-five actions and scored by the same checkpoint
under two contexts: the original no-demo prompt and one ICL prompt.  The
reported forward divergence is ``KL(prompted || no_demo)`` at latent
next-action positions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .icl_runtime import (
    RUNTIME_PROTOCOL,
    RUNTIME_SCHEMA_VERSION,
    SamplingSettings,
    TrajectoryMetadata,
    canonical_json_bytes,
    sha256_file,
    stable_request_seed,
)


REPLAY_SCHEMA_VERSION = 3
REPLAY_PROTOCOL = "opd-softgrpo-native-soft-icl-replay-v2"
ACTOR_ACTIVE_PROBABILITY_THRESHOLD = 1e-7


@dataclass(frozen=True)
class ActorAgreementTolerances:
    """Declared cross-backend smoke tolerances.

    SGLang and Transformers use different kernels, so this is an observational
    agreement gate rather than a bitwise equality claim.  Only the active
    (non-floor) portion of the released top-five policy support is compared.
    """

    min_active_support_exact_rate: float = 0.95
    max_centered_logprob_mae: float = 0.05
    max_centered_logprob_abs_error: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_active_support_exact_rate <= 1.0:
            raise ValueError("support agreement tolerance must be in [0, 1]")
        for name in (
            "max_centered_logprob_mae",
            "max_centered_logprob_abs_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and nonnegative" % name)


@dataclass(frozen=True)
class ReplayRecord:
    model_label: str
    benchmark: str
    example_id: str
    sample_index: int
    prompted_condition: str
    request_seed: int
    latent_token_count: int
    replay_exclusion_reason: str | None
    forward_kl_mean: float | None
    forward_kl_sum: float | None
    reverse_kl_mean: float | None
    reverse_kl_sum: float | None
    prompted_entropy_mean: float | None
    prompted_top1_probability_mean: float | None
    sglang_hf_active_support_exact_slots: int
    sglang_hf_centered_logprob_value_count: int
    sglang_hf_centered_logprob_abs_error_sum: float | None
    sglang_hf_centered_logprob_abs_error_max: float | None
    elapsed_seconds: float
    source_runtime_protocol: str = RUNTIME_PROTOCOL
    schema_version: int = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay record schema")
        if self.source_runtime_protocol != RUNTIME_PROTOCOL:
            raise ValueError("replay source runtime protocol changed")
        if self.model_label not in {"starting", "softgrpo"}:
            raise ValueError("unsupported replay model")
        if not self.benchmark or not self.example_id or not self.prompted_condition:
            raise ValueError("replay identity fields must be nonempty")
        if not 0 <= self.sample_index < 8:
            raise ValueError("sample_index must be in [0, 8)")
        if self.request_seed != stable_request_seed(
            benchmark=self.benchmark,
            example_id=self.example_id,
            sample_index=self.sample_index,
        ):
            raise ValueError("request seed is not paired with the source trajectory")
        if (
            isinstance(self.latent_token_count, bool)
            or not isinstance(self.latent_token_count, int)
            or self.latent_token_count < 0
        ):
            raise ValueError("latent_token_count must be a nonnegative integer")
        for name in (
            "sglang_hf_active_support_exact_slots",
            "sglang_hf_centered_logprob_value_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("%s must be a nonnegative integer" % name)
        if self.sglang_hf_active_support_exact_slots > self.latent_token_count:
            raise ValueError("exact-support slots cannot exceed latent slots")
        optional_metrics = (
            self.forward_kl_mean,
            self.forward_kl_sum,
            self.reverse_kl_mean,
            self.reverse_kl_sum,
            self.prompted_entropy_mean,
            self.prompted_top1_probability_mean,
            self.sglang_hf_centered_logprob_abs_error_sum,
            self.sglang_hf_centered_logprob_abs_error_max,
        )
        if self.replay_exclusion_reason is not None:
            if self.replay_exclusion_reason != "zero_latent_slots":
                raise ValueError("unsupported replay exclusion reason")
            if self.latent_token_count != 0:
                raise ValueError("excluded replay must have zero latent slots")
            if any(value is not None for value in optional_metrics):
                raise ValueError("excluded replay cannot define KL/actor metrics")
            if (
                self.sglang_hf_active_support_exact_slots != 0
                or self.sglang_hf_centered_logprob_value_count != 0
                or self.elapsed_seconds != 0.0
            ):
                raise ValueError("excluded replay must have zero work counters")
            return
        if self.latent_token_count == 0:
            raise ValueError("valid replay requires at least one latent action")
        if any(value is None for value in optional_metrics):
            raise ValueError("valid replay requires every KL/actor metric")
        if self.sglang_hf_centered_logprob_value_count < self.latent_token_count:
            raise ValueError("every latent slot must have an active support value")
        values = (
            *optional_metrics,
            self.elapsed_seconds,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("replay metrics must be finite")
        if min(
            float(self.forward_kl_mean),
            float(self.forward_kl_sum),
            float(self.reverse_kl_mean),
            float(self.reverse_kl_sum),
        ) < -1e-5:
            raise ValueError("categorical KL cannot be materially negative")
        if float(self.prompted_entropy_mean) < 0:
            raise ValueError("entropy cannot be negative")
        if min(
            float(self.sglang_hf_centered_logprob_abs_error_sum),
            float(self.sglang_hf_centered_logprob_abs_error_max),
        ) < 0:
            raise ValueError("actor-agreement errors cannot be negative")
        if not 0.0 <= float(self.prompted_top1_probability_mean) <= 1.0:
            raise ValueError("top-one probability must be in [0, 1]")
        if self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayRecord":
        if not isinstance(value, Mapping):
            raise TypeError("replay record must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError(
                "replay record fields differ: missing=%s unknown=%s"
                % (sorted(expected - set(value)), sorted(set(value) - expected))
            )
        return cls(**dict(value))


def reconstruct_latent_embeddings(
    model: Any,
    trajectory: TrajectoryMetadata,
    *,
    gumbel_temperature: float,
    device: Any | None = None,
) -> Any:
    """Reconstruct and detach the exact released continuous action prefix."""

    import torch

    if not trajectory.latent_support_ids:
        raise ValueError("cannot replay a trajectory with no latent actions")
    if not math.isfinite(float(gumbel_temperature)) or gumbel_temperature <= 0:
        raise ValueError("Gumbel temperature must be finite and positive")
    embedding = model.get_input_embeddings()
    target_device = device or embedding.weight.device
    support_ids = torch.tensor(
        trajectory.latent_support_ids, dtype=torch.long, device=target_device
    )
    perturbed = torch.tensor(
        trajectory.latent_perturbed_logits,
        dtype=torch.float32,
        device=target_device,
    )
    if support_ids.ndim != 2 or support_ids.shape[1] != 5 or perturbed.shape != support_ids.shape:
        raise ValueError("replay actions must have shape [latent, 5]")
    with torch.no_grad():
        support_embeddings = embedding(support_ids)
        weights = torch.softmax(perturbed / float(gumbel_temperature), dim=-1)
        # Match upstream ``weighted_forward`` exactly: multiplication may
        # promote to FP32, then the reduction explicitly returns model dtype.
        actions = torch.sum(
            weights.unsqueeze(-1) * support_embeddings,
            dim=1,
            dtype=support_embeddings.dtype,
        )
    return actions.detach()


def full_vocab_pair_metrics(
    no_demo_logits: Any,
    prompted_logits: Any,
    *,
    vocab_chunk_size: int = 8192,
) -> dict[str, Any]:
    """Compute exact FP32 KL/entropy values without materializing FP32 vocab copies."""

    import torch

    if no_demo_logits.shape != prompted_logits.shape or no_demo_logits.ndim != 2:
        raise ValueError("paired logits must have shape [slots, vocabulary]")
    if no_demo_logits.device != prompted_logits.device:
        raise ValueError("paired logits must share a device")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    vocabulary = no_demo_logits.shape[-1]

    def normalizer(logits: Any) -> Any:
        result = None
        for start in range(0, vocabulary, vocab_chunk_size):
            stop = min(start + vocab_chunk_size, vocabulary)
            current = torch.logsumexp(logits[:, start:stop].float(), dim=-1)
            result = current if result is None else torch.logaddexp(result, current)
        return result

    p_norm = normalizer(no_demo_logits)
    q_norm = normalizer(prompted_logits)
    forward = torch.zeros_like(p_norm, dtype=torch.float32)
    reverse = torch.zeros_like(p_norm, dtype=torch.float32)
    entropy = torch.zeros_like(p_norm, dtype=torch.float32)
    q_max = torch.full_like(q_norm, -torch.inf, dtype=torch.float32)
    for start in range(0, vocabulary, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, vocabulary)
        p_log = no_demo_logits[:, start:stop].float() - p_norm.unsqueeze(-1)
        q_log = prompted_logits[:, start:stop].float() - q_norm.unsqueeze(-1)
        p_probability = p_log.exp()
        q_probability = q_log.exp()
        forward.add_((q_probability * (q_log - p_log)).sum(dim=-1))
        reverse.add_((p_probability * (p_log - q_log)).sum(dim=-1))
        entropy.sub_((q_probability * q_log).sum(dim=-1))
        q_max = torch.maximum(q_max, prompted_logits[:, start:stop].float().max(dim=-1).values)
    top1 = torch.exp(q_max - q_norm)
    # Tiny negative values can arise from FP32 summation of otherwise exact KL.
    forward = forward.clamp_min(0.0)
    reverse = reverse.clamp_min(0.0)
    return {
        "forward_kl": forward,
        "reverse_kl": reverse,
        "prompted_entropy": entropy,
        "prompted_top1_probability": top1,
    }


def _hf_released_filtered_topk(logits: Any, settings: SamplingSettings) -> tuple[Any, Any]:
    """Reproduce released temperature -> top-k -> top-p probabilities in FP32."""

    import torch

    if logits.ndim != 2:
        raise ValueError("actor-agreement logits must have shape [slots, vocabulary]")
    probabilities = torch.softmax(logits.float() / float(settings.temperature), dim=-1)
    values, token_ids = torch.topk(probabilities, k=settings.top_k, dim=-1)
    values = values / values.sum(dim=-1, keepdim=True)

    # Match sgl_kernel.top_p_renorm_prob: sort ascending and retain entries
    # whose low-tail CDF is at least 1-p, then renormalize.
    ascending, order = torch.sort(values, dim=-1, descending=False)
    keep_ascending = torch.cumsum(ascending, dim=-1) >= (1.0 - float(settings.top_p))
    keep = torch.zeros_like(keep_ascending)
    keep.scatter_(1, order, keep_ascending)
    values = torch.where(keep, values, torch.zeros_like(values))
    values = values / values.sum(dim=-1, keepdim=True)
    return token_ids, values


def compare_sglang_hf_actor(
    hf_logits: Any,
    *,
    support_ids: Any,
    perturbed_logits: Any,
    gumbel_noise: Any,
    settings: SamplingSettings,
) -> dict[str, int | float]:
    """Compare SGLang's generating actor with HF replay on aligned soft slots.

    The bundled sampler stores ``perturbed = log(filtered_probability + 1e-6)
    + clipped_gumbel``.  Subtracting the compact clipped noise recovers the
    SGLang-side pre-noise values.  The comparison never stores HF vocabulary
    logits; it returns only aggregate counts and errors.
    """

    import torch

    if support_ids.ndim != 2 or support_ids.shape[-1] != settings.top_k:
        raise ValueError("SGLang support must have shape [slots, top_k]")
    if perturbed_logits.shape != support_ids.shape or gumbel_noise.shape != support_ids.shape:
        raise ValueError("SGLang support, perturbed logits, and noise must align")
    if hf_logits.shape[0] != support_ids.shape[0]:
        raise ValueError("HF and SGLang latent-query counts differ")
    if support_ids.device != hf_logits.device:
        support_ids = support_ids.to(hf_logits.device)
    perturbed_logits = perturbed_logits.to(device=hf_logits.device, dtype=torch.float32)
    gumbel_noise = gumbel_noise.to(device=hf_logits.device, dtype=torch.float32)

    hf_ids, hf_values = _hf_released_filtered_topk(hf_logits, settings)
    sglang_log = perturbed_logits - gumbel_noise
    # SGLang adds 1e-6 before log.  Leave 1e-7 of headroom so BF16/FP32
    # serialization roundoff at the exact zero-probability floor is not
    # mistaken for active mass.  A vocabulary top-five probability is well
    # above this threshold for the supported checkpoint.
    active = sglang_log > math.log(1e-6 + ACTOR_ACTIVE_PROBABILITY_THRESHOLD)
    active_count = active.sum(dim=-1)
    if torch.any(active_count == 0):
        raise RuntimeError("released SGLang metadata has an empty active support")

    hf_active = hf_values > 0
    membership = support_ids.unsqueeze(-1) == hf_ids.unsqueeze(1)
    hf_on_sglang_support = (
        membership.to(hf_values.dtype) * hf_values.unsqueeze(1)
    ).sum(dim=-1)
    hf_log = torch.log(hf_on_sglang_support + 1e-6)

    sglang_sets = active.unsqueeze(-1) & membership
    every_sglang_active_found = sglang_sets.any(dim=-1).logical_or(~active).all(dim=-1)
    hf_sets = hf_active.unsqueeze(1) & membership
    every_hf_active_found = hf_sets.any(dim=1).logical_or(~hf_active).all(dim=-1)
    same_cardinality = active_count == hf_active.sum(dim=-1)
    exact_support = every_sglang_active_found & every_hf_active_found & same_cardinality

    count = active_count.to(torch.float32).unsqueeze(-1)
    sglang_centered = sglang_log - (sglang_log * active).sum(dim=-1, keepdim=True) / count
    hf_centered = hf_log - (hf_log * active).sum(dim=-1, keepdim=True) / count
    errors = torch.abs(sglang_centered - hf_centered)[active]
    if errors.numel() == 0 or not torch.isfinite(errors).all():
        raise RuntimeError("actor-agreement comparison produced invalid errors")
    return {
        "latent_slots": int(support_ids.shape[0]),
        "active_support_exact_slots": int(exact_support.sum().item()),
        "centered_logprob_value_count": int(errors.numel()),
        "centered_logprob_abs_error_sum": float(errors.double().sum().item()),
        "centered_logprob_abs_error_max": float(errors.max().item()),
    }


def actor_agreement_gate(
    records: Sequence[ReplayRecord],
    *,
    tolerances: ActorAgreementTolerances | None = None,
) -> dict[str, Any]:
    """Summarize SGLang/HF agreement; callers decide whether to hard-fail."""

    if not records:
        raise ValueError("actor-agreement gate requires replay records")
    if any(record.prompted_condition != "no_demo" for record in records):
        raise ValueError(
            "actor-agreement gate accepts one no_demo record per source trajectory"
        )
    if any(record.replay_exclusion_reason is not None for record in records):
        raise ValueError("actor-agreement gate accepts valid replay records only")
    identities = {
        (
            record.model_label,
            record.benchmark,
            record.example_id,
            record.sample_index,
        )
        for record in records
    }
    if len(identities) != len(records):
        raise ValueError("actor-agreement gate received duplicate source trajectories")
    tolerances = tolerances or ActorAgreementTolerances()
    slots = sum(record.latent_token_count for record in records)
    exact = sum(record.sglang_hf_active_support_exact_slots for record in records)
    values = sum(record.sglang_hf_centered_logprob_value_count for record in records)
    error_sum = sum(
        float(record.sglang_hf_centered_logprob_abs_error_sum)
        for record in records
    )
    error_max = max(
        float(record.sglang_hf_centered_logprob_abs_error_max)
        for record in records
    )
    support_rate = exact / slots
    mae = error_sum / values
    valid = bool(
        support_rate >= tolerances.min_active_support_exact_rate
        and mae <= tolerances.max_centered_logprob_mae
        and error_max <= tolerances.max_centered_logprob_abs_error
    )
    return {
        "valid": valid,
        "latent_slots": slots,
        "active_support_exact_rate": support_rate,
        "centered_logprob_mae": mae,
        "centered_logprob_abs_error_max": error_max,
        "tolerances": asdict(tolerances),
        "active_probability_threshold": ACTOR_ACTIVE_PROBABILITY_THRESHOLD,
    }


class _CausalContext:
    """KV-cached backbone state that never applies an LM head to prompt rows."""

    def __init__(self, model: Any, prompt_ids: Any):
        import torch

        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must have shape [1, nonempty]")
        self.model = model
        self.backbone = getattr(model, "model", None)
        if self.backbone is None:
            raise TypeError("causal LM must expose its decoder backbone as .model")
        self.length = int(prompt_ids.shape[1])
        device = prompt_ids.device
        positions = torch.arange(self.length, device=device, dtype=torch.long)
        output = self.backbone(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
            return_dict=True,
        )
        self.past_key_values = output.past_key_values
        self.next_hidden = output.last_hidden_state[:, -1, :].detach()

    def advance(self, embeddings: Any) -> Any:
        import torch

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("action embeddings must have shape [nonempty, hidden]")
        count = int(embeddings.shape[0])
        positions = torch.arange(
            self.length,
            self.length + count,
            device=embeddings.device,
            dtype=torch.long,
        )
        attention = torch.ones(
            (1, self.length + count), dtype=torch.long, device=embeddings.device
        )
        output = self.backbone(
            inputs_embeds=embeddings.detach().unsqueeze(0),
            attention_mask=attention,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        self.past_key_values = output.past_key_values
        self.length += count
        return output.last_hidden_state.squeeze(0).detach()


def _tokenize_prompt(tokenizer: Any, prompt: str, device: Any) -> Any:
    import torch

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("rendered replay prompt must be nonempty")
    # The released SGLang evaluator accepts text, and its TokenizerManager
    # calls tokenizer.encode(text) with tokenizer defaults.  This checkpoint's
    # rendered template already contains BOS, so exact upstream replay includes
    # the same additional BOS rather than silently "repairing" the prompt.
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("rendered replay prompt tokenized to empty")
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def _hidden_chunks(
    context: _CausalContext, actions: Any, *, hidden_chunk_size: int
) -> Iterator[Any]:
    """Yield hidden states predicting latent actions in exact causal order."""

    if hidden_chunk_size <= 0:
        raise ValueError("hidden_chunk_size must be positive")
    # Last prompt position predicts action zero.
    yield context.next_hidden
    # Action i predicts action i+1, so the final latent action need not be fed.
    for start in range(0, max(int(actions.shape[0]) - 1, 0), hidden_chunk_size):
        stop = min(start + hidden_chunk_size, int(actions.shape[0]) - 1)
        yield context.advance(actions[start:stop])


def replay_trajectory(
    *,
    model: Any,
    tokenizer: Any,
    trajectory: TrajectoryMetadata,
    no_demo_prompt: str,
    prompted_prompt: str,
    model_label: str,
    benchmark: str,
    example_id: str,
    sample_index: int,
    prompted_condition: str,
    settings: SamplingSettings | None = None,
    hidden_chunk_size: int = 32,
    vocab_chunk_size: int = 8192,
) -> ReplayRecord:
    """Replay one no-demo soft prefix and compute prompted-vs-no-demo KL."""

    return replay_trajectory_many(
        model=model,
        tokenizer=tokenizer,
        trajectory=trajectory,
        no_demo_prompt=no_demo_prompt,
        prompted_prompts={prompted_condition: prompted_prompt},
        model_label=model_label,
        benchmark=benchmark,
        example_id=example_id,
        sample_index=sample_index,
        settings=settings,
        hidden_chunk_size=hidden_chunk_size,
        vocab_chunk_size=vocab_chunk_size,
    )[0]


def replay_trajectory_many(
    *,
    model: Any,
    tokenizer: Any,
    trajectory: TrajectoryMetadata,
    no_demo_prompt: str,
    prompted_prompts: Mapping[str, str],
    model_label: str,
    benchmark: str,
    example_id: str,
    sample_index: int,
    settings: SamplingSettings | None = None,
    hidden_chunk_size: int = 32,
    vocab_chunk_size: int = 8192,
) -> list[ReplayRecord]:
    """Replay several contexts while sharing the expensive no-demo forward."""

    import torch

    settings = settings or SamplingSettings()
    allowed = {
        "no_demo",
        "sdft_matched",
        "sdft_shuffled",
        "sdpg_matched",
        "sdpg_shuffled",
    }
    if not prompted_prompts or any(condition not in allowed for condition in prompted_prompts):
        raise ValueError("replay supports no-demo and the five core contexts only")
    if "no_demo" in prompted_prompts and prompted_prompts["no_demo"] != no_demo_prompt:
        raise ValueError("the no_demo replay context must exactly equal its source prompt")
    if not trajectory.latent_support_ids:
        request_seed = stable_request_seed(
            benchmark=benchmark,
            example_id=example_id,
            sample_index=sample_index,
        )
        return [
            ReplayRecord(
                model_label=model_label,
                benchmark=benchmark,
                example_id=example_id,
                sample_index=sample_index,
                prompted_condition=condition,
                request_seed=request_seed,
                latent_token_count=0,
                replay_exclusion_reason="zero_latent_slots",
                forward_kl_mean=None,
                forward_kl_sum=None,
                reverse_kl_mean=None,
                reverse_kl_sum=None,
                prompted_entropy_mean=None,
                prompted_top1_probability_mean=None,
                sglang_hf_active_support_exact_slots=0,
                sglang_hf_centered_logprob_value_count=0,
                sglang_hf_centered_logprob_abs_error_sum=None,
                sglang_hf_centered_logprob_abs_error_max=None,
                elapsed_seconds=0.0,
            )
            for condition in prompted_prompts
        ]
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("replay model has stale gradients")
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    actions = reconstruct_latent_embeddings(
        model,
        trajectory,
        gumbel_temperature=settings.gumbel_temperature,
        device=device,
    )
    source_support_ids = torch.tensor(
        trajectory.latent_support_ids, dtype=torch.long, device=device
    )
    source_perturbed_logits = torch.tensor(
        trajectory.latent_perturbed_logits, dtype=torch.float32, device=device
    )
    source_gumbel_noise = torch.tensor(
        trajectory.latent_gumbel_noise, dtype=torch.float32, device=device
    )
    p_ids = _tokenize_prompt(tokenizer, no_demo_prompt, device)
    q_ids = {
        condition: _tokenize_prompt(tokenizer, prompt, device)
        for condition, prompt in prompted_prompts.items()
        if condition != "no_demo"
    }
    maximum_positions = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    prompt_lengths = [int(p_ids.shape[1])] + [int(value.shape[1]) for value in q_ids.values()]
    required = max(prompt_lengths) + actions.shape[0]
    if maximum_positions and required > maximum_positions:
        raise RuntimeError(
            "replay requires %d positions but checkpoint supports %d"
            % (required, maximum_positions)
        )

    started = time.perf_counter()
    values = {
        condition: {
            "forward": [],
            "reverse": [],
            "entropy": [],
            "top1": [],
        }
        for condition in prompted_prompts
    }
    actor_agreement = {
        "latent_slots": 0,
        "active_support_exact_slots": 0,
        "centered_logprob_value_count": 0,
        "centered_logprob_abs_error_sum": 0.0,
        "centered_logprob_abs_error_max": 0.0,
    }
    actor_offset = 0
    model.eval()
    with torch.inference_mode():
        p_context = _CausalContext(model, p_ids)
        q_contexts = {
            condition: _CausalContext(model, ids)
            for condition, ids in q_ids.items()
        }
        p_chunks = _hidden_chunks(
            p_context, actions, hidden_chunk_size=hidden_chunk_size
        )
        q_chunks = {
            condition: iter(
                _hidden_chunks(context, actions, hidden_chunk_size=hidden_chunk_size)
            )
            for condition, context in q_contexts.items()
        }
        for p_hidden in p_chunks:
            p_logits = model.lm_head(p_hidden)
            actor_stop = actor_offset + int(p_logits.shape[0])
            observation = compare_sglang_hf_actor(
                p_logits,
                support_ids=source_support_ids[actor_offset:actor_stop],
                perturbed_logits=source_perturbed_logits[actor_offset:actor_stop],
                gumbel_noise=source_gumbel_noise[actor_offset:actor_stop],
                settings=settings,
            )
            actor_agreement["latent_slots"] += observation["latent_slots"]
            actor_agreement["active_support_exact_slots"] += observation[
                "active_support_exact_slots"
            ]
            actor_agreement["centered_logprob_value_count"] += observation[
                "centered_logprob_value_count"
            ]
            actor_agreement["centered_logprob_abs_error_sum"] += observation[
                "centered_logprob_abs_error_sum"
            ]
            actor_agreement["centered_logprob_abs_error_max"] = max(
                actor_agreement["centered_logprob_abs_error_max"],
                observation["centered_logprob_abs_error_max"],
            )
            actor_offset = actor_stop
            for condition in prompted_prompts:
                if condition == "no_demo":
                    q_logits = p_logits
                else:
                    q_hidden = next(q_chunks[condition])
                    if p_hidden.shape != q_hidden.shape:
                        raise RuntimeError("paired causal replay hidden chunks differ")
                    # Apply the LM head only to selected latent-query states.
                    q_logits = model.lm_head(q_hidden)
                metrics = full_vocab_pair_metrics(
                    p_logits, q_logits, vocab_chunk_size=vocab_chunk_size
                )
                values[condition]["forward"].append(metrics["forward_kl"].cpu())
                values[condition]["reverse"].append(metrics["reverse_kl"].cpu())
                values[condition]["entropy"].append(metrics["prompted_entropy"].cpu())
                values[condition]["top1"].append(metrics["prompted_top1_probability"].cpu())
                if condition != "no_demo":
                    del q_logits
                del metrics
            del p_logits
        if actor_offset != len(trajectory.latent_support_ids):
            raise RuntimeError("actor-agreement comparison missed latent query slots")
        for condition, chunks in q_chunks.items():
            try:
                next(chunks)
            except StopIteration:
                pass
            else:
                raise RuntimeError("prompted replay produced extra causal chunks: %s" % condition)
    elapsed_per_condition = max(time.perf_counter() - started, 1e-12) / len(prompted_prompts)
    result = []
    for condition in prompted_prompts:
        forward = torch.cat(values[condition]["forward"]).double()
        reverse = torch.cat(values[condition]["reverse"]).double()
        entropy = torch.cat(values[condition]["entropy"]).double()
        top1 = torch.cat(values[condition]["top1"]).double()
        if forward.numel() != len(trajectory.latent_support_ids):
            raise RuntimeError("causal replay did not score every latent action exactly once")
        forward_sum = max(float(forward.sum().item()), 0.0)
        reverse_sum = max(float(reverse.sum().item()), 0.0)
        result.append(
            ReplayRecord(
                model_label=model_label,
                benchmark=benchmark,
                example_id=example_id,
                sample_index=sample_index,
                prompted_condition=condition,
                request_seed=stable_request_seed(
                    benchmark=benchmark,
                    example_id=example_id,
                    sample_index=sample_index,
                ),
                latent_token_count=int(forward.numel()),
                replay_exclusion_reason=None,
                forward_kl_mean=forward_sum / forward.numel(),
                forward_kl_sum=forward_sum,
                reverse_kl_mean=reverse_sum / reverse.numel(),
                reverse_kl_sum=reverse_sum,
                prompted_entropy_mean=float(entropy.mean().item()),
                prompted_top1_probability_mean=float(top1.mean().item()),
                sglang_hf_active_support_exact_slots=int(
                    actor_agreement["active_support_exact_slots"]
                ),
                sglang_hf_centered_logprob_value_count=int(
                    actor_agreement["centered_logprob_value_count"]
                ),
                sglang_hf_centered_logprob_abs_error_sum=float(
                    actor_agreement["centered_logprob_abs_error_sum"]
                ),
                sglang_hf_centered_logprob_abs_error_max=float(
                    actor_agreement["centered_logprob_abs_error_max"]
                ),
                elapsed_seconds=elapsed_per_condition,
            )
        )
    return result


def replay_chunk_metrics(records: Sequence[ReplayRecord]) -> dict[str, float | int]:
    if not records:
        raise ValueError("replay metrics require records")
    source_by_identity: dict[tuple[str, str, str, int], ReplayRecord] = {}
    for record in records:
        identity = (
            record.model_label,
            record.benchmark,
            record.example_id,
            record.sample_index,
        )
        prior = source_by_identity.setdefault(identity, record)
        if (
            prior.replay_exclusion_reason != record.replay_exclusion_reason
            or prior.latent_token_count != record.latent_token_count
        ):
            raise ValueError("replay contexts disagree about source eligibility")
    source_records = list(source_by_identity.values())
    valid_records = [
        record for record in records if record.replay_exclusion_reason is None
    ]
    valid_sources = [
        record
        for record in source_records
        if record.replay_exclusion_reason is None
    ]
    excluded_sources = len(source_records) - len(valid_sources)
    result: dict[str, float | int] = {
        "replay/records_committed": len(records),
        "replay/valid_context_records": len(valid_records),
        "replay/excluded_context_records": len(records) - len(valid_records),
        "replay/source_trajectory_count": len(source_records),
        "replay/source_valid_trajectory_count": len(valid_sources),
        "replay/source_excluded_trajectory_count": excluded_sources,
        "replay/source_zero_latent_excluded_count": excluded_sources,
        "replay/source_latent_slots": sum(
            record.latent_token_count for record in valid_sources
        ),
        "replay/seconds": sum(record.elapsed_seconds for record in records),
    }
    if not valid_records:
        return result

    slots = sum(record.latent_token_count for record in valid_records)
    actor_values = sum(
        record.sglang_hf_centered_logprob_value_count for record in valid_sources
    )
    result.update({
        "replay/latent_slots": slots,
        "replay/forward_kl_slot_mean": sum(
            float(record.forward_kl_sum) for record in valid_records
        ) / slots,
        "replay/reverse_kl_slot_mean": sum(
            float(record.reverse_kl_sum) for record in valid_records
        ) / slots,
        "replay/forward_kl_sequence_mean": sum(
            float(record.forward_kl_sum) for record in valid_records
        ) / len(valid_records),
        "replay/prompted_entropy_slot_mean": sum(
            float(record.prompted_entropy_mean) * record.latent_token_count
            for record in valid_records
        )
        / slots,
        "replay/prompted_top1_probability_slot_mean": sum(
            float(record.prompted_top1_probability_mean) * record.latent_token_count
            for record in valid_records
        )
        / slots,
        "integrity/sglang_hf_active_support_exact_rate": sum(
            record.sglang_hf_active_support_exact_slots for record in valid_sources
        )
        / sum(record.latent_token_count for record in valid_sources),
        "integrity/sglang_hf_centered_logprob_mae": sum(
            float(record.sglang_hf_centered_logprob_abs_error_sum)
            for record in valid_sources
        )
        / actor_values,
        "integrity/sglang_hf_centered_logprob_abs_error_max": max(
            float(record.sglang_hf_centered_logprob_abs_error_max)
            for record in valid_sources
        ),
    })
    return result


class AtomicReplayStore:
    """Authenticated JSONL replay chunks with manifest-last commits."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def paths(self, key: str) -> tuple[Path, Path]:
        relative = Path(key)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("replay key must be a safe relative path")
        base = self.root / relative
        return base.with_suffix(".jsonl"), base.with_suffix(".manifest.json")

    def resume_state(
        self, key: str, *, expected_identity: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        data, sidecar = self.paths(key)
        if sidecar.exists():
            return self.verify(key, expected_identity=expected_identity)
        if data.exists():
            if not data.is_file() or data.is_symlink():
                raise RuntimeError("uncommitted replay path is not a regular file")
            data.unlink()
        if data.parent.exists():
            prefix = ".%s." % data.name
            for temporary in data.parent.iterdir():
                if temporary.name.startswith(prefix) and temporary.name.endswith(".tmp"):
                    if not temporary.is_file() or temporary.is_symlink():
                        raise RuntimeError("unexpected replay temporary path")
                    temporary.unlink()
        return None

    def verify(self, key: str, *, expected_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data, sidecar = self.paths(key)
        if not data.is_file() or not sidecar.is_file():
            if data.exists() or sidecar.exists():
                raise RuntimeError("replay chunk is only partially committed")
            raise FileNotFoundError(sidecar)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise RuntimeError("replay chunk schema changed")
        if expected_identity is not None and manifest.get("identity") != dict(expected_identity):
            raise RuntimeError("replay chunk identity changed")
        if manifest.get("size") != data.stat().st_size or manifest.get("sha256") != sha256_file(data):
            raise RuntimeError("replay chunk authentication failed")
        with data.open("r", encoding="utf-8") as stream:
            rows = [ReplayRecord.from_mapping(json.loads(line)) for line in stream]
        if len(rows) != manifest.get("row_count"):
            raise RuntimeError("replay row count changed")
        return manifest

    def load(self, key: str) -> list[ReplayRecord]:
        self.verify(key)
        data, _ = self.paths(key)
        with data.open("r", encoding="utf-8") as stream:
            return [ReplayRecord.from_mapping(json.loads(line)) for line in stream]

    def commit(
        self,
        key: str,
        records: Sequence[ReplayRecord],
        *,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("cannot commit an empty replay chunk")
        data, sidecar = self.paths(key)
        if data.exists() or sidecar.exists():
            return self.verify(key, expected_identity=identity)
        data.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(canonical_json_bytes(record.to_dict()) for record in records)
        fd, temporary_name = tempfile.mkstemp(
            dir=str(data.parent), prefix=".%s." % data.name, suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, data)
        finally:
            if temporary.exists():
                temporary.unlink()
        manifest = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "protocol": REPLAY_PROTOCOL,
            "source_runtime_schema": RUNTIME_SCHEMA_VERSION,
            "identity": dict(identity),
            "row_count": len(records),
            "size": data.stat().st_size,
            "sha256": sha256_file(data),
        }
        from .icl_runtime import _atomic_bytes

        _atomic_bytes(sidecar, canonical_json_bytes(manifest))
        return self.verify(key, expected_identity=identity)


def stable_replay_run_id(config: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(dict(config))).hexdigest()[:20]
    return "icl-replay-%s" % digest
