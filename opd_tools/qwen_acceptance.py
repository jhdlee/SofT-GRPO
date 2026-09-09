"""CPU-only acceptance checks shared by Qwen correctness and production."""
from __future__ import annotations

import json
import math

PHYSICAL_RESOURCE_POLICY = {"mode": "physical_device_v1", "max_device_used_fraction": 0.98,
                            "sample_interval_seconds": 0.1}
PHYSICAL_MONITOR_RESOURCE_POLICY = {**PHYSICAL_RESOURCE_POLICY, "mode": "physical_device_monitor_v1"}


def _validate_world_size(world_size):
    if type(world_size) is not int or world_size not in (2, 4):
        raise ValueError("Qwen acceptance requires world size two or four")


def number(mapping, name):
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing finite production evidence: {name}")
    return float(value)


def validate_fsdp_probe(value, *, world_size=2):
    """Require actual mixed frozen/trainable FSDP evidence on every requested rank."""
    _validate_world_size(world_size)
    if (not isinstance(value, dict) or value.get("status") != "passed"
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or type(value.get("world_size")) is not int or value["world_size"] != world_size):
        raise ValueError("GPU allocation lacks native LoRA/FSDP acceptance")
    ranks = value.get("ranks")
    if (not isinstance(ranks, list) or len(ranks) != world_size
            or any(not isinstance(row, dict) or type(row.get("rank")) is not int for row in ranks)
            or {row["rank"] for row in ranks} != set(range(world_size))):
        raise ValueError("native LoRA/FSDP acceptance requires all distinct training ranks")
    for row in ranks:
        for key in ("frozen_base_unchanged", "base_gradients_absent", "adapter_gradients_finite",
                    "adapter_update_nonzero", "disabled_reference_exact", "dense_export_exact",
                    "current_actor_detached"):
            if row.get(key) is not True:
                raise ValueError("native LoRA/FSDP acceptance failed: " + key)
        for key, expected in (("optimizer_steps", 2), ("dense_ema_updates", 1), ("wrapper_count", 3)):
            if type(row.get(key)) is not int or row[key] != expected:
                raise ValueError("native LoRA/FSDP acceptance cadence differs: " + key)
    return value


def resource_policy_from_arguments(arguments, *, required=False):
    """Read the policy from authenticated argv, never from observed statistics."""
    prefix = "trainer.resource_policy"
    expected = PHYSICAL_RESOURCE_POLICY
    values, leaves = [], {}
    for argument in arguments:
        key, separator, value = argument.partition("=")
        key = key.lstrip("+")
        if key != prefix and not key.startswith(prefix + "."):
            continue
        if not separator:
            raise ValueError("malformed sealed physical resource policy")
        if key == prefix:
            values.append(json.loads(value))
        else:
            field = key[len(prefix) + 1:]
            if field not in expected:
                raise ValueError("unknown sealed physical resource policy field")
            if field in leaves:
                raise ValueError("duplicate sealed physical resource policy field")
            leaves[field] = json.loads(value)
    if len(values) > 1:
        raise ValueError("duplicate sealed physical resource policy")
    if values and leaves:
        raise ValueError("mixed nested and leaf sealed physical resource policy")
    if leaves and set(leaves) != set(expected):
        raise ValueError("partial sealed physical resource policy")
    policy = values[0] if values else leaves
    if policy == {}:
        if required:
            raise ValueError("missing sealed physical resource policy")
        return None
    if policy not in (PHYSICAL_RESOURCE_POLICY, PHYSICAL_MONITOR_RESOURCE_POLICY):
        raise ValueError("unsupported sealed physical resource policy")
    return policy


def validate_physical_memory(record, *, policy, world_size=2):
    """Verify sampler evidence; enforce the fraction only for the historical gate."""
    _validate_world_size(world_size)
    if policy not in (PHYSICAL_RESOURCE_POLICY, PHYSICAL_MONITOR_RESOURCE_POLICY):
        raise ValueError("unsupported sealed physical resource policy")
    if not isinstance(record, dict) or not isinstance(record.get("actor_update_timing"), dict):
        raise ValueError("missing physical memory timing evidence")
    timing = record["actor_update_timing"]
    scope = "update_actor entry through policy completion; excludes rollout and final offload"
    sampling = "start, periodic, final; observed peak may miss sub-interval spikes"
    host_scope = "whole-node utilization is diagnostic; Slurm enforces the job memory allocation"
    if (timing.get("resource_policy") != policy
            or timing.get("logical_allocator_peaks_diagnostic_only") is not True
            or timing.get("physical_memory_scope") != scope):
        raise ValueError("physical resource policy or accounting scope differs")
    ranks = timing.get("ranks")
    if (not isinstance(ranks, list) or len(ranks) != world_size
            or any(not isinstance(rank, dict) or type(rank.get("rank")) is not int for rank in ranks)
            or {rank["rank"] for rank in ranks} != set(range(world_size))):
        raise ValueError("physical memory requires all distinct training ranks")
    observations = []
    for rank in ranks:
        observation = rank.get("physical_memory")
        if not isinstance(observation, dict):
            raise ValueError("missing physical memory evidence for training rank")
        if (observation.get("source") != "cuda_mem_get_info"
                or observation.get("scope") != scope or observation.get("sampling") != sampling
                or observation.get("host_ram_scope") != host_scope
                or number(observation, "sample_interval_seconds") != policy["sample_interval_seconds"]):
            raise ValueError("physical memory sampling identity differs")
        fields = ("device_total_bytes", "device_used_peak_bytes", "device_free_min_bytes",
                  "start_free_bytes", "final_free_bytes", "sample_count")
        if any(type(observation.get(key)) is not int for key in fields):
            raise ValueError("physical memory bytes and sample count must be integers")
        total, used, free, first, last, count = (observation[key] for key in fields)
        if (total <= 0 or not 0 <= used <= total or free != total - used
                or not 0 <= free <= first <= total or not free <= last <= total or count < 2
                or number(observation, "observed_seconds") < 0):
            raise ValueError("invalid physical memory sample bounds")
        if policy["mode"] == "physical_device_v1" and used / total >= policy["max_device_used_fraction"]:
            raise ValueError("physical memory exceeds the sealed device fraction")
        observations.append(observation)
    aggregates = {
        "physical_device_used_peak_gib": max(x["device_used_peak_bytes"] for x in observations) / 1024**3,
        "physical_device_free_min_gib": min(x["device_free_min_bytes"] for x in observations) / 1024**3,
        "physical_device_used_fraction_peak": max(x["device_used_peak_bytes"] / x["device_total_bytes"] for x in observations),
    }
    for key, expected in aggregates.items():
        if number(timing, key) != expected:
            raise ValueError("physical memory aggregate differs from per-rank evidence: " + key)


def phase_resource_policy(row, phase):
    """Read correctness policy from its authenticated command."""
    return resource_policy_from_arguments(row["phase_commands"][phase][3:])


def validate_measurement_physical_memory(measured, arguments, *, world_size, required=False):
    """Compare observed policy with sealed argv and recompute every rank's peak."""
    policy = resource_policy_from_arguments(arguments, required=required)
    if policy is None:
        return
    config = measured.get("configuration", {})
    if (any(config.get(owner, {}).get("resource_policy") != policy
            for owner in ("trainer", "actor_rollout_ref"))
            or measured.get("acceptance_policy", {}).get("resource_policy") != policy):
        raise ValueError("physical resource policy is missing from completed measurement")
    for record in measured.get("iterations", []):
        validate_physical_memory(record, policy=policy, world_size=world_size)
