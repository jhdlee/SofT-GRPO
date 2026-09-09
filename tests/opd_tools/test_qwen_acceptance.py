"""Both sealed memory policies authenticate telemetry; only the legacy policy gates it."""
import copy
import json

import pytest

from opd_tools.qwen_acceptance import (
    PHYSICAL_MONITOR_RESOURCE_POLICY,
    PHYSICAL_RESOURCE_POLICY,
    resource_policy_from_arguments,
    validate_measurement_physical_memory,
    validate_physical_memory,
)


POLICIES = (PHYSICAL_RESOURCE_POLICY, PHYSICAL_MONITOR_RESOURCE_POLICY)
SCOPE = "update_actor entry through policy completion; excludes rollout and final offload"


def memory_record(policy, *, percent=99, world_size=4):
    total, used = 100 * 1024**3, percent * 1024**3
    timing = {
        "resource_policy": dict(policy), "logical_allocator_peaks_diagnostic_only": True,
        "physical_memory_scope": SCOPE,
        "physical_device_used_peak_gib": used / 1024**3,
        "physical_device_free_min_gib": (total - used) / 1024**3,
        "physical_device_used_fraction_peak": used / total,
        "ranks": [],
    }
    for rank in range(world_size):
        timing["ranks"].append({"rank": rank, "physical_memory": {
            "source": "cuda_mem_get_info", "scope": SCOPE,
            "sampling": "start, periodic, final; observed peak may miss sub-interval spikes",
            "host_ram_scope": "whole-node utilization is diagnostic; Slurm enforces the job memory allocation",
            "sample_interval_seconds": 0.1, "observed_seconds": 0.5, "sample_count": 6,
            "device_total_bytes": total, "device_used_peak_bytes": used,
            "device_free_min_bytes": total - used, "start_free_bytes": total, "final_free_bytes": total,
        }})
    return {"actor_update_timing": timing}


def arguments_for(policy, *, leaf=False):
    if leaf:
        return ["++trainer.resource_policy." + key + "=" + json.dumps(value) for key, value in policy.items()]
    return ["++trainer.resource_policy=" + json.dumps(policy)]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("leaf", [False, True])
def test_both_policies_are_read_from_sealed_nested_or_leaf_arguments(policy, leaf):
    assert resource_policy_from_arguments(arguments_for(policy, leaf=leaf), required=True) == policy


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("percent", [97, 98, 99, 100])
@pytest.mark.parametrize("world_size", [2, 4])
def test_only_legacy_policy_rejects_high_physical_usage(policy, percent, world_size):
    record = memory_record(policy, percent=percent, world_size=world_size)
    if policy == PHYSICAL_RESOURCE_POLICY and percent >= 98:
        with pytest.raises(ValueError, match="exceeds the sealed device fraction"):
            validate_physical_memory(record, policy=policy, world_size=world_size)
    else:
        validate_physical_memory(record, policy=policy, world_size=world_size)


@pytest.mark.parametrize("mutation", ["unknown_mode", "fraction", "interval", "extra_field", "missing_field"])
def test_monitor_policy_cannot_change_the_sealed_sampling_contract(mutation):
    policy = dict(PHYSICAL_MONITOR_RESOURCE_POLICY)
    if mutation == "unknown_mode": policy["mode"] = "disabled"
    elif mutation == "fraction": policy["max_device_used_fraction"] = 1.0
    elif mutation == "interval": policy["sample_interval_seconds"] = 1.0
    elif mutation == "extra_field": policy["abort"] = False
    else: del policy["max_device_used_fraction"]
    with pytest.raises(ValueError, match="sealed physical resource policy"):
        resource_policy_from_arguments(arguments_for(policy), required=True)
    with pytest.raises(ValueError, match="sealed physical resource policy"):
        validate_physical_memory(memory_record(PHYSICAL_MONITOR_RESOURCE_POLICY), policy=policy, world_size=4)


@pytest.mark.parametrize("case", ["missing", "duplicate", "mixed", "partial", "duplicate_leaf"])
def test_monitor_policy_arguments_must_be_complete_and_unambiguous(case):
    args = arguments_for(PHYSICAL_MONITOR_RESOURCE_POLICY)
    leaves = arguments_for(PHYSICAL_MONITOR_RESOURCE_POLICY, leaf=True)
    if case == "missing": args = []
    elif case == "duplicate": args *= 2
    elif case == "mixed": args += leaves
    elif case == "partial": args = leaves[:-1]
    else: args = leaves + leaves[:1]
    with pytest.raises(ValueError, match="sealed physical resource policy"):
        resource_policy_from_arguments(args, required=True)


@pytest.mark.parametrize("mutation", ["timing", "mode", "scope", "allocator", "missing_rank", "duplicate_rank",
                                     "missing_observation", "source", "sampling", "host_scope", "interval",
                                     "bool_bytes", "used_above_total", "negative_free", "free_identity",
                                     "start_below_min", "final_above_total", "sample_count", "negative_seconds",
                                     "nonfinite_seconds", "used_aggregate", "free_aggregate", "fraction_aggregate"])
def test_monitor_mode_still_rejects_malformed_or_inconsistent_rank_evidence(mutation):
    policy = PHYSICAL_MONITOR_RESOURCE_POLICY
    record = memory_record(policy)
    timing = record["actor_update_timing"]
    observation = timing["ranks"][-1]["physical_memory"]
    if mutation == "timing": record["actor_update_timing"] = None
    elif mutation == "mode": timing["resource_policy"] = dict(PHYSICAL_RESOURCE_POLICY)
    elif mutation == "scope": timing["physical_memory_scope"] = "rollout"
    elif mutation == "allocator": timing["logical_allocator_peaks_diagnostic_only"] = False
    elif mutation == "missing_rank": timing["ranks"].pop()
    elif mutation == "duplicate_rank": timing["ranks"][-1]["rank"] = 0
    elif mutation == "missing_observation": del timing["ranks"][-1]["physical_memory"]
    elif mutation == "source": observation["source"] = "torch_allocator"
    elif mutation == "sampling": observation["sampling"] = "start only"
    elif mutation == "host_scope": observation["host_ram_scope"] = "job only"
    elif mutation == "interval": observation["sample_interval_seconds"] = 0.2
    elif mutation == "bool_bytes": observation["device_used_peak_bytes"] = True
    elif mutation == "used_above_total": observation["device_used_peak_bytes"] = observation["device_total_bytes"] + 1
    elif mutation == "negative_free": observation["device_free_min_bytes"] = -1
    elif mutation == "free_identity": observation["device_free_min_bytes"] += 1
    elif mutation == "start_below_min": observation["start_free_bytes"] = 0
    elif mutation == "final_above_total": observation["final_free_bytes"] += 1
    elif mutation == "sample_count": observation["sample_count"] = 1
    elif mutation == "negative_seconds": observation["observed_seconds"] = -1
    elif mutation == "nonfinite_seconds": observation["observed_seconds"] = float("nan")
    elif mutation == "used_aggregate": timing["physical_device_used_peak_gib"] += 1
    elif mutation == "free_aggregate": timing["physical_device_free_min_gib"] += 1
    else: timing["physical_device_used_fraction_peak"] = 0.97
    with pytest.raises(ValueError):
        validate_physical_memory(record, policy=policy, world_size=4)


@pytest.mark.parametrize("mutation", [None, "trainer", "actor_rollout_ref", "acceptance_policy", "sealed_arguments"])
def test_completed_monitor_measurement_must_match_sealed_policy_everywhere(mutation):
    policy = PHYSICAL_MONITOR_RESOURCE_POLICY
    measured = {
        "configuration": {owner: {"resource_policy": dict(policy)} for owner in ("trainer", "actor_rollout_ref")},
        "acceptance_policy": {"resource_policy": dict(policy)},
        "iterations": [memory_record(policy)],
    }
    arguments = arguments_for(policy)
    if mutation in ("trainer", "actor_rollout_ref"):
        measured["configuration"][mutation]["resource_policy"] = copy.deepcopy(PHYSICAL_RESOURCE_POLICY)
    elif mutation == "acceptance_policy":
        measured["acceptance_policy"]["resource_policy"] = copy.deepcopy(PHYSICAL_RESOURCE_POLICY)
    elif mutation == "sealed_arguments":
        arguments = arguments_for(PHYSICAL_RESOURCE_POLICY)
    if mutation is None:
        validate_measurement_physical_memory(measured, arguments, world_size=4, required=True)
    else:
        with pytest.raises(ValueError, match="missing from completed measurement"):
            validate_measurement_physical_memory(measured, arguments, world_size=4, required=True)
