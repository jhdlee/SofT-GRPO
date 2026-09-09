"""Physical CUDA sampling and matched post-update resource decisions.

Logical PyTorch allocation counters include vLLM's live, physically unmapped
sleeping pools. They remain useful diagnostics, but cannot bound physical HBM.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import math
from numbers import Integral, Real
import threading
import time

from .metrics import validate_physical_resource_limits, validate_resource_limits

logger = logging.getLogger(__name__)
PHYSICAL_RESOURCE_MODES = ("physical_device_v1", "physical_device_monitor_v1")


def normalize_resource_policy(value):
    if value is None or isinstance(value, Mapping) and not value:
        return None
    keys = {"mode", "max_device_used_fraction", "sample_interval_seconds"}
    if not isinstance(value, Mapping) or set(value) != keys or value["mode"] not in PHYSICAL_RESOURCE_MODES:
        raise ValueError("resource-integrity policy must explicitly select a supported physical-device mode and both bounds")
    fraction, interval = value["max_device_used_fraction"], value["sample_interval_seconds"]
    if (isinstance(fraction, bool) or not isinstance(fraction, Real) or not math.isfinite(fraction)
            or not 0 < fraction < 1 or isinstance(interval, bool) or not isinstance(interval, Real)
            or not math.isfinite(interval) or not .01 <= interval <= 1):
        raise ValueError("resource-integrity policy has invalid physical fraction or sampling interval")
    return {"mode": value["mode"], "max_device_used_fraction": float(fraction),
            "sample_interval_seconds": float(interval)}


class PhysicalMemorySampler:
    """Bounded state: first/final observations and extrema, no tensor retention."""

    def __init__(self, read_memory, interval):
        self.read_memory = read_memory
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.total = None
        self.count = 0
        self.minimum_free = None
        self.first_free = self.last_free = None
        self.started = None
        self.ended = None
        self.stopped = False

    def _sample(self):
        free, total = self.read_memory()
        if (any(isinstance(value, bool) or not isinstance(value, Integral) for value in (free, total))
                or total <= 0 or not 0 <= free <= total):
            raise RuntimeError("physical-device resource gate received invalid CUDA free/total bytes")
        if self.total is not None and total != self.total:
            raise RuntimeError("physical-device resource gate observed changing device capacity")
        self.total, self.last_free = int(total), int(free)
        if self.count == 0:
            self.first_free = self.minimum_free = self.last_free
        self.minimum_free = min(self.minimum_free, self.last_free)
        self.count += 1

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except BaseException as error:
                self.error = error
                return

    def start(self):
        self.started = time.monotonic()
        self._sample()  # Mandatory synchronous entry observation.
        self.thread = threading.Thread(target=self._run, name="opd-physical-memory", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if not self.stopped:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=2)
                if self.thread.is_alive():
                    raise RuntimeError("physical-device resource gate sampler did not stop")
            self.stopped = True
            if self.error is not None:
                raise RuntimeError("physical-device resource gate sampling failed: " + str(self.error)) from self.error
            self._sample()  # Mandatory final observation, including short updates.
            self.ended = time.monotonic()
        if self.error is not None:
            raise RuntimeError("physical-device resource gate sampling failed: " + str(self.error)) from self.error
        return {"source": "cuda_mem_get_info", "scope": "update_actor entry through policy completion; excludes rollout and final offload",
                "sampling": "start, periodic, final; observed peak may miss sub-interval spikes",
                "sample_interval_seconds": self.interval, "sample_count": self.count,
                "device_total_bytes": self.total, "device_used_peak_bytes": self.total - self.minimum_free,
                "device_free_min_bytes": self.minimum_free, "start_free_bytes": self.first_free,
                "final_free_bytes": self.last_free,
                "observed_seconds": self.ended - self.started,
                "host_ram_scope": "whole-node utilization is diagnostic; Slurm enforces the job memory allocation"}


def collective_resource_stage(operation, *, distributed, stage):
    """Only wrap local resource operations, never FSDP training collectives."""
    result, error = None, None
    try:
        result = operation()
    except BaseException as failure:
        error = failure
    message = None if error is None else f"{type(error).__name__}: {error}"[:3000]
    messages = [message]
    if distributed.is_initialized():
        messages = [None] * distributed.get_world_size()
        try:
            distributed.all_gather_object(messages, message)
        except BaseException as failure:
            raise RuntimeError(f"resource-integrity {stage} collective failed: {failure}") from failure
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(messages) if message is not None]
    if failures:
        raise RuntimeError(f"resource-integrity {stage} failed: " + "; ".join(failures)) from error
    return result


class ResourceGuard:
    """Opt-in sampler lifetime; local/peer resource errors poison future updates."""

    def __init__(self, worker, *, distributed, device):
        self.worker, self.distributed, self.device = worker, distributed, device
        self.policy = None
        self.monitor = None
        self.physical_memory = None
        self.enabled = bool(worker.config.get("rollout_integrity", {}).get("enabled", False))

    def _stage(self, operation, name):
        try:
            return collective_resource_stage(operation, distributed=self.distributed, stage=name)
        except BaseException as error:
            self.worker._opd_resource_failure = str(error)
            raise

    def __enter__(self):
        configured = self.worker.config.get("resource_policy", None)
        if not self.enabled and not configured:
            return self
        def start():
            if getattr(self.worker, "_opd_resource_failure", None):
                raise RuntimeError("resource-integrity worker cannot be reused after failure: " + self.worker._opd_resource_failure)
            import torch
            self.policy = normalize_resource_policy(configured)
            if self.policy is None:
                return
            if not self.enabled:
                raise RuntimeError("physical-device resource gate requires rollout integrity enabled")
            self.monitor = PhysicalMemorySampler(lambda: torch.cuda.mem_get_info(self.device),
                                                 self.policy["sample_interval_seconds"])
            self.monitor.start()
        try:
            self._stage(start, "physical sampling start")
        except BaseException:
            self._cleanup()
            raise
        return self

    def finish(self, metrics, *, collect_metrics=None):
        if not self.enabled:
            if collect_metrics is not None:
                metrics.update(collect_metrics())
            return None
        def check():
            if collect_metrics is not None:
                metrics.update(collect_metrics())
            for name in ("perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb",
                         "perf/cpu_memory_used_gb", "perf/host_memory_percent", "perf/cpu_utilization_percent"):
                value = float(metrics[name])
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError("resource-integrity requires finite nonnegative metric " + name)
            if self.policy is None:
                validate_resource_limits(hbm_peak_gib=metrics["perf/max_memory_allocated_gb"],
                                         host_ram_percent=metrics["perf/host_memory_percent"])
                return None
            self.physical_memory = self.monitor.stop()
            try:
                monitor_only = self.policy["mode"] == "physical_device_monitor_v1"
                validate_physical_resource_limits(self.physical_memory,
                    max_device_used_fraction=self.policy["max_device_used_fraction"],
                    enforce_limit=not monitor_only)
                total = self.physical_memory["device_total_bytes"]
                used = self.physical_memory["device_used_peak_bytes"]
                if monitor_only and used / total >= self.policy["max_device_used_fraction"]:
                    logger.warning(
                        "Physical-device memory warning on device %s: sampled used %.3f GiB / %.3f GiB "
                        "(%.6f) >= %.6f; minimum free %.3f GiB, %d samples; training continues",
                        self.device, used / 1024**3, total / 1024**3, used / total,
                        self.policy["max_device_used_fraction"],
                        self.physical_memory["device_free_min_bytes"] / 1024**3,
                        self.physical_memory["sample_count"],
                    )
            except BaseException as error:
                evidence = {"policy": self.policy, "physical_memory": self.physical_memory,
                            "logical_allocated_peak_gib": metrics["perf/max_memory_allocated_gb"],
                            "logical_reserved_peak_gib": metrics["perf/max_memory_reserved_gb"]}
                raise RuntimeError(str(error) + "; evidence=" + json.dumps(evidence, sort_keys=True, allow_nan=False)) from error
            return self.physical_memory
        return self._stage(check, "post-update resource validation")

    def _cleanup(self):
        if self.monitor is not None and not self.monitor.stopped:
            try:
                self.monitor.stop()
            except BaseException as error:
                # Preserve the original actor/update error. The controller
                # terminates the process; this is not permission to reuse it.
                self.worker._opd_resource_failure = "resource-integrity sampler cleanup failed: " + str(error)

    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()
        if exc_type is None and self.monitor is not None and self.physical_memory is None:
            self.worker._opd_resource_failure = "resource-integrity physical sampling exited without a completed collective validation"
            raise RuntimeError(self.worker._opd_resource_failure)
        return False
