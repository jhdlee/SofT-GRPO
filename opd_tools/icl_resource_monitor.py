"""Small Linux resource sampler for long-running ICL generation jobs.

The monitor has no mandatory third-party dependency.  Host metrics use
``psutil`` when it is importable, while GPU metrics use the stable CSV output
of ``nvidia-smi``.  Missing or temporarily failing providers are represented
as unavailable values in the result instead of terminating generation.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from typing import Any, Mapping


GIB = float(1024**3)


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _load_psutil() -> tuple[Any | None, str | None]:
    try:
        import psutil
    except (ImportError, ModuleNotFoundError) as error:
        return None, "%s: %s" % (type(error).__name__, error)
    return psutil, None


class _GPUQueryUnavailable(RuntimeError):
    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


def _gpu_visibility_from_environment(
    environment: Mapping[str, str] | None = None,
) -> tuple[str | None, tuple[str, ...] | None, int | None]:
    """Return the most specific GPU assignment published by Slurm/CUDA.

    ``SLURM_STEP_GPUS`` is preferred because a nested ``srun`` can expose fewer
    devices than the enclosing allocation.  ``CUDA_VISIBLE_DEVICES`` is the
    next-best process-level contract, followed by the allocation-wide Slurm
    value.  The function only reads these variables; it never mutates CUDA
    visibility.
    """

    values = os.environ if environment is None else environment
    source = None
    raw = None
    for name in ("SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS"):
        if name in values:
            source = name
            raw = str(values[name]).strip()
            break
    selectors: tuple[str, ...] | None = None
    if source is not None:
        if not raw or raw in {"-1", "NoDevFiles"}:
            selectors = ()
        else:
            parsed = []
            for value in raw.split(","):
                token = value.strip()
                if token.lower().startswith("gpu:"):
                    token = token[4:]
                if not token:
                    raise ValueError("%s contains an empty GPU selector" % source)
                parsed.append(token)
            if len(set(parsed)) != len(parsed):
                raise ValueError("%s contains duplicate GPU selectors" % source)
            selectors = tuple(parsed)

    expected_count = None
    raw_expected = values.get("OPD_EXPECTED_VISIBLE_GPUS")
    if raw_expected is not None:
        try:
            expected_count = int(str(raw_expected))
        except ValueError as error:
            raise ValueError(
                "OPD_EXPECTED_VISIBLE_GPUS must be a positive integer"
            ) from error
        if expected_count <= 0:
            raise ValueError("OPD_EXPECTED_VISIBLE_GPUS must be a positive integer")
    return source, selectors, expected_count


def _selector_matches_gpu(selector: str, *, index: str, uuid: str) -> bool:
    normalized = selector.strip()
    if normalized == index or normalized == uuid:
        return True
    # Slurm/NVML configurations can publish the UUID either with or without
    # NVIDIA's conventional GPU- prefix.
    return (
        uuid.startswith("GPU-")
        and normalized == uuid.removeprefix("GPU-")
    )


def _query_nvidia_smi(
    timeout_seconds: float,
    *,
    selectors: tuple[str, ...] | None = None,
    expected_count: int | None = None,
) -> dict[str, tuple[float, float | None]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise _GPUQueryUnavailable(
            "nvidia-smi is not installed or not on PATH", permanent=True
        ) from error
    except subprocess.TimeoutExpired as error:
        raise _GPUQueryUnavailable("nvidia-smi timed out") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        message = "nvidia-smi failed with status %d" % error.returncode
        if detail:
            message += ": " + detail
        raise _GPUQueryUnavailable(message) from error
    except OSError as error:
        raise _GPUQueryUnavailable(
            "%s: %s" % (type(error).__name__, error)
        ) from error

    discovered: dict[str, tuple[str, float, float | None]] = {}
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise _GPUQueryUnavailable(
                "nvidia-smi returned malformed CSV on line %d" % line_number
            )
        gpu_id, gpu_uuid, memory_text, utilization_text = fields
        if not gpu_id or gpu_id in discovered:
            raise _GPUQueryUnavailable(
                "nvidia-smi returned a missing or repeated GPU index"
            )
        if not gpu_uuid:
            raise _GPUQueryUnavailable("nvidia-smi returned a missing GPU UUID")
        try:
            memory_mib = float(memory_text)
        except ValueError as error:
            raise _GPUQueryUnavailable(
                "nvidia-smi returned nonnumeric GPU memory"
            ) from error
        if not math.isfinite(memory_mib) or memory_mib < 0.0:
            raise _GPUQueryUnavailable("nvidia-smi returned invalid GPU memory")
        utilization: float | None
        if utilization_text.lower() in {"n/a", "[n/a]", "not supported"}:
            utilization = None
        else:
            try:
                utilization = float(utilization_text)
            except ValueError as error:
                raise _GPUQueryUnavailable(
                    "nvidia-smi returned nonnumeric GPU utilization"
                ) from error
            if not math.isfinite(utilization) or not 0.0 <= utilization <= 100.0:
                raise _GPUQueryUnavailable(
                    "nvidia-smi returned invalid GPU utilization"
                )
        discovered[gpu_id] = (gpu_uuid, memory_mib / 1024.0, utilization)
    result = {
        gpu_id: (memory_gib, utilization)
        for gpu_id, (gpu_uuid, memory_gib, utilization) in discovered.items()
        if selectors is None
        or any(
            _selector_matches_gpu(selector, index=gpu_id, uuid=gpu_uuid)
            for selector in selectors
        )
    }
    target_count = expected_count
    if target_count is None and selectors is not None:
        target_count = len(selectors)
    # Some Slurm device-cgroup configurations expose only assigned devices but
    # renumber NVIDIA-SMI indices locally. If the command itself returned
    # exactly the assigned cardinality, accepting all its rows is safer than
    # either dropping telemetry or accidentally querying the rest of the node.
    if (
        selectors is not None
        and target_count is not None
        and len(result) != target_count
        and len(discovered) == target_count
    ):
        result = {
            gpu_id: (memory_gib, utilization)
            for gpu_id, (_, memory_gib, utilization) in discovered.items()
        }
    if not result:
        raise _GPUQueryUnavailable(
            "nvidia-smi returned no rows matching the visible GPU assignment"
        )
    if expected_count is not None and len(result) != expected_count:
        raise _GPUQueryUnavailable(
            "nvidia-smi matched %d GPUs, expected %d from the Slurm step"
            % (len(result), expected_count)
        )
    return result


@dataclass(frozen=True)
class ResourceMonitorResult:
    """Aggregated samples collected between ``start`` and ``stop``."""

    sample_count: int
    host_sample_count: int
    gpu_sample_count: int
    peak_hbm_gib_per_gpu: Mapping[str, float] | None
    peak_hbm_gib_aggregate: float | None
    peak_host_ram_gib: float | None
    cpu_utilization_mean: float | None
    cpu_utilization_peak: float | None
    gpu_utilization_mean: float | None
    gpu_utilization_mean_per_gpu: Mapping[str, float] | None
    host_metrics_available: bool
    gpu_metrics_available: bool
    host_error: str | None
    gpu_error: str | None
    monitor_error: str | None
    gpu_selection_source: str | None
    gpu_selectors: tuple[str, ...] | None
    expected_gpu_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """Poll host and GPU resources on a private, bounded-lifetime thread.

    The default one-second cadence is suitable for evaluation jobs.  A monitor
    is one-shot, but ``stop`` is idempotent and returns the same immutable
    result on repeated calls.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = 1.0,
        command_timeout_seconds: float = 2.0,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or float(interval_seconds) <= 0.0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        if (
            isinstance(command_timeout_seconds, bool)
            or not isinstance(command_timeout_seconds, (int, float))
            or not math.isfinite(float(command_timeout_seconds))
            or float(command_timeout_seconds) <= 0.0
        ):
            raise ValueError(
                "command_timeout_seconds must be finite and positive"
            )
        self.interval_seconds = float(interval_seconds)
        self.command_timeout_seconds = float(command_timeout_seconds)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_sample_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._result: ResourceMonitorResult | None = None

        self._sample_count = 0
        self._host_sample_count = 0
        self._gpu_sample_count = 0
        self._peak_host_ram_gib: float | None = None
        self._cpu_sum = 0.0
        self._cpu_peak: float | None = None
        self._gpu_peak: dict[str, float] = {}
        self._aggregate_gpu_peak: float | None = None
        self._gpu_util_sum: dict[str, float] = {}
        self._gpu_util_count: dict[str, int] = {}
        self._host_error: str | None = None
        self._gpu_error: str | None = None
        self._monitor_error: str | None = None
        self._host_disabled = False
        self._gpu_disabled = False
        self._psutil: Any | None = None
        self._gpu_selection_source: str | None = None
        self._gpu_selectors: tuple[str, ...] | None = None
        self._expected_gpu_count: int | None = None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def result(self) -> ResourceMonitorResult | None:
        return self._result

    def start(self) -> ResourceMonitor:
        with self._lock:
            if self._started:
                raise RuntimeError("resource monitor is one-shot and already started")
            self._started = True
            if not _is_linux():
                message = "resource monitoring is available only on Linux"
                self._host_error = message
                self._gpu_error = message
                self._host_disabled = True
                self._gpu_disabled = True
            else:
                self._psutil, self._host_error = _load_psutil()
                self._host_disabled = self._psutil is None
                try:
                    (
                        self._gpu_selection_source,
                        self._gpu_selectors,
                        self._expected_gpu_count,
                    ) = _gpu_visibility_from_environment()
                except ValueError as error:
                    self._gpu_error = "%s: %s" % (type(error).__name__, error)
                    self._gpu_disabled = True
            self._thread = threading.Thread(
                target=self._run,
                name="opd-icl-resource-monitor",
                daemon=True,
            )
            self._thread.start()
        return self

    def _sample_host(self) -> None:
        if self._host_disabled:
            return
        try:
            used_gib = float(self._psutil.virtual_memory().used) / GIB
            cpu = float(self._psutil.cpu_percent(interval=None))
            if (
                not math.isfinite(used_gib)
                or used_gib < 0.0
                or not math.isfinite(cpu)
                or not 0.0 <= cpu <= 100.0
            ):
                raise ValueError("psutil returned an invalid host sample")
        except Exception as error:
            self._host_error = "%s: %s" % (type(error).__name__, error)
            return
        self._host_sample_count += 1
        self._peak_host_ram_gib = (
            used_gib
            if self._peak_host_ram_gib is None
            else max(self._peak_host_ram_gib, used_gib)
        )
        self._cpu_sum += cpu
        self._cpu_peak = cpu if self._cpu_peak is None else max(self._cpu_peak, cpu)

    def _sample_gpu(self) -> None:
        if self._gpu_disabled:
            return
        try:
            sample = _query_nvidia_smi(
                self.command_timeout_seconds,
                selectors=self._gpu_selectors,
                expected_count=self._expected_gpu_count,
            )
        except _GPUQueryUnavailable as error:
            self._gpu_error = str(error)
            self._gpu_disabled = error.permanent
            return
        self._gpu_sample_count += 1
        aggregate = 0.0
        for gpu_id, (memory_gib, utilization) in sample.items():
            aggregate += memory_gib
            self._gpu_peak[gpu_id] = max(
                self._gpu_peak.get(gpu_id, 0.0), memory_gib
            )
            if utilization is not None:
                self._gpu_util_sum[gpu_id] = (
                    self._gpu_util_sum.get(gpu_id, 0.0) + utilization
                )
                self._gpu_util_count[gpu_id] = (
                    self._gpu_util_count.get(gpu_id, 0) + 1
                )
        self._aggregate_gpu_peak = (
            aggregate
            if self._aggregate_gpu_peak is None
            else max(self._aggregate_gpu_peak, aggregate)
        )

    def _run(self) -> None:
        while True:
            try:
                with self._lock:
                    self._sample_count += 1
                    self._sample_host()
                    self._sample_gpu()
            except Exception as error:  # pragma: no cover - last-resort guard
                with self._lock:
                    self._monitor_error = "%s: %s" % (
                        type(error).__name__,
                        error,
                    )
            finally:
                self._first_sample_event.set()
            if self._stop_event.wait(self.interval_seconds):
                return

    def _build_result(self) -> ResourceMonitorResult:
        per_gpu_utilization = {
            gpu_id: self._gpu_util_sum[gpu_id] / count
            for gpu_id, count in self._gpu_util_count.items()
            if count
        }
        total_gpu_utilization_count = sum(self._gpu_util_count.values())
        return ResourceMonitorResult(
            sample_count=self._sample_count,
            host_sample_count=self._host_sample_count,
            gpu_sample_count=self._gpu_sample_count,
            peak_hbm_gib_per_gpu=(
                dict(sorted(self._gpu_peak.items())) if self._gpu_peak else None
            ),
            peak_hbm_gib_aggregate=self._aggregate_gpu_peak,
            peak_host_ram_gib=self._peak_host_ram_gib,
            cpu_utilization_mean=(
                self._cpu_sum / self._host_sample_count
                if self._host_sample_count
                else None
            ),
            cpu_utilization_peak=self._cpu_peak,
            gpu_utilization_mean=(
                sum(self._gpu_util_sum.values()) / total_gpu_utilization_count
                if total_gpu_utilization_count
                else None
            ),
            gpu_utilization_mean_per_gpu=(
                dict(sorted(per_gpu_utilization.items()))
                if per_gpu_utilization
                else None
            ),
            host_metrics_available=bool(self._host_sample_count),
            gpu_metrics_available=bool(self._gpu_sample_count),
            host_error=self._host_error,
            gpu_error=self._gpu_error,
            monitor_error=self._monitor_error,
            gpu_selection_source=self._gpu_selection_source,
            gpu_selectors=self._gpu_selectors,
            expected_gpu_count=self._expected_gpu_count,
        )

    def stop(self) -> ResourceMonitorResult:
        with self._lock:
            if not self._started:
                raise RuntimeError("resource monitor has not been started")
            if self._result is not None:
                return self._result
            thread = self._thread
            self._stop_event.set()
        if thread is None:  # pragma: no cover - state invariant
            raise RuntimeError("resource monitor thread was not created")
        thread.join(
            timeout=self.command_timeout_seconds + self.interval_seconds + 1.0
        )
        if thread.is_alive():
            raise RuntimeError("resource monitor thread did not stop")
        with self._lock:
            if self._result is None:
                self._result = self._build_result()
            return self._result

    def __enter__(self) -> ResourceMonitor:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        return False


__all__ = ["ResourceMonitor", "ResourceMonitorResult"]
