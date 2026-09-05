import subprocess
import threading
import types

import pytest

import opd_tools.icl_resource_monitor as resource_monitor
from opd_tools.icl_resource_monitor import ResourceMonitor


@pytest.fixture(autouse=True)
def _clear_gpu_assignment_environment(monkeypatch):
    for name in (
        "SLURM_STEP_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_GPUS",
        "OPD_EXPECTED_VISIBLE_GPUS",
    ):
        monkeypatch.delenv(name, raising=False)


class _FakePsutil:
    def __init__(self):
        self.samples = []

    def virtual_memory(self):
        index = len(self.samples)
        used_gib = (1.0, 3.0, 2.0)[min(index, 2)]
        return types.SimpleNamespace(used=int(used_gib * 1024**3))

    def cpu_percent(self, interval=None):
        assert interval is None
        index = len(self.samples)
        value = (10.0, 50.0, 30.0)[min(index, 2)]
        self.samples.append((value, index))
        return value


def _monitor_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "opd-icl-resource-monitor" and thread.is_alive()
    ]


def test_resource_monitor_collects_peaks_means_and_stops_cleanly(monkeypatch):
    fake_psutil = _FakePsutil()
    gpu_outputs = [
        "0, GPU-zero, 1024, 20\n1, GPU-one, 2048, 40\n",
        "0, GPU-zero, 3072, 60\n1, GPU-one, 1024, 80\n",
        "0, GPU-zero, 2048, 40\n1, GPU-one, 1536, 60\n",
    ]
    queried = []
    enough_samples = threading.Event()

    def run(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 0.2,
        }
        index = min(len(queried), len(gpu_outputs) - 1)
        queried.append(index)
        if len(queried) >= 3:
            enough_samples.set()
        return types.SimpleNamespace(stdout=gpu_outputs[index])

    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: True)
    monkeypatch.setattr(
        resource_monitor,
        "_load_psutil",
        lambda: (fake_psutil, None),
    )
    monkeypatch.setattr(resource_monitor.subprocess, "run", run)

    monitor = ResourceMonitor(
        interval_seconds=0.01,
        command_timeout_seconds=0.2,
    )
    with monitor:
        assert enough_samples.wait(timeout=1.0)
        assert monitor.is_running

    result = monitor.result
    assert result is not None
    assert result.sample_count == len(queried) == len(fake_psutil.samples)
    assert result.sample_count >= 3
    assert result.host_sample_count == result.sample_count
    assert result.gpu_sample_count == result.sample_count
    assert result.peak_host_ram_gib == 3.0
    assert result.cpu_utilization_peak == 50.0
    expected_cpu_mean = sum(value for value, _ in fake_psutil.samples) / len(
        fake_psutil.samples
    )
    assert result.cpu_utilization_mean == pytest.approx(expected_cpu_mean)
    assert result.peak_hbm_gib_per_gpu == {"0": 3.0, "1": 2.0}
    assert result.peak_hbm_gib_aggregate == 4.0

    gpu0_values = (20.0, 60.0) + (40.0,) * (len(queried) - 2)
    gpu1_values = (40.0, 80.0) + (60.0,) * (len(queried) - 2)
    assert result.gpu_utilization_mean_per_gpu == pytest.approx(
        {
            "0": sum(gpu0_values) / len(gpu0_values),
            "1": sum(gpu1_values) / len(gpu1_values),
        }
    )
    assert result.gpu_utilization_mean == pytest.approx(
        (sum(gpu0_values) + sum(gpu1_values))
        / (len(gpu0_values) + len(gpu1_values))
    )
    assert result.host_metrics_available
    assert result.gpu_metrics_available
    assert result.host_error is None
    assert result.gpu_error is None
    assert result.monitor_error is None
    assert not monitor.is_running
    assert not _monitor_threads()
    assert monitor.stop() is result
    assert result.to_dict()["sample_count"] == result.sample_count


def test_resource_monitor_marks_missing_providers_without_leaking(monkeypatch):
    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: True)
    monkeypatch.setattr(
        resource_monitor,
        "_load_psutil",
        lambda: (None, "ImportError: psutil unavailable"),
    )

    def missing_nvidia_smi(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(
        resource_monitor.subprocess,
        "run",
        missing_nvidia_smi,
    )
    monitor = ResourceMonitor(
        interval_seconds=0.01,
        command_timeout_seconds=0.2,
    ).start()
    assert monitor._first_sample_event.wait(timeout=1.0)
    result = monitor.stop()

    assert result.sample_count >= 1
    assert result.host_sample_count == 0
    assert result.gpu_sample_count == 0
    assert result.peak_host_ram_gib is None
    assert result.cpu_utilization_mean is None
    assert result.cpu_utilization_peak is None
    assert result.peak_hbm_gib_per_gpu is None
    assert result.peak_hbm_gib_aggregate is None
    assert result.gpu_utilization_mean is None
    assert result.gpu_utilization_mean_per_gpu is None
    assert not result.host_metrics_available
    assert not result.gpu_metrics_available
    assert "psutil unavailable" in result.host_error
    assert "not installed" in result.gpu_error
    assert not monitor.is_running
    assert not _monitor_threads()


def test_resource_monitor_stops_when_context_body_raises(monkeypatch):
    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: False)
    monitor = ResourceMonitor(interval_seconds=0.01)
    with pytest.raises(ValueError, match="body failed"):
        with monitor:
            raise ValueError("body failed")
    assert monitor.result is not None
    assert monitor.result.sample_count >= 1
    assert "only on Linux" in monitor.result.host_error
    assert "only on Linux" in monitor.result.gpu_error
    assert not monitor.is_running
    assert not _monitor_threads()


def test_resource_monitor_retries_transient_nvidia_smi_failure(monkeypatch):
    fake_psutil = _FakePsutil()
    second_call = threading.Event()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("nvidia-smi", timeout=0.2)
        second_call.set()
        return types.SimpleNamespace(stdout="0, GPU-zero, 512, N/A\n")

    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: True)
    monkeypatch.setattr(
        resource_monitor,
        "_load_psutil",
        lambda: (fake_psutil, None),
    )
    monkeypatch.setattr(resource_monitor.subprocess, "run", run)
    monitor = ResourceMonitor(
        interval_seconds=0.01,
        command_timeout_seconds=0.2,
    ).start()
    assert second_call.wait(timeout=1.0)
    result = monitor.stop()

    assert len(calls) >= 2
    assert result.gpu_metrics_available
    assert result.peak_hbm_gib_per_gpu == {"0": 0.5}
    assert result.peak_hbm_gib_aggregate == 0.5
    assert result.gpu_utilization_mean is None
    assert result.gpu_utilization_mean_per_gpu is None
    assert "timed out" in result.gpu_error
    assert not monitor.is_running
    assert not _monitor_threads()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_seconds": 0},
        {"interval_seconds": True},
        {"command_timeout_seconds": float("nan")},
    ],
)
def test_resource_monitor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="finite and positive"):
        ResourceMonitor(**kwargs)


def test_resource_monitor_rejects_restart_and_stop_before_start():
    monitor = ResourceMonitor()
    with pytest.raises(RuntimeError, match="not been started"):
        monitor.stop()
    monitor.start()
    with pytest.raises(RuntimeError, match="one-shot"):
        monitor.start()
    monitor.stop()


def test_resource_monitor_filters_to_slurm_step_before_cuda_visibility(
    monkeypatch,
):
    fake_psutil = _FakePsutil()
    sampled = threading.Event()

    def run(*_args, **_kwargs):
        sampled.set()
        return types.SimpleNamespace(
            stdout=(
                "0, GPU-zero, 1024, 10\n"
                "1, GPU-one, 2048, 20\n"
                "3, GPU-three, 3072, 30\n"
                "5, GPU-five, 4096, 40\n"
            )
        )

    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: True)
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: (fake_psutil, None))
    monkeypatch.setattr(resource_monitor.subprocess, "run", run)
    monkeypatch.setenv("SLURM_STEP_GPUS", "3,5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,3,5")
    monkeypatch.setenv("OPD_EXPECTED_VISIBLE_GPUS", "2")

    monitor = ResourceMonitor(
        interval_seconds=0.01,
        command_timeout_seconds=0.2,
    ).start()
    assert sampled.wait(timeout=1.0)
    result = monitor.stop()

    assert result.gpu_metrics_available
    assert result.peak_hbm_gib_per_gpu == {"3": 3.0, "5": 4.0}
    assert result.peak_hbm_gib_aggregate == 7.0
    assert result.gpu_utilization_mean_per_gpu == {"3": 30.0, "5": 40.0}
    assert result.gpu_selection_source == "SLURM_STEP_GPUS"
    assert result.gpu_selectors == ("3", "5")
    assert result.expected_gpu_count == 2
    assert not _monitor_threads()


def test_resource_monitor_filters_cuda_uuid_when_no_slurm_step(monkeypatch):
    fake_psutil = _FakePsutil()
    sampled = threading.Event()

    def run(*_args, **_kwargs):
        sampled.set()
        return types.SimpleNamespace(
            stdout=(
                "0, GPU-zero, 1024, 10\n"
                "1, GPU-one, 2048, 20\n"
            )
        )

    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: True)
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: (fake_psutil, None))
    monkeypatch.setattr(resource_monitor.subprocess, "run", run)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one")
    monkeypatch.setenv("OPD_EXPECTED_VISIBLE_GPUS", "1")

    monitor = ResourceMonitor(
        interval_seconds=0.01,
        command_timeout_seconds=0.2,
    ).start()
    assert sampled.wait(timeout=1.0)
    result = monitor.stop()

    assert result.gpu_metrics_available
    assert result.peak_hbm_gib_per_gpu == {"1": 2.0}
    assert result.gpu_selection_source == "CUDA_VISIBLE_DEVICES"
    assert result.gpu_selectors == ("GPU-one",)
    assert result.expected_gpu_count == 1


def test_resource_monitor_marks_assignment_count_mismatch_unavailable(monkeypatch):
    fake_psutil = _FakePsutil()

    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: True)
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: (fake_psutil, None))
    monkeypatch.setattr(
        resource_monitor.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            stdout="0, GPU-zero, 1024, 10\n"
        ),
    )
    monkeypatch.setenv("SLURM_STEP_GPUS", "0,1")
    monkeypatch.setenv("OPD_EXPECTED_VISIBLE_GPUS", "2")

    monitor = ResourceMonitor(
        interval_seconds=0.01,
        command_timeout_seconds=0.2,
    ).start()
    assert monitor._first_sample_event.wait(timeout=1.0)
    result = monitor.stop()

    assert not result.gpu_metrics_available
    assert result.peak_hbm_gib_per_gpu is None
    assert "matched 1 GPUs, expected 2" in result.gpu_error
    assert not _monitor_threads()


def test_resource_monitor_accepts_step_cgroup_local_index_remapping(monkeypatch):
    fake_psutil = _FakePsutil()
    monkeypatch.setattr(resource_monitor, "_is_linux", lambda: True)
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: (fake_psutil, None))
    monkeypatch.setattr(
        resource_monitor.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            stdout=(
                "0, GPU-physical-three, 1024, 10\n"
                "1, GPU-physical-five, 2048, 20\n"
            )
        ),
    )
    monkeypatch.setenv("SLURM_STEP_GPUS", "3,5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("OPD_EXPECTED_VISIBLE_GPUS", "2")

    monitor = ResourceMonitor(
        interval_seconds=0.01,
        command_timeout_seconds=0.2,
    ).start()
    assert monitor._first_sample_event.wait(timeout=1.0)
    result = monitor.stop()

    assert result.gpu_metrics_available
    assert result.peak_hbm_gib_per_gpu == {"0": 1.0, "1": 2.0}
    assert result.peak_hbm_gib_aggregate == 3.0
