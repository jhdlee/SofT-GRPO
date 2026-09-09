"""Failure recovery must reuse authenticated production state and stay bounded."""

import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from opd_tools import qwen_production_controller as controller
from opd_tools.manifest import file_sha256

_expected_provenance = controller._recovery_expected_provenance


@pytest.fixture
def recovery_run(tmp_path, monkeypatch):
    manifest = {"profile_id": "qwen3-math-seven-arm-lora-fa3-v1",
                "parent_commit": "a" * 40, "fork_commit": "b" * 40}
    manifest_path = tmp_path / "manifest.json"
    controller.write_json(manifest_path, manifest)
    root = tmp_path / "arm"
    production = root / "production"
    training = production / "training"
    training.mkdir(parents=True)
    row = {"arm_id": "softopd_math_s11", "run_root": str(root),
           "wandb_run_id": "production-test",
           "production_overrides": [],
           "phases": {"production": {"directory": str(production),
                       "run_dir": str(training), "output": str(production / "measurement.json")}}}
    admission = {"manifest_content_sha256": "c" * 64}
    provenance = {"resume_identity_sha256": "d" * 64}
    selected = {"global_step": 25, "resume_from_path": str(training / "global_step_25"),
                "checkpoint_manifest_sha256": "e" * 64,
                "admission_sha256": admission["manifest_content_sha256"],
                "resume_provenance_sha256": provenance["resume_identity_sha256"]}

    def segment(restart=0, **updates):
        value = {"arm_id": row["arm_id"], "job_id": "42", "restart_count": restart,
                 "submission_manifest_sha256": file_sha256(manifest_path),
                 "parent_commit": manifest["parent_commit"], "fork_commit": manifest["fork_commit"],
                 "status": "failed", "phases": {"production": {"status": "failed"}}}
        value.update(updates)
        controller.write_json(root / f"segment-{restart}.json", value)
        return value

    monkeypatch.setattr(controller, "verify_admission", lambda *args: admission)
    monkeypatch.setattr(controller, "_recovery_expected_provenance", lambda *args: provenance)
    segment()
    return SimpleNamespace(manifest=manifest, manifest_path=manifest_path, row=row, root=root,
                           training=training, admission=admission, provenance=provenance,
                           selected=selected, segment=segment)


def prepare(run, restart=0, exit_code=1):
    return controller.prepare_recovery(run.manifest_path, run.manifest, run.row,
        job_id="42", restart_count=restart, failure_exit_code=exit_code)


def mock_selected(run, monkeypatch):
    monkeypatch.setattr(controller, "select_recovery_checkpoint", lambda *args: copy.deepcopy(run.selected))


def test_selection_reauthenticates_the_production_boundary(recovery_run, monkeypatch):
    run = recovery_run
    path = run.training / "global_step_25"
    path.mkdir()
    controller.write_json(path / "checkpoint_manifest.json", {"global_step": 25})
    calls = []

    def find(root, expected, *, require_semantic):
        assert root == run.training and expected == run.provenance and require_semantic
        return str(path), {"global_step": 25}

    def authenticate(root, step, **kwargs):
        calls.append((root, step, kwargs))
        return {"resume_provenance_sha256": run.provenance["resume_identity_sha256"]}

    monkeypatch.setattr(controller, "_find_recovery_checkpoint", find)
    monkeypatch.setattr(controller, "authenticate_checkpoint", authenticate)
    result = controller.select_recovery_checkpoint(run.manifest_path, run.manifest, run.row)
    assert result["resume_from_path"] == str(path)
    assert result["checkpoint_manifest_sha256"] == file_sha256(path / "checkpoint_manifest.json")
    assert result["admission_sha256"] == run.admission["manifest_content_sha256"]
    assert calls == [(run.training, 25, {"arm_id": run.row["arm_id"], "require_semantic": True,
                                         "expected_provenance": run.provenance})]


@pytest.mark.parametrize("selected", [None, ("/other/global_step_25", {"global_step": 25}),
                                      ("/other/global_step_0", {"global_step": 0}),
                                      ("/other/global_step_110", {"global_step": 110}),
                                      ("/other/global_step_1", {"global_step": True})])
def test_no_checkpoint_or_wrong_boundary_cannot_recover(recovery_run, monkeypatch, selected):
    run = recovery_run
    monkeypatch.setattr(controller, "_find_recovery_checkpoint", lambda *args, **kwargs: selected)
    monkeypatch.setattr(controller, "authenticate_checkpoint", lambda *args, **kwargs: pytest.fail("invalid boundary"))
    with pytest.raises(ValueError):
        controller.select_recovery_checkpoint(run.manifest_path, run.manifest, run.row)


def test_prologue_checkpoint_root_cannot_be_used_for_production(recovery_run, monkeypatch):
    run = recovery_run
    run.row["phases"]["production"]["run_dir"] = str(run.root / "prologue/uninterrupted/training")
    monkeypatch.setattr(controller, "_find_recovery_checkpoint", lambda *args, **kwargs: pytest.fail("prologue state"))
    with pytest.raises(ValueError, match="outside"):
        controller.select_recovery_checkpoint(run.manifest_path, run.manifest, run.row)


def test_failed_admission_prevents_checkpoint_recovery(recovery_run, monkeypatch):
    run = recovery_run
    def fail(*args):
        raise ValueError("admission evidence changed")
    monkeypatch.setattr(controller, "verify_admission", fail)
    with pytest.raises(ValueError, match="admission"):
        controller.select_recovery_checkpoint(run.manifest_path, run.manifest, run.row)


def test_retry_requests_are_idempotent_and_stop_after_three_without_progress(recovery_run, monkeypatch):
    run = recovery_run
    mock_selected(run, monkeypatch)
    for restart in range(3):
        run.segment(restart)
        result = prepare(run, restart)
        assert result["consecutive_failure_retries"] == restart + 1
        assert result["status"] == "recovery_ready"
        path = run.root / "recovery" / f"restart-{restart}.json"
        original = path.read_bytes()
        assert prepare(run, restart, exit_code=None) == result
        assert path.read_bytes() == original
    run.segment(3)
    with pytest.raises(ValueError, match="exhausted"):
        prepare(run, 3)
    assert not (run.root / "recovery/restart-3.json").exists()


@pytest.mark.parametrize("next_step,expected_count", [(50, 1), (25, 2), (10, 2)])
def test_only_a_newer_checkpoint_resets_retry_budget(recovery_run, monkeypatch, next_step, expected_count):
    run = recovery_run
    mock_selected(run, monkeypatch)
    prepare(run)
    run.selected.update(global_step=next_step, resume_from_path=str(run.training / f"global_step_{next_step}"))
    run.segment(1)
    assert prepare(run, 1)["consecutive_failure_retries"] == expected_count


def test_existing_request_cannot_silently_switch_checkpoint(recovery_run, monkeypatch):
    run = recovery_run
    mock_selected(run, monkeypatch)
    prepare(run)
    run.selected["checkpoint_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="existing recovery request differs"):
        prepare(run)


@pytest.mark.parametrize("key,value", [("job_id", "43"), ("restart_count", 1),
    ("arm_id", "softgrpo_math_s11"), ("parent_commit", "f" * 40),
    ("fork_commit", "f" * 40), ("submission_manifest_sha256", "f" * 64)])
def test_recovery_refuses_a_segment_from_another_allocation_or_source(recovery_run, monkeypatch, key, value):
    run = recovery_run
    mock_selected(run, monkeypatch)
    report = run.segment()
    report[key] = value
    controller.write_json(run.root / "segment-0.json", report)
    with pytest.raises(ValueError, match="another source, arm or allocation"):
        prepare(run)


@pytest.mark.parametrize("exit_code", [0, -9, 75, 130, 143, True, "1"])
def test_completion_continuation_and_cancellation_exits_do_not_retry(recovery_run, monkeypatch, exit_code):
    mock_selected(recovery_run, monkeypatch)
    with pytest.raises(ValueError, match="not failure retries"):
        prepare(recovery_run, exit_code=exit_code)


@pytest.mark.parametrize("updates", [
    {"phases": {"uninterrupted": {"status": "failed"}}},
    {"cleanup_error": "trainer could not be stopped"},
    {"persistence_error": "disk full"},
    {"failure": {"category": "authentication", "error": "payload digest differs"}},
    *[{"failure": {"error": message}} for message in
      ("interrupted by signal 15", "interrupted by signal 2", "KeyboardInterrupt", "cancelled", "canceled")],
])
def test_nonproduction_and_unsafe_failures_do_not_retry(recovery_run, monkeypatch, updates):
    run = recovery_run
    mock_selected(run, monkeypatch)
    run.segment(**updates)
    with pytest.raises(ValueError):
        prepare(run)
    assert not (run.root / "recovery").exists()


def test_recovered_final_checkpoint_is_marked_complete(recovery_run, monkeypatch):
    run = recovery_run
    mock_selected(run, monkeypatch)
    run.selected.update(global_step=109, resume_from_path=str(run.training / "global_step_109"))
    assert prepare(run)["status"] == "complete"


def test_cleanup_ignores_reused_pid_without_signalling_it(recovery_run, monkeypatch):
    run = recovery_run
    run.segment(child_process={"pid": 777, "process_group": 777,
                               "hostname": socket.gethostname(), "start_ticks": 10})
    monkeypatch.setattr(controller, "_process_identity", lambda pid:
                        {"start_ticks": 11, "process_group": 888, "state": "S"} if pid == 777 else None)
    def nonexistent_group(group, signum):
        assert signum == 0, "must not send a signal to a reused PID"
        raise ProcessLookupError()
    monkeypatch.setattr(os, "killpg", nonexistent_group)
    result = controller.cleanup_trainer(run.manifest_path, run.manifest, run.row, job_id="42", restart_count=0)
    assert result["status"] == "recorded_trainer_exited"


def test_cleanup_refuses_another_allocation_owner(recovery_run, monkeypatch):
    run = recovery_run
    pid = os.getpid()
    owned = {"pid": pid, "process_group": pid, "hostname": socket.gethostname(), "start_ticks": 10}
    run.segment(child_process=owned)
    monkeypatch.setattr(controller, "_process_identity", lambda pid: {**owned, "state": "S"})
    original_read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"OPD_QPROD_TRAINER_OWNER=another-job\0"
                        if str(path) == f"/proc/{pid}/environ" else original_read_bytes(path))
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("must not signal another owner"))
    with pytest.raises(ValueError, match="another allocation"):
        controller.cleanup_trainer(run.manifest_path, run.manifest, run.row, job_id="42", restart_count=0)


def test_cleanup_signals_only_authenticated_trainer_group(recovery_run, monkeypatch):
    run = recovery_run
    pid = os.getpid()
    owned = {"pid": pid, "process_group": pid, "hostname": socket.gethostname(), "start_ticks": 10}
    run.segment(child_process=owned)
    monkeypatch.setattr(controller, "_process_identity", lambda pid: {**owned, "state": "S"})
    owner = f"OPD_QPROD_TRAINER_OWNER={file_sha256(run.manifest_path)}:42:0\0".encode()
    original_read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: owner if str(path) == f"/proc/{pid}/environ"
                        else original_read_bytes(path))
    calls = []
    def killpg(group, signum):
        calls.append((group, signum))
        if signum == 0:
            raise ProcessLookupError()
    monkeypatch.setattr(os, "killpg", killpg)
    result = controller.cleanup_trainer(run.manifest_path, run.manifest, run.row, job_id="42", restart_count=0)
    assert result["status"] == "trainer_stopped"
    assert calls == [(pid, signal.SIGTERM), (pid, 0)]


def provenance_measurement(run, monkeypatch):
    config = {"data": {"train_batch_size": 64}, "trainer": {"n_gpus_per_node": 4,
              "resume_mode": "resume_path", "resume_from_path": "/previous/checkpoint",
              "production_output": "/next/measurement.json", "requeue_signal_file": "/new/signal"},
              "actor_rollout_ref": {"actor": {"optim": {"total_training_steps": 109}}},
              "critic": {"optim": {"total_training_steps": 109}}, "algorithm": {"opd": {"enabled": True}}}
    run.row["production_overrides"] = ["data.train_batch_size=64", "trainer.n_gpus_per_node=4",
        "trainer.resume_mode=disable", "trainer.resume_from_path=null", "trainer.production_output=old",
        "trainer.requeue_signal_file=old", "hydra.run.dir=unused", "++algorithm.opd.enabled=true",
        "actor_rollout_ref.actor.optim.total_training_steps=null", "critic.optim.total_training_steps=null"]
    provenance = {"source": {"commit": run.manifest["fork_commit"]}, "identity": "expected"}
    built, compared = [], []
    fake = SimpleNamespace(build_checkpoint_provenance=lambda value: built.append(value) or provenance,
        assert_checkpoint_provenance_matches=lambda observed, expected: compared.append((observed, expected)))
    monkeypatch.setitem(sys.modules, "verl.opd.provenance", fake)
    measured = {"phase": "production", "arm_id": run.row["arm_id"],
                "configuration": config, "checkpoint_provenance": copy.deepcopy(provenance)}
    path = Path(run.row["phases"]["production"]["output"])
    controller.write_json(path, measured)
    return path, measured, provenance, built, compared


def test_recovery_rebuilds_provenance_and_checks_recipe_with_resolved_horizons(recovery_run, monkeypatch):
    run = recovery_run
    _, measured, provenance, built, compared = provenance_measurement(run, monkeypatch)
    result = _expected_provenance(run.manifest, run.row)
    assert result == provenance
    assert built == [measured["configuration"]]
    assert compared == [(measured["checkpoint_provenance"], provenance)]


@pytest.mark.parametrize("mutation", ["phase", "arm", "batch", "gpu_count", "missing_override",
                                     "boolean_type", "horizon", "source", "missing_provenance"])
def test_recovery_rejects_wrong_resolved_production_identity(recovery_run, monkeypatch, mutation):
    run = recovery_run
    path, measured, provenance, _, _ = provenance_measurement(run, monkeypatch)
    if mutation == "phase": measured["phase"] = "split"
    elif mutation == "arm": measured["arm_id"] = "another_arm"
    elif mutation == "batch": measured["configuration"]["data"]["train_batch_size"] = 16
    elif mutation == "gpu_count": measured["configuration"]["trainer"]["n_gpus_per_node"] = 2
    elif mutation == "missing_override": del measured["configuration"]["data"]["train_batch_size"]
    elif mutation == "boolean_type": measured["configuration"]["algorithm"]["opd"]["enabled"] = 1
    elif mutation == "horizon": measured["configuration"]["critic"]["optim"]["total_training_steps"] = 2
    elif mutation == "source": provenance["source"]["commit"] = "f" * 40
    elif mutation == "missing_provenance": del measured["checkpoint_provenance"]
    controller.write_json(path, measured)
    with pytest.raises(ValueError):
        _expected_provenance(run.manifest, run.row)


def restarted_controller(run, restart=2):
    value = controller.ProductionController.__new__(controller.ProductionController)
    value.args = SimpleNamespace(manifest=run.manifest_path, arm=run.row["arm_id"], signal_file=run.root / "signal")
    value.root = run.root
    value.manifest = run.manifest
    value.row = run.row
    value.child = None
    value.restart = restart
    value.report = {"job_id": "42", "phases": {}}
    value.phase = "initialization"
    value.deadline = 10**12
    value.persist = lambda: None
    value.admission = lambda: pytest.fail("restart must not repeat admission")
    return value


@pytest.mark.parametrize("candidate", [None, {"restart_count": 0, "job_id": "42"},
                                     {"restart_count": 1, "job_id": "another-job"}])
def test_restart_uses_latest_valid_checkpoint_when_clean_continuation_is_absent_or_stale(
        recovery_run, monkeypatch, candidate):
    run = recovery_run
    value = restarted_controller(run)
    if candidate is not None:
        controller.write_json(run.root / "continuation.json", candidate, seal=True)
    calls = []
    monkeypatch.setattr(controller, "verify_continuation", lambda *args: pytest.fail("stale continuation"))
    monkeypatch.setattr(controller, "prepare_recovery", lambda *args, **kwargs: calls.append(kwargs) or run.selected)
    monkeypatch.setattr(controller, "authenticate_checkpoint", lambda *args, **kwargs: {})
    value.args.signal_file.write_text("old signal")
    invoked = []
    value.invoke = lambda phase, **kwargs: invoked.append((phase, kwargs)) or {"iterations": [{"rollout_iteration": 108}]}
    assert value.run() == 0
    assert calls == [{"job_id": "42", "restart_count": 1}]
    assert invoked == [("production", {"resume_from_path": Path(run.selected["resume_from_path"])})]
    assert value.report["resume_reason"] == "latest_valid_checkpoint"
    assert not value.args.signal_file.exists()


def test_restart_keeps_a_current_authenticated_clean_continuation(recovery_run, monkeypatch):
    run = recovery_run
    value = restarted_controller(run)
    controller.write_json(run.root / "continuation.json", {"restart_count": 1, "job_id": "42"}, seal=True)
    monkeypatch.setattr(controller, "verify_continuation", lambda *args: {"global_step": 50})
    monkeypatch.setattr(controller, "prepare_recovery", lambda *args, **kwargs: pytest.fail("clean continuation should not consume retry budget"))
    monkeypatch.setattr(controller, "authenticate_checkpoint", lambda *args, **kwargs: {})
    resumed = []
    value.invoke = lambda phase, **kwargs: resumed.append(kwargs["resume_from_path"]) or {"iterations": [{"rollout_iteration": 108}]}
    assert value.run() == 0
    assert resumed == [run.training / "global_step_50"]
    assert value.report["resume_reason"] == "clean_continuation"


@pytest.mark.parametrize("failure", ["broken_json", "unusable_checkpoint"])
def test_invalid_continuation_can_fall_back_to_independently_verified_recovery(recovery_run, monkeypatch, failure):
    run = recovery_run
    value = restarted_controller(run)
    path = run.root / "continuation.json"
    if failure == "broken_json":
        path.write_text("{")
    else:
        controller.write_json(path, {"restart_count": 1, "job_id": "42"}, seal=True)
    def invalid(*args):
        raise RuntimeError("continuation checkpoint is incomplete")
    monkeypatch.setattr(controller, "verify_continuation", invalid)
    monkeypatch.setattr(controller, "prepare_recovery", lambda *args, **kwargs: run.selected)
    monkeypatch.setattr(controller, "authenticate_checkpoint", lambda *args, **kwargs: {})
    resumed = []
    value.invoke = lambda phase, **kwargs: resumed.append(kwargs["resume_from_path"]) or {"iterations": [{"rollout_iteration": 108}]}
    assert value.run() == 0
    assert resumed == [Path(run.selected["resume_from_path"])]
    assert value.report["unusable_continuation"]
    assert value.report["resume_reason"] == "latest_valid_checkpoint"


def test_recovered_step_109_finishes_without_restarting_training(recovery_run, monkeypatch):
    run = recovery_run
    value = restarted_controller(run)
    monkeypatch.setattr(controller, "prepare_recovery", lambda *args, **kwargs: {"global_step": 109})
    repaired = []
    monkeypatch.setattr(controller, "finalize_recovered_checkpoint", lambda *args:
                        repaired.append(args) or {"status": "complete", "global_step": 109})
    value.invoke = lambda *args, **kwargs: pytest.fail("completed training must not repeat")
    assert value.run() == 0
    assert repaired == [(run.manifest, run.row, {"global_step": 109})]
    assert value.report["final_checkpoint_repair"] == {"status": "complete", "global_step": 109}
    assert value.report["status"] == "complete"
    assert value.report["recovered_completed_checkpoint"] is True


@pytest.mark.parametrize("changed", [None, "digest", "provenance", "step", "path"])
def test_final_recovery_authenticates_before_repair_and_retention(recovery_run, monkeypatch, changed):
    run = recovery_run
    path = run.training / "global_step_109"
    path.mkdir()
    controller.write_json(path / "checkpoint_manifest.json", {"global_step": 109})
    selected = {**run.selected, "global_step": 109, "resume_from_path": str(path),
                "checkpoint_manifest_sha256": file_sha256(path / "checkpoint_manifest.json")}
    checkpoint = {"global_step": 109, "resume_provenance_sha256": selected["resume_provenance_sha256"]}
    calls = []
    def authenticate(root, step, **kwargs):
        calls.append(("authenticate", root, step, kwargs))
        return checkpoint
    def repair(root, **kwargs):
        calls.append(("repair", root, kwargs))
        return ["abandoned temporary directory"]
    def prune(root, **kwargs):
        calls.append(("prune", root, kwargs))
        return ["obsolete checkpoint"]
    monkeypatch.setattr(controller, "authenticate_checkpoint", authenticate)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", SimpleNamespace(
        _repair_checkpoint_history=repair, _prune_committed_checkpoints=prune))
    if changed == "digest": selected["checkpoint_manifest_sha256"] = "f" * 64
    elif changed == "provenance": selected["resume_provenance_sha256"] = "f" * 64
    elif changed == "step": selected["global_step"] = 100
    elif changed == "path": selected["resume_from_path"] = str(run.root / "prologue/global_step_109")
    if changed:
        with pytest.raises(ValueError, match="final recovery"):
            controller.finalize_recovered_checkpoint(run.manifest, run.row, selected)
        assert all(call[0] == "authenticate" for call in calls)
    else:
        result = controller.finalize_recovered_checkpoint(run.manifest, run.row, selected)
        assert calls == [
            ("authenticate", run.training, 109, {"arm_id": run.row["arm_id"],
                "require_semantic": True, "expected_provenance": run.provenance}),
            ("repair", str(run.training), {"resumed_manifest": checkpoint, "best_mode": "max"}),
            ("prune", str(run.training), {"keep_latest": 2}),
        ]
        assert result == {"status": "complete", "global_step": 109,
            "history_removed": ["abandoned temporary directory"], "retention_removed": ["obsolete checkpoint"]}


def test_failed_final_metadata_repair_does_not_report_success(recovery_run, monkeypatch):
    run = recovery_run
    value = restarted_controller(run)
    monkeypatch.setattr(controller, "prepare_recovery", lambda *args, **kwargs: {"global_step": 109})
    def failed(*args):
        raise OSError("checkpoint metadata publication failed")
    monkeypatch.setattr(controller, "finalize_recovered_checkpoint", failed)
    value.invoke = lambda *args, **kwargs: pytest.fail("completed training must not repeat")
    assert value.run() == 1
    assert value.report["status"] == "failed"
    assert not value.report.get("recovered_completed_checkpoint")


def test_cleanup_stops_a_real_owned_child_session(recovery_run):
    run = recovery_run
    owner = f"{file_sha256(run.manifest_path)}:42:0"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             start_new_session=True, env={**os.environ, "OPD_QPROD_TRAINER_OWNER": owner})
    reaper = threading.Thread(target=child.wait, daemon=True)
    reaper.start()
    try:
        run.segment(child_process=controller._process_identity(child.pid))
        result = controller.cleanup_trainer(run.manifest_path, run.manifest, run.row, job_id="42", restart_count=0)
        reaper.join(timeout=5)
        assert result["status"] == "trainer_stopped"
        assert child.returncode == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
        reaper.join(timeout=5)


@pytest.mark.parametrize("foreign_owner", [False, True])
def test_cleanup_handles_real_orphan_group_and_refuses_foreign_members(recovery_run, foreign_owner):
    """A separate subreaper owns the orphan, so the test leaves no zombies."""
    run = recovery_run
    expected_owner = f"{file_sha256(run.manifest_path)}:42:0"
    child_owner = "foreign-allocation" if foreign_owner else expected_owner
    leader_code = "\n".join([
        "import json, os, socket, subprocess, sys",
        "from pathlib import Path",
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],",
        "    env={**os.environ, 'OPD_QPROD_TRAINER_OWNER': " + repr(child_owner) + "})",
        "fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()",
        "print(json.dumps({'pid': os.getpid(), 'process_group': os.getpgrp(),",
        "    'hostname': socket.gethostname(), 'start_ticks': int(fields[19]),",
        "    'child_pid': child.pid}), flush=True)",
    ])
    wrapper_code = "\n".join([
        "import ctypes, json, os, subprocess, sys",
        "assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0",  # PR_SET_CHILD_SUBREAPER
        "leader = subprocess.Popen([sys.executable, '-c', " + repr(leader_code) + "],",
        "    stdout=subprocess.PIPE, text=True, start_new_session=True)",
        "identity = json.loads(leader.stdout.readline())",
        "assert leader.wait() == 0",
        "print(json.dumps(identity), flush=True)",
        "pid, status = os.waitpid(-1, 0)",
        "print(json.dumps({'pid': pid, 'status': status}), flush=True)",
    ])
    wrapper = subprocess.Popen([sys.executable, "-c", wrapper_code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               env={**os.environ, "OPD_QPROD_TRAINER_OWNER": expected_owner})
    owned = None
    try:
        line = wrapper.stdout.readline()
        assert line, wrapper.stderr.read()
        owned = json.loads(line)
        child_pid = owned.pop("child_pid")
        assert controller._process_identity(owned["pid"]) is None
        run.segment(child_process=owned)
        if foreign_owner:
            with pytest.raises(ValueError, match="allocation|ownership|owner|foreign"):
                controller.cleanup_trainer(run.manifest_path, run.manifest, run.row, job_id="42", restart_count=0)
            assert controller._process_identity(child_pid)["state"] != "Z"
        else:
            result = controller.cleanup_trainer(run.manifest_path, run.manifest, run.row, job_id="42", restart_count=0)
            assert result["status"] == "trainer_stopped"
            stdout, stderr = wrapper.communicate(timeout=5)
            assert wrapper.returncode == 0, stderr
            ended = json.loads(stdout)
            assert ended["pid"] == child_pid
            assert os.WIFSIGNALED(ended["status"]) and os.WTERMSIG(ended["status"]) == signal.SIGTERM
    finally:
        if owned is not None:
            try:
                os.killpg(owned["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if wrapper.poll() is None:
            try:
                wrapper.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                wrapper.kill()
                wrapper.communicate(timeout=5)
