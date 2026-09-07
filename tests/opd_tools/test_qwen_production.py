"""Seven-arm Qwen production recipes and immutable launcher contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opd_tools import qwen_production as production
from opd_tools import qwen_training, study
from opd_tools.manifest import canonical_sha256


def values(overrides):
    result = {}
    for item in overrides:
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        assert key not in result, f"ambiguous duplicate override: {key}"
        result[key] = json.loads(value)
    return result


def test_order_objectives_and_historical_registry_are_isolated():
    historical = [spec.as_manifest() for spec in study.ARM_SPECS]
    assert production.ARM_IDS == (
        "hardgrpo_math_s11", "softgrpo_math_s11", "softopd_math_s11",
        "softgrpo_math_opd_s11", "softgrpo_math_opd_posadv_s11",
        "softgrpo_math_opd_current_s11", "softgrpo_math_opd_beta0p1_s11",
    )
    contracts = [production.arm_contract(arm) for arm in production.ARM_IDS]
    assert [row["group_size"] for row in contracts] == [8, 8, 1, 8, 8, 8, 8]
    assert [row["beta_base"] for row in contracts[2:]] == [1, 1, 1, 1, 0.1]
    assert contracts[4]["trajectory_gate"] == "positive_advantage"
    assert contracts[5]["teacher_type"] == "current_actor"
    assert contracts[2]["schedule"] == "constant"
    assert all(row["warmup_iterations"] == 11 for row in contracts[3:])
    assert all(row["protocol"] == production.PROFILE_ID for row in contracts)
    assert [spec.as_manifest() for spec in study.ARM_SPECS] == historical
    assert study.resolve_arm("softgrpo_math_opd_s11").spec.beta_base == 0.001
    with pytest.raises(ValueError):
        qwen_training.profile_overrides("hybrid", 4, "/assets", "/run")


@pytest.mark.parametrize("arm", production.ARM_IDS)
def test_production_preserves_full_training_recipe_and_backend(tmp_path, arm):
    spec = production.resolve_arm(arm)
    config = values(production.production_overrides(arm, tmp_path / "assets", tmp_path / "run"))
    assert config["data.train_batch_size"] == 64
    assert config["data.val_batch_size"] == 128
    assert config["data.max_response_length"] == 8192
    assert config["data.max_prompt_length"] == 2048
    assert config["data.seed"] == 11
    assert config["trainer.total_epochs"] == 1
    assert config["trainer.total_training_steps"] is None
    assert config["trainer.max_rollout_iterations_per_invocation"] is None
    assert config["trainer.n_gpus_per_node"] == 4 and config["ray_init.num_cpus"] == 56
    assert config["actor_rollout_ref.actor.ppo_mini_batch_size"] == 32
    assert config["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 2
    assert config["actor_rollout_ref.actor.ppo_epochs"] == 1
    assert config["actor_rollout_ref.actor.optim.lr"] == 1e-6
    assert config["actor_rollout_ref.actor.optim.weight_decay"] == 0.01
    assert config["actor_rollout_ref.actor.grad_clip"] == 1.0
    assert config["actor_rollout_ref.rollout.tensor_model_parallel_size"] == 1
    assert config["actor_rollout_ref.rollout.n"] == spec.group_size
    assert config["algorithm.opd.beta_base"] == spec.beta_base
    assert config["algorithm.opd.prompt_profile"] == config["data.prompt_profile"] == qwen_training.PROFILE_ID
    assert config["trainer.training_profile"] == production.PROFILE_ID
    assert config["trainer.production_gradient_policy"] == "diagnostic_clipping"
    assert not config["trainer.rollout_integrity.completion_gate_enabled"]
    assert not config["trainer.rollout_integrity.full_dose_gradient_gate_enabled"]
    assert config["trainer.rollout_integrity.enabled"]
    assert config["trainer.rollout_integrity.gate_first_n_iterations"] == 109
    assert config["trainer.val_before_train"] and config["trainer.test_freq"] == 25
    assert config["trainer.save_freq"] == 25
    assert config["trainer.validation_seed"] == 11 and config["trainer.validation_seed_iteration"] == 0
    assert not any("benchmark" in key or "capacity" in key for key in config)
    if spec.rollout_kind == "categorical":
        assert config["actor_rollout_ref.rollout.name"] == "vllm"
        assert config["actor_rollout_ref.model.qwen_replay_backend"] == "disabled"
        assert not config["actor_rollout_ref.rollout.enable_soft_thinking"]
        assert not config["actor_rollout_ref.rollout.require_retained_support"]
        assert not config["actor_rollout_ref.rollout.production_engine_isolation"]
    else:
        assert config["actor_rollout_ref.rollout.name"] == "sglang"
        assert config["actor_rollout_ref.model.qwen_replay_backend"] == "native_fa3_v1"
        assert config["actor_rollout_ref.rollout.require_retained_support"]
        assert config["actor_rollout_ref.rollout.production_engine_isolation"]
        assert config["actor_rollout_ref.rollout.dispatch_mode"] == "bounded_async"
        assert config["actor_rollout_ref.rollout.max_running_requests"] == 32
        assert config["actor_rollout_ref.rollout.async_queue_size"] == 64
    assert config["actor_rollout_ref.actor.use_kl_loss"] == (arm != "softopd_math_s11")


@pytest.mark.parametrize("arm", production.ARM_IDS)
@pytest.mark.parametrize("phase", ["production", "uninterrupted", "split", "resume"])
def test_actual_hydra_composition_and_resume_horizon(tmp_path, arm, phase):
    import hydra
    from omegaconf import OmegaConf
    from verl.opd.config import OPDConfig

    resume = tmp_path / "run/prologue/resume/training/global_step_1" if phase == "resume" else None
    overrides = production.production_overrides(arm, tmp_path / "assets", tmp_path / "run", phase=phase, resume_from_path=resume)
    directory = Path(qwen_training.__file__).resolve().parents[1] / "verl-0.4.x/verl/trainer/config"
    with hydra.initialize_config_dir(config_dir=str(directory), version_base=None):
        config = hydra.compose(config_name="ppo_trainer", overrides=overrides)
    opd = OPDConfig.from_mapping(OmegaConf.to_container(config.algorithm.opd, resolve=True))
    assert opd.prompt_profile == qwen_training.PROFILE_ID
    assert config.trainer.total_training_steps is None and config.trainer.total_epochs == 1
    assert config.trainer.max_rollout_iterations_per_invocation == (None if phase == "production" else 2 if phase == "uninterrupted" else 1)
    assert config.trainer.production_mode and config.trainer.production_arm_id == arm
    assert config.actor_rollout_ref.rollout.qwen_replay_backend == config.actor_rollout_ref.model.qwen_replay_backend
    if production.resolve_arm(arm).opd_mode == "auxiliary":
        assert config.algorithm.opd.warmup_fraction == 0.1
        assert config.algorithm.opd.schedule == "warmup_constant"
    if phase != "production":
        assert not config.trainer.val_before_train and config.trainer.test_freq == -1
        assert config.trainer.save_freq == 1
        assert config.trainer.project_name == production.PRODUCTION_PROJECT + "-prologue"
    if phase == "resume":
        assert config.trainer.resume_mode == "resume_path"
        assert config.trainer.resume_from_path == str(resume)


@pytest.mark.parametrize("arm", production.ARM_IDS[3:])
def test_disposable_full_dose_and_zero_dose_match_intended_objectives(tmp_path, arm):
    full = values(production.production_overrides(arm, tmp_path / "assets", tmp_path / "run", phase="full_dose"))
    zero = values(production.production_overrides(arm, tmp_path / "assets", tmp_path / "run", phase="zero_dose"))
    direct = values(production.production_overrides(arm, tmp_path / "assets", tmp_path / "run", phase="uninterrupted"))
    assert full["algorithm.opd.enabled"] and full["algorithm.opd.schedule"] == "constant"
    assert full["algorithm.opd.beta_base"] == production.resolve_arm(arm).beta_base
    assert full["trainer.max_rollout_iterations_per_invocation"] == 1
    assert full["trainer.total_training_steps"] is None
    assert not zero["algorithm.opd.enabled"]
    assert zero["actor_rollout_ref.rollout.n"] == 8
    for key in direct:
        if key.startswith(("data.", "actor_rollout_ref.actor.")):
            assert zero[key] == direct[key], key
    assert production.phase_metadata(arm, tmp_path, "zero_dose")["effective_arm_id"] == "softgrpo_math_s11"


def test_phase_outputs_are_isolated_but_split_resume_share_checkpoint_root(tmp_path):
    arm = production.ARM_IDS[3]
    phases = {name: production.phase_metadata(arm, tmp_path, name) for name in production.PHASES}
    assert phases["split"]["run_dir"] == phases["resume"]["run_dir"]
    assert phases["split"]["output"] != phases["resume"]["output"]
    assert len({row["output"] for row in phases.values()}) == len(production.PHASES)
    assert phases["production"]["run_dir"] == str(tmp_path / "production/training")
    with pytest.raises(ValueError, match="explicit committed checkpoint"):
        production.production_overrides(arm, tmp_path, tmp_path, phase="resume")
    with pytest.raises(ValueError, match="only resume or production"):
        production.production_overrides(arm, tmp_path, tmp_path, phase="split", resume_from_path=tmp_path)
    for baseline in production.ARM_IDS[:3]:
        with pytest.raises(ValueError, match="only applies to hybrid"):
            production.production_overrides(baseline, tmp_path, tmp_path, phase="full_dose")


@pytest.fixture
def manifest_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    config = source / "3rdparty/SofT-GRPO/verl-0.4.x/verl/trainer/config/ppo_trainer.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("committed source configuration")
    assets = tmp_path / "assets"
    assets.mkdir()
    asset_manifest = production._seal({"profile_id": qwen_training.PROFILE_ID, "model": production.MODEL})
    (assets / "manifest.json").write_text(json.dumps(asset_manifest))
    monkeypatch.setattr(qwen_training, "verify", lambda path: asset_manifest)
    monkeypatch.setattr(production, "_verify_source", lambda *args: None)
    return dict(assets_root=assets, study_root=tmp_path / "study", source_root=source,
                parent_commit="a" * 40, fork_commit="b" * 40, soft_env=tmp_path / "soft env", hard_env=tmp_path / "hard env")


def test_manifest_materialization_authenticates_source_assets_order_and_environment(manifest_inputs):
    manifest = production.materialize_manifest(**manifest_inputs)
    path = manifest_inputs["study_root"] / "manifest.json"
    assert production.materialize_manifest(**manifest_inputs) == manifest
    assert production.verify_manifest(path) == manifest
    assert production.verify_manifest(path, verify_assets=False, verify_source=False) == manifest
    assert [row["arm_id"] for row in manifest["arms"]] == list(production.ARM_IDS)
    assert manifest["resources"] == {"gpus": 4, "cpus": 56, "memory_gib": 768, "time_limit_seconds": 129600, "exclusive": False}
    assert manifest["prologue_limit_seconds"] == 7200
    assert len({row["wandb_run_id"] for row in manifest["arms"]}) == 7
    for index, arm in enumerate(manifest["arms"]):
        assert arm["environment_root"] == str(manifest_inputs["hard_env" if index == 0 else "soft_env"])
        command = production.phase_command(manifest, arm["arm_id"], "production")
        assert command[:3] == [arm["python_bin"], "-m", "verl.trainer.main_ppo"]
        assert values(command[3:])["custom_reward_function.path"] == str(manifest_inputs["source_root"] / "3rdparty/SofT-GRPO/opd_tools/reward.py")
        assert arm["production_overrides_sha256"] == canonical_sha256(command[3:])
    changed = production.build_manifest(**{**manifest_inputs, "hard_env": manifest_inputs["hard_env"] / "new"})
    assert changed["arms"][0]["wandb_run_id"] != manifest["arms"][0]["wandb_run_id"]


@pytest.mark.parametrize("tamper", ["beta", "resources", "order", "profile", "config", "arm_file"])
def test_self_consistently_resealed_manifest_drift_is_rejected(manifest_inputs, tamper):
    manifest = production.materialize_manifest(**manifest_inputs)
    path = manifest_inputs["study_root"] / "manifest.json"
    if tamper == "beta":
        manifest["arms"][3]["contract"]["beta_base"] = 0.001
    elif tamper == "resources":
        manifest["resources"]["gpus"] = 8
    elif tamper == "order":
        manifest["arms"].reverse()
    elif tamper == "profile":
        manifest["profile_id"] = qwen_training.PROFILE_ID
    elif tamper == "config":
        (manifest_inputs["source_root"] / "3rdparty/SofT-GRPO/verl-0.4.x/verl/trainer/config/ppo_trainer.yaml").write_text("changed")
    else:
        (manifest_inputs["study_root"] / "arms" / production.ARM_IDS[0] / "profile.json").write_text("{}")
    manifest.pop("manifest_content_sha256")
    path.write_text(json.dumps(production._seal(manifest)))
    with pytest.raises(ValueError):
        production.verify_manifest(path)


def test_manifest_refuses_overwrite_symlink_and_source_mismatch(manifest_inputs, tmp_path):
    production.materialize_manifest(**manifest_inputs)
    with pytest.raises(ValueError, match="overwrite"):
        production.materialize_manifest(**{**manifest_inputs, "fork_commit": "c" * 40})
    with pytest.raises(ValueError, match="Gitlink"):
        production.build_manifest(**manifest_inputs, parent_gitlink="c" * 40)
    with pytest.raises(ValueError, match="full lowercase"):
        production.build_manifest(**{**manifest_inputs, "parent_commit": "short"})
    link = tmp_path / "link.json"
    link.symlink_to(manifest_inputs["study_root"] / "manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        production._write_immutable(link, {})
    with pytest.raises(ValueError, match="regular file"):
        production.verify_manifest(link)


def test_source_verification_rejects_untracked_source(monkeypatch, tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        args = command[3:]
        if args == ["rev-parse", "HEAD"]:
            output = "b" * 40 if command[2].endswith("SofT-GRPO") else "a" * 40
        elif args[0] == "ls-tree":
            output = "160000 commit " + "b" * 40 + "\t3rdparty/SofT-GRPO"
        else:
            assert args == ["status", "--porcelain", "--untracked-files=all"]
            output = "?? uncommitted_source.py"
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(production.subprocess, "run", run)
    with pytest.raises(ValueError, match="untracked"):
        production._verify_source(tmp_path, "a" * 40, "b" * 40)
    assert calls


def test_asset_cli_reuses_authenticated_qwen_preparation(monkeypatch, tmp_path, capsys):
    called = []
    monkeypatch.setattr(qwen_training, "prepare", lambda root, cache: called.append((root, cache)) or {"ready": True})
    assert production.main(["prepare-assets", "--assets-root", str(tmp_path / "assets"), "--cache-dir", str(tmp_path / "cache")]) == 0
    assert json.loads(capsys.readouterr().out) == {"ready": True}
    assert called == [(tmp_path / "assets", tmp_path / "cache")]


def test_cli_manifest_verify_and_phase_argv(manifest_inputs, capsys):
    argv = ["materialize"]
    for key, value in manifest_inputs.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    assert production.main(argv) == 0
    manifest = json.loads(capsys.readouterr().out)
    path = manifest_inputs["study_root"] / "manifest.json"
    assert production.main(["verify", "--manifest", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == manifest
    assert production.main(["command", "--manifest", str(path), "--arm", production.ARM_IDS[3], "--phase", "full_dose"]) == 0
    command = json.loads(capsys.readouterr().out)
    assert values(command[3:])["algorithm.opd.beta_base"] == 1.0
    assert production.main(["overrides", "--manifest", str(path), "--arm", production.ARM_IDS[2]]) == 0
    assert values(json.loads(capsys.readouterr().out))["actor_rollout_ref.rollout.n"] == 1
