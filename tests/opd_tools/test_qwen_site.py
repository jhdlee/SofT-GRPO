"""Cluster identity and shared-storage containment without GPU imports."""

import importlib.util
from pathlib import Path

import pytest

from opd_tools import qwen_site


@pytest.fixture
def shared(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setattr(qwen_site, "shared_storage_root", lambda: root)
    return root


def test_h200_profile_resolves_shared_storage_and_does_not_guess_wandb_identity(shared):
    site = qwen_site.resolve_site("mbzuai-h200")
    assert site["artifact_root"] == str(shared / "opd-latent-reasoning")
    assert site["scheduler"] == dict(account="k2m", partition="main", qos="k2m", constraint="nvidia_h200")
    assert site["gpu"] == dict(name_contains="H200", compute_capability=[9, 0])
    assert site["modules"] == []
    assert site["production_prologue_limit_seconds"] == 10800
    assert site["wandb_entity"] is None
    explicit = qwen_site.resolve_site("mbzuai-h200", shared / "custom", "research-team")
    assert explicit["wandb_entity"] == "research-team"
    assert qwen_site.site_from_manifest({"site": explicit}) == explicit


def test_missing_site_retains_marlowe_defaults():
    site = qwen_site.site_from_manifest({})
    assert site["site_id"] == "marlowe-h100"
    assert site["scheduler"]["account"] is None  # Historical per-arm accounts remain authoritative.
    assert site["gpu"]["name_contains"] == "H100"
    assert site["wandb_entity"] == "jhdlee"
    assert site["artifact_root"] is None
    assert site["production_prologue_limit_seconds"] == 7200


def test_local_root_must_be_dedicated_shared_directory_and_cannot_escape_via_symlink(shared, tmp_path):
    with pytest.raises(ValueError, match="dedicated directory"):
        qwen_site.resolve_site("mbzuai-h200", shared)
    with pytest.raises(ValueError, match="dedicated directory"):
        qwen_site.resolve_site("mbzuai-h200", tmp_path / "home")
    escaped = shared / "escape"
    escaped.symlink_to(tmp_path / "home")
    with pytest.raises(ValueError, match="dedicated directory"):
        qwen_site.resolve_site("mbzuai-h200", escaped / "experiments")
    site = qwen_site.resolve_site("mbzuai-h200")
    root = Path(site["artifact_root"])
    root.mkdir()
    (root / "escape").symlink_to(tmp_path / "home")
    with pytest.raises(ValueError, match="sealed artifact root"):
        qwen_site.validate_artifact_path(root / "escape" / "checkpoint", site)
    assert qwen_site.validate_artifact_path(root / "runs" / "checkpoint", site) == root / "runs" / "checkpoint"


def test_artifact_tree_checks_nested_aliases_without_following_directory_cycles(shared, tmp_path):
    site = qwen_site.resolve_site("mbzuai-h200")
    root = Path(site["artifact_root"])
    run, cache = root / "run", root / "cache"
    run.mkdir(parents=True)
    cache.mkdir()
    (run / "cache").symlink_to(cache)
    (cache / "latest-run").symlink_to(run)
    assert qwen_site.validate_artifact_tree(run, site) == run
    (cache / "escape").symlink_to(tmp_path / "home")
    with pytest.raises(ValueError, match="symlink escapes"):
        qwen_site.validate_artifact_tree(run, site)


@pytest.mark.parametrize("field,value", [
    ("scheduler", dict(account="wrong", partition="main", qos="k2m", constraint="nvidia_h200")),
    ("gpu", dict(name_contains="H100", compute_capability=[9, 0])),
    ("modules", ["stockcuda/12.6.2"]),
    ("production_prologue_limit_seconds", 7200),
    ("wandb_entity", "../other-team"),
    ("unexpected", True),
])
def test_even_resealed_site_cannot_change_cluster_profile(shared, field, value):
    site = qwen_site.resolve_site("mbzuai-h200")
    site[field] = value
    with pytest.raises(ValueError):
        qwen_site.validate_site(site)


def test_site_validation_returns_independent_copy(shared):
    site = qwen_site.resolve_site("mbzuai-h200")
    validated = qwen_site.validate_site(site)
    validated["gpu"]["compute_capability"][0] = 1
    assert site == qwen_site.resolve_site("mbzuai-h200")


def test_shared_symlink_must_resolve_outside_home(tmp_path, monkeypatch):
    # Launcher tests replace the module's shared-root resolver. Exercise the
    # filesystem implementation in isolation from those imported aliases.
    spec = importlib.util.spec_from_file_location("isolated_qwen_site", qwen_site.__file__)
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    with pytest.raises(ValueError, match="existing ~/shrd"):
        isolated.shared_storage_root()
    inside = home / "ordinary-directory"
    inside.mkdir()
    (home / "shrd").symlink_to(inside)
    with pytest.raises(ValueError, match="outside the home"):
        isolated.shared_storage_root()
    (home / "shrd").unlink()
    external = tmp_path / "shared"
    external.mkdir()
    (home / "shrd").symlink_to(external)
    assert isolated.shared_storage_root() == external


def test_prologue_budget_requires_explicit_supported_seconds(shared):
    site = qwen_site.resolve_site("mbzuai-h200")
    assert qwen_site.validate_prologue_limit(7200, site) == 7200
    assert qwen_site.validate_prologue_limit(10800, site) == 10800
    for value in (True, "10800", 1, 7201, 10800.0):
        with pytest.raises(ValueError, match="explicitly sealed"):
            qwen_site.validate_prologue_limit(value, site)


@pytest.mark.parametrize("field,value", [
    ("Partition", "batch"), ("QOS", "medium"), ("Account", "another-account"),
    ("Features", "(null)"), ("Features", "nvidia_h200_fake"),
    ("Features", "nvidia_h200|nvidia_h100"),
])
def test_live_scheduler_fields_cannot_drift_from_local_site(shared, field, value):
    site = qwen_site.resolve_site("mbzuai-h200")
    fields = dict(Partition="main", QOS="k2m", Account="k2m", Features="nvidia_h200")
    assert qwen_site.validate_scheduler_allocation(fields, site, account="k2m") == fields
    fields[field] = value
    with pytest.raises(ValueError, match="live Slurm"):
        qwen_site.validate_scheduler_allocation(fields, site, account="k2m")


def test_allocation_account_is_sealed_locally_but_per_arm_for_marlowe(shared):
    site = qwen_site.resolve_site("mbzuai-h200")
    fields = dict(Partition="main", QOS="k2m", Account="k2m", Features="nvidia_h200")
    with pytest.raises(ValueError, match="allocation account"):
        qwen_site.validate_scheduler_allocation(fields, site, account="different")
    historical = dict(Partition="batch", QOS="medium", Account="marlowe-m000215-pm06")
    assert qwen_site.validate_scheduler_allocation(
        historical, qwen_site.resolve_site(), account="marlowe-m000215-pm06") == historical
    with pytest.raises(ValueError, match="must be a mapping"):
        qwen_site.validate_scheduler_allocation(None, site, account="k2m")
