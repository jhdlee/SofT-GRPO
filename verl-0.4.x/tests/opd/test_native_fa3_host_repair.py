"""The native host repair authenticates its input and preserves CUDA source."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


PACKAGE = (Path(__file__).resolve().parents[3] / "Soft-Thinking+noise+loss-main" /
           "sglang_soft_thinking_pkg/sgl-kernel/native-fa3")


@pytest.fixture
def support():
    spec = importlib.util.spec_from_file_location("native_fa3_host_repair_fixture", PACKAGE / "build_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(support, monkeypatch, tmp_path):
    # Keep the exact real allocation block; surrounding text verifies that the
    # transformation changes no unrelated host or arithmetic statements.
    before = b"// unchanged pinned forward\nvoid forward();\n"
    after = b"\n// unchanged deterministic reductions\nrun_mha_bwd(params, stream);\n"
    original = before + support._MISSING_SEMAPHORE + after
    repaired = before + support._INITIALIZED_SEMAPHORE + after
    monkeypatch.setattr(support, "UPSTREAM_FLASH_API_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(support, "PATCHED_FLASH_API_SHA256", hashlib.sha256(repaired).hexdigest())
    root = tmp_path / "upstream"
    path = root / "hopper/flash_api.cpp"
    path.parent.mkdir(parents=True)
    path.write_bytes(original)
    return root, path, original, repaired


def test_patch_changes_only_missing_host_workspace(support, source):
    _, _, original, repaired = source
    assert support.patched_flash_api(original) == repaired
    assert repaired.replace(support._INITIALIZED_SEMAPHORE, support._MISSING_SEMAPHORE) == original


def test_unrecognized_upstream_bytes_fail_closed(support, source):
    with pytest.raises(RuntimeError, match="upstream.*hash differs"):
        support.patched_flash_api(source[2] + b"\n")


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_patch_anchor_fails_closed(support, monkeypatch, count):
    content = support._MISSING_SEMAPHORE * count
    monkeypatch.setattr(support, "UPSTREAM_FLASH_API_SHA256", hashlib.sha256(content).hexdigest())
    with pytest.raises(RuntimeError, match="exactly one"):
        support.patched_flash_api(content)


def test_unexpected_generated_hash_is_rejected(support, source, monkeypatch):
    monkeypatch.setattr(support, "PATCHED_FLASH_API_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="generated.*hash differs"):
        support.patched_flash_api(source[2])


def test_generated_include_is_outside_upstream_and_idempotent(support, source, tmp_path):
    root, upstream, original, repaired = source
    generated = support.prepare_host_source(root, tmp_path / "build/generated")
    assert generated.name == "opd_flash_api.cpp" and generated.read_bytes() == repaired
    assert upstream.read_bytes() == original
    assert list(root.rglob("*")) == [root / "hopper", upstream]
    assert support.prepare_host_source(root, generated.parent) == generated
    assert upstream.read_bytes() == original


def test_generation_inside_upstream_is_forbidden(support, source):
    with pytest.raises(RuntimeError, match="outside the upstream"):
        support.prepare_host_source(source[0], source[0] / "generated")


def test_existing_changed_generated_include_is_not_overwritten(support, source, tmp_path):
    path = support.prepare_host_source(source[0], tmp_path / "generated")
    path.write_bytes(b"different source")
    with pytest.raises(RuntimeError, match="differs from the authenticated"):
        support.prepare_host_source(source[0], path.parent)
    assert path.read_bytes() == b"different source"


def test_generated_symlink_is_rejected(support, source, tmp_path):
    directory = tmp_path / "generated"
    directory.mkdir()
    (directory / "opd_flash_api.cpp").symlink_to(source[1])
    with pytest.raises(RuntimeError, match="cannot be a symlink"):
        support.prepare_host_source(source[0], directory)
    assert source[1].read_bytes() == source[2]
