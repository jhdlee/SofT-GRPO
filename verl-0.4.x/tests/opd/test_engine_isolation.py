"""Cohosted engine starts coordinate without consuming sampling RNG state."""

import importlib.util
import os
from pathlib import Path
import random
import sys
import threading

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "verl/workers/rollout/sglang_rollout/engine_isolation.py"
SPEC = importlib.util.spec_from_file_location("opd_engine_isolation_tested", SOURCE)
isolation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(isolation)


@pytest.fixture(autouse=True)
def fake_port_reservation(monkeypatch):
    # Unit tests exercise real advisory locks without opening a network socket
    # in the filesystem-only local test sandbox. GPU admission uses real binds.
    class Reservation:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            assert address == ("127.0.0.1", 0)
        def getsockname(self):
            return ("127.0.0.1", 54000)
    monkeypatch.setattr(isolation.socket, "socket", lambda *args: Reservation())


def test_cohosted_starts_are_serialized_until_constructor_returns(tmp_path):
    path = tmp_path / "engine.lock"
    ready, release = threading.Event(), threading.Event()
    failures = []

    def first_constructor():
        try:
            with isolation.isolated_engine_startup(lock_path=path) as port:
                assert 1024 <= port <= 64535
                ready.set()
                assert release.wait(2)
        except BaseException as error:
            failures.append(error)
            ready.set()

    before = random.getstate()
    thread = threading.Thread(target=first_constructor)
    thread.start()
    try:
        assert ready.wait(2)
        with pytest.raises(TimeoutError, match="cohosted"):
            with isolation.isolated_engine_startup(lock_path=path, timeout_seconds=0):
                pytest.fail("second engine entered before first constructor completed")
    finally:
        release.set()
        thread.join(2)
    assert not failures and not thread.is_alive()
    with isolation.isolated_engine_startup(lock_path=path) as port:
        assert 1024 <= port <= 64535
    assert random.getstate() == before


def test_constructor_failure_releases_startup_lock(tmp_path):
    path = tmp_path / "engine.lock"
    with pytest.raises(RuntimeError, match="constructor"):
        with isolation.isolated_engine_startup(lock_path=path):
            raise RuntimeError("constructor failed")
    with isolation.isolated_engine_startup(lock_path=path, timeout_seconds=0):
        pass


def test_startup_lock_refuses_symlinks(tmp_path):
    target = tmp_path / "target"
    target.write_text("protected")
    path = tmp_path / "engine.lock"
    path.symlink_to(target)
    with pytest.raises(OSError):
        with isolation.isolated_engine_startup(lock_path=path):
            pytest.fail("symlink accepted")
    assert target.read_text() == "protected"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux inherited-lock regression; fork after macOS Torch/ObjC initialization is unsafe")
def test_parent_explicitly_unlocks_while_forked_child_retains_descriptor(tmp_path):
    path = tmp_path / "engine.lock"
    reader, writer = os.pipe()
    child = None
    try:
        with isolation.isolated_engine_startup(lock_path=path):
            child = os.fork()
            if child == 0:
                os.close(writer)
                os.read(reader, 1)
                os._exit(0)
        # The child is still alive holding inherited FDs. Explicit LOCK_UN
        # must allow the next engine to enter before the child closes them.
        with isolation.isolated_engine_startup(lock_path=path, timeout_seconds=0):
            pass
    finally:
        os.close(reader)
        if child:
            os.write(writer, b"x")
            os.waitpid(child, 0)
        os.close(writer)
