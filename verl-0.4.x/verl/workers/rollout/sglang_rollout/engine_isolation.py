"""Coordinate local SGLang rendezvous allocation across cohosted study jobs."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import socket
import stat
import time


@contextmanager
def isolated_engine_startup(*, timeout_seconds=600.0, lock_path=None):
    """Hold a shared host/user lock until the engine has bound its rendezvous.

    PortArgs selects and checks the actual NCCL port during Engine.__init__.
    Keeping this lock through construction closes that check/bind race between
    participating production jobs. OS-selected base ports do not consume the
    model's Python/NumPy/Torch random streams. IPC files use job-local TMPDIR.
    """

    path = Path(lock_path) if lock_path is not None else Path(f"/tmp/opd-sglang-engine-start-{os.getuid()}.lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    acquired = False
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("SGLang startup lock must be a regular file owned by this user")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for cohosted SGLang engine initialization")
                time.sleep(0.05)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            # PortArgs adds 100..1000 to this base, then checks availability.
            base_port = max(1024, reservation.getsockname()[1] - 1000)
        yield base_port
    finally:
        # A forked engine child may inherit the open file description even
        # with CLOEXEC. Explicit unlock releases it before that child exits.
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
