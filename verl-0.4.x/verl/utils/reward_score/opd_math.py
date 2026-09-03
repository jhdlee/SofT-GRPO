"""File-path-loadable VERL reward entry point for the OPD study.

VERL loads custom rewards with ``spec_from_file_location`` and may be launched
from ``verl-0.4.x``. Bootstrap the repository root explicitly so the isolated
``opd_tools`` package is resolvable without relying on the caller's cwd.
"""

import sys
from pathlib import Path

_SOFTGRPO_ROOT = Path(__file__).resolve().parents[4]
if str(_SOFTGRPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOFTGRPO_ROOT))

from opd_tools.reward import compute_score  # noqa: E402,F401

__all__ = ["compute_score"]
