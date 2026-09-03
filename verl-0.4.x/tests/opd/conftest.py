"""Load only the dependency-light ``verl.opd`` package in unit tests."""

import sys
import types
from pathlib import Path

# The upstream top-level ``verl`` package eagerly imports optional production
# dependencies.  OPD primitives intentionally need only PyTorch, so avoid making
# those unrelated packages prerequisites for this focused unit suite.
if "verl" not in sys.modules:
    verl_package = types.ModuleType("verl")
    verl_package.__path__ = [str(Path(__file__).resolve().parents[2] / "verl")]
    sys.modules["verl"] = verl_package
