"""VERL custom reward preserving released reward plus validation diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# VERL loads custom reward files through ``spec_from_file_location`` under the
# synthetic module name ``custom_module``. An absolute import therefore works
# both as a package import and through VERL's file-path loader.
from opd_tools.graders import math_verify_full_response_grade, released_last_boxed_grade


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Return the released scalar reward and numeric auxiliary metrics.

    This signature is directly loadable through VERL's
    ``custom_reward_function`` configuration. The decoded ``solution_str`` is
    passed intact to both graders. ``data_source`` and ``extra_info`` remain in
    the signature for compatibility, but do not alter grading semantics.
    """

    del data_source
    if not isinstance(solution_str, str) or not isinstance(ground_truth, str):
        raise TypeError("solution_str and ground_truth must be strings")
    released = float(released_last_boxed_grade(solution_str, ground_truth).correct)
    result = {
        "score": released,
        "released_reward": released,
    }
    split = extra_info.get("split") if isinstance(extra_info, Mapping) else None
    if split in {"validation", "val", "test"}:
        result["math_verify"] = float(
            math_verify_full_response_grade(solution_str, ground_truth).correct
        )
    return result
