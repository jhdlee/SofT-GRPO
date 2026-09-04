"""Regression tests for Ray-safe Hugging Face dataset filtering."""

import ast
from pathlib import Path
from unittest.mock import patch

import datasets
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "verl" / "utils" / "dataset" / "rl_dataset.py"


def _load_filter_kwargs_helper():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_dataset_filter_kwargs"
    )
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_dataset_filter_kwargs"], tree


def test_single_worker_filter_does_not_start_multiprocessing_manager():
    filter_kwargs, _ = _load_filter_kwargs_helper()
    dataframe = datasets.Dataset.from_dict({"keep": [True, False]})

    with patch("multiprocess.Manager", side_effect=AssertionError("multiprocessing used")):
        filtered = dataframe.filter(
            lambda row: row["keep"],
            load_from_cache_file=False,
            **filter_kwargs(1),
        )

    assert len(filtered) == 1


def test_parallel_filter_preserves_requested_worker_count():
    filter_kwargs, _ = _load_filter_kwargs_helper()
    assert filter_kwargs(2) == {"num_proc": 2}
    with pytest.raises(ValueError, match="must be positive"):
        filter_kwargs(0)


def test_rlhf_dataset_routes_filter_through_ray_safe_kwargs():
    _, tree = _load_filter_kwargs_helper()
    read_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_read_files_and_tokenize"
    )
    filter_calls = [
        node
        for node in ast.walk(read_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "filter"
    ]

    assert len(filter_calls) == 1
    call = filter_calls[0]
    assert not any(keyword.arg == "num_proc" for keyword in call.keywords)
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "filter_kwargs"
        for keyword in call.keywords
    )
