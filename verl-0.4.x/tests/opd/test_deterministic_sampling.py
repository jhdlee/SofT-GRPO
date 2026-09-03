import importlib.util
import sys
import types
from pathlib import Path

import torch

SGLANG_PYTHON = (
    Path(__file__).resolve().parents[3]
    / "Soft-Thinking+noise+loss-main"
    / "sglang_soft_thinking_pkg"
    / "python"
)
sys.path.insert(0, str(SGLANG_PYTHON))
if "sglang" not in sys.modules:
    sglang_package = types.ModuleType("sglang")
    sglang_package.__path__ = [str(SGLANG_PYTHON / "sglang")]
    sys.modules["sglang"] = sglang_package

from sglang.srt.sampling.stateless_random import (  # noqa: E402
    derive_parallel_seed,
    derive_seed,
    expand_parallel_sampling_params,
    stateless_categorical,
    stateless_gumbel,
    stateless_uniform,
)

VERL_SEED_MODULE = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "workers"
    / "rollout"
    / "sglang_rollout"
    / "deterministic_sampling.py"
)
seed_spec = importlib.util.spec_from_file_location(
    "opd_test_deterministic_sampling", VERL_SEED_MODULE
)
seed_module = importlib.util.module_from_spec(seed_spec)
assert seed_spec.loader is not None
seed_spec.loader.exec_module(seed_module)
build_request_sampling_params = seed_module.build_request_sampling_params
expand_parallel_seeds = seed_module.expand_parallel_seeds


def test_seed_derivation_is_stable_and_identity_sensitive():
    first = derive_seed(11, "rollout", 4, {"index": 9})
    assert first == derive_seed(11, "rollout", 4, {"index": 9})
    assert first != derive_seed(11, "rollout", 5, {"index": 9})
    assert first != derive_seed(11, "rollout", 4, {"index": 10})
    assert derive_parallel_seed(first, 0) != derive_parallel_seed(first, 1)


def test_stateless_stream_does_not_consume_or_depend_on_global_rng():
    seeds = torch.tensor([derive_seed(11, "a"), derive_seed(11, "b")])
    counters = torch.tensor([7, 7])

    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)
    first = stateless_uniform(seeds, counters, width=5, stream=0)
    actual_next = torch.rand(4)
    torch.rand(1000)
    second = stateless_uniform(seeds, counters, width=5, stream=0)

    assert torch.equal(first, second)
    assert torch.equal(expected_next, actual_next)


def test_counter_and_named_streams_separate_draws_for_exact_resume():
    seeds = torch.tensor([derive_seed(11, "request")])
    counter = torch.tensor([12])
    gumbel_before = stateless_gumbel(seeds, counter, width=5, stream=0)
    categorical_uniform = stateless_uniform(seeds, counter, width=1, stream=1)

    torch.manual_seed(999)
    torch.randn(4096)
    gumbel_after_resume = stateless_gumbel(seeds, counter, width=5, stream=0)

    assert torch.equal(gumbel_before, gumbel_after_resume)
    assert not torch.equal(gumbel_before[:, :1], categorical_uniform)
    assert torch.isfinite(gumbel_before).all()


def test_resumed_soft_then_hard_sequence_matches_uninterrupted_suffix():
    seed = torch.tensor([derive_seed(11, "trajectory", 31)])

    def draw(counter_value):
        counter = torch.tensor([counter_value])
        if counter_value < 4:
            return stateless_gumbel(seed, counter, width=5, stream=0)
        probs = torch.tensor([[0.2, 0.3, 0.5]])
        return stateless_categorical(probs, seed, counter, stream=1)

    uninterrupted = [draw(counter) for counter in range(9)]
    _checkpoint_after_four_tokens = uninterrupted[:4]

    torch.manual_seed(404)
    torch.rand(10000)
    resumed_suffix = [draw(counter) for counter in range(4, 9)]

    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(uninterrupted[4:], resumed_suffix)
    )


def test_stateless_categorical_uses_each_request_seed_and_counter():
    probs = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    seeds = torch.tensor([derive_seed(11, "x"), derive_seed(11, "y")])
    counters = torch.tensor([3, 9])
    sampled = stateless_categorical(probs, seeds, counters)

    assert sampled.shape == (2, 1)
    assert torch.equal(sampled, stateless_categorical(probs, seeds, counters))
    assert ((0 <= sampled) & (sampled < probs.shape[-1])).all()


def test_parallel_params_are_unaliased_and_children_are_prompt_major():
    base = [{"n": 3, "seed": 101}, {"n": 3, "seed": 202}]
    normalized = expand_parallel_sampling_params(
        base, batch_size=2, parallel_sample_num=3
    )
    normalized[0]["temperature"] = 0.5

    assert len(normalized) == 6
    assert "temperature" not in normalized[2]
    assert [item["seed"] for item in normalized] == [101, 202, 101, 202, 101, 202]

    actual = expand_parallel_seeds([101, 202], 3)
    expected = [
        derive_parallel_seed(101, 0),
        derive_parallel_seed(101, 1),
        derive_parallel_seed(101, 2),
        derive_parallel_seed(202, 0),
        derive_parallel_seed(202, 1),
        derive_parallel_seed(202, 2),
    ]
    assert actual == expected


def test_request_seed_includes_iteration_identity_prompt_and_external_sample():
    kwargs = dict(
        base_sampling_params={"n": 1, "temperature": 1.0},
        root_seed=11,
        rollout_iteration=8,
        example_identities=["math-3", "math-3"],
        prompt_token_ids=[[1, 2, 3], [1, 2, 3]],
        external_sample_indices=[0, 1],
    )
    params, seeds = build_request_sampling_params(**kwargs)

    assert params[0]["seed"] == seeds[0]
    assert params[1]["seed"] == seeds[1]
    assert seeds[0] != seeds[1]
    assert build_request_sampling_params(**kwargs)[1] == seeds
    assert build_request_sampling_params(
        **{**kwargs, "rollout_iteration": 9}
    )[1] != seeds
