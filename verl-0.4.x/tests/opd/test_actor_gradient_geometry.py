"""CPU checks for the actor's fixed-support gradient diagnostics.

Importing ``dp_actor`` requires the full Ray/tensordict/FlashAttention runtime,
which is intentionally absent from the dependency-light CPU test environment.
The loader below compiles the function definition directly from that module so
the test still exercises the checked-in actor implementation rather than a
copy maintained by the test.
"""

import ast
import math
from pathlib import Path

import torch


def _load_geometry_function():
    actor_path = Path(__file__).resolve().parents[2] / "verl" / "workers" / "actor" / "dp_actor.py"
    module = ast.parse(actor_path.read_text(encoding="utf-8"), filename=str(actor_path))
    function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_continuous_support_gradient_geometry"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"torch": torch, "math": math}
    exec(compile(isolated, str(actor_path), "exec"), namespace)
    return namespace["_continuous_support_gradient_geometry"]


def _direct_policy_gradient(logits, stored, sensitivity, scale):
    direct_logits = logits.detach().float().requires_grad_(True)
    support_log_probs = (torch.softmax(direct_logits, dim=-1) + 1e-6).log()
    reparameterized = (stored.float() - support_log_probs).clamp(-1.5, 3.0)
    support_mask = (support_log_probs > -3.0).float()
    log_density = -reparameterized - (-reparameterized).exp()
    log_density = (log_density * support_mask).sum(-1) / support_mask.sum(-1).clamp_min(1.0)
    objective = (log_density * sensitivity.detach().float()).sum()
    return torch.autograd.grad(objective, direct_logits)[0] * scale


def _tiny_inputs():
    logits = torch.tensor([[0.8, -0.1, 0.3], [-0.2, 0.5, 0.1]], dtype=torch.float32)
    stored = torch.tensor([[1.0, -0.4, 0.2], [-0.5, 0.7, 0.0]], dtype=torch.float32)
    sensitivity = torch.tensor([0.75, -1.25], dtype=torch.float32)
    return logits, stored, sensitivity


def test_continuous_support_geometry_matches_policy_autograd_without_opd():
    geometry = _load_geometry_function()
    logits, stored, sensitivity = _tiny_inputs()
    policy_scale = 0.25
    expected_policy = _direct_policy_gradient(logits, stored, sensitivity, policy_scale)

    policy_sq, opd_sq, dot = geometry(
        support_logits=logits,
        stored_perturbed_logits=stored,
        policy_log_density_sensitivity=sensitivity,
        opd_support_gradient=None,
        gumbel_temperature=0.1,
        policy_scale=policy_scale,
        opd_scale=0.0,
    )

    torch.testing.assert_close(policy_sq, expected_policy.square().sum(dtype=torch.float64))
    assert policy_sq.dtype is torch.float64
    assert opd_sq.item() == 0.0
    assert dot.item() == 0.0


def test_continuous_support_geometry_matches_direct_policy_opd_norm_and_dot():
    geometry = _load_geometry_function()
    logits, stored, sensitivity = _tiny_inputs()
    policy_scale = 0.5
    opd_scale = 0.003
    raw_opd_gradient = torch.tensor(
        [[0.2, -0.3, 0.1], [-0.4, 0.15, 0.25]],
        dtype=torch.float32,
    )
    expected_policy = _direct_policy_gradient(logits, stored, sensitivity, policy_scale)
    expected_opd = raw_opd_gradient * opd_scale

    policy_sq, opd_sq, dot = geometry(
        support_logits=logits,
        stored_perturbed_logits=stored,
        policy_log_density_sensitivity=sensitivity,
        opd_support_gradient=raw_opd_gradient,
        gumbel_temperature=0.1,
        policy_scale=policy_scale,
        opd_scale=opd_scale,
    )

    torch.testing.assert_close(policy_sq, expected_policy.square().sum(dtype=torch.float64))
    torch.testing.assert_close(opd_sq, expected_opd.square().sum(dtype=torch.float64))
    torch.testing.assert_close(dot, (expected_policy * expected_opd).sum(dtype=torch.float64))
