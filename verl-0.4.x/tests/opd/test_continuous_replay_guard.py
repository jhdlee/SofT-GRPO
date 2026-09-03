import sys
import types

import pytest
import torch

# This focused guard test does not exercise TensorDict.  Keep it runnable in the
# dependency-light CPU test environment used by the parent repository while the
# production environment installs the real package through VERL.
if "tensordict" not in sys.modules:
    try:
        import tensordict  # noqa: F401
    except ImportError:
        tensordict_stub = types.ModuleType("tensordict")
        tensordict_stub.TensorDict = object
        sys.modules["tensordict"] = tensordict_stub

from verl.utils import torch_functional


def test_topk_gumbel_replay_hard_fails_without_flash_attention(monkeypatch):
    monkeypatch.setattr(
        torch_functional,
        "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE",
        False,
    )

    with pytest.raises(RuntimeError, match="categorical fallback is forbidden"):
        torch_functional.logprobs_from_logits_topk_gumbel(
            logits=torch.zeros(2, 7),
            rollout_topk_ids=torch.tensor([[0, 1], [1, 2]]),
            rollout_topk_gumbels=torch.zeros(2, 2),
            labels=torch.tensor([0, 1]),
        )
