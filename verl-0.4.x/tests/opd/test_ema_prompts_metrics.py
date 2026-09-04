import pytest
import torch
from torch import nn

from verl.opd import (
    FORBIDDEN_METRICS,
    ITERATION_METRICS,
    STANDALONE_INAPPLICABLE_METRICS,
    STANDALONE_ITERATION_METRICS,
    VALIDATION_METRICS,
    EMAUpdateState,
    OPDConfig,
    freeze_teacher_,
    opd_schedule_metrics,
    parameter_squared_distance_sum_and_count,
    render_privileged_prompt,
    render_sdft_prompt,
    required_iteration_metrics,
    rms_from_squared_sum_and_count,
    teacher_gradient_isolation_violations,
    update_ema_once_,
    validate_metric_payload,
    validation_pass_at_1_aliases,
)


class TinyModel(nn.Module):
    def __init__(self, weight, running, batches):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([weight], dtype=torch.float32))
        self.register_buffer("running", torch.tensor([running], dtype=torch.float32))
        self.register_buffer("batches", torch.tensor(batches, dtype=torch.int64))


def test_ema_updates_float_state_copies_integer_state_and_only_runs_once():
    teacher = TinyModel(0.0, 2.0, 1)
    student = TinyModel(10.0, 6.0, 7)
    freeze_teacher_(teacher)
    state = EMAUpdateState()

    report = update_ema_once_(teacher, student, decay=0.75, rollout_iteration=0, state=state)
    torch.testing.assert_close(teacher.weight, torch.tensor([2.5]))
    torch.testing.assert_close(teacher.running, torch.tensor([3.0]))
    assert teacher.batches.item() == 7
    assert not teacher.training
    assert not teacher.weight.requires_grad
    assert report.parameter_tensors == 1
    assert report.averaged_buffers == 1
    assert report.copied_buffers == 1
    assert state.state_dict() == {"update_count": 1, "last_rollout_iteration": 0}

    with pytest.raises(RuntimeError, match="already been updated"):
        update_ema_once_(teacher, student, decay=0.75, rollout_iteration=0, state=state)


def test_ema_state_round_trip():
    state = EMAUpdateState(update_count=12, last_rollout_iteration=11)
    restored = EMAUpdateState()
    restored.load_state_dict(state.state_dict())
    assert restored == state


def test_ema_identical_model_is_bit_exact_fixed_point():
    student = TinyModel(0.1234567, -0.7654321, 9)
    teacher = TinyModel(0.1234567, -0.7654321, 9)
    weight_bits = teacher.weight.detach().view(torch.int32).clone()
    buffer_bits = teacher.running.view(torch.int32).clone()

    update_ema_once_(teacher, student, decay=0.99, rollout_iteration=0, state=EMAUpdateState())

    assert torch.equal(teacher.weight.detach().view(torch.int32), weight_bits)
    assert torch.equal(teacher.running.view(torch.int32), buffer_bits)


def test_parameter_distance_exposes_distributed_sum_and_count():
    teacher = TinyModel(1.0, 0.0, 0)
    student = TinyModel(4.0, 0.0, 0)
    squared_sum, count = parameter_squared_distance_sum_and_count(teacher, student)

    assert squared_sum.dtype is torch.float64
    assert squared_sum.item() == 9.0
    assert count.item() == 1
    assert rms_from_squared_sum_and_count(squared_sum, count).item() == 3.0


def test_teacher_gradient_isolation_detects_frozen_state_and_stale_gradients():
    teacher = freeze_teacher_(TinyModel(1.0, 0.0, 0))
    assert teacher_gradient_isolation_violations(teacher) == (0, 0)

    teacher.weight.grad = torch.ones_like(teacher.weight)
    assert teacher_gradient_isolation_violations(teacher) == (0, 1)

    teacher.weight.grad = None
    teacher.weight.requires_grad_(True)
    assert teacher_gradient_isolation_violations(teacher) == (1, 0)

    with pytest.raises(ValueError, match="parameterless teacher"):
        teacher_gradient_isolation_violations(nn.Identity())


def test_sdft_prompt_is_exact_and_does_not_duplicate_answer():
    question = "Compute 2+2."
    cot = "Add the integers."
    expected = (
        "\nCompute 2+2.\n\n"
        "This is an example for a response to the question:\n"
        "Add the integers.\n"
        "The final answer is: \\boxed{4}\n\n"
        "Now answer with a response of your own, including the thinking process.\n"
    )
    prompt = render_sdft_prompt(question, cot, "4")
    assert prompt == expected
    assert prompt.count(cot) == 1
    assert prompt.count(r"\boxed{4}") == 1


def test_sdpg_prompt_requires_answer_and_dispatches():
    with pytest.raises(ValueError, match="gold_answer"):
        render_privileged_prompt("Question", "Solution", template="sdpg")
    rendered = render_privileged_prompt("Question", "Solution", template="sdpg", gold_answer="1")
    assert "[Hint] The correct answer is 1." in rendered
    assert "[Instruction]" in rendered


def test_default_privileged_prompt_dispatches_to_sdpg():
    rendered = render_privileged_prompt(
        "Question", "Solution", gold_answer="1"
    )
    assert "[Hint] The correct answer is 1." in rendered
    assert "This is an example for a response" not in rendered


def test_metric_contract_contains_required_names_and_separates_weight_terms():
    assert "opd/kl_weight" in FORBIDDEN_METRICS
    assert "opd/schedule_multiplier" in ITERATION_METRICS
    assert "opd/beta_effective" in ITERATION_METRICS
    assert "algorithm/objective_mode" in ITERATION_METRICS
    assert "opd/loss_support" in ITERATION_METRICS
    assert "loss/opd_kl_latent" in ITERATION_METRICS
    assert "loss/opd_kl_answer" in ITERATION_METRICS
    assert "opd/latent_slot_count" in ITERATION_METRICS
    assert "opd/answer_slot_count" in ITERATION_METRICS
    assert "opd/selected_slot_fraction" in ITERATION_METRICS
    assert "val/math_verify/mean_at_1" in VALIDATION_METRICS
    assert "val/math_verify/pass_at_1" in VALIDATION_METRICS
    assert "val/released_reward/pass_at_1" in VALIDATION_METRICS

    metrics = opd_schedule_metrics(OPDConfig(), rollout_iteration=11, total_iterations=109)
    assert metrics["opd/warmup_iterations"] == 11
    assert metrics["opd/schedule_multiplier"] == 1.0
    assert metrics["opd/beta_effective"] == 0.001
    assert metrics["algorithm/objective_mode"] == "auxiliary"
    assert metrics["opd/loss_support"] == "latent_only"
    validate_metric_payload(metrics, required=metrics.keys())


def test_constant_schedule_metrics_report_no_warmup():
    metrics = opd_schedule_metrics(
        OPDConfig(
            mode="standalone",
            loss_support="all_response",
            beta_base=1.0,
            schedule="constant",
        ),
        rollout_iteration=0,
        total_iterations=109,
    )
    assert metrics["algorithm/objective_mode"] == "standalone"
    assert metrics["opd/loss_support"] == "all_response"
    assert metrics["opd/warmup_iterations"] == 0
    assert metrics["opd/schedule_multiplier"] == 1.0
    assert metrics["opd/beta_effective"] == 1.0


def test_standalone_metric_contract_omits_inapplicable_grpo_and_ppo_fields():
    assert required_iteration_metrics("auxiliary") is ITERATION_METRICS
    assert required_iteration_metrics("standalone") is STANDALONE_ITERATION_METRICS
    assert STANDALONE_INAPPLICABLE_METRICS.isdisjoint(STANDALONE_ITERATION_METRICS)
    assert "loss/grpo" in STANDALONE_INAPPLICABLE_METRICS
    assert "loss/native_ref_kl" in STANDALONE_INAPPLICABLE_METRICS
    assert "ppo/ratio_mean" in STANDALONE_INAPPLICABLE_METRICS
    assert "grad/grpo_norm" in STANDALONE_INAPPLICABLE_METRICS
    assert "train/reward_mean" in STANDALONE_ITERATION_METRICS
    assert "loss/opd_kl_unweighted" in STANDALONE_ITERATION_METRICS


def test_validation_pass_at_1_is_an_exact_mean_at_1_alias():
    aliases = validation_pass_at_1_aliases(
        {
            "val/math_verify/mean_at_1": 0.625,
            "val/released_reward/mean_at_1": 0.5,
        }
    )
    assert aliases == {
        "val/math_verify/pass_at_1": 0.625,
        "val/released_reward/pass_at_1": 0.5,
    }

    with pytest.raises(ValueError, match="disagrees"):
        validation_pass_at_1_aliases(
            {
                "val/math_verify/mean_at_1": 0.625,
                "val/math_verify/pass_at_1": 0.5,
            }
        )


def test_zero_beta_schedule_metrics_are_zero_without_hiding_base_config():
    metrics = opd_schedule_metrics(OPDConfig(beta_base=0.0), rollout_iteration=11, total_iterations=109)
    assert metrics["opd/beta_base"] == 0.0
    assert metrics["opd/schedule_multiplier"] == 0.0
    assert metrics["opd/beta_effective"] == 0.0


def test_metric_payload_rejects_ambiguous_and_missing_names():
    with pytest.raises(ValueError, match="ambiguous"):
        validate_metric_payload({"opd/kl_weight": 1.0})
    with pytest.raises(ValueError, match="missing"):
        validate_metric_payload({}, required={"loss/opd_weighted"})
