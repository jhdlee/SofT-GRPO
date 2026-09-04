from __future__ import annotations

import hashlib
import math
import re
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from opd_tools.constants import (
    MATH_TRAIN_SIZE,
    STUDENT_PROMPT_SUFFIX,
    TRAIN_MAX_PROMPT_TOKENS,
    TRAIN_MAX_RESPONSE_TOKENS,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_RECIPE = SOURCE_ROOT / "SofT-GRPO-deepscaler-8k.sh"
STUDY_RECIPE = SOURCE_ROOT / "scripts" / "opd" / "run_math_seed11.sh"
TRAINER_CONFIG = (
    SOURCE_ROOT / "verl-0.4.x" / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
)

# This is the byte hash of the released recipe at upstream commit
# 8d3c61380b15c3400818da5ce41c62c293a1bfb4.  The comparison below is only
# meaningful if the local reference script itself remains untouched.
UPSTREAM_RECIPE_SHA256 = (
    "66c59bfc2980b2802fc276aa73eeb739ddb8e1fe585408ae09d7a7cecc6da149"
)


def _shell_overrides(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.]*)=(.*)", line)
        if match is None or "." not in match.group(1):
            continue
        value = match.group(2).strip()
        try:
            parsed = yaml.safe_load(value)
            # Bash removes one quote layer before passing the Hydra argument.
            # Parse the resulting scalar once more so quoted list overrides
            # such as ``"[console,wandb]"`` compare as their runtime value.
            if (
                isinstance(parsed, str)
                and len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                parsed = yaml.safe_load(parsed)
        except yaml.YAMLError:
            parsed = value.strip("\"'")
        result[match.group(1)] = parsed
    return result


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        qualified = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, qualified))
        else:
            result[qualified] = value
    return result


class ReleasedTrainingRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream_text = UPSTREAM_RECIPE.read_text(encoding="utf-8")
        cls.study_text = STUDY_RECIPE.read_text(encoding="utf-8")
        cls.upstream = _shell_overrides(UPSTREAM_RECIPE)
        cls.study_overrides = _shell_overrides(STUDY_RECIPE)
        defaults = yaml.safe_load(TRAINER_CONFIG.read_text(encoding="utf-8"))
        cls.study_effective = _flatten(defaults)
        cls.study_effective.update(cls.study_overrides)

    def test_reference_is_the_byte_exact_pinned_upstream_recipe(self):
        observed = hashlib.sha256(UPSTREAM_RECIPE.read_bytes()).hexdigest()
        self.assertEqual(observed, UPSTREAM_RECIPE_SHA256)

    def test_every_released_explicit_mechanic_is_preserved(self):
        # These are deliberate study/runtime deviations, audited separately
        # below.  Every other explicit release override must resolve to the
        # same scalar in our launcher (possibly through the pinned base config).
        deliberate = {
            "data.train_files",
            "data.val_files",
            "data.max_prompt_length",
            "actor_rollout_ref.model.path",
            "trainer.logger",
            "trainer.project_name",
            "trainer.experiment_name",
            "trainer.n_gpus_per_node",
            "trainer.save_freq",
            "trainer.test_freq",
            "trainer.default_local_dir",
        }
        compared = set(self.upstream) - deliberate
        self.assertGreaterEqual(len(compared), 45)
        for key in sorted(compared):
            with self.subTest(key=key):
                self.assertIn(key, self.study_effective)
                self.assertEqual(self.study_effective[key], self.upstream[key])

    def test_released_inherited_mechanics_are_locked(self):
        expected = {
            "actor_rollout_ref.hybrid_engine": True,
            "actor_rollout_ref.actor.strategy": "fsdp",
            "actor_rollout_ref.actor.use_dynamic_bsz": False,
            "actor_rollout_ref.actor.grad_clip": 1.0,
            "actor_rollout_ref.actor.clip_ratio": 0.2,
            "actor_rollout_ref.actor.clip_ratio_low": 0.2,
            "actor_rollout_ref.actor.clip_ratio_high": 0.2,
            "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
            "actor_rollout_ref.actor.use_torch_compile": True,
            "actor_rollout_ref.actor.ppo_epochs": 1,
            "actor_rollout_ref.actor.shuffle": False,
            "actor_rollout_ref.actor.optim.weight_decay": 0.01,
            "actor_rollout_ref.actor.optim.warmup_style": "constant",
            "actor_rollout_ref.rollout.mode": "sync",
            "actor_rollout_ref.rollout.dtype": "bfloat16",
            "actor_rollout_ref.rollout.ignore_eos": False,
            "actor_rollout_ref.rollout.enforce_eager": True,
            "actor_rollout_ref.rollout.free_cache_engine": True,
            "actor_rollout_ref.rollout.enable_chunked_prefill": True,
            "actor_rollout_ref.rollout.do_sample": True,
            "actor_rollout_ref.rollout.add_noise_dirichlet": False,
            "actor_rollout_ref.rollout.noise_gaussian": False,
            "actor_rollout_ref.rollout.noise_on_inputs": False,
            "actor_rollout_ref.rollout.val_kwargs.n": 1,
            "data.truncation": "error",
            "algorithm.gamma": 1.0,
            "algorithm.lam": 1.0,
            "algorithm.norm_adv_by_std_in_grpo": True,
            "trainer.balance_batch": True,
            "trainer.critic_warmup": 0,
            "trainer.default_hdfs_dir": None,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(self.study_effective[key], value)

        # The release delegates -1 to a zero warm-up ratio.  The study locks
        # the resulting zero steps explicitly, which is semantically equal.
        self.assertEqual(self.study_overrides["actor_rollout_ref.actor.optim.lr_warmup_steps"], 0)
        self.assertEqual(
            self.study_effective["actor_rollout_ref.actor.optim.lr_warmup_steps_ratio"],
            0.0,
        )

    def test_deliberate_study_and_runtime_deviations_are_explicit(self):
        expected = {
            "data.train_files": "${DATA_ROOT}/math_lighteval_train.parquet",
            "data.val_files": "${DATA_ROOT}/math_lighteval_validation.parquet",
            "data.max_prompt_length": TRAIN_MAX_PROMPT_TOKENS,
            "data.seed": 11,
            "actor_rollout_ref.model.path": "${MODEL_ROOT}",
            "actor_rollout_ref.rollout.deterministic_sampling": True,
            "actor_rollout_ref.actor.checkpoint.contents": [
                "model",
                "optimizer",
                "extra",
                "hf_model",
            ],
            "custom_reward_function.path": "${REWARD_FILE}",
            "custom_reward_function.name": "compute_score",
            "trainer.logger": ["console", "wandb"],
            "trainer.project_name": "opd-softgrpo-math",
            "trainer.total_epochs": 1,
            "trainer.total_training_steps": None,
            "trainer.save_freq": 25,
            "trainer.test_freq": 25,
            "trainer.default_local_dir": "${SCRATCH_ROOT}/runs/${EXPERIMENT_NAME}",
            "trainer.checkpoint_keep_latest": 2,
            "trainer.rollout_integrity.enabled": True,
            "trainer.rollout_integrity.gate_first_n_iterations": 1,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(self.study_overrides[key], value)

        # OPD is the treatment and the baseline switches only this feature off.
        self.assertEqual(self.study_overrides["algorithm.opd.enabled"], "${OPD_ENABLED}")
        self.assertEqual(self.study_overrides["algorithm.opd.beta_base"], 0.001)
        self.assertEqual(self.study_overrides["algorithm.opd.warmup_fraction"], 0.10)
        self.assertEqual(self.study_overrides["algorithm.opd.teacher.type"], "ema")
        self.assertEqual(self.study_overrides["algorithm.opd.prompt_template"], "sdpg")

        # Slurm owns device visibility; the released hard-coded mask must not
        # leak into the study launcher.
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.study_text)
        self.assertIn('WANDB_MODE must be online', self.study_text)

    def test_obsolete_worker_checkpoint_rotation_is_not_overridden(self):
        self.assertNotIn("trainer.max_actor_ckpt_to_keep", self.study_overrides)
        self.assertNotIn("trainer.max_critic_ckpt_to_keep", self.study_overrides)

    def test_study_horizon_and_prompt_match_the_locked_contract(self):
        prompt_batch = self.study_overrides["data.train_batch_size"]
        epochs = self.study_overrides["trainer.total_epochs"]
        mini_batch = self.study_overrides[
            "actor_rollout_ref.actor.ppo_mini_batch_size"
        ]
        rollout_iterations = (MATH_TRAIN_SIZE // prompt_batch) * epochs
        optimizer_steps = rollout_iterations * (prompt_batch // mini_batch)
        self.assertEqual(rollout_iterations, 109)
        self.assertEqual(optimizer_steps, 218)
        self.assertEqual(
            math.ceil(
                self.study_overrides["algorithm.opd.warmup_fraction"]
                * rollout_iterations
            ),
            11,
        )
        self.assertEqual(
            self.study_overrides["data.max_prompt_length"], TRAIN_MAX_PROMPT_TOKENS
        )
        self.assertEqual(
            self.study_overrides["data.max_response_length"],
            TRAIN_MAX_RESPONSE_TOKENS,
        )
        self.assertLessEqual(
            TRAIN_MAX_PROMPT_TOKENS + TRAIN_MAX_RESPONSE_TOKENS,
            self.study_overrides["actor_rollout_ref.rollout.max_model_len"],
        )
        self.assertEqual(
            STUDENT_PROMPT_SUFFIX,
            " Let's think step by step and output the final answer within \\boxed{}.",
        )


if __name__ == "__main__":
    unittest.main()
