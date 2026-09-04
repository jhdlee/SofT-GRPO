from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from opd_tools.study import (
    ACCOUNT_M000120,
    ACCOUNT_M000215,
    ARM_IDS,
    ARM_SPECS,
    TOTAL_ROLLOUT_ITERATIONS,
    WARMUP_ITERATIONS,
    arm_contract_sha256,
    build_submission_manifest,
    hydra_overrides,
    materialize_submission_manifest,
    resolve_arm,
    stable_wandb_run_id,
    validate_registry,
    verify_submission_manifest,
)


def override_mapping(identifier):
    return dict(value.split("=", 1) for value in hydra_overrides(identifier))


class SevenArmStudyRegistryTest(unittest.TestCase):
    def test_exact_arm_matrix_and_account_routing(self):
        validate_registry()
        self.assertEqual(TOTAL_ROLLOUT_ITERATIONS, 109)
        self.assertEqual(WARMUP_ITERATIONS, 11)
        self.assertEqual(
            ARM_IDS,
            (
                "hardgrpo_math_s11",
                "softgrpo_math_s11",
                "softgrpo_math_opd_s11",
                "softgrpo_math_opd_posadv_s11",
                "softgrpo_math_opd_current_s11",
                "softgrpo_math_opd_beta0p1_s11",
                "softopd_math_s11",
            ),
        )
        by_account = {
            ACCOUNT_M000120: {
                "hardgrpo_math_s11",
                "softgrpo_math_opd_s11",
                "softgrpo_math_opd_current_s11",
                "softopd_math_s11",
            },
            ACCOUNT_M000215: {
                "softgrpo_math_s11",
                "softgrpo_math_opd_posadv_s11",
                "softgrpo_math_opd_beta0p1_s11",
            },
        }
        self.assertEqual(
            {
                account: {spec.arm_id for spec in ARM_SPECS if spec.account == account}
                for account in by_account
            },
            by_account,
        )

    def test_all_new_opd_arms_explicitly_use_all_response(self):
        opd_specs = [spec for spec in ARM_SPECS if spec.opd_enabled]
        self.assertEqual(len(opd_specs), 5)
        self.assertTrue(all(spec.loss_support == "all_response" for spec in opd_specs))
        self.assertTrue(
            all(
                override_mapping(spec.arm_id)["algorithm.opd.loss_support"]
                == "all_response"
                for spec in opd_specs
            )
        )

    def test_primary_and_each_single_factor_ablation(self):
        primary = resolve_arm("softgrpo_math_opd_s11").spec
        positive = resolve_arm("softgrpo_math_opd_posadv_s11").spec
        current = resolve_arm("softgrpo_math_opd_current_s11").spec
        high_dose = resolve_arm("softgrpo_math_opd_beta0p1_s11").spec
        ignored = {
            "arm_id",
            "account",
            "experiment_name",
            "trajectory_gate",
            "teacher_type",
            "beta_base",
            "wandb_tags",
        }

        def common(spec):
            return {
                key: value
                for key, value in spec.as_manifest().items()
                if key not in ignored
            }

        self.assertEqual(common(primary), common(positive))
        self.assertEqual(common(primary), common(current))
        self.assertEqual(common(primary), common(high_dose))
        self.assertEqual(positive.trajectory_gate, "positive_advantage")
        self.assertEqual(current.teacher_type, "current_actor")
        self.assertEqual(high_dose.beta_base, 0.1)

    def test_standalone_is_g1_without_grpo_reference_loss(self):
        standalone = resolve_arm("softopd_math_s11").spec
        overrides = override_mapping(standalone.arm_id)
        self.assertEqual(standalone.objective_mode, "standalone")
        self.assertEqual(standalone.group_size, 1)
        self.assertFalse(standalone.native_reference_kl)
        self.assertEqual(standalone.beta_base, 1.0)
        self.assertEqual(standalone.schedule, "constant")
        self.assertEqual(overrides["algorithm.opd.mode"], "standalone")
        self.assertEqual(overrides["actor_rollout_ref.rollout.n"], "1")
        self.assertEqual(overrides["actor_rollout_ref.actor.use_kl_loss"], "false")
        self.assertEqual(overrides["actor_rollout_ref.actor.kl_loss_coef"], "0.0")

    def test_hard_arm_uses_real_categorical_vllm_route(self):
        overrides = override_mapping("hardgrpo_math_s11")
        self.assertEqual(overrides["actor_rollout_ref.rollout.name"], "vllm")
        self.assertEqual(overrides["actor_rollout_ref.rollout.enable_soft_thinking"], "false")
        self.assertEqual(overrides["actor_rollout_ref.rollout.add_noise_gumbel_softmax"], "false")
        self.assertEqual(overrides["actor_rollout_ref.rollout.noise_on_logits"], "false")
        self.assertEqual(overrides["actor_rollout_ref.rollout.top_p"], "1.0")
        self.assertEqual(overrides["actor_rollout_ref.rollout.top_k"], "-1")
        self.assertEqual(overrides["actor_rollout_ref.rollout.n"], "8")
        self.assertEqual(overrides["actor_rollout_ref.rollout.val_kwargs.top_k"], "30")

    def test_legacy_aliases_are_rejected_and_new_opd_is_all_response(self):
        self.assertNotIn("baseline", ARM_IDS)
        self.assertNotIn("opd", ARM_IDS)
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            resolve_arm("baseline")
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            resolve_arm("opd")
        self.assertEqual(
            override_mapping("softgrpo_math_opd_s11")["algorithm.opd.loss_support"],
            "all_response",
        )

    def test_manifest_hash_and_wandb_id_are_stable_and_arm_specific(self):
        first = arm_contract_sha256("softgrpo_math_opd_s11")
        second = arm_contract_sha256("softgrpo_math_opd_s11")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        manifest_hash = "a" * 64
        run_id = stable_wandb_run_id("softgrpo_math_opd_s11", manifest_hash)
        self.assertEqual(
            run_id,
            stable_wandb_run_id("softgrpo_math_opd_s11", manifest_hash),
        )
        self.assertNotEqual(
            run_id,
            stable_wandb_run_id(
                "softgrpo_math_opd_beta0p1_s11", manifest_hash
            ),
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            stable_wandb_run_id("softgrpo_math_opd_s11", "abc123")

    def test_unknown_arm_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            resolve_arm("softgrpo_math_s22")

    def test_submission_manifest_is_immutable_and_source_bound(self):
        parent_commit = "1" * 40
        submodule_commit = "2" * 40
        manifest = build_submission_manifest(
            "softgrpo_math_opd_s11",
            parent_commit=parent_commit,
            submodule_commit=submodule_commit,
            parent_gitlink=submodule_commit,
        )
        self.assertEqual(
            manifest["arm_contract_sha256"],
            arm_contract_sha256("softgrpo_math_opd_s11"),
        )
        verified = verify_submission_manifest(
            manifest,
            identifier="softgrpo_math_opd_s11",
            parent_commit=parent_commit,
            submodule_commit=submodule_commit,
        )
        self.assertEqual(verified, manifest)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            first_hash = materialize_submission_manifest(path, manifest)
            self.assertEqual(first_hash, materialize_submission_manifest(path, manifest))
            changed = dict(manifest)
            changed["arm_contract_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                materialize_submission_manifest(path, changed)

        tampered = dict(manifest)
        tampered["arm_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            verify_submission_manifest(
                tampered,
                identifier="softgrpo_math_opd_s11",
                parent_commit=parent_commit,
                submodule_commit=submodule_commit,
            )

    def test_submission_manifest_rejects_gitlink_drift(self):
        with self.assertRaisesRegex(ValueError, "Gitlink"):
            build_submission_manifest(
                "softgrpo_math_s11",
                parent_commit="1" * 40,
                submodule_commit="2" * 40,
                parent_gitlink="3" * 40,
            )


if __name__ == "__main__":
    unittest.main()
