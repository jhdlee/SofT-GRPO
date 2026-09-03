#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {baseline|opd} [Hydra overrides...]\n' "${0##*/}" >&2
}

if (($# == 0)); then
  usage
  exit 2
fi

readonly ARM="$1"
shift
if [[ "${ARM}" != "baseline" && "${ARM}" != "opd" ]]; then
  usage
  exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly VERL_ROOT="${SOURCE_ROOT}/verl-0.4.x"
readonly SCRATCH_ROOT="${OPD_SCRATCH_ROOT:-/scratch/m000120-pm06/hdlee/opd_latent_reasoning/softgrpo-math}"
readonly MODEL_ROOT="${SCRATCH_ROOT}/assets/model"
readonly DATA_ROOT="${SCRATCH_ROOT}/data"
readonly CACHE_ROOT="${SCRATCH_ROOT}/cache/huggingface"
readonly REWARD_FILE="${SOURCE_ROOT}/opd_tools/reward.py"
readonly PYTHON_BIN="${OPD_PYTHON:-python}"

if [[ "${WANDB_MODE:-}" != "online" ]]; then
  printf 'WANDB_MODE must be online for every study run.\n' >&2
  exit 1
fi
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  printf 'WANDB_RUN_ID must be stable and explicitly set.\n' >&2
  exit 1
fi

mkdir -p "${SCRATCH_ROOT}" "${CACHE_ROOT}" "${SCRATCH_ROOT}/wandb"
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
export PYTHONPATH="${SOURCE_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

if [[ "${OPD_ASSETS_PREFLIGHTED:-0}" != 1 ]]; then
  "${PYTHON_BIN}" -m opd_tools.assets \
    --output-dir "${MODEL_ROOT}" \
    --cache-dir "${CACHE_ROOT}"
  "${PYTHON_BIN}" -m opd_tools.prepare \
    --output-dir "${DATA_ROOT}" \
    --cache-dir "${CACHE_ROOT}"
  "${PYTHON_BIN}" -m opd_tools.preflight \
    --model-dir "${MODEL_ROOT}" \
    --data-dir "${DATA_ROOT}"
fi

if [[ "${ARM}" == "baseline" ]]; then
  readonly EXPERIMENT_NAME="softgrpo_math_s11"
  readonly OPD_ENABLED=false
else
  readonly EXPERIMENT_NAME="softgrpo_math_opd_s11"
  readonly OPD_ENABLED=true
fi

readonly RAY_CPUS="${SLURM_CPUS_PER_TASK:-8}"

cd "${VERL_ROOT}"
exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  algorithm.opd.enabled="${OPD_ENABLED}" \
  algorithm.opd.beta_base=0.001 \
  algorithm.opd.schedule=warmup_constant \
  algorithm.opd.warmup_fraction=0.10 \
  algorithm.opd.teacher.type=ema \
  algorithm.opd.teacher.ema_decay=0.99 \
  algorithm.opd.kl_direction=teacher_to_student \
  algorithm.opd.trajectory_gate=all \
  algorithm.opd.prompt_template=sdft \
  algorithm.opd.temperature=1.0 \
  data.train_files="${DATA_ROOT}/math_lighteval_train.parquet" \
  data.val_files="${DATA_ROOT}/math_lighteval_validation.parquet" \
  data.train_batch_size=64 \
  data.val_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.seed=11 \
  actor_rollout_ref.model.path="${MODEL_ROOT}" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.checkpoint.contents="[model,optimizer,extra,hf_model]" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.deterministic_sampling=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.max_model_len=12000 \
  actor_rollout_ref.rollout.max_num_batched_tokens=12000 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=5 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.after_thinking_temperature=1.0 \
  actor_rollout_ref.rollout.after_thinking_top_p=0.95 \
  actor_rollout_ref.rollout.after_thinking_top_k=5 \
  actor_rollout_ref.rollout.gumbel_softmax_temperature=0.1 \
  actor_rollout_ref.rollout.enable_soft_thinking=true \
  actor_rollout_ref.rollout.add_noise_gumbel_softmax=true \
  actor_rollout_ref.rollout.add_noise_dirichlet=false \
  actor_rollout_ref.rollout.noise_gumbel=true \
  actor_rollout_ref.rollout.noise_gaussian=false \
  actor_rollout_ref.rollout.noise_on_logits=true \
  actor_rollout_ref.rollout.noise_on_inputs=false \
  actor_rollout_ref.rollout.noise_factor=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=5 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.strategy=fsdp2 \
  custom_reward_function.path="${REWARD_FILE}" \
  custom_reward_function.name=compute_score \
  trainer.logger="[console,wandb]" \
  trainer.project_name=opd-softgrpo-math \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.val_before_train=true \
  trainer.total_epochs=3 \
  trainer.total_training_steps=null \
  trainer.n_gpus_per_node="${SLURM_GPUS_ON_NODE:-8}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${SCRATCH_ROOT}/runs/${EXPERIMENT_NAME}" \
  trainer.save_freq=25 \
  trainer.test_freq=25 \
  trainer.checkpoint_keep_latest=2 \
  trainer.rollout_integrity.enabled=true \
  trainer.rollout_integrity.gate_first_n_iterations=1 \
  ray_init.num_cpus="${RAY_CPUS}" \
  "$@"
