#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  ${0##*/} generate MODEL_LABEL MODEL_PATH {native_soft|hard_token} [generate_eval options...]" \
    "  ${0##*/} aggregate [aggregate_eval options...]" >&2
}

if (($# == 0)); then
  usage
  exit 2
fi

readonly COMMAND="$1"
shift
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly VERL_ROOT="${SOURCE_ROOT}/verl-0.4.x"
readonly SGLANG_ROOT="${SOURCE_ROOT}/Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/python"
readonly SCRATCH_ROOT="${OPD_SCRATCH_ROOT:-/scratch/m000120-pm06/hdlee/opd_latent_reasoning/softgrpo-math}"
readonly DATA_ROOT="${SCRATCH_ROOT}/data"
readonly EVALUATION_ROOT="${SCRATCH_ROOT}/evaluation"
readonly PYTHON_BIN="${OPD_PYTHON:-/projects/m000215/hdlee/miniconda3/envs/opd-softgrpo/bin/python}"

if [[ "${WANDB_MODE:-}" != "online" ]]; then
  printf 'Evaluation requires WANDB_MODE=online.\n' >&2
  exit 1
fi
if [[ "${WANDB_RESUME:-allow}" != "allow" ]]; then
  printf 'Evaluation requires WANDB_RESUME=allow.\n' >&2
  exit 1
fi
export WANDB_PROJECT="${WANDB_PROJECT:-opd-softgrpo-math}"
export WANDB_RESUME=allow
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
export PYTHONPATH="${SGLANG_ROOT}:${SOURCE_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${EVALUATION_ROOT}" "${WANDB_DIR}"

case "${COMMAND}" in
  generate)
    if (($# < 3)); then
      usage
      exit 2
    fi
    readonly MODEL_LABEL="$1"
    readonly MODEL_PATH="$2"
    readonly MODE="$3"
    shift 3
    exec "${PYTHON_BIN}" -m opd_tools.generate_eval \
      --model-label "${MODEL_LABEL}" \
      --model-path "${MODEL_PATH}" \
      --mode "${MODE}" \
      --data-dir "${DATA_ROOT}" \
      --output-dir "${EVALUATION_ROOT}" \
      --tensor-parallel-size "${OPD_EVAL_TENSOR_PARALLEL_SIZE:-1}" \
      --data-parallel-size "${OPD_EVAL_DATA_PARALLEL_SIZE:-${SLURM_GPUS_ON_NODE:-1}}" \
      "$@"
    ;;
  aggregate)
    exec "${PYTHON_BIN}" -m opd_tools.aggregate_eval \
      --input-dir "${EVALUATION_ROOT}" \
      --output-dir "${EVALUATION_ROOT}/reports" \
      --workers "${SLURM_CPUS_PER_TASK:-112}" \
      "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
