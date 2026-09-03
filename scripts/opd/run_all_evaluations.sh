#!/usr/bin/env bash
set -euo pipefail

if (($# != 3)); then
  printf 'Usage: %s INITIAL_MODEL BASELINE_HF_EXPORT OPD_HF_EXPORT\n' "${0##*/}" >&2
  exit 2
fi

readonly INITIAL_MODEL="$1"
readonly BASELINE_MODEL="$2"
readonly OPD_MODEL="$3"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Each command is idempotent at benchmark/seed shard boundaries.  Keeping the
# modes separate ensures the SGLang engine is initialized with one unambiguous
# soft-thinking setting.
"${SCRIPT_DIR}/run_evaluation.sh" generate initial "${INITIAL_MODEL}" native_soft
"${SCRIPT_DIR}/run_evaluation.sh" generate initial "${INITIAL_MODEL}" hard_token
"${SCRIPT_DIR}/run_evaluation.sh" generate baseline "${BASELINE_MODEL}" native_soft
"${SCRIPT_DIR}/run_evaluation.sh" generate baseline "${BASELINE_MODEL}" hard_token
"${SCRIPT_DIR}/run_evaluation.sh" generate opd "${OPD_MODEL}" native_soft
"${SCRIPT_DIR}/run_evaluation.sh" generate opd "${OPD_MODEL}" hard_token
"${SCRIPT_DIR}/run_evaluation.sh" aggregate
