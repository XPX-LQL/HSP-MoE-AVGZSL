#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate_run.sh <UCF|VGGSound|ActivityNet> <stage1_dir> <stage2_dir> [device]

Examples:
  bash scripts/evaluate_run.sh ActivityNet \
    logs/ClipClap_ActivityNet/STAGE1 \
    logs/ClipClap_ActivityNet/STAGE2 \
    cuda

Environment overrides:
  SEMANTIC_FAR_FUSION           default: False
  SEMANTIC_FAR_ADAPTED_WEIGHT   default: 1.0
  SEMANTIC_FAR_SCALE_RAW        default: True

EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 1
fi

DATASET="$1"
STAGE1_DIR="$2"
STAGE2_DIR="$3"
DEVICE="${4:-cuda}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_OWNER_HOME=""
case "${REPO_ROOT}" in
  /home/*/*)
    REPO_OWNER_HOME="$(printf '%s\n' "${REPO_ROOT}" | cut -d/ -f1-3)"
    ;;
esac
if [[ -x "${HOME}/anaconda3/envs/clipclap/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/clipclap/bin/python}"
elif [[ -x "${HOME}/miniconda3/envs/clipclap/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/clipclap/bin/python}"
elif [[ -n "${REPO_OWNER_HOME}" && -x "${REPO_OWNER_HOME}/anaconda3/envs/clipclap/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_OWNER_HOME}/anaconda3/envs/clipclap/bin/python}"
elif [[ -n "${REPO_OWNER_HOME}" && -x "${REPO_OWNER_HOME}/miniconda3/envs/clipclap/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_OWNER_HOME}/miniconda3/envs/clipclap/bin/python}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

case "${DATASET}" in
  UCF)
    ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/UCF}"
    ;;
  VGGSound)
    ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/VGGSound}"
    ;;
  ActivityNet)
    ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/ActivityNet}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    usage
    exit 1
    ;;
esac

EXTRA_ARGS=()
if [[ -n "${SEMANTIC_FAR_FUSION:-}" ]]; then
  EXTRA_ARGS+=(
    --eval_semantic_far_fusion "${SEMANTIC_FAR_FUSION}"
    --eval_semantic_far_adapted_weight "${SEMANTIC_FAR_ADAPTED_WEIGHT:-1.0}"
    --eval_semantic_far_scale_raw "${SEMANTIC_FAR_SCALE_RAW:-True}"
  )
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" get_evaluation.py \
  --cfg config/clipclap.yaml \
  --load_path_stage_A "${STAGE1_DIR}" \
  --load_path_stage_B "${STAGE2_DIR}" \
  --dataset_name "${DATASET}" \
  --root_dir "${ROOT_DIR}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
