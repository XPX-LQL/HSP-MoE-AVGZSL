#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce.sh <UCF|VGGSound|ActivityNet> [device] [run_mode]

Examples:
  bash scripts/reproduce.sh UCF cuda
  bash scripts/reproduce.sh VGGSound cuda:0 all
  bash scripts/reproduce.sh ActivityNet cpu stage-1

Arguments:
  dataset   One of: UCF, VGGSound, ActivityNet
  device    Optional, defaults to cuda
  run_mode  Optional, defaults to all
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 1
fi

DATASET="$1"
DEVICE="${2:-cuda}"
RUN_MODE="${3:-all}"

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
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ClipClap_UCF}"
    EPOCHS="${EPOCHS:-20}"
    LR="${LR:-0.00007}"
    ;;
  VGGSound)
    ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/VGGSound}"
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ClipClap_VGGSound}"
    EPOCHS="${EPOCHS:-15}"
    LR="${LR:-0.0001}"
    ;;
  ActivityNet)
    ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/ActivityNet}"
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ClipClap_ActivityNet}"
    EPOCHS="${EPOCHS:-15}"
    LR="${LR:-0.0001}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    usage
    exit 1
    ;;
esac

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "Dataset directory not found: ${ROOT_DIR}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Running reproduction for ${DATASET}"
echo "Python:  ${PYTHON_BIN}"
echo "Device:  ${DEVICE}"
echo "Root:    ${ROOT_DIR}"
echo "Logs:    ${LOG_DIR}"
echo "Epochs:  ${EPOCHS}"
echo "LR:      ${LR}"
echo "Run:     ${RUN_MODE}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" main.py \
  --cfg config/clipclap.yaml \
  --device "${DEVICE}" \
  --root_dir "${ROOT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --dataset_name "${DATASET}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --use_wavcaps_embeddings True \
  --modality both \
  --word_embeddings both \
  --run "${RUN_MODE}"
