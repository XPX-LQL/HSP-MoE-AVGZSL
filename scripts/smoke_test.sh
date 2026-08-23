#!/usr/bin/env bash
set -euo pipefail

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

DEVICE="${1:-cpu}"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/UCF}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/smoke_test_ucf}"

mkdir -p "${LOG_DIR}"

echo "Running a tiny stage-1 smoke test on UCF"
echo "Device: ${DEVICE}"
echo "Root:   ${ROOT_DIR}"
echo "Logs:   ${LOG_DIR}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" main.py \
  --cfg config/clipclap.yaml \
  --device "${DEVICE}" \
  --root_dir "${ROOT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --dataset_name UCF \
  --epochs 1 \
  --n_batches 1 \
  --bs 4 \
  --eval_bs 4 \
  --eval_num_workers 0 \
  --use_wavcaps_embeddings True \
  --modality both \
  --word_embeddings both \
  --run stage-1
