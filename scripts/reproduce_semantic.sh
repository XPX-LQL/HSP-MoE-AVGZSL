#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_semantic.sh <UCF|VGGSound|ActivityNet> [device] [seed]

Examples:
  bash scripts/reproduce_semantic.sh UCF cuda
  bash scripts/reproduce_semantic.sh UCF cuda 42

Environment overrides:
  SEMANTIC_DISTILL_WEIGHT       default: 0.005
  SEMANTIC_DISTRIBUTION_WEIGHT  optional override for distribution distillation
  SEMANTIC_RELATION_WEIGHT      optional override for relation distillation
  SEMANTIC_DISTILL_MODE         default: distribution
  SEMANTIC_HYPERSPHERICAL       default: True
  SEMANTIC_HYPERSPHERICAL_TEMP  default: 0.2 for UCF, 1.0 otherwise
  SEMANTIC_RADIAL_MODE          default: ce (none=hard hypersphere, norm=legacy radius in embeddings)
  SEMANTIC_RADIAL_ALPHA         default: 0.5
  AV_FUSION_MODULE              default: False
  AV_FUSION_DROPOUT             default: 0.1
  SEMANTIC_CONSENSUS_ROUTING    default: False
  SEMANTIC_CONSENSUS_WEIGHT     default: 0.0
  SEMANTIC_CONSENSUS_TEMP       default: 1.0
  SEMANTIC_HUP_WEIGHT           default: 0.0
  SEMANTIC_HUP_MODE             default: uniformity
  SEMANTIC_HUP_TEMPERATURE      default: 2.0
  SEMANTIC_HUP_MARGIN           default: 0.0
  SEMANTIC_HUP_CROSS_CLASS_ONLY default: True
  SEMANTIC_FAR_FUSION           default: False
  SEMANTIC_FAR_ADAPTED_WEIGHT   default: 1.0
  SEMANTIC_FAR_SCALE_RAW        default: True
  INIT_FROM_AUTHOR              set to True to fine-tune from the author's released checkpoints
  STAGE2_EPOCHS                 optional override, e.g. 15 for UCF
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 1
fi

DATASET="$1"
DEVICE="${2:-cuda}"
SEED="${3:-42}"

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
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ClipClap_UCF_semantic}"
    AUTHOR_MODEL_DIR="${REPO_ROOT}/models/ClipClap_ucf"
    EPOCHS="${EPOCHS:-20}"
    LR="${LR:-0.00007}"
    HYPER_TEMP_DEFAULT="0.2"
    ;;
  VGGSound)
    ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/VGGSound}"
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ClipClap_VGGSound_semantic}"
    AUTHOR_MODEL_DIR="${REPO_ROOT}/models/ClipClap_vggsound"
    EPOCHS="${EPOCHS:-15}"
    LR="${LR:-0.0001}"
    HYPER_TEMP_DEFAULT="1.0"
    ;;
  ActivityNet)
    ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/data/ActivityNet}"
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ClipClap_ActivityNet_semantic}"
    AUTHOR_MODEL_DIR="${REPO_ROOT}/models/ClipClap_activitynet"
    EPOCHS="${EPOCHS:-15}"
    LR="${LR:-0.0001}"
    HYPER_TEMP_DEFAULT="1.0"
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
HYPER_TEMP="${SEMANTIC_HYPERSPHERICAL_TEMP:-${HYPER_TEMP_DEFAULT}}"
RADIAL_MODE="${SEMANTIC_RADIAL_MODE:-ce}"
RADIAL_ALPHA="${SEMANTIC_RADIAL_ALPHA:-0.5}"
DISTILL_NAME="${SEMANTIC_DISTILL_MODE:-distribution}"
if [[ -n "${SEMANTIC_DISTRIBUTION_WEIGHT:-}" || -n "${SEMANTIC_RELATION_WEIGHT:-}" ]]; then
  DISTILL_NAME="${DISTILL_NAME}_dw_${SEMANTIC_DISTRIBUTION_WEIGHT:-fallback}_rw_${SEMANTIC_RELATION_WEIGHT:-fallback}"
fi

EXTRA_ARGS=()
if [[ -n "${STAGE2_EPOCHS:-}" ]]; then
  EXTRA_ARGS+=(--stage2_epochs "${STAGE2_EPOCHS}")
fi
if [[ -n "${SEMANTIC_DISTRIBUTION_WEIGHT:-}" ]]; then
  EXTRA_ARGS+=(--semantic_distribution_weight "${SEMANTIC_DISTRIBUTION_WEIGHT}")
fi
if [[ -n "${SEMANTIC_RELATION_WEIGHT:-}" ]]; then
  EXTRA_ARGS+=(--semantic_relation_weight "${SEMANTIC_RELATION_WEIGHT}")
fi
if [[ "${INIT_FROM_AUTHOR:-False}" == "True" || "${INIT_FROM_AUTHOR:-False}" == "true" || "${INIT_FROM_AUTHOR:-False}" == "1" ]]; then
  EXTRA_ARGS+=(
    --init_model_path_stage_A "${AUTHOR_MODEL_DIR}/stage1/Clip_model_score.pt"
    --init_model_path_stage_B "${AUTHOR_MODEL_DIR}/stage2/Clip_model_score.pt"
  )
fi

echo "Running semantic improved CLIPCLAP for ${DATASET}"
echo "Python:        ${PYTHON_BIN}"
echo "Device:        ${DEVICE}"
echo "Root:          ${ROOT_DIR}"
echo "Logs:          ${LOG_DIR}"
echo "Epochs:        ${EPOCHS}"
echo "LR:            ${LR}"
echo "Seed:          ${SEED}"
echo "Distill weight:${SEMANTIC_DISTILL_WEIGHT:-0.005}"
echo "Distribution w:${SEMANTIC_DISTRIBUTION_WEIGHT:-fallback}"
echo "Relation w:    ${SEMANTIC_RELATION_WEIGHT:-fallback}"
echo "Distill mode:  ${SEMANTIC_DISTILL_MODE:-distribution}"
echo "Hypersphere:   ${SEMANTIC_HYPERSPHERICAL:-True}"
echo "Hyper temp:    ${HYPER_TEMP}"
echo "Radial mode:   ${RADIAL_MODE}"
echo "Radial alpha:  ${RADIAL_ALPHA}"
echo "AV fusion:     ${AV_FUSION_MODULE:-False}"
echo "AV fuse drop:  ${AV_FUSION_DROPOUT:-0.1}"
echo "Consensus:     ${SEMANTIC_CONSENSUS_ROUTING:-False}"
echo "Consensus w:   ${SEMANTIC_CONSENSUS_WEIGHT:-0.0}"
echo "Consensus tmp: ${SEMANTIC_CONSENSUS_TEMP:-1.0}"
echo "HUP weight:    ${SEMANTIC_HUP_WEIGHT:-0.0}"
echo "HUP mode:      ${SEMANTIC_HUP_MODE:-uniformity}"
echo "HUP temp:      ${SEMANTIC_HUP_TEMPERATURE:-2.0}"
echo "HUP margin:    ${SEMANTIC_HUP_MARGIN:-0.0}"
echo "FAR fusion:    ${SEMANTIC_FAR_FUSION:-False}"
echo "FAR adapted w: ${SEMANTIC_FAR_ADAPTED_WEIGHT:-1.0}"
echo "Author init:   ${INIT_FROM_AUTHOR:-False}"

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
  --av_fusion_module "${AV_FUSION_MODULE:-False}" \
  --av_fusion_dropout "${AV_FUSION_DROPOUT:-0.1}" \
  --semantic_consensus_routing "${SEMANTIC_CONSENSUS_ROUTING:-False}" \
  --semantic_consensus_weight "${SEMANTIC_CONSENSUS_WEIGHT:-0.0}" \
  --semantic_consensus_temperature "${SEMANTIC_CONSENSUS_TEMP:-1.0}" \
  --seed "${SEED}" \
  --semantic_distill_loss True \
  --semantic_distill_weight "${SEMANTIC_DISTILL_WEIGHT:-0.005}" \
  --semantic_distill_mode "${SEMANTIC_DISTILL_MODE:-distribution}" \
  --semantic_distill_temperature 2.0 \
  --semantic_hyperspherical "${SEMANTIC_HYPERSPHERICAL:-True}" \
  --semantic_hyperspherical_temperature "${HYPER_TEMP}" \
  --semantic_radial_mode "${RADIAL_MODE}" \
  --semantic_radial_alpha "${RADIAL_ALPHA}" \
  --semantic_hup_loss True \
  --semantic_hup_weight "${SEMANTIC_HUP_WEIGHT:-0.0}" \
  --semantic_hup_mode "${SEMANTIC_HUP_MODE:-uniformity}" \
  --semantic_hup_temperature "${SEMANTIC_HUP_TEMPERATURE:-2.0}" \
  --semantic_hup_margin "${SEMANTIC_HUP_MARGIN:-0.0}" \
  --semantic_hup_cross_class_only "${SEMANTIC_HUP_CROSS_CLASS_ONLY:-True}" \
  --semantic_far_fusion "${SEMANTIC_FAR_FUSION:-False}" \
  --semantic_far_adapted_weight "${SEMANTIC_FAR_ADAPTED_WEIGHT:-1.0}" \
  --semantic_far_scale_raw "${SEMANTIC_FAR_SCALE_RAW:-True}" \
  "${EXTRA_ARGS[@]}" \
  --exp_name "hyper_${SEMANTIC_HYPERSPHERICAL:-True}_temp_${HYPER_TEMP}_radial_${RADIAL_MODE}_${RADIAL_ALPHA}_${DISTILL_NAME}_consensus_${SEMANTIC_CONSENSUS_ROUTING:-False}_${SEMANTIC_CONSENSUS_WEIGHT:-0.0}_${SEMANTIC_CONSENSUS_TEMP:-1.0}_hup_${SEMANTIC_HUP_MODE:-uniformity}_${SEMANTIC_HUP_WEIGHT:-0.0}_seed${SEED}" \
  --run all
