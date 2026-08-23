#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda}"
SEED="${2:-42}"

run_one() {
  local dataset="$1"
  local name="$2"
  local distill_mode="$3"
  local distribution_weight="$4"
  local relation_weight="$5"
  local hup_weight="$6"

  echo "========== ${dataset} : ${name} =========="
  SEMANTIC_HYPERSPHERICAL=True \
  SEMANTIC_HYPERSPHERICAL_TEMP=0.1 \
  SEMANTIC_RADIAL_MODE=none \
  SEMANTIC_RADIAL_ALPHA=0.0 \
  SEMANTIC_DISTILL_MODE="${distill_mode}" \
  SEMANTIC_DISTILL_WEIGHT=0 \
  SEMANTIC_DISTRIBUTION_WEIGHT="${distribution_weight}" \
  SEMANTIC_RELATION_WEIGHT="${relation_weight}" \
  SEMANTIC_HUP_WEIGHT="${hup_weight}" \
  SEMANTIC_HUP_MODE=uniformity \
  SEMANTIC_HUP_TEMPERATURE=2.0 \
  SEMANTIC_FAR_FUSION=True \
  SEMANTIC_FAR_ADAPTED_WEIGHT=0.5 \
  SEMANTIC_FAR_SCALE_RAW=True \
  bash scripts/reproduce_semantic.sh "${dataset}" "${DEVICE}" "${SEED}"
}

for dataset in UCF ActivityNet VGGSound; do
  run_one "${dataset}" "-Distribution" "relation" "0" "0.0002" "0.0002"
  run_one "${dataset}" "-Relation" "distribution" "0.001" "0" "0.0002"
  run_one "${dataset}" "-HUP" "both" "0.001" "0.0002" "0"
done
