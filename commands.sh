#!/usr/bin/env bash

# Clean command examples for the GitHub release.
# Run commands from the repository root.

set -euo pipefail

###############################################################################
# 1. Baseline CLIPCLAP-GZSL
###############################################################################

bash scripts/reproduce.sh UCF cuda all
bash scripts/reproduce.sh ActivityNet cuda all
bash scripts/reproduce.sh VGGSound cuda all

###############################################################################
# 2. Full model: hypersphere + DP/RP/HUP + FAR + AV-MoE
###############################################################################

for DATASET in UCF ActivityNet VGGSound; do
  SEMANTIC_HYPERSPHERICAL=True \
  SEMANTIC_HYPERSPHERICAL_TEMP=0.1 \
  SEMANTIC_RADIAL_MODE=none \
  SEMANTIC_RADIAL_ALPHA=0.0 \
  SEMANTIC_DISTILL_MODE=both \
  SEMANTIC_DISTILL_WEIGHT=0 \
  SEMANTIC_DISTRIBUTION_WEIGHT=0.001 \
  SEMANTIC_RELATION_WEIGHT=0.0002 \
  SEMANTIC_HUP_WEIGHT=0.0002 \
  SEMANTIC_HUP_MODE=uniformity \
  SEMANTIC_HUP_TEMPERATURE=2.0 \
  SEMANTIC_FAR_FUSION=True \
  SEMANTIC_FAR_ADAPTED_WEIGHT=0.5 \
  SEMANTIC_FAR_SCALE_RAW=True \
  SEMANTIC_CONSENSUS_ROUTING=True \
  SEMANTIC_CONSENSUS_WEIGHT=0.5 \
  SEMANTIC_CONSENSUS_TEMP=1.0 \
  bash scripts/reproduce_semantic.sh "$DATASET" cuda 42
done

###############################################################################
# 3. Evaluate an existing run
###############################################################################

# bash scripts/evaluate_run.sh UCF /path/to/stage1 /path/to/stage2 cuda

# FAR-enabled evaluation:
# SEMANTIC_FAR_FUSION=True \
# SEMANTIC_FAR_ADAPTED_WEIGHT=0.5 \
# SEMANTIC_FAR_SCALE_RAW=True \
# bash scripts/evaluate_run.sh UCF /path/to/stage1 /path/to/stage2 cuda

###############################################################################
# 4. Smoke test
###############################################################################

# bash scripts/smoke_test.sh cuda
