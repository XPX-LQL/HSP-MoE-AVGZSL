# Hyperspherical Semantic-Preserving AVGZSL with AV-MoE Routing

This repository contains the cleaned research codebase for our audio-visual generalized zero-shot learning (AVGZSL) paper line:

**Hyperspherical semantic preservation for mitigating seen-class bias, together with semantic divergence-aware Audio-Visual Mixture-of-Experts (AV-MoE) routing for adaptive audio-visual fusion.**

The implementation is built on top of **CLIPCLAP-GZSL**: *Audio-Visual Generalized Zero-Shot Learning using Pre-Trained Large Multi-Modal Models* (Kurzendoerfer et al., CVPRW 2024). Our code keeps the original two-stage CLIPCLAP training pipeline and adds the current paper-critical modules.

## 1. Hardware and Software

The following machine was used during code cleanup and experiment organization. It can be treated as a reference environment for reproduction.

| Item | Specification |
| --- | --- |
| OS | Ubuntu 22.04 on WSL |
| CPU | 12th Gen Intel Core i7-12800HX |
| CPU Cores / Threads | 12 Cores / 24 Threads |
| Memory | 32 GB RAM |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB |
| Python | 3.8.3 |
| PyTorch | 1.7.1 + CUDA 11.0 |
| NumPy | 1.20.2 |
| Pandas | 1.2.4 |
| Scikit-learn | 1.3.0 |

Notes:

- GPU training is recommended.
- The scripts default to a local conda environment named `clipclap` when it exists.
- Feature extraction from raw audio/video uses a separate environment because the CLIP/CLAP/WavCaps dependencies are heavier.

## 2. Code Structure

```text
CLIPCLAP-GZSL/
|-- config/
|   |-- clipclap.yaml
|   `-- default.yaml
|-- scripts/
|   |-- reproduce.sh
|   |-- reproduce_semantic.sh
|   |-- evaluate_run.sh
|   |-- run_ablation_split_weight.sh
|   `-- smoke_test.sh
|-- src/
|   |-- args.py
|   |-- clipclap_model.py
|   |-- dataset.py
|   |-- loss.py
|   |-- train.py
|   |-- test.py
|   |-- utils.py
|   `-- utils_improvements.py
|-- clip_feature_extraction/
|-- clip_embeddings_extraction/
|-- splitting_scripts_cls/
|-- WavCaps/                         # optional external dependency, not tracked
|-- avgzsl_benchmark_non_averaged_datasets/
|-- data/
|-- models/
|-- logs/
|-- main.py
|-- get_evaluation.py
|-- clipclap.yml
|-- clipclap_feature_extraction.yml
|-- commands.sh
`-- README.md
```

The current GitHub version keeps the paper-critical and minimally reproducible workflow. Local datasets, extracted features, checkpoints, tensorboard events, and training logs are excluded from version control.

Core directories:

- `config/`: default model and training hyperparameters.
- `scripts/`: reproduction, evaluation, smoke-test, and ablation entry points.
- `src/`: model, data loading, losses, training, evaluation, and utility code.
- `clip_feature_extraction/`: optional feature extraction from raw videos/audio.
- `clip_embeddings_extraction/`: optional class text embedding extraction.
- `splitting_scripts_cls/`: dataset splitting and feature packaging helpers.
- `WavCaps/`: optional external CLAP/WavCaps code used by the upstream feature extraction workflow. It is not tracked by Git in this cleaned release.
- `data/`: local extracted CLIP/CLAP features, not tracked by Git.
- `models/`: local checkpoints, not tracked by Git.
- `logs/`: local training/evaluation logs, not tracked by Git.

## 3. Method Scope

The current paper workflow contains the following components:

1. **CLIPCLAP baseline**  
   The original two-stage AVGZSL pipeline based on CLIP/CLAP features and class-name embeddings.

2. **Hyperspherical semantic space**  
   Audio-visual representations and text class prototypes are normalized onto the unit hypersphere, making semantic matching depend mainly on angular direction instead of unconstrained feature norm.

3. **Distribution Preservation (DP)**  
   Distills the class-similarity distribution from raw foundation features, reducing the risk that seen-only training degenerates into hard-label fitting.

4. **Relation Preservation (RP)**  
   Preserves sample-level neighborhood relations from the foundation feature space to reduce semantic structure distortion after adaptation.

5. **Hyperspherical Uniformity Preservation (HUP)**  
   Encourages adapted features to remain appropriately dispersed on the unit hypersphere, alleviating over-concentration around seen-class regions.

6. **Foundation-Anchored Routing (FAR)**  
   Fuses adapted and raw foundation branches during evaluation to balance seen-class discrimination and unseen-class transfer.

7. **Semantic Divergence-aware AV-MoE Routing**  
   Keeps the original module interface name `semantic_consensus_routing`, while replacing its internal implementation with an Audio-Visual Mixture-of-Experts style router. The module uses unimodal and cross-modal experts, then dynamically weights expert outputs according to sample-level audio-visual semantic states.

## 4. Dataset

Raw datasets and extracted features are not shipped with this repository.

The training code expects pre-extracted CLIP/CLAP features under:

```text
data/
|-- UCF/
|-- ActivityNet/
`-- VGGSound/
```

Each dataset directory should contain:

```text
class-split/
features/
_features_processed/
```

You can use the original CLIPCLAP-GZSL released features or your own extracted features. The upstream CLIPCLAP project provides the prepared CLIP/CLAP features via Google Drive:

```text
https://drive.google.com/uc?export=download&id=1fNb3WvbN76yuPVi4MeVtgDycdX0jAE2G
```

After downloading, unzip the data and either place it under `data/` or pass a custom path through `ROOT_DIR` / `--root_dir`.

## 5. Environment Setup

We recommend using Anaconda or Miniconda.

### 5.1 Main training environment

```bash
conda env create -f clipclap.yml
conda activate clipclap
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 5.2 Optional feature-extraction environment

If you only train from already extracted CLIP/CLAP features, the main `clipclap` environment is enough.

If you need to extract CLIP/CLAP features from raw audio/video, create the separate feature-extraction environment:

```bash
conda env create -f clipclap_feature_extraction.yml
conda activate clipclap_feature_extraction
```

## 6. Training and Evaluation

First enter the project directory:

```bash
cd /path/to/CLIPCLAP-GZSL
```

### 6.1 Original CLIPCLAP baseline

```bash
bash scripts/reproduce.sh UCF cuda all
bash scripts/reproduce.sh ActivityNet cuda all
bash scripts/reproduce.sh VGGSound cuda all
```

### 6.2 Full model: hypersphere + DP/RP/HUP + FAR + AV-MoE

Single-dataset example:

```bash
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
bash scripts/reproduce_semantic.sh UCF cuda 42
```

Run all three datasets:

```bash
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
```

If a run is launched with `--run all`, evaluation is executed automatically after training.

### 6.3 Evaluate existing checkpoints

```bash
python get_evaluation.py \
  --cfg config/clipclap.yaml \
  --root_dir data/UCF \
  --dataset_name UCF \
  --load_path_stage_A /path/to/stage1 \
  --load_path_stage_B /path/to/stage2
```

You can also use the wrapper:

```bash
bash scripts/evaluate_run.sh UCF /path/to/stage1 /path/to/stage2 cuda
```

Enable FAR during evaluation:

```bash
SEMANTIC_FAR_FUSION=True \
SEMANTIC_FAR_ADAPTED_WEIGHT=0.5 \
SEMANTIC_FAR_SCALE_RAW=True \
bash scripts/evaluate_run.sh UCF /path/to/stage1 /path/to/stage2 cuda
```

### 6.4 Smoke test

For a tiny UCF stage-1 smoke test:

```bash
bash scripts/smoke_test.sh cuda
```

## 7. Current Results

The following numbers are from the current cleaned run with seed `42`. They are included to document the code state used during paper development.

### 7.1 CLIPCLAP baseline

| Dataset | Seen | Unseen | GZSL-HM | ZSL |
| --- | ---: | ---: | ---: | ---: |
| UCF | 77.14 | 43.91 | 55.97 | 46.96 |
| ActivityNet | 45.98 | 20.08 | 27.95 | 22.78 |
| VGGSound | 29.68 | 11.09 | 16.15 | 11.50 |

### 7.2 Full model

Full model means: hyperspherical semantic space + DP + RP + HUP + FAR + AV-MoE routing.

| Dataset | Seen | Unseen | GZSL-HM | ZSL | Best Beta |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCF | 81.54 | 56.90 | 67.03 | 59.16 | 0.2667 |
| ActivityNet | 52.74 | 27.99 | 36.57 | 31.28 | 0.1333 |
| VGGSound | 31.49 | 13.78 | 19.17 | 14.97 | 0.2000 |

### 7.3 Gain over baseline

| Dataset | Seen | Unseen | GZSL-HM | ZSL |
| --- | ---: | ---: | ---: | ---: |
| UCF | +4.40 | +12.99 | +11.06 | +12.20 |
| ActivityNet | +6.76 | +7.91 | +8.62 | +8.50 |
| VGGSound | +1.81 | +2.69 | +3.02 | +3.47 |

## 8. Ablation

The split-weight ablation script is provided for testing the contribution of the semantic preservation losses:

```bash
bash scripts/run_ablation_split_weight.sh cuda 42
```

Recommended ablation groups for the paper:

| Variant | Hypersphere | DP | RP | HUP | FAR | AV-MoE |
| --- | --- | --- | --- | --- | --- | --- |
| CLIPCLAP baseline | - | - | - | - | - | - |
| + Hypersphere | yes | - | - | - | - | - |
| + DP | yes | yes | - | - | - | - |
| + DP + RP | yes | yes | yes | - | - | - |
| + DP + RP + HUP | yes | yes | yes | yes | - | - |
| + FAR | yes | yes | yes | yes | yes | - |
| Full model | yes | yes | yes | yes | yes | yes |

## 9. Important Options

| Option | Meaning |
| --- | --- |
| `SEMANTIC_HYPERSPHERICAL=True` | Enables hyperspherical semantic matching. |
| `SEMANTIC_RADIAL_MODE=none` | Uses the direction-only hyperspherical setting. |
| `SEMANTIC_DISTILL_MODE=distribution` | Enables only distribution preservation. |
| `SEMANTIC_DISTILL_MODE=relation` | Enables only relation preservation. |
| `SEMANTIC_DISTILL_MODE=both` | Enables both DP and RP. |
| `SEMANTIC_DISTRIBUTION_WEIGHT` | Weight for distribution preservation. |
| `SEMANTIC_RELATION_WEIGHT` | Weight for relation preservation. |
| `SEMANTIC_HUP_WEIGHT` | Weight for HUP. |
| `SEMANTIC_FAR_FUSION=True` | Enables FAR during evaluation. |
| `SEMANTIC_FAR_ADAPTED_WEIGHT` | Weight of the adapted branch in FAR. |
| `SEMANTIC_CONSENSUS_ROUTING=True` | Enables the semantic divergence-aware AV-MoE module. |
| `SEMANTIC_CONSENSUS_WEIGHT` | Mixing weight of the AV-MoE-routed direction. |
| `SEMANTIC_CONSENSUS_TEMP` | Router temperature. |

## 10. Feature Extraction From Scratch

Feature extraction is optional. If needed, prepare the feature-extraction environment, place the external WavCaps code under `WavCaps/`, download the required WavCaps checkpoints, then run:

```bash
python clip_feature_extraction/get_clip_features_activitynet.py
python clip_feature_extraction/get_clip_features_ucf.py
python clip_feature_extraction/get_clip_features_vggsound.py
```

Package features into the expected dataset format:

```bash
python splitting_scripts_cls/create_pkl_files_cls.py \
  --dataset_name UCF \
  --path_original_dataset /path/to/original/features \
  --path_splitted_dataset /path/to/output/UCF
```

Extract class text embeddings:

```bash
python clip_embeddings_extraction/get_clip_embeddings_activitynet.py
python clip_embeddings_extraction/get_clip_embeddings_ucf.py
python clip_embeddings_extraction/get_clip_embeddings_vggsound.py
```

Some feature-extraction scripts are inherited from the upstream project and may contain original absolute paths. Update those paths before running them on a new machine.

## 11. Checkpoints and Logs

This repository does not track local checkpoints, extracted features, or training logs.

Use these local directories for experiments:

```text
data/
models/
logs/
```

If releasing trained weights, upload them to GitHub Releases, Google Drive, Hugging Face, or another external storage service, then link them here.


## 12. GitHub Release Notes

- Raw datasets and extracted CLIP/CLAP features are excluded from version control.
- Generated checkpoints, tensorboard events, logs, and result pickles are excluded from version control.
- Historical exploratory outputs are not part of the GitHub release.
- The repository is organized around the current paper workflow: hyperspherical semantic preservation, DP/RP/HUP losses, FAR, and AV-MoE routing.
