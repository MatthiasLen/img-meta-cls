# `net4_duke` – Network v04 for the Duke Liver Dataset

This sub-package provides the complete training, inference, and cross-validation
evaluation pipeline for Network v04 applied specifically to the
**Duke Liver MRI Dataset**.

The scripts use the Duke-specific `LiverDataset` dataloader and run **5-fold
cross-validation** (CV) as the standard training protocol.

---

## Architecture overview

```
Multiple MRI slices (B x N x H x W)
	-> Shared CNN Backbone (for example DenseNet-121)
		-> Slice embeddings
			-> SliceFeatureFusion
				-> Fused image representation

DICOM metadata vector
	-> Metadata encoder (imputer or sparse)
		-> Metadata embedding

Cross-modal fusion
	-> image <-> metadata interaction
		-> MultiTaskHead
			-> per-task logits
```

The v04 family supports three experiment modes:

| `--modality` | Description |
|---|---|
| `combined` | Image + metadata fusion with cross-modal interaction |
| `image` | Image-only ablation |
| `metadata` | Metadata-only ablation |

For `combined` and `metadata`, the metadata branch can use either:

- an imputer-based encoder via `--metadata_enc_type imputer`, or
- a sparse encoder via `--metadata_enc_type sparse`.

---

## Relation to the paper

This module implements the proposed method and its direct ablations from
[arXiv:2602.23833](https://arxiv.org/abs/2602.23833). The table below maps each
paper experiment to its command-line configuration. All experiments use the
**Duke Liver MRI dataset** with 5-fold cross-validation and `SequenceType_Code_norm`
as the primary target.

| Paper exp. | Modality | Encoder | Imputer | Fusion | `--n_slices` | Duke F1 (%) |
|---|---|---|---|---|---|---|
| (1) 2D Image-only | image | — | — | — | 1 | 85.09 ± 1.31 |
| (4) Joint: Concat + zero imputation | combined | `imputer` | `ignore` | `concat` | 3 | 93.51 ± 1.89 |
| (5) Joint: Concat + learned imputation | combined | `imputer` | `contextual` | `concat` | 3 | 93.21 ± 3.48 |
| **Ours** (SME + BCA, proposed) | combined | `sparse` | — | `v1` | 10 | **96.66 ± 1.03** |

### Exact commands for paper experiments

**Experiment (1) — 2D image-only (`SimpleImageBasedClassifier`):**
```bash
python -m IMC.net4_duke.train \
    --modality image \
    --vanilla_image_classifier \
    --img_enc_backbone densenet121 \
    --n_slices 1 \
    --batch_size 64 \
    --gpu 0
```

**Experiment (4) — Joint with zero imputation:**
```bash
python -m IMC.net4_duke.train \
    --modality combined \
    --metadata_enc_type imputer \
    --imputer_type ignore \
    --fusion_module_version concat \
    --n_slices 3 \
    --batch_size 64 \
    --gpu 0
```

**Experiment (5) — Joint with learned (MLP) imputation:**
```bash
python -m IMC.net4_duke.train \
    --modality combined \
    --metadata_enc_type imputer \
    --imputer_type contextual \
    --fusion_module_version concat \
    --n_slices 3 \
    --batch_size 64 \
    --gpu 0
```

**Proposed method (Ours) — Sparse Metadata Encoder + Bi-directional Cross-modal Attention:**
```bash
python -m IMC.net4_duke.train \
    --modality combined \
    --metadata_enc_type sparse \
    --sparse_enc_version v1 \
    --fusion_module_version v2 \
    --n_slices 10 \
    --batch_size 16 \
    --gpu 0
```
The batch size is adjusted to 16 for the proposed method due to GPU memory constraints with 10 slices.
---

## Directory layout

```
IMC/net4_duke/
├── helper.py        – Shared build_model factory used by train and infer
├── train.py         – 5-fold CV training entry point
├── infer.py         – Inference entry point
├── summarize_cv.py  – CV fold aggregation and report generation
└── README.md        – This file
```
---

## 1. Training (5-fold cross-validation)

Training is launched with `python -m IMC.net4_duke.train`.  The script iterates
over the requested folds and for each fold:

- Assigns **test = fold_idx**, **val = (fold_idx + 1) % 5**, **train = remaining**.
- Creates a timestamped log directory `<log_dir>/<timestamp>_5fold_cv/fold_<i>/`.
- Writes `config.json` and `model_architecture.txt` for reproducibility.
- Trains with mixed precision, warmup + cosine-decay LR, and early stopping.
- Evaluates the best checkpoint on the test fold.
- Saves a `cv_summary.csv` at the top-level experiment directory.

### Minimal example – combined mode (all defaults)

```bash
python -m IMC.net4_duke.train \
    --modality combined \
    --gpu 0
```

### Combined mode with sparse encoder

```bash
python -m IMC.net4_duke.train \
    --modality combined \
    --img_enc_backbone densenet121 \
    --metadata_enc_type sparse \
    --sparse_enc_version v1 \
    --fusion_module_version v2 \
    --gpu 0
```

### Image-only with ResNet50

```bash
python -m IMC.net4_duke.train \
    --modality image \
    --img_enc_backbone resnet50 \
    --gpu 0
```

### Metadata-only

```bash
python -m IMC.net4_duke.train \
    --modality metadata \
    --metadata_enc_type sparse \
    --gpu 0
```

### Restart a single failed fold (e.g. fold 3)

```bash
python -m IMC.net4_duke.train \
    --modality combined \
    --folds 3 \
    --log_dir ./logs/ \
    --gpu 0
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--modality` | *(required)* | `combined` / `image` / `metadata` |
| `--img_enc_backbone` | `densenet121` | CNN backbone |
| `--batch_size` | `16` | Mini-batch size |
| `--num_epochs` | `30` | Maximum epochs per fold |
| `--lr` | `1e-6` | Base learning rate |
| `--gpu` | `0` | CUDA device index (`-1` = CPU) |
| `--folds` | all | Comma-separated fold indices, e.g. `0,1,2` |
| `--n_folds` | `5` | Total number of folds |
| `--patience` | `30` | Early-stopping patience (epochs) |
| `--log_dir` | `./logs` | Root output directory |
| `--metadata_enc_type` | `sparse` | `imputer` / `sparse` |
| `--sparse_enc_version` | `v1` | `v1` / `v2` / `v5` |
| `--fusion_module_version` | `v2` | `v1` / `v2` / `concat` |
| `--n_slices` | `10` | MRI slices sampled per volume (use `10` for the proposed method) |
| `--dataset_path` | env default | Override `LOCAL_DATASET_PATH` |
| `--label_csv_path` | env default | Override `LABEL_CSV_PATH` |
| `--metadata_path` | env default | Override `METADATA_PATH` |

---

## 2. Inference

Inference is launched with `python -m IMC.net4_duke.infer`.  The script:

- Loads the Duke dataset (optionally filtered to specific folds via `--folds`).
- Loads the trained checkpoint and builds the matching model architecture.
- Runs batch inference and writes `predictions.csv` to `--output_dir`.
- Optionally evaluates against ground-truth labels (`--run_eval`).

> **All model architecture arguments must exactly match the training invocation.**

### Infer on all data (no fold filter)

```bash
python -m IMC.net4_duke.infer \
    --ckpt ./logs/<run>/fold_0/best_model.pth \
    --output_dir ./results/fold_0 \
    --modality combined \
    --gpu 0
```

### Infer on fold 4 only and evaluate

```bash
python -m IMC.net4_duke.infer \
    --ckpt ./logs/<run>/fold_4/best_model.pth \
    --output_dir ./results/fold_4/evaluation \
    --modality combined \
    --folds 4 \
    --run_eval \
    --gpu 0
```

### Run inference for all 5 folds (CV evaluation loop)

```bash
for fold in 0 1 2 3 4; do
    python -m IMC.net4_duke.infer \
        --ckpt ./logs/<run>/fold_${fold}/best_model.pth \
        --output_dir ./logs/<run>/fold_${fold}/evaluation \
        --modality combined \
        --folds ${fold} \
        --run_eval \
        --gpu 0
done
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | *(required)* | Path to trained checkpoint |
| `--output_dir` | *(required)* | Directory for predictions.csv |
| `--modality` | *(required)* | `combined` / `image` / `metadata` |
| `--img_enc_backbone` | `densenet121` | Must match training |
| `--folds` | all data | Comma-separated fold indices |
| `--gpu` | `0` | CUDA device index |
| `--batch_size` | `16` | Inference mini-batch size |
| `--run_eval` | off | Evaluate against ground-truth labels |
| `--metadata_enc_type` | `sparse` | Must match training |
| `--fusion_module_version` | `v2` | Must match training |
| `--n_slices` | `10` | Must match training |

---

## 3. Cross-validation evaluation (`summarize_cv.py`)

After running inference with `--run_eval` for all folds, aggregate the
per-fold evaluation results into a single cross-validation report.

### Usage

```bash
python -m IMC.net4_duke.summarize_cv \
    ./logs/<timestamp>_5fold_cv \
    --num_folds 5 \
    --output_dir ./logs/<timestamp>_5fold_cv/cv_summary
```

Or more simply, using the default output directory (`<cv_dir>/cv_summary`):

```bash
python -m IMC.net4_duke.summarize_cv ./logs/<timestamp>_5fold_cv
```

### What it does

Reads `fold_*/evaluation/summary_metrics.csv` and
`fold_*/evaluation/detailed_metrics.csv` from each fold sub-directory, then:

| Output file | Description |
|---|---|
| `cv_statistics.csv` | Mean ± std of each metric across all folds |
| `per_class_statistics.csv` | Per-class precision / recall / F1 across folds |
| `per_fold_summary.csv` | Raw per-fold aggregate metrics |
| `per_fold_detailed_metrics.csv` | Raw per-fold per-class metrics |
| `fold_comparison_accuracy.png` | Accuracy per task across folds |
| `fold_comparison_macro_f1.png` | Macro F1 per task across folds |
| `fold_distribution_boxplots.png` | Box plots of accuracy and macro F1 |
| `performance_heatmap.png` | Tasks × Folds heatmap |
| `per_class_<task>.png` | Per-class F1 / precision / recall bar charts |
| `per_class_heatmap_all_tasks.png` | All-tasks per-class heatmap |
| `cv_summary_report.md` | Human-readable Markdown summary |

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `cv_dir` | *(required)* | Path to `<timestamp>_5fold_cv/` directory |
| `--num_folds` | `5` | Number of folds |
| `--output_dir` | `<cv_dir>/cv_summary` | Where to save outputs |
| `--eval_fold` | `evaluation` | Name of the evaluation sub-directory |

---

## Full end-to-end example

```bash
# 1. Train all 5 folds
python -m IMC.net4_duke.train \
    --modality combined \
    --img_enc_backbone densenet121 \
    --num_epochs 30 \
    --gpu 0

# 2. Run per-fold inference + evaluation
for fold in 0 1 2 3 4; do
    python -m IMC.net4_duke.infer \
        --ckpt ./logs/<run>/fold_${fold}/best_model.pth \
        --output_dir ./logs/<run>/fold_${fold}/evaluation \
        --modality combined \
        --folds ${fold} \
        --run_eval \
        --gpu 0
done

# 3. Aggregate CV results
python -m IMC.net4_duke.summarize_cv ./logs/<run>
```

---

## Environment variables

The dataset paths can be configured via environment variables.  CLI arguments
always take precedence:

| Variable | Default used by scripts | Override via |
|---|---|---|
| `LOCAL_DATASET_PATH` | Duke MRI root dir | `--dataset_path` |
| `LABEL_CSV_PATH` | Duke label CSV | `--label_csv_path` |
| `METADATA_PATH` | Duke encoded metadata parquet | `--metadata_path` |
| `DEBUG_MODE` | `"0"` (set to `"1"` with `--debug` in train) | `--debug` |
