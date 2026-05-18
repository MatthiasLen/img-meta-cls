# IMC Net6 – Two-Stage CNN + RF Baseline

## Overview

`net6` implements the two-stage CNN + RF baseline for Duke Liver MRI series
classification:

1. **CNN image classifier** (`train_duke.py` / `infer_duke.py`): a 2D,
   single-slice image-only classifier based on
   [`IMC.network06.PixelOnlyModel`](../network06.py) with a DenseNet-121 backbone.
2. **Two-stage CNN + RF ensemble** (`train_rf_duke.py` + `infer_duke.py`): a
   separately-trained Random Forest on tabular DICOM metadata that gates the CNN
   predictions at inference time.

Both components target `SequenceType_Code_norm` only.  Focal loss (α = 1.0,
γ = 2.0) is used when training the CNN to handle the class-imbalanced
distribution.

> **Note:** Experiment (1) — the 2D image-only baseline — is implemented via
> `net4_duke` using `SimpleImageBasedClassifier` from `network04.py` with
> `--modality image`.  The CNN trained here in `net6` is used as the pixel
> classifier component of the two-stage system (experiment 6) only.

---

## Relation to the paper

These scripts implement baseline (6) from
[arXiv:2602.23833](https://arxiv.org/abs/2602.23833):

| Paper exp. | Description | Scripts | Duke weighted F1 (%) |
|---|---|---|
| (6) Two-stage [Miller et al.] | RF gates CNN predictions by metadata confidence | `train_duke.py` + `train_rf_duke.py` → `infer_duke.py --rf_model` | 87.01 ± 0.97 |

Experiment (6) re-implements the two-stage approach from:
> Miller et al., "Automated selection of abdominal MRI series using a DICOM
> metadata classifier and selective use of a pixel-based classifier."
> *Abdominal Radiology*, 2024.

---

## Directory layout

```
IMC/net6/
├── __init__.py          – Package entry point
├── train_duke.py        – 5-fold CV training of the CNN (PixelOnlyModel)
├── train_rf_duke.py     – Per-fold Random Forest training on DICOM metadata
├── infer_duke.py        – Inference: CNN only, or CNN gated by RF
└── README.md            – This file
```

---

## Architecture

### Component A – CNN image classifier (Experiment 1)

```
Input: (B, 1, C, H, W)   (single slice; n_slices=1)
          │
          ▼
MultiSliceImageEncoder    ← 2-D CNN backbone (default: DenseNet-121)
(B, 1, feat_dim)
          │
   Mean-pool over slices
(B, feat_dim)
          │
          ▼
MultiTaskHead             ← linear classifier for SequenceType_Code_norm
[(B, 13)]                 ← logits
```

### Component B – Random Forest metadata classifier (Experiment 6)

Trained separately on the encoded tabular DICOM metadata parquet (same file as
all other experiments).  Produces per-class probability scores for
`SequenceType_Code_norm`.

### Inference fusion – RF gate (Experiment 6)

At inference time, the RF gates CNN predictions per sample:

- If the RF's maximum class probability ≥ `--threshold` (default 0.7), the
  **RF prediction** is used.
- Otherwise, the **CNN prediction** is used.

This confidence-based gating follows the approach of Miller et al. [8].

---

## Quick-start

### Experiment (6): Two-stage CNN + RF baseline

**Step 1 – Train the CNN (5-fold CV)**

```bash
python -m IMC.net6.train_duke \
    --img_enc_backbone densenet121 \
    --gpu 0 \
    --num_epochs 25
```

**Step 2 – Run inference (CNN only, no RF gate)**

```bash
for fold in 0 1 2 3 4; do
    python -m IMC.net6.infer_duke \
        --ckpt ./logs/<run>/fold_${fold}/best_model.pth \
        --output_dir ./infer_out/net6/fold_${fold} \
        --folds ${fold} \
        --eval \
        --gpu 0
done
```

**Step 3 – Train the Random Forest**

```bash
python -m IMC.net6.train_rf_duke \
    --out_dir ./rf_checkpoints/duke_net6
```

**Step 3 – Run inference with RF gate**

```bash
for fold in 0 1 2 3 4; do
    python -m IMC.net6.infer_duke \
        --ckpt ./logs/<run>/fold_${fold}/best_model.pth \
        --output_dir ./infer_out/net6_rf/fold_${fold} \
        --rf_model ./rf_checkpoints/duke_net6/duke_metadata_rf_fold_${fold}.joblib \
        --threshold 0.7 \
        --folds ${fold} \
        --eval \
        --gpu 0
done
```

---

## Key arguments

### `train_duke.py` – CNN training

| Argument | Default | Description |
|---|---|---|
| `--img_enc_backbone` | `densenet121` | CNN backbone (e.g. `resnet50`, `densenet201`) |
| `--batch_size` | `8` | Mini-batch size |
| `--num_epochs` | `25` | Maximum epochs per fold |
| `--lr` | `1e-6` | Base learning rate (AdamW) |
| `--weight_decay` | `1e-2` | AdamW weight decay |
| `--patience` | `30` | Early-stopping patience (epochs) |
| `--focal_gamma` | `2.0` | Focal loss γ parameter |
| `--folds` | *(all)* | Comma-separated fold indices, e.g. `0,1` |
| `--gpu` | `0` | CUDA device index (`-1` for CPU) |
| `--log_dir` | `./logs` | Root log/checkpoint directory |

### `train_rf_duke.py` – RF training

| Argument | Default | Description |
|---|---|---|
| `--out_dir` | *(required)* | Directory for per-fold RF joblib files |
| `--folds` | *(all)* | Comma-separated fold indices |

### `infer_duke.py` – inference

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | *(required)* | Path to CNN checkpoint (`best_model.pth`) |
| `--output_dir` | *(required)* | Directory for `predictions.csv` |
| `--img_enc_backbone` | `densenet121` | CNN backbone (must match training) |
| `--batch_size` | `16` | Inference batch size |
| `--rf_model` | `""` | Path to RF joblib file (omit for CNN-only) |
| `--threshold` | `0.7` | RF confidence threshold for gating |
| `--folds` | *(all)* | Comma-separated fold indices to infer on |
| `--eval` | `false` | Evaluate against ground truth |
| `--gpu` | `0` | CUDA device index |

---

## Dataset environment variables

| Variable | Description |
|---|---|
| `LOCAL_DATASET_PATH` | Root folder of the Duke MRI image dataset |
| `METADATA_PATH` | Path to the encoded metadata parquet file |
| `LABEL_CSV_PATH` | Path to the Duke label CSV |
| `DEBUG_MODE` | Set to `"1"` for extra debug output |

CLI overrides: `--dataset_path`, `--metadata_path`, `--label_csv_path`.

---

## Output structure

```
logs/<timestamp>_net6_duke_5fold/
├── fold_0/
│   ├── config.json              – Hyperparameters for this fold
│   ├── model_architecture.txt   – Model string representation
│   ├── best_model.pth           – Best checkpoint (lowest val loss)
│   └── *.log
├── fold_1/ … fold_4/
└── cv_summary.csv               – Per-fold test results

rf_checkpoints/duke_net6/
├── duke_metadata_rf_fold_0.joblib
├── duke_metadata_rf_fold_1.joblib
└── …

infer_out/fold_0/
├── predictions.csv              – Filepath + predicted class
└── inference_metadata.json      – Checkpoint path, task names, sample count
```
