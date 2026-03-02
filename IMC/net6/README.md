# IMC Net6 – PixelOnlyModel on Duke Liver MRI

## Overview

`net6` contains experiment scripts for training and evaluating
**Network version 6** ([`IMC.network06.PixelOnlyModel`](../network06.py)) on the
**Duke Liver MRI** dataset.

`PixelOnlyModel` is a **2-D image-only** classifier.  It encodes a single
equidistantly-sampled middle slice through a shared CNN backbone
(`MultiSliceImageEncoder`), mean-pools per-slice features, and passes the
pooled representation to a multi-task classification head.

Only the `SequenceType_Code_norm` task is trained in this experiment.  Focal
loss (α = 1.0, γ = 2.0) is used to handle the class-imbalanced distribution of
sequence types.

---

## Directory layout

```
IMC/net6/
├── __init__.py          – Package entry point (sub-module documentation)
├── train_duke.py        – 5-fold CV training of PixelOnlyModel on Duke
├── infer_duke.py        – Batch inference with optional RF gate
├── train_rf_duke.py     – Train per-fold Random Forest metadata classifiers
└── README.md            – This file
```

Superseded originals are kept as `.bak` files for reference.

---

## Architecture

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
MultiTaskHead             ← single linear classifier for SequenceType_Code_norm
[(B, n_classes)]          ← list of per-task logit tensors
```

---

## Heuristic RF gate (optional)

At inference time a separately-trained **Random Forest** can gate
`SequenceType_Code_norm` predictions:

- For each sample the RF scores its tabular metadata.
- If the RF's maximum class probability ≥ `--threshold` (default 0.7) the
  **RF prediction** is used.
- Otherwise the **image-model prediction** is used.

All other tasks always use the image-model predictions.

---

## Quick-start

### 1 – Train the image model (5-fold CV, all folds)

```bash
python -m IMC.net6.train_duke \
    --img_enc_backbone densenet121 \
    --gpu 0 \
    --num_epochs 25 \
    --batch_size 8
```

| Argument             | Default      | Description                                       |
|----------------------|--------------|---------------------------------------------------|
| `--img_enc_backbone` | `densenet121`| CNN backbone (e.g. `resnet50`, `densenet201`)     |
| `--batch_size`       | `8`          | Mini-batch size                                   |
| `--num_epochs`       | `25`         | Maximum epochs per fold                           |
| `--lr`               | `1e-6`       | Base learning rate for AdamW                      |
| `--weight_decay`     | `1e-2`       | AdamW weight decay                                |
| `--patience`         | `30`         | Early-stopping patience (epochs)                  |
| `--focal_gamma`      | `2.0`        | Focal loss γ parameter                            |
| `--folds`            | *(all)*      | Comma-separated fold indices, e.g. `0,1`          |
| `--gpu`              | `0`          | CUDA device index (`-1` for CPU)                  |
| `--log_dir`          | `./logs`     | Root log/checkpoint directory                     |

### 2 – Train the Random Forest metadata classifiers (optional)

```bash
python -m IMC.net6.train_rf_duke \
    --out_dir ./rf_checkpoints/duke_net6
```

| Argument       | Default    | Description                                              |
|----------------|------------|----------------------------------------------------------|
| `--out_dir`    | *(required)*| Directory to save per-fold RF joblib files              |
| `--use_selected`| `false`  | Use pre-selected metadata feature subset                 |
| `--folds`      | *(all)*    | Comma-separated fold indices                             |

### 3 – Inference (image model only)

```bash
python -m IMC.net6.infer_duke \
    --ckpt ./logs/.../fold_0/best_model.pth \
    --output_dir ./infer_out/fold_0
```

### 3b – Inference with RF gate

```bash
python -m IMC.net6.infer_duke \
    --ckpt ./logs/.../fold_0/best_model.pth \
    --output_dir ./infer_out/fold_0 \
    --rf_model ./rf_checkpoints/duke_net6/duke_metadata_rf_fold_0.joblib \
    --threshold 0.7
```

| Argument             | Default      | Description                                         |
|----------------------|--------------|-----------------------------------------------------|
| `--ckpt`             | *(required)* | Path to checkpoint (`best_model.pth`)               |
| `--output_dir`       | *(required)* | Directory where `predictions.csv` is saved          |
| `--img_enc_backbone` | `densenet121`| CNN backbone (must match checkpoint)                |
| `--batch_size`       | `16`         | Batch size for inference                            |
| `--rf_model`         | `""`         | Path to RF joblib file (leave empty to disable gate)|
| `--threshold`        | `0.7`        | Confidence threshold for RF gate                    |
| `--folds`            | *(all data)* | Comma-separated fold indices to infer on            |
| `--eval`             | `false`      | Run evaluation against ground truth after inference |
| `--gpu`              | `0`          | CUDA device index                                   |

---

## Dataset environment variables

| Variable             | Description                                   |
|----------------------|-----------------------------------------------|
| `LOCAL_DATASET_PATH` | Root folder of the Duke MRI image dataset     |
| `METADATA_PATH`      | Path to the Duke encoded metadata parquet file|
| `LABEL_CSV_PATH`     | Path to the Duke label CSV                    |
| `DEBUG_MODE`         | Set to `"1"` for extra debug output           |

All variables can be overridden via the corresponding `--dataset_path`,
`--metadata_path`, `--label_csv_path` CLI arguments.

---

## Output structure

```
logs/<timestamp>_net6_duke_5fold/
├── fold_0/
│   ├── config.json              – All hyperparameters for this fold
│   ├── model_architecture.txt   – Model string representation
│   ├── best_model.pth           – Best checkpoint (lowest val loss)
│   └── *.log / events.out.tfevents.*
├── fold_1/ … fold_4/
└── cv_summary.csv               – Per-fold test results
```

```
infer_out/fold_0/
├── predictions.csv              – Filepath + predicted class per task
└── inference_metadata.json      – Checkpoint path, task names, sample count
```
