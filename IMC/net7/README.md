# IMC Net7 – PyramidPooling3DClassifier on Duke and ADNI

## Overview

`net7` contains experiment scripts for training and evaluating
**Network version 7** ([`IMC.network07.PyramidPooling3DClassifier`](../network07.py))
on two volumetric MRI datasets:

| Dataset  | Tasks                                                                    |
|----------|--------------------------------------------------------------------------|
| **Duke** | All Duke liver MRI sequence classification tasks                          |
| **ADNI** | `label_AcquisitionPlane`, `label_SequenceContrast`, `label_Localizer`    |

Both datasets share the identical network architecture and training recipe.
One script invocation trains a single fold; run five invocations in parallel
(``--fold 0 … 4``) to complete a full 5-fold cross-validation.

---

## Directory layout

```
IMC/net7/
├── __init__.py        – Package entry point (sub-module documentation)
├── train_duke.py      – Single-fold CV training on Duke
├── infer_duke.py      – Batch inference on Duke
├── train_adni.py      – Single-fold CV training on ADNI
├── infer_adni.py      – Batch inference on ADNI
└── README.md          – This file
```

Superseded originals from `net7_duke/` and `net7_adni/` are kept in those
directories as reference (archived with `.bak` extension).

---

## Architecture

```
Input: (B, 1, D, H, W)    volumetric MRI series
            │
            ▼
3-D CNN Backbone           ← ResNet3D or DenseNet3D variant
(B, C_f, D', H', W')
            │
            ▼
3-D Pyramid Pooling        ← multi-scale spatial aggregation
(B, C_pp)
            │
            ▼
MLP Projection (optional)  ← Linear → BN → GELU → Dropout
(B, embedding_dim)
            │
            ▼
MultiTaskHead              ← one linear classifier per task
list[(B, n_classes_i)]     ← per-task logit tensors
```

**Backbone options** (``--backbone_type``):

| Value             | Description                                   |
|-------------------|-----------------------------------------------|
| `resnet`          | Custom ResNet3D (configurable via `--backbone_channels`, `--backbone_blocks`) |
| `densenet121`     | DenseNet-121 3-D variant                      |
| `densenet169`     | DenseNet-169 3-D variant                      |
| `densenet201`     | DenseNet-201 3-D variant                      |
| `densenet_custom` | DenseNet with custom block layout             |

---

## Training configuration

- AdamW optimiser with per-parameter-group weight decay (biases/norms excluded).
- Linear-warmup + cosine-decay LR schedule (10 % warmup steps).
- Multi-task cross-entropy loss with label smoothing = 0.1.
- Mixed precision via `torch.amp.GradScaler`.
- Early stopping controlled by `--patience`.
- Per-fold `config.json`, `model_architecture.txt`, and `fold_results.json`.

---

## Quick-start: Duke

### Train (single fold)

```bash
python -m IMC.net7.train_duke \
    --fold 0 \
    --backbone_type resnet \
    --gpu 0 \
    --num_epochs 50
```

### Train (all 5 folds in parallel)

```bash
for i in 0 1 2 3 4; do
    python -m IMC.net7.train_duke \
        --fold $i \
        --base_log_dir ./logs/net07_duke_cv \
        --gpu 0 &
done
```

| Argument           | Default       | Description                                          |
|--------------------|---------------|------------------------------------------------------|
| `--fold`           | *(required)*  | Test fold index (0–4)                                |
| `--backbone_type`  | `resnet`      | 3-D CNN backbone                                     |
| `--backbone_channels`| `32`        | Initial backbone channels                            |
| `--backbone_blocks`| `2 2 2 2`    | ResNet block counts per stage                        |
| `--growth_rate`    | `12`          | DenseNet growth rate k                               |
| `--embedding_dim`  | `512`         | MLP projection dimension                             |
| `--target_depth`   | `64`          | Target depth for 3-D volumes                         |
| `--augment_config` | `DEFAULT3D`   | 3-D augmentation config for train splits             |
| `--batch_size`     | `4`           | Mini-batch size                                      |
| `--num_epochs`     | `50`          | Maximum epochs per fold                              |
| `--lr`             | `1e-6`        | Base learning rate for AdamW                         |
| `--patience`       | `30`          | Early-stopping patience (epochs)                     |
| `--gpu`            | `0`           | CUDA device index (`-1` for CPU)                     |
| `--base_log_dir`   | auto          | Shared root log dir for all folds                    |

### Inference (Duke)

```bash
python -m IMC.net7.infer_duke \
    --ckpt ./logs/net07_duke_cv/fold_0/best_model.pth \
    --output_dir ./infer_out/duke/fold_0 \
    --split fold_0
```

| Argument           | Default       | Description                                          |
|--------------------|---------------|------------------------------------------------------|
| `--ckpt`              | *(required)*  | Path to checkpoint (`best_model.pth`)                |
| `--output_dir`        | *(required)*  | Directory where `predictions.csv` is saved           |
| `--split`             | *(all data)*  | Comma-separated fold names, e.g. `fold_0,fold_1`     |
| `--target_depth`      | `64`          | Must match training                                  |
| `--backbone_type`     | `resnet`      | Must match training                                  |
| `--backbone_channels` | `32`          | Must match training                                  |
| `--backbone_blocks`   | `2 2 2 2`     | ResNet block counts per stage (must match training)  |
| `--growth_rate`       | `12`          | DenseNet growth rate (must match training)           |
| `--embedding_dim`     | `512`         | MLP projection dimension (must match training)       |
| `--batch_size`        | `8`           | Batch size for inference                             |
| `--eval`              | `false`       | Run evaluation against ground truth                  |
| `--gpu`               | `0`           | CUDA device index                                    |

---

## Quick-start: ADNI

### Train (single fold)

```bash
python -m IMC.net7.train_adni \
    --fold 0 \
    --backbone_type resnet \
    --gpu 0 \
    --num_epochs 50
```

### Train (all 5 folds in parallel)

```bash
for i in 0 1 2 3 4; do
    python -m IMC.net7.train_adni \
        --fold $i \
        --base_log_dir ./logs/net07_adni_cv \
        --gpu 0 &
done
```

| Argument           | Default       | Description                                          |
|--------------------|---------------|------------------------------------------------------|
| `--fold`           | *(required)*  | Test fold index (0–4)                                |
| `--n_slices`       | `16`          | Slices per series (= depth for network07)            |
| `--img_size`       | `224`         | Spatial resolution                                   |
| `--backbone_type`  | `resnet`      | 3-D CNN backbone                                     |
| `--backbone_channels`| `32`        | Initial backbone channels                            |
| `--backbone_blocks`| `2 2 2 2`    | ResNet block counts per stage                        |
| `--growth_rate`    | `12`          | DenseNet growth rate k                               |
| `--embedding_dim`  | `512`         | MLP projection dimension                             |
| `--augment_config` | `DEFAULT3D`   | 3-D augmentation config for train splits             |
| `--batch_size`     | `4`           | Mini-batch size                                      |
| `--num_epochs`     | `50`          | Maximum epochs per fold                              |
| `--lr`             | `1e-6`        | Base learning rate for AdamW                         |
| `--patience`       | `30`          | Early-stopping patience (epochs)                     |
| `--gpu`            | `0`           | CUDA device index (`-1` for CPU)                     |
| `--base_log_dir`   | auto          | Shared root log dir for all folds                    |

### Inference (ADNI)

```bash
python -m IMC.net7.infer_adni \
    --ckpt ./logs/net07_adni_cv/fold_0/best_model.pth \
    --output_dir ./infer_out/adni/fold_0 \
    --split fold_0
```

| Argument           | Default       | Description                                          |
|--------------------|---------------|------------------------------------------------------|
| `--ckpt`              | *(required)*  | Path to checkpoint (`best_model.pth`)                |
| `--output_dir`        | *(required)*  | Directory where `predictions.csv` is saved           |
| `--split`             | *(all data)*  | Comma-separated fold names, e.g. `fold_0,fold_1`     |
| `--n_slices`          | `16`          | Must match training                                  |
| `--img_size`          | `224`         | Must match training                                  |
| `--backbone_type`     | `resnet`      | Must match training                                  |
| `--backbone_channels` | `32`          | Must match training                                  |
| `--backbone_blocks`   | `2 2 2 2`     | ResNet block counts per stage (must match training)  |
| `--growth_rate`       | `12`          | DenseNet growth rate (must match training)           |
| `--embedding_dim`     | `512`         | MLP projection dimension (must match training)       |
| `--batch_size`        | `8`           | Batch size for inference                             |
| `--eval`              | `false`       | Evaluate against ground truth (accuracy + macro-F1)  |
| `--gpu`               | `0`           | CUDA device index                                    |

---

## Dataset environment variables

### Duke

| Variable             | Description                                   |
|----------------------|-----------------------------------------------|
| `LOCAL_DATASET_PATH` | Root folder of the Duke MRI image dataset     |
| `LABEL_CSV_PATH`     | Path to the Duke label CSV                    |
| `DEBUG_MODE`         | Set to `"1"` for extra debug output           |

### ADNI

| Variable                 | Description                                   |
|--------------------------|-----------------------------------------------|
| `ADNI_LOCAL_DATASET_PATH`| Root folder of the ADNI brain MRI dataset     |
| `ADNI_LABEL_CSV_PATH`    | Path to the ADNI label CSV                    |
| `DEBUG_MODE`             | Set to `"1"` for extra debug output           |

All variables can be overridden via the corresponding `--dataset_path` and
`--label_csv_path` CLI arguments.

---

## Output structure

```
logs/net07_<dataset>_cv/
├── fold_0/
│   ├── config.json              – All hyperparameters for this fold
│   ├── model_architecture.txt   – Model string representation
│   ├── best_model.pth           – Best checkpoint (lowest val loss)
│   ├── fold_results.json        – Test-set results for this fold
│   └── *.log / events.out.tfevents.*
├── fold_1/ … fold_4/

infer_out/<dataset>/fold_0/
├── predictions.csv              – Filepath + predicted class per task
└── evaluation.csv               – Per-task accuracy and macro-F1 (if --eval)
```
