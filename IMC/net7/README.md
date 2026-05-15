# IMC Net7 - PyramidPooling3DClassifier on Duke

`net7` contains the Duke Liver MRI training and inference scripts for the 3D
volumetric baseline described in the paper.

Unlike `net4_duke`, each invocation of `train_duke.py` trains a single fold.
Run one process per fold to complete the full 5-fold Duke evaluation.

---

## Included entry points

```text
IMC/net7/
├── __init__.py
├── train_duke.py
├── infer_duke.py
└── README.md
```

---

## Architecture

`IMC.network07.PyramidPooling3DClassifier` combines:

- a 3D CNN backbone,
- 3D pyramid pooling,
- an optional projection MLP,
- a multi-task classification head.

```text
Input volume (B, 1, D, H, W)
    -> 3D CNN backbone
    -> 3D pyramid pooling
    -> projection MLP
    -> MultiTaskHead
    -> per-task logits
```

Supported backbone choices:

| `--backbone_type` | Description |
|---|---|
| `resnet` | Custom ResNet3D backbone |
| `densenet121` | DenseNet-121 3D variant |
| `densenet169` | DenseNet-169 3D variant |
| `densenet201` | DenseNet-201 3D variant |
| `densenet_custom` | DenseNet with custom block layout |

---

## Training configuration

The Duke net7 training script uses:

- AdamW with weight-decay grouping,
- linear warmup plus cosine LR decay,
- multi-task cross-entropy with label smoothing,
- mixed precision,
- early stopping,
- per-fold config and checkpoint outputs.

---

## Quick-start

### Train one fold

```bash
python -m IMC.net7.train_duke \
    --fold 0 \
    --backbone_type resnet \
    --gpu 0 \
    --num_epochs 50
```

### Train all 5 folds

```bash
for i in 0 1 2 3 4; do
    python -m IMC.net7.train_duke \
        --fold $i \
        --base_log_dir ./logs/net07_duke_cv \
        --gpu 0 &
done
```

### Run inference

```bash
python -m IMC.net7.infer_duke \
    --ckpt ./logs/net07_duke_cv/fold_0/best_model.pth \
    --output_dir ./infer_out/duke/fold_0 \
    --split fold_0
```

---

## Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--fold` | required | Test fold index |
| `--n_folds` | `5` | Number of folds |
| `--backbone_type` | `resnet` | 3D CNN backbone |
| `--backbone_channels` | `32` | Initial backbone channels |
| `--backbone_blocks` | `2 2 2 2` | ResNet block counts |
| `--growth_rate` | `12` | DenseNet growth rate |
| `--embedding_dim` | `512` | Projection dimension |
| `--target_depth` | `64` | Target depth for input volumes |
| `--augment_config` | `DEFAULT3D` | 3D augmentation preset |
| `--batch_size` | `4` | Mini-batch size |
| `--num_epochs` | `50` | Maximum training epochs |
| `--lr` | `1e-6` | Base learning rate |
| `--patience` | `30` | Early-stopping patience |
| `--base_log_dir` | auto | Shared output directory for all folds |

## Key inference arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | required | Path to trained checkpoint |
| `--output_dir` | required | Directory for predictions |
| `--split` | all data | Comma-separated fold names |
| `--target_depth` | `64` | Must match training |
| `--backbone_type` | `resnet` | Must match training |
| `--backbone_channels` | `32` | Must match training |
| `--backbone_blocks` | `2 2 2 2` | Must match training |
| `--growth_rate` | `12` | Must match training |
| `--embedding_dim` | `512` | Must match training |
| `--batch_size` | `8` | Inference batch size |
| `--eval` | off | Evaluate predictions against labels |

---

## Required dataset configuration

The Duke net7 scripts expect explicit dataset configuration through environment
variables or CLI overrides.

Required environment variables:

- `LOCAL_DATASET_PATH`
- `LABEL_CSV_PATH`

Equivalent CLI overrides:

- `--dataset_path`
- `--label_csv_path`

---

## Output structure

```text
logs/net07_duke_cv/
├── fold_0/
│   ├── config.json
│   ├── model_architecture.txt
│   ├── best_model.pth
│   └── fold_results.json
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

```text
infer_out/duke/fold_0/
├── predictions.csv
└── evaluation.csv  # when --eval is enabled
```

---

## Notes

- This public repository snapshot documents only the Duke workflow.
- The scripts no longer assume author-specific filesystem defaults; dataset
  paths must be provided explicitly.
