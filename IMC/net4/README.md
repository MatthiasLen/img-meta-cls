# Network v04 – Training & Inference Guide

Network **v04** is the cross-attention fusion architecture for DICOM MRI series
classification.  This directory contains a unified `train.py` and `infer.py`
that supersede the four separate training scripts
(`train04_baseline_multi_slices.py`, `train04_image.py`,
`train04_metadata.py`, `train04_sparse_metadata_encoder.py`).

Model construction is delegated to `helper.py:build_model`, which is shared by
both `train.py` and `infer.py` to guarantee architectural parity.

---

## Architecture overview

```
Multiple MRI slices (B × N × H × W)
    └─► Shared CNN Backbone (e.g. DenseNet-121)
            └─► Slice Embeddings
                    └─► SliceFeatureFusion (multi-head self-attention)
                                └─► Fused image vector (f_img)

DICOM Metadata vector
    └─► Metadata Encoder (sparse / imputer)
                └─► Metadata embedding (f_meta)

BiDirectionalCrossModalAttentionFusion / SimpleConcatFusion
    └─► f_img ⟷ f_meta
            └─► MultiTaskHead
                    ├─► label_SequenceType  (softmax)
                    ├─► label_AcquisitionPlane (softmax)
                    ├─► label_BodyRegion   (softmax)
                    ├─► label_ContrastPhase (softmax or regression)
                    └─► … (additional tasks)
```

---

## Training modalities

| `--modality` | Model class(es) | Description |
|---|---|---|
| `combined` | `MRISequenceClassifier` | Image + metadata fusion. Metadata encoding determined by `--metadata_enc_type` (`sparse` or `imputer`). Fusion style set by `--fusion_module_version`. |
| `image` | `ImageBasedClassifier` / `SimpleImageBasedClassifier` | Image-only ablation (no metadata). Pass `--vanilla_image_classifier` for the simpler variant. |
| `metadata` | `MetadataBasedClassifier` | Metadata-only ablation (no image). |

> **Migration note** – the former `baseline` mode (dense metadata + imputer) is
> now `--modality combined --metadata_enc_type imputer`, and the former `sparse`
> mode is `--modality combined --metadata_enc_type sparse` (the default).

---

## Prerequisites

Install the package and its dependencies:

```bash
cd /path/to/IMC
pip install -e .
```

Set the following environment variables (or pass the equivalent CLI flags):

```bash
export LOCAL_DATASET_PATH=/path/to/image/data
export METADATA_PATH=/path/to/encoded_metadata.parquet
export LABEL_CSV_PATH=/path/to/labels.csv
```

---

## Training

All training is done through `train.py`.  The `--modality` argument selects the
model variant.  Architecture arguments **must** be noted and passed identically
to `infer.py` later.

### Common arguments

| Argument | Default | Description |
|---|---|---|
| `--modality` | *(required)* | Training modality: `combined`, `image`, or `metadata`. |
| `--img_enc_backbone` | `densenet121` | CNN backbone identifier. |
| `--batch_size` | `16` | Mini-batch size. |
| `--num_epochs` | `15` | Maximum training epochs. |
| `--lr` | `1e-6` | Base AdamW learning rate. |
| `--gpu` | `0` | CUDA device index (`-1` → CPU). |
| `--ckpt` | `None` | Resume from checkpoint path. |
| `--patience` | `5` | Early-stopping patience (epochs). |
| `--log_dir` | `./logs` | Root directory for logs and checkpoints. |
| `--incl_regression` | off | Add regression head for `label_ContrastPhase`. |
| `--debug` | off | Enable verbose debug output. |
| `--dataset_path` | env | Override `LOCAL_DATASET_PATH`. |
| `--metadata_path` | env | Override `METADATA_PATH`. |
| `--label_csv_path` | env | Override `LABEL_CSV_PATH`. |
| `--use_preselected_features` | off | Use reduced metadata feature set. |
| `--num_workers` | `4` | DataLoader workers. |

### Modality-specific arguments

#### `combined`

| Argument | Default | Description |
|---|---|---|
| `--metadata_enc_type` | `sparse` | `sparse` (sparse encoder) or `imputer` (contextual imputer). |
| `--fusion_module_version` | `concat` | `v1`, `v2`, or `concat`. |
| `--sparse_enc_version` | `v1` | Sparse encoder version: `v1`, `v2`, or `v5` (used when `--metadata_enc_type sparse`). |
| `--imputer_type` | `contextual` | Imputer variant: `contextual` or `ignore` (used when `--metadata_enc_type imputer`). |
| `--metadata_dropout` | off | Apply dropout to metadata features. |

#### `image`

| Argument | Default | Description |
|---|---|---|
| `--vanilla_image_classifier` | off | Use the simpler `SimpleImageBasedClassifier` (no fusion module). |

#### `metadata`

| Argument | Default | Description |
|---|---|---|
| `--metadata_enc_type` | `sparse` | `imputer` or `sparse`. |
| `--sparse_enc_version` | `v1` | Sparse encoder version (when `--metadata_enc_type sparse`). |
| `--metadata_embed_dim` | `128` | Metadata encoder output dimension. |
| `--output_emb_dim` | `256` | Final projection dimension. |

### Examples

```bash
# Combined mode – image + sparse metadata encoder v1, concat fusion
python -m IMC.net4.train \
    --modality combined \
    --img_enc_backbone densenet121 \
    --sparse_enc_version v1 \
    --fusion_module_version concat \
    --batch_size 16 \
    --num_epochs 15 \
    --gpu 0

# Combined mode with contextual imputer, bi-directional cross-attention v1
python -m IMC.net4.train \
    --modality combined \
    --metadata_enc_type imputer \
    --fusion_module_version v1 \
    --num_epochs 20 \
    --gpu 1

# Image-only ablation (no metadata)
python -m IMC.net4.train \
    --modality image \
    --batch_size 32 \
    --num_epochs 20 \
    --gpu 0

# Image-only – simpler vanilla variant
python -m IMC.net4.train \
    --modality image \
    --vanilla_image_classifier \
    --batch_size 32 \
    --gpu 0

# Metadata-only with sparse encoder
python -m IMC.net4.train \
    --modality metadata \
    --metadata_enc_type sparse \
    --num_epochs 30 \
    --gpu 0

# Resume from checkpoint
python -m IMC.net4.train \
    --modality combined \
    --ckpt /path/to/logs/combined_densenet121_20260101_120000/best_model.pth \
    --gpu 0
```

### Outputs

Each run creates a timestamped sub-directory under `--log_dir`:

```
logs/
└── combined_densenet121_20260101_120000/
    ├── config.json            # full configuration snapshot
    ├── model_architecture.txt # model.__str__() output
    ├── best_model.pth         # best checkpoint (model + scaler state)
    ├── <experiment>.log       # text log
    └── events.out.tfevents.*  # TensorBoard events
```

Visualise training with TensorBoard:

```bash
tensorboard --logdir ./logs
```

---

## Inference

`infer.py` runs batch inference on a dataset split.  **All architecture
arguments must match the training run.**

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | *(required)* | Path to trained checkpoint `.pth`. |
| `--output_dir` | *(required)* | Directory to save `predictions.csv`. |
| `--modality` | *(required)* | Must match training modality. |
| `--img_enc_backbone` | `densenet121` | Must match training. |
| `--fusion_module_version` | `concat` | Must match training (combined mode). |
| `--metadata_enc_type` | `sparse` | Must match training (combined / metadata). |
| `--sparse_enc_version` | `v1` | Must match training (combined / metadata with sparse). |
| `--metadata_embed_dim` | `128` | Must match training (metadata mode). |
| `--output_emb_dim` | `256` | Must match training (metadata mode). |
| `--vanilla_image_classifier` | off | Must match training (image mode). |
| `--incl_regression` | off | Must match training. |
| `--dataset` | env / `pvai` | `pvai`, `duke`, `prostate_x`, `prostate_mri`, `lhic`. |
| `--dataset_path` | env | Override `LOCAL_DATASET_PATH`. |
| `--metadata_path` | env | Override `METADATA_PATH`. |
| `--label_csv_path` | env | Override `LABEL_CSV_PATH`. |
| `--batch_size` | `16` | Inference mini-batch size. |
| `--num_workers` | `4` | DataLoader workers. |
| `--gpu` | `0` | CUDA device index (`-1` → CPU). |
| `--use_preselected_features` | off | Must match training. |
| `--run_eval` | off | Run evaluation after inference. |

### Examples

```bash
# Combined model (sparse encoder) on PV.AI (default)
python -m IMC.net4.infer \
    --ckpt ./logs/combined_densenet121_20260101_120000/best_model.pth \
    --output_dir ./results/combined_pvai \
    --modality combined \
    --img_enc_backbone densenet121 \
    --sparse_enc_version v1 \
    --fusion_module_version concat

# Combined model (sparse encoder) on Duke dataset with evaluation
python -m IMC.net4.infer \
    --ckpt ./logs/combined_densenet121_20260101_120000/best_model.pth \
    --output_dir ./results/combined_duke \
    --modality combined \
    --dataset duke \
    --run_eval

# Combined model with contextual imputer on ProstateX
python -m IMC.net4.infer \
    --ckpt ./logs/combined_imputer_densenet121_20260101_130000/best_model.pth \
    --output_dir ./results/combined_imputer_prostate_x \
    --modality combined \
    --metadata_enc_type imputer \
    --fusion_module_version v1 \
    --dataset prostate_x

# Image-only
python -m IMC.net4.infer \
    --ckpt ./logs/image_densenet121_20260101_140000/best_model.pth \
    --output_dir ./results/image_pvai \
    --modality image

# Metadata-only
python -m IMC.net4.infer \
    --ckpt ./logs/metadata_densenet121_20260101_150000/best_model.pth \
    --output_dir ./results/metadata_pvai \
    --modality metadata \
    --metadata_enc_type sparse
```

### Output

`predictions.csv` is written to `--output_dir`:

| Filepath | label_SequenceType | label_AcquisitionPlane | label_BodyRegion | label_ContrastPhase | label_Contrast | … |
|---|---|---|---|---|---|---|
| /data/series_001 | T1 | AX | ABD | pre | pre | … |

If `--run_eval` is passed and a ground-truth CSV exists at `LABEL_CSV_PATH`,
evaluation metrics are also written to the same directory.

---

## Running tests

Unit and integration tests are located under `tests/` at the root of the IMC
repository:

```bash
# Run all net4 tests from the repo root
pytest tests/test_net4.py -v

# Skip slow forward-pass tests (faster CI)
pytest tests/test_net4.py -v -m "not slow"

# Run with coverage
pytest tests/test_net4.py --cov=IMC.net4 --cov-report=term-missing
```

---

## File overview

| File | Description |
|---|---|
| `train.py` | Unified training entry point (all modalities). |
| `infer.py` | Unified inference entry point (all modalities, all datasets). |
| `helper.py` | Shared `build_model` factory used by both `train.py` and `infer.py`. |
| `onnx_export.py` | Export a trained model to ONNX format. |
| `infer_onnx.py` | Run inference with an ONNX-exported model. |
| `train04_*.py` | Legacy per-mode training scripts (kept for reference). |
