# Network v04 - Architecture and Duke Usage

Network v04 is the multimodal cross-attention architecture for DICOM MRI series
classification.

In this public repository snapshot, this directory contains the shared model
construction logic used by the Duke experiment pipeline rather than standalone
training or inference entry points. The runnable Duke scripts live in:

- `IMC.net4_duke.train`
- `IMC.net4_duke.infer`
- `IMC.net4_duke.summarize_cv`

The central helper here is `helper.py:build_model`, which is shared by the Duke
training and inference code to keep checkpoint loading and model construction in
sync.

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

The v04 family is designed to support three experiment modes:

| `--modality` | Description |
|---|---|
| `combined` | Image + metadata fusion with cross-modal interaction |
| `image` | Image-only ablation |
| `metadata` | Metadata-only ablation |

For `combined` and `metadata`, the metadata branch can use either:

- an imputer-based encoder via `--metadata_enc_type imputer`, or
- a sparse encoder via `--metadata_enc_type sparse`.

---

## Duke training and inference

Although the actual CLI entry points are in `IMC.net4_duke`, the main v04 model
choices are configured there.

### Training examples

```bash
# Combined image + metadata fusion
python -m IMC.net4_duke.train \
	--modality combined \
	--img_enc_backbone densenet121 \
	--metadata_enc_type sparse \
	--sparse_enc_version v1 \
	--fusion_module_version v1 \
	--gpu 0

# Image-only ablation
python -m IMC.net4_duke.train \
	--modality image \
	--img_enc_backbone resnet50 \
	--gpu 0

# Metadata-only ablation
python -m IMC.net4_duke.train \
	--modality metadata \
	--metadata_enc_type sparse \
	--gpu 0
```

### Inference example

```bash
python -m IMC.net4_duke.infer \
	--ckpt ./logs/<run>/fold_0/best_model.pth \
	--output_dir ./results/fold_0 \
	--modality combined \
	--folds 0 \
	--run_eval \
	--gpu 0
```

### Cross-validation summary

```bash
python -m IMC.net4_duke.summarize_cv ./logs/<timestamp>_5fold_cv
```

---

## Important arguments

These are the most relevant v04 configuration switches exposed through the Duke
entry points:

| Argument | Description |
|---|---|
| `--modality` | `combined`, `image`, or `metadata` |
| `--img_enc_backbone` | CNN backbone for the image encoder |
| `--metadata_enc_type` | `imputer` or `sparse` |
| `--sparse_enc_version` | sparse metadata encoder variant |
| `--fusion_module_version` | fusion module variant for combined mode |
| `--vanilla_image_classifier` | simpler image-only classifier |
| `--n_slices` | number of slices sampled from each MRI series |
| `--incl_regression` | add regression head for contrast phase |

All architecture-related arguments used at inference time must match the
training run that produced the checkpoint.

---

## Required dataset configuration

The Duke scripts expect explicit dataset configuration via environment variables
or CLI overrides. They no longer assume machine-specific defaults.

Required environment variables:

- `LOCAL_DATASET_PATH`
- `METADATA_PATH`
- `LABEL_CSV_PATH`

Equivalent CLI overrides:

- `--dataset_path`
- `--metadata_path`
- `--label_csv_path`

---

## File overview

| File | Description |
|---|---|
| `helper.py` | Shared `build_model` factory for the Duke v04 training and inference pipeline |

For the end-to-end Duke workflow, see `IMC/net4_duke/README.md`.
