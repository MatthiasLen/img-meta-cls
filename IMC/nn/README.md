# IMC Neural Network Components

This directory contains reusable neural network building blocks shared across
the multimodal and baseline models in this repository.

The components in `IMC/nn` are not themselves end-to-end experiments. They are
assembled by the model definitions in:

- `IMC/network04.py` for the multimodal image + metadata architecture,
- `IMC/network06.py` for the 2D image-only baseline,
- `IMC/network07.py` for the 3D volumetric baseline.

The table below is written from the perspective of the paper companion release.
For the metadata encoders in particular, the important distinction is between:

- the dense imputer-style metadata branch,
- the sparse metadata branch that represents the paper's main metadata idea,
- later experimental sparse variants that are still available in the codebase.

## Component Overview

| File | Main class(es) | What it does | Paper role |
|---|---|---|---|
| `image_encoder.py` | `MultiSliceImageEncoder` | Encodes one or more 2D slices with a shared CNN backbone and returns per-slice features | Core image encoder for the multimodal model and the 2D image-only baseline |
| `metadata_encoder.py` | `ContextualImputer`, `NanIgnorer`, `MetadataEncoder` | Dense metadata encoder with either learned contextual imputation or zero-fill handling, followed by MLP projection | Used for imputer-based metadata experiments and ablations; not the paper's main sparse metadata contribution |
| `sparse_metadata_encoder.py` | `SparseMetadataEncoder` | Sparse metadata encoder that skips NaNs, uses feature embeddings plus FiLM-style value modulation, and aggregates observed features | Core sparse metadata family aligned with the paper's missingness-aware metadata modeling |
| `sparse_metadata_encoder_v2.py` | `SparseMetadataEncoder` | Transformer-based variant with CLS-token aggregation over a dense feature sequence | Experimental alternative sparse encoder; available in the repo but not the default Duke path |
| `sparse_metadata_encoder_v5.py` | `SparseMetadataEncoder` | Later sparse encoder with stabilized FiLM, feature self-attention, and deeper post-processing | Experimental later sparse variant; selectable in Duke scripts but not the default setting |
| `multi_task_head.py` | `MultiTaskHead`, `SimplifiedMultiTaskHead` | Shared classification heads for one or more tasks | Used across the paper models and baselines |
| `multi_task_loss.py` | `MultiTaskLoss` | Combined loss for multi-class, binary, and optional regression tasks with masking | Used across the paper models and baselines |
| `resnet_3d.py` | `ResNet3D` | 3D residual backbone for volumetric MRI inputs | Used by the 3D volumetric baseline |
| `densenet_3d.py` | `DenseNet3D`, `densenet121_3d`, `densenet169_3d`, `densenet201_3d` | 3D DenseNet backbones for volumetric MRI inputs | Alternative backbones for the 3D volumetric baseline |
| `pyramid_pooling_3d.py` | `PyramidPooling3D` | Multi-scale volumetric pooling over 3D feature maps | Core pooling module in the 3D volumetric baseline |

## Metadata Encoders

The metadata encoders are the most important source of architectural variation
inside `IMC/nn`.

| Encoder | Missing-data strategy | Main mechanism | Typical use in repo | Paper relation |
|---|---|---|---|---|
| `MetadataEncoder` with `ContextualImputer` | Replace NaNs with learned, context-dependent estimates | Dense MLP encoder after imputation | `--metadata_enc_type imputer --imputer_type contextual` | Metadata baseline or ablation, not the main sparse contribution |
| `MetadataEncoder` with `NanIgnorer` | Replace NaNs with zeros | Dense MLP encoder without learned imputation | `--metadata_enc_type imputer --imputer_type ignore` | Simpler ablation / control setting |
| `SparseMetadataEncoder` v1 | Skip NaNs entirely, process only observed features | Feature embeddings + FiLM-style modulation + sum/mean aggregation | `--metadata_enc_type sparse --sparse_enc_version v1` | Closest released sparse baseline to the paper's missingness-aware idea |
| `SparseMetadataEncoder` v2 | Represent all features, including missing ones, in a transformer sequence | Dense transformer with CLS token aggregation | `--metadata_enc_type sparse --sparse_enc_version v2` | Experimental alternative |
| `SparseMetadataEncoder` v5 | Zero-mask missing values, then refine observed features with self-attention | Stabilized FiLM + feature self-attention + transformer-style blocks | `--metadata_enc_type sparse --sparse_enc_version v5` | Experimental later sparse variant |

### Practical interpretation

- If you want the dense imputation-style metadata branch, use `MetadataEncoder`.
- If you want the sparse metadata branch highlighted by the paper, use one of the `SparseMetadataEncoder*` variants.
- In the current public Duke scripts, the sparse path is selectable but not the CLI default. The released Duke training defaults are:
  - `--metadata_enc_type imputer`
  - `--fusion_module_version v1`
  - `--sparse_enc_version v1` when the sparse branch is selected

## Image-side Components

### `MultiSliceImageEncoder`

`MultiSliceImageEncoder` in `image_encoder.py` is the shared 2D image encoder for
slice-based models. It applies a common backbone to each slice independently and
returns a sequence of slice embeddings.

This component is used by:

- `MRISequenceClassifier` in `network04.py`,
- `ImageBasedClassifier` in `network04.py`,
- `PixelOnlyModel` in `network06.py`.

It supports multiple 2D backbones including DenseNet, ResNet, EfficientNet, and
optional DINOv3-based backbones when local environment variables are provided.

## 3D Components

The 3D baseline in `network07.py` uses a separate family of components:

- `ResNet3D` and `DenseNet3D` backbones for volumetric feature extraction,
- `PyramidPooling3D` for multi-scale volumetric aggregation,
- `MultiTaskHead` for prediction.

These are relevant to the paper as image-only volumetric baselines rather than
to the multimodal sparse metadata contribution.

## Shared Heads and Losses

### `MultiTaskHead`

`MultiTaskHead` builds one prediction head per task name. It is shared across
the multimodal model and the baseline models and is the standard output layer
used by the paper experiments.

### `MultiTaskLoss`

`MultiTaskLoss` combines per-task losses while supporting:

- multi-class classification,
- binary classification,
- optional regression for `label_ContrastPhase`,
- label masks for partially available annotations,
- optional task-specific class weighting.

## Where The Fusion Modules Live

The cross-modal fusion blocks used by the multimodal paper model are defined in
`IMC/network04.py`, not in `IMC/nn`.

In particular:

- `SliceFeatureFusion` fuses per-slice image embeddings,
- `BiDirectionalCrossModalAttentionFusion` and `BiDirectionalCrossModalAttentionFusionV2` combine image and metadata embeddings,
- `MRISequenceClassifier` assembles image encoder, metadata encoder, fusion, and task heads into the full multimodal model.

## Recommended Reading Order

If you want to understand the paper model from the bottom up, a good order is:

1. `image_encoder.py`
2. `metadata_encoder.py`
3. `sparse_metadata_encoder.py`
4. `multi_task_head.py`
5. `network04.py`
6. `IMC/net4_duke/README.md`