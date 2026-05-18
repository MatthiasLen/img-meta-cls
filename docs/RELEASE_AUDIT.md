

## Priority 1: Release Blockers

### 1. Combined-model embedding flags are advertised but not fully wired into the public builder

**Location**

- [IMC/net4_duke/helper.py](IMC/net4_duke/helper.py#L90)
- [IMC/net4_duke/train.py](IMC/net4_duke/train.py#L510)
- [IMC/net4_duke/infer.py](IMC/net4_duke/infer.py#L372)
- [IMC/network04.py](IMC/network04.py#L494)

**Issue Type**

Critical Flaw

**Description**

The `net4_duke` train and inference entry points expose `--metadata_embed_dim` and `--output_emb_dim`, and inference states that these values must exactly match training. However, in the `combined` branch of `build_model`, the `MRISequenceClassifier` constructor is called without forwarding `output_emb_dim` or `metadata_embed_dim` explicitly. During validation, the combined model was instantiated with `output_emb_dim=256` and still produced an effective 128-dimensional head input. That means the public CLI/config surface can claim one architecture while actually constructing another.

This is a reproducibility and checkpoint-compatibility problem, not just a documentation issue.

**Actionable Fix**

Update the combined branch in [IMC/net4_duke/helper.py](IMC/net4_duke/helper.py#L90) so the public arguments are actually applied:

```python
elif modality == "combined":
    model = MRISequenceClassifier(
        metadata_input_dim=metadata_input_dim,
        num_classes_dict=num_classes_dict,
        img_enc_backbone=img_enc_backbone,
        metadata_embed_dim=metadata_embed_dim,
        output_emb_dim=output_emb_dim,
        metadata_encoder_type=metadata_enc_type,
        imputer_type=imputer_type,
        sparse_enc_version=sparse_enc_version,
        fusion_module_version=fusion_module_version,
        dropout_metadata=metadata_dropout,
        scalar_modulation=kwargs.get("scalar_modulation", False),
        n_channels=kwargs.get("n_channels", 1),
        learn_missing_embed=kwargs.get("learn_missing_embed", False),
        pre_processors=pre_processors,
        post_processors=post_processors,
    )
```

Add a regression test that constructs a combined model with non-default `output_emb_dim` and asserts that the final task head input width matches the requested value.

### 2. The public `fusion_module_version=v2` path is broken at runtime

**Location**

- [IMC/network04.py](IMC/network04.py#L260)
- [IMC/network04.py](IMC/network04.py#L323)
- [IMC/network04.py](IMC/network04.py#L367)
- [IMC/network04.py](IMC/network04.py#L368)
- [IMC/net4_duke/train.py](IMC/net4_duke/train.py#L227)
- [IMC/net4_duke/infer.py](IMC/net4_duke/infer.py#L149)

**Issue Type**

Critical Flaw

**Description**

`fusion_module_version=v2` is exposed as a supported CLI option in both training and inference. A direct runtime probe of `BiDirectionalCrossModalAttentionFusionV2` fails with a matrix-shape error. The implementation defines `self.weighted_pooling = nn.Linear(embed_dim, 1)` and then applies it to the post-projection tensor `output`, whose last dimension is `output_dim`, not `embed_dim`. The path is therefore broken on the public API boundary.

Because this path is not part of the paper reproduction surface, the safest release action is to disable it until it is repaired and tested.

**Actionable Fix**

For the public release, remove `v2` from the supported CLI choices and fail fast in the model builder:

```python
# IMC/net4_duke/train.py and IMC/net4_duke/infer.py
parser.add_argument(
    "--fusion_module_version",
    type=str,
    default="v1",
    choices=["v1", "concat"],
    help="Publicly supported fusion module version (combined mode only).",
)
```

```python
# IMC/network04.py or IMC/net4_duke/helper.py
elif fusion_module_version == "v2":
    raise NotImplementedError(
        "fusion_module_version='v2' is experimental and disabled for the public release."
    )
```

If `v2` must stay public, it needs a proper shape-correct implementation plus an executable regression test.

### 3. The paper-facing reproduction commands do not fully match the stated training setup

**Location**

- [README.md](README.md#L167)
- [IMC/net4_duke/README.md](IMC/net4_duke/README.md#L97)
- [IMC/net4_duke/train.py](IMC/net4_duke/train.py#L130)
- [IMC/net4_duke/train.py](IMC/net4_duke/train.py#L137)
- [IMC/helper.py](IMC/helper.py#L490)

**Issue Type**

Documentation Gap

**Description**

The paper states the proposed method uses `S=10`, batch size `64`, and a learning rate setting equivalent to a peak of `1e-4`. The public reproduction commands set `--n_slices 10`, but they do not set `--batch_size 64`. In addition, the documentation describes `--lr` as the base learning rate while the scheduler multiplies it by `peak_scale_factor=100` during warmup. That makes the public reproduction surface ambiguous: a reader cannot tell whether the intended paper setting is `--lr 1e-4` directly or `--lr 1e-6` with a 100x scheduler peak.

This does not just affect clarity. It affects whether a third party can actually reproduce the paper’s Duke setup from the release docs alone.

**Actionable Fix**

Replace the proposed-method command in [README.md](README.md#L167) and [IMC/net4_duke/README.md](IMC/net4_duke/README.md#L97) with explicit paper-aligned values:

```bash
uv run python -m IMC.net4_duke.train \
  --modality combined \
  --metadata_enc_type sparse \
  --sparse_enc_version v1 \
  --fusion_module_version v1 \
  --n_slices 10 \
  --batch_size 64 \
  --lr 1e-6 \
  --gpu 0
```

Add this exact documentation note:

```md
The scheduler multiplies `--lr` by 100 at the end of warmup. The paper's peak LR of `1e-4` therefore corresponds to `--lr 1e-6` with the current scheduler defaults.
```

## Priority 2: High-Value Reproducibility and Portability Issues

### 4. The XGBoost Duke baseline is hard-wired to CUDA

**Location**

- [IMC/xgboost/cv.py](IMC/xgboost/cv.py#L127)
- [IMC/xgboost/README.md](IMC/xgboost/README.md#L31)

**Issue Type**

Risk

**Description**

The metadata-only baseline is part of the paper comparison table, but `IMC/xgboost/cv.py` unconditionally configures `XGBClassifier(..., device="cuda")`. That makes one of the headline public baselines unusable on CPU-only systems and unnecessarily reduces portability for reproducibility reviewers.

COMMENT MATTHIAS: FOR ME IT WOULD BE FINE TO KEEP IT AS IT IS. I LEFT THIS ISSUE JUST FYI.

**Actionable Fix**

Add a CLI or environment-controlled device selection with a CPU default:

```python
device = os.environ.get("XGBOOST_DEVICE", "cpu")

model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    device=device,
)
```

Add this README text:

```md
The XGBoost baseline supports both CPU and CUDA. Set `XGBOOST_DEVICE=cpu` on systems without a compatible GPU.
```

### 5. Public training entry points do not expose or document seed control

**Location**

- [IMC/net4_duke/train.py](IMC/net4_duke/train.py)
- [IMC/net6/train_duke.py](IMC/net6/train_duke.py)
- [IMC/net7/train_duke.py](IMC/net7/train_duke.py)

**Issue Type**

Documentation Gap

**Description**

No explicit seed argument or seed initialization was found in the public training entry points. That means weight initialization, data-order randomness, and some augmentation behavior can vary across runs without a documented control surface. For a paper companion release, this is a real reproducibility gap even if it does not break execution.

**Actionable Fix**

Add a shared seed helper and expose `--seed` in all public training scripts:

```python
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Then document it with text like:

```md
For reproducible runs, pass `--seed <int>` and record the resulting `config.json`. Note that exact bitwise reproducibility can still depend on CUDA, cuDNN, and hardware-specific kernels.
```

### 6. There is no single top-level paper-reproduction pipeline for the main Duke benchmark table

**Location**

- [README.md](README.md)
- [IMC/net4_duke/README.md](IMC/net4_duke/README.md#L323)
- [IMC/net6/README.md](IMC/net6/README.md#L86)
- [IMC/net7/README.md](IMC/net7/README.md#L73)
- [IMC/xgboost/README.md](IMC/xgboost/README.md#L31)

**Issue Type**

Documentation Gap

**Description**

The repository contains the pieces needed to run each model family, but it does not currently provide one top-level workflow that regenerates the full Duke comparison table from the paper. The public release therefore requires manual assembly across multiple READMEs and output formats.

That is acceptable for internal research code, but weak for a public paper companion repository.

**Actionable Fix**

Add a top-level section to [README.md](README.md) with this structure:

```md
## Reproducing Duke Table 2

1. Generate the slicewise metadata parquet for `net4`, `net6`, and `net7`.
2. Generate the series-level metadata parquet for `xgboost`.
3. Run the proposed `net4_duke` command and aggregate with `IMC.net4_duke.summarize_cv`.
4. Run the `net6` two-stage baseline.
5. Run the `net7` 3D baseline.
6. Run the `xgboost` baseline.
7. Collect fold-wise weighted F1 scores into one CSV and compare them to Table 2.
```

If statistical-significance claims are meant to be public, add a small script that computes the Wilcoxon signed-rank test from the fold-level outputs.

## Priority 3: Release Polish

### 7. Library code still contains import-time debug output, stray prints, and mutable default list arguments

**Location**

- [IMC/trainer.py](IMC/trainer.py#L16)
- [IMC/trainer.py](IMC/trainer.py#L42)
- [IMC/nn/sparse_metadata_encoder.py](IMC/nn/sparse_metadata_encoder.py#L54)
- [IMC/data/duke_dataloader_local.py](IMC/data/duke_dataloader_local.py#L127)
- [IMC/data/duke_dataloader_local.py](IMC/data/duke_dataloader_local.py#L495)
- [IMC/network04.py](IMC/network04.py#L894)

**Issue Type**

Polish

**Description**

The codebase still contains a few internal-development leftovers:

- `trainer.py` prints `DEBUG_MODE` at import time
- `classification_losses` prints per-task accuracy directly
- `SparseMetadataEncoder` prints `Value Network output dim` during construction
- `network04.py` contains an inline output-shape print in the demo block
- the local dataloader helpers use mutable list defaults

These do not invalidate results, but they are avoidable rough edges in a public release and make the code look less curated than the README suggests.

**Actionable Fix**

Replace unconditional prints with logger-guarded messages and remove mutable defaults. For example:

```python
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
if DEBUG_MODE:
    print(f"DEBUG_MODE is {'ON' if DEBUG_MODE else 'OFF'}")
```

```python
def __init__(self, ..., split: list[str] | None = None, ...):
    split = ["fold_0"] if split is None else split
```

```python
def get_train_dataloader(..., folder_split: list[str] | None = None, ...):
    folder_split = folder_split or ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
```