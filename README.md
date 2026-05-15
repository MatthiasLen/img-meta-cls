# img-meta-cls

Research code accompanying the preprint "Revisiting Integration of Image and Metadata for DICOM Series Classification: Cross-Attention and Dictionary Learning" (arXiv:2602.23833).

The repository implements multimodal DICOM series classification models that combine image content and acquisition metadata, together with image-only and metadata-only baselines. The main public experiment surface is the Duke Liver MRI benchmark used in the paper.

## Paper Summary

The paper studies DICOM series classification under three practical failure modes: heterogeneous slice content, variable series length, and missing or inconsistent metadata. The proposed approach addresses these issues with:

- a 2.5D image encoder operating on equidistantly sampled slices,
- a sparse, missingness-aware metadata encoder that avoids explicit imputation,
- cross-modal fusion through bi-directional attention between image and metadata representations.

The codebase also includes Duke baselines used for comparison in the paper:

- [`net4_duke`](IMC/net4_duke/README.md): Duke training, inference, and CV workflow for the multimodal cross-attention model,
- [`net6`](IMC/net6/README.md): 2D image-only baseline,
- [`net7`](IMC/net7/README.md): 3D volumetric image-only baseline,
- `xgboost`: metadata-only baseline.

## Repository Scope

This public release is focused on code needed to understand and reproduce the Duke experiments and the model components described in the preprint.

- Duke dataset files are not included.
- The large in-house multi-institutional cohort from the paper is not part of this repository.
- You must provide dataset and metadata locations through environment variables or CLI flags.

## Setup

This repository uses `uv` and targets Python 3.11+.

```bash
uv sync
```

Activate the environment if you want an interactive shell:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

## Required Dataset Configuration

Most training and inference entry points expect these environment variables:

```bash
LOCAL_DATASET_PATH=/path/to/duke_images
METADATA_PATH=/path/to/encoded_metadata.parquet
LABEL_CSV_PATH=/path/to/labels.csv
```

You can also pass the corresponding CLI overrides:

- `--dataset_path`
- `--metadata_path`
- `--label_csv_path`

## Quick Validation

Run the test suite:

```bash
uv run pytest -q
```

## Main Experiment Entry Points

### Network 4: multimodal image + metadata fusion on Duke

See the [Network v04 architecture README](IMC/net4/README.md) for the shared multimodal model design and the [Duke workflow README](IMC/net4_duke/README.md) for training, inference, and CV details.

Train 5-fold cross-validation:

```bash
uv run python -m IMC.net4_duke.train --modality combined --gpu 0
```

Run inference for a trained fold:

```bash
uv run python -m IMC.net4_duke.infer \
  --ckpt ./logs/<run>/fold_0/best_model.pth \
  --output_dir ./infer_out/net4/fold_0 \
  --modality combined \
  --folds 0 \
  --run_eval
```

Aggregate fold-level evaluation results:

```bash
uv run python -m IMC.net4_duke.summarize_cv ./logs/<run>
```

### Network 6: 2D image-only Duke baseline

See the [Network 6 README](IMC/net6/README.md) for architecture notes, RF gating, and full Duke usage.

```bash
uv run python -m IMC.net6.train_duke --gpu 0
```

Optional metadata gate:

```bash
uv run python -m IMC.net6.train_rf_duke --out_dir ./rf_checkpoints/duke_net6
```

Inference:

```bash
uv run python -m IMC.net6.infer_duke \
  --ckpt ./logs/<run>/fold_0/best_model.pth \
  --output_dir ./infer_out/net6/fold_0
```

### Network 7: 3D volumetric Duke baseline

See the [Network 7 README](IMC/net7/README.md) for the volumetric architecture, fold-wise training pattern, and CLI details.

Train a single fold:

```bash
uv run python -m IMC.net7.train_duke --fold 0 --gpu 0
```

Inference:

```bash
uv run python -m IMC.net7.infer_duke \
  --ckpt ./logs/<run>/fold_0/best_model.pth \
  --output_dir ./infer_out/net7/fold_0 \
  --split fold_0
```

### Metadata-only XGBoost baseline

```bash
uv run python -m IMC.xgboost.cv
```

## Repository Layout

```text
IMC/
  data/        Duke data loading, metadata encoding, image I/O
  nn/          reusable model components
  network04.py multimodal cross-attention architecture
  network06.py 2D image-only baseline
  network07.py 3D volumetric baseline
  net4_duke/   Duke training, inference, and CV summarization for network 4
  net6/        Duke training and inference for network 6
  net7/        Duke training and inference for network 7
  xgboost/     metadata-only baseline
tests/         unit tests for loaders, encoders, and Duke workflows
```

For a more detailed breakdown of the reusable neural network modules, including
the different metadata encoder variants and their role in the paper code, see
[`IMC/nn/README.md`](IMC/nn/README.md).

## Notes On Reproducibility

- Dataset configuration must be supplied explicitly through env vars or CLI args.
- Some optional backbones and older experimental modules remain in the codebase but are not the main public reproduction path for the paper.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{truong2026revisiting,
  title={Revisiting Integration of Image and Metadata for DICOM Series Classification: Cross-Attention and Dictionary Learning},
  author={Truong, Tuan and Dohmen, Melanie and Lorio, Sara and Lenga, Matthias},
  journal={arXiv preprint arXiv:2602.23833},
  year={2026}
}
```

## License

This repository is licensed under the Apache License 2.0. See `LICENSE`.
