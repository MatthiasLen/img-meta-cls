# IMC XGBoost - Metadata-Only Duke Baseline

`IMC/xgboost/cv.py` provides the metadata-only baseline referenced in the paper companion release.

## Relation to the paper

The paper's main contribution is a multimodal model with sparse, missingness-aware metadata encoding and cross-modal attention. This XGBoost script is not that proposed model. Instead, it serves as a metadata-only comparison baseline on the Duke Liver MRI benchmark, matching the paper's comparison setting against image-only and multimodal approaches.

| Paper exp. | Description | Duke weighted F1 (%) |
|---|---|---|
| (3) Metadata-only | XGBoost on tabular DICOM metadata, no image pixels | 74.71 ± 2.34 |

The metadata-only result confirms that acquisition metadata alone is insufficient for reliable series identification: image-only baselines score around 85–88% and the proposed method reaches 96.66%.

## What the script does

- Loads Duke labels from `LABEL_CSV_PATH` and **series-level** encoded metadata from `XGBOOST_METADATA_PATH` (one row per series, not per slice).
- Aligns both tables by `Filepath` and prefixes image-relative paths with `LOCAL_DATASET_PATH`.
- Trains an `xgboost.XGBClassifier` using the predefined 5-fold split stored in the label CSV.
- Uses fold `i` for test, fold `(i + 1) % 5` for validation bookkeeping, and the remaining folds for training.
- Predicts the `SequenceType_Code_norm` target only.
- Writes per-fold predictions, aggregate metrics, confusion matrices, and a Markdown cross-validation report.

## Usage

### Metadata format

XGBoost requires a **series-level** parquet: one row per series folder, with
DICOM metadata aggregated across all slices in that series.  This is different
from the slicewise parquet (`METADATA_PATH`) used by the neural network
experiments.  Generate it with:

```bash
uv run python -m IMC.data.encode_metadata \
    --dicom_root "$LOCAL_DATASET_PATH" \
    --output encoded_metadata_series.parquet \
    --mode series

export XGBOOST_METADATA_PATH=/path/to/encoded_metadata_series.parquet
```

See [`IMC/data/README.md`](../data/README.md) for full encoding documentation.

### Required environment variables

- `LABEL_CSV_PATH`
- `XGBOOST_METADATA_PATH` — series-level metadata parquet
- `LOCAL_DATASET_PATH`

Optional:

- `XGBOOST_OUTPUT_DIR` — override the default output directory under `logs/`

Run the baseline with:

```bash
uv run python -m IMC.xgboost.cv
```

## Outputs

The script creates an output directory such as `logs/xgboost_cv_<timestamp>/` containing:

- `predictions_fold_*.csv` and `predictions_all_folds.csv`
- `fold_metrics_summary.csv`, `overall_metrics.json`, and `detailed_metrics.csv`
- per-fold and overall confusion matrices
- summary visualizations across folds and classes
- `cv_report.md`, a self-contained Markdown report with the aggregated results

## Notes

- The current implementation assumes `device="cuda"` in the XGBoost classifier configuration.
- This baseline operates on tabular metadata only and does not use image pixels or the sparse metadata encoder proposed by the paper.
- The series-level parquet (`XGBOOST_METADATA_PATH`) is distinct from the slicewise parquet (`METADATA_PATH`) used by the neural network experiments; make sure to generate both if running all experiments.
