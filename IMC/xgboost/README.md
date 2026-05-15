# IMC XGBoost - Metadata-Only Duke Baseline

`IMC/xgboost/cv.py` provides the metadata-only baseline referenced in the paper companion release.

## Relation to the paper

The paper's main contribution is a multimodal model with sparse, missingness-aware metadata encoding and cross-modal attention. This XGBoost script is not that proposed model. Instead, it serves as a metadata-only comparison baseline on the Duke Liver MRI benchmark, matching the paper's comparison setting against image-only and multimodal approaches.

## What the script does

- Loads Duke labels from `LABEL_CSV_PATH` and encoded metadata from `METADATA_PATH`.
- Aligns both tables by `Filepath` and prefixes image-relative paths with `LOCAL_DATASET_PATH`.
- Trains an `xgboost.XGBClassifier` using the predefined 5-fold split stored in the label CSV.
- Uses fold `i` for test, fold `(i + 1) % 5` for validation bookkeeping, and the remaining folds for training.
- Predicts the `SequenceType_Code_norm` target only.
- Writes per-fold predictions, aggregate metrics, confusion matrices, and a Markdown cross-validation report.

## Usage

Required environment variables:

- `LABEL_CSV_PATH`
- `METADATA_PATH`
- `LOCAL_DATASET_PATH`

Optional environment variable:

- `XGBOOST_OUTPUT_DIR` to override the default output directory under `logs/`

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
