# IMC Data Module

This directory contains the data loading and preprocessing utilities for the
IMC image-metadata classification pipeline.

---

## 1. Creating the label CSV

The label CSV is the entry point for all dataset operations. It maps each DICOM
series folder to a classification label and a cross-validation fold.

### Required columns

| Column | Type | Description |
|---|---|---|
| `Filepath` | `str` | Relative path to the DICOM series folder (relative to `LOCAL_DATASET_PATH`). |
| `SequenceType_Code` | `str` | Fine-grained letter code assigned during annotation (e.g. `"C"`, `"O"`, `"Q"`). No grouping is applied; each annotated sub-type has its own letter. |
| `SequenceType_Code_norm` | `str` | Normalised letter code (`A`–`M`, see table below). Multiple fine-grained codes representing the same sequence family are merged into a single letter. |
| `split` | `str` | Cross-validation fold assignment (`fold_0` … `fold_4`). |

### Class mapping

The normalised letter code is defined by `LABEL_NAME_MAPS` in
[`constants.py`](constants.py). Fine-grained `SequenceType_Code` letters that
represent the same sequence family are merged into a single
`SequenceType_Code_norm` letter (e.g. Arterial sub-types `C`, `O`, `Q` in the
original annotation all map to the same normalised class).

| Letter(s) | Human-readable name |
|---|---|
| `A` | AX T2w |
| `B` | AX FatSat T1w |
| `C`, `O`, `Q` | Arterial T1w |
| `D` | MRCP |
| `E`, `N`, `P` | Late T1w |
| `F` | Other |
| `G` | AX Dixon In |
| `H` | AX Dixon Opp |
| `I` | AX DWI |
| `J` | COR T2w |
| `K` | Portven T1w |
| `L` | Localizer |
| `M` | AX ADC |

### Example CSV

```
Filepath,SequenceType_Code,SequenceType_Code_norm,split
series/patient001/T2ax,A,A,fold_0
series/patient001/T1fat,B,B,fold_0
series/patient002/T1art,O,C,fold_1
series/patient002/dwi,I,I,fold_2
series/patient003/mrcp,D,D,fold_3
series/patient003/loc,L,L,fold_4
```

> **Note:** The `Filepath` values must match the paths stored in the metadata
> Parquet file (see Section 2). Use consistent relative or absolute paths
> throughout both files.

---

## 2. Generating the metadata Parquet

The metadata Parquet file contains encoded numerical features derived from
DICOM header tags. It is generated offline and loaded at training time by
[`LiverDataset`](duke_dataloader_local.py) via the `METADATA_PATH` environment
variable (or `--metadata_path` CLI flag).

Use [`encode_metadata.py`](encode_metadata.py) to produce this file:

```bash
python -m IMC.data.encode_metadata \
    --label_file /path/to/labels.csv \
    --data_folder /path/to/dicom_root/ \
    --output_folder /path/to/output/ \
    --output_name encoded_metadata.parquet \
    --mode slicewise
```

### Modes

| Mode | Description |
|---|---|
| `slicewise` | One row per individual DICOM file. Use this for the Duke pipeline. |
| `series` | One row per series folder (all slice tags aggregated). |

### Full argument reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--label_file` | ✓ | — | Path to the label CSV (see Section 1). |
| `--data_folder` | ✓ | — | Root folder; relative `Filepath` values from the CSV are resolved against it. |
| `--output_folder` | ✓ | — | Directory where the output Parquet is written. |
| `--output_name` | | `encoded_metadata_<timestamp>.parquet` | Output filename. |
| `--filepath_col` | | `Filepath` | Column name in the label CSV that contains paths. |
| `--mode` | | `slicewise` | `slicewise` or `series`. |
| `--checkpoint_every` | | `5000` | Save an intermediate checkpoint every N series (set to `0` to disable). |

### Output format

The Parquet file contains:

- **`Filepath`** — path to the DICOM file (slicewise) or series folder (series
  mode). Must match the `Filepath` column in the label CSV.
- **`SeriesInstanceUID`** *(optional)* — present only when the input label CSV
  includes this column. It is dropped by `LiverDataset` before being used as
  model input.
- **`enc_*` columns** — encoded numerical features (≈ 88 columns in v1
  encoding). These are the features consumed by the model.

### Example (series mode)

```bash
python -m IMC.data.encode_metadata \
    --label_file /path/to/labels.csv \
    --data_folder /path/to/dicom_root/ \
    --output_folder /path/to/output/ \
    --mode series
```

### Matching paths between CSV and Parquet

The `Filepath` values in both files must resolve to the same paths after the
normalisation applied by `LiverDataset._normalize_dataset_index`. The simplest
approach is to use the same relative paths everywhere.
