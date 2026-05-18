"""Encode DICOM metadata for a set of series listed in a CSV label file.

Reads DICOM tags from each series folder listed in a label CSV and writes
encoded feature vectors to a Parquet file that can be consumed by
:class:`~IMC.data.duke_dataloader_local.LiverDataset` at training time.

Usage
-----
Slicewise mode (one row per DICOM file):

.. code-block:: bash

    python -m IMC.data.encode_metadata \\
        --label_file /path/to/labels.csv \\
        --data_folder /path/to/dicom_root/ \\
        --output_folder /path/to/output/ \\
        --output_name encoded_metadata.parquet \\
        --mode slicewise

Series mode (one row per series folder, tags aggregated):

.. code-block:: bash

    python -m IMC.data.encode_metadata \\
        --label_file /path/to/labels.csv \\
        --data_folder /path/to/dicom_root/ \\
        --output_folder /path/to/output/ \\
        --mode series
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from IMC.data.dicom_tag_encoding_v2 import (
    generate_series_metadata_vector,
    generate_slicewise_metadata_vector,
)

UID_TAG = (0x0020, 0x000E)  # SeriesInstanceUID


def _filter_slices_by_uid(slices: List[str], series_uid: str) -> List[str]:
    """Return only slices whose SeriesInstanceUID header matches *series_uid*.

    Reads only tag (0x0020, 0x000E) so no pixel data is loaded.
    Unreadable files are skipped silently.
    """
    import pydicom  # lazy import

    matched: List[str] = []
    for path in slices:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=[UID_TAG])
            uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
            if uid == series_uid:
                matched.append(path)
        except Exception:
            pass
    return matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode DICOM metadata for a set of series listed in a CSV label file."
    )
    parser.add_argument(
        "--label_file",
        required=True,
        type=Path,
        help="Path to the CSV file with series metadata. Must contain a filepath column.",
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        type=Path,
        help="Root folder that the relative paths in the label file are resolved against.",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        type=Path,
        help="Folder where the output parquet file will be written.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output parquet filename.  Defaults to 'encoded_metadata_<timestamp>.parquet'.",
    )
    parser.add_argument(
        "--filepath_col",
        type=str,
        default="Filepath",
        help="Name of the column in the label CSV that contains the series folder paths. Default: 'Filepath'.",
    )
    parser.add_argument(
        "--mode",
        choices=["slicewise", "series"],
        default="slicewise",
        help="Encoding mode: 'slicewise' produces one row per DICOM file; "
        "'series' aggregates all slices into one row per series. Default: slicewise.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Save an intermediate checkpoint parquet every N series. Set to 0 to disable. Default: 5000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    label_df = pd.read_csv(args.label_file)
    if args.filepath_col not in label_df.columns:
        raise ValueError(
            f"Column '{args.filepath_col}' not found in {args.label_file}. Available columns: {list(label_df.columns)}"
        )

    has_uid_col = "SeriesInstanceUID" in label_df.columns
    if has_uid_col:
        cases = label_df[[args.filepath_col, "SeriesInstanceUID"]].drop_duplicates().itertuples(index=False, name=None)
        cases = list(cases)  # list of (filepath, uid) tuples
    else:
        cases = [(fp, None) for fp in label_df[args.filepath_col].drop_duplicates()]

    print(f"Total number of cases: {len(cases)}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"encoded_metadata_{timestamp}.parquet"
    output_path = args.output_folder / output_name

    checkpoint_path = (
        output_path.parent / (output_path.stem + "_checkpoint" + output_path.suffix) if args.checkpoint_every else None
    )

    all_metadata = []
    for series_count, (series, series_uid) in enumerate(tqdm(cases), start=1):
        series_path = args.data_folder / series
        if series_path.is_file():
            print(f"Warning: expected a directory for series path but found a file: {series_path}. Skipping.")
            continue
        elif not series_path.is_dir():
            print(f"Warning: series path does not exist: {series_path}. Skipping.")
            continue
        slices = natsorted(str(s) for ext in ("*.dcm", "*.dicom") for s in series_path.rglob(ext))
        if not slices:
            print(f"Warning: no .dcm files found in {series_path}, skipping.")
            continue

        if series_uid is not None:
            slices = _filter_slices_by_uid(slices, series_uid)
            if not slices:
                print(f"Warning: no .dcm files with SeriesInstanceUID={series_uid!r} found in {series_path}, skipping.")
                continue

        if args.mode == "slicewise":
            out, _ = generate_slicewise_metadata_vector(slice_filenames=slices)
            for o, s in zip(out, slices):
                o["Filepath"] = s
                if series_uid is not None:
                    o["SeriesInstanceUID"] = series_uid
        else:
            out, _ = generate_series_metadata_vector(slice_filenames=slices)
            out["Filepath"] = str(series_path)
            if series_uid is not None:
                out["SeriesInstanceUID"] = series_uid
            out = [out]

        all_metadata.append(pd.DataFrame(out))

        if args.checkpoint_every and series_count % args.checkpoint_every == 0:
            pd.concat(all_metadata, ignore_index=True).to_parquet(checkpoint_path, index=False)
            print(f"\nCheckpoint saved at {series_count} series → {checkpoint_path}")

    if not all_metadata:
        print("No metadata was encoded. Check that the data folder and label file are correct.")
        return

    all_metadata_df = pd.concat(all_metadata, ignore_index=True)
    all_metadata_df.to_parquet(output_path, index=False)
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Aggregated metadata saved to {output_path}")
    print("\nMetadata statistics:")
    print(all_metadata_df.describe())


if __name__ == "__main__":
    main()
