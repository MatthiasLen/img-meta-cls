"""Train per-fold Random Forest metadata classifiers for the Duke pipeline.

This script trains one :class:`sklearn.ensemble.RandomForestClassifier`
pipeline per CV fold to classify ``SequenceType_Code_norm`` from Duke DICOM
tabular metadata.  The cross-validation fold layout matches
:mod:`IMC.net6.train` exactly (test = fold *i*, val = (i+1) % 5, train = the
rest), so the saved RF checkpoints can be used as a heuristic gate at
inference time via :mod:`IMC.net6.infer`.

Typical usage
-------------
::

    python -m IMC.net6.train_rf --out_dir ./rf_checkpoints/duke_net6

Environment variables
---------------------
The following variables configure Duke dataset paths.  They can be overridden
via CLI arguments ``--dataset_path``, ``--metadata_path``, ``--label_csv_path``
which take precedence::

    LOCAL_DATASET_PATH – root folder of the Duke MRI image dataset
    METADATA_PATH      – path to the Duke encoded metadata parquet file
    LABEL_CSV_PATH     – path to the Duke label CSV
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from IMC.data.constants import DUKE_ORIGINAL_LABEL_NAMES

# ---------------------------------------------------------------------------
# Default Duke dataset paths
# ---------------------------------------------------------------------------
_DATASET_PATH = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
_LABEL_CSV_PATH = (
    "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"
)
_METADATA_PATH = (
    "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_X_y(dataset: Dataset, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) arrays from a :class:`~IMC.data.duke_dataloader_local.LiverDataset`.

    Only ``SequenceType_Code_norm`` targets are extracted.  For each sample
    the metadata vector of the *first* (and typically only) slice is used.

    Args:
        dataset: A configured :class:`LiverDataset` instance.
        batch_size: Batch size used when iterating the dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix ``X`` of shape
        ``(N, D)`` and label vector ``y`` of shape ``(N,)``.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    label_keys = list(dataset.label_names.keys())
    seq_idx = (
        label_keys.index("SequenceType_Code_norm")
        if "SequenceType_Code_norm" in label_keys
        else 0
    )

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    with torch.no_grad():
        for images, metadata, targets, masks in loader:
            # metadata: (B, N_slices, D) → take first slice → (B, D)
            if metadata.dim() == 3:
                meta_first = metadata[:, 0, :].numpy()
            else:
                meta_first = metadata.numpy()
            y = targets[seq_idx].numpy()
            X_rows.append(meta_first)
            y_rows.append(y)

    return np.concatenate(X_rows, axis=0), np.concatenate(y_rows, axis=0)


def train_fold_rf(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = 64,
) -> tuple[Pipeline, float]:
    """Fit a :class:`sklearn.pipeline.Pipeline` (scaler + RF) for one fold.

    Args:
        train_ds: Training split dataset.
        val_ds: Validation split dataset.
        batch_size: Batch size used for feature extraction.

    Returns:
        tuple[Pipeline, float]: The fitted pipeline and its validation
        accuracy on *val_ds*.
    """
    X_train, y_train = extract_X_y(train_ds, batch_size=batch_size)
    X_val, y_val = extract_X_y(val_ds, batch_size=batch_size)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=373,
                    random_state=42,
                    n_jobs=2,
                    min_samples_leaf=2,
                    min_samples_split=10,
                    bootstrap=False,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    val_acc = float(pipeline.score(X_val, y_val))
    return pipeline, val_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RF training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train per-fold RandomForest metadata classifiers for the Duke net6 pipeline."
        )
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save the RF model checkpoints (one .joblib file per fold).",
    )
    parser.add_argument(
        "--use_selected",
        action="store_true",
        help="Use the pre-selected metadata feature subset instead of all features.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help=(
            "Comma-separated list of fold indices to train (e.g. '0,1,2'). "
            "Defaults to all 5 folds."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for metadata extraction.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override LOCAL_DATASET_PATH.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Override METADATA_PATH.")
    parser.add_argument("--label_csv_path", type=str, default=None, help="Override LABEL_CSV_PATH.")
    return parser.parse_args()


def _configure_dataset_env(args: argparse.Namespace) -> None:
    """Set Duke dataset environment variables from CLI args (if provided)."""
    os.environ["LOCAL_DATASET_PATH"] = args.dataset_path or _DATASET_PATH
    os.environ["METADATA_PATH"] = args.metadata_path or _METADATA_PATH
    os.environ["LABEL_CSV_PATH"] = args.label_csv_path or _LABEL_CSV_PATH


def main(args: argparse.Namespace) -> None:
    """Run 5-fold RF training for the Duke net6 pipeline.

    For each fold the following split is used (consistent with :mod:`IMC.net6.train`):

    - **test fold** : ``fold_idx``  (held out – not used here, only for model saving)
    - **val fold**  : ``(fold_idx + 1) % 5``
    - **train folds**: all remaining folds

    Args:
        args: Parsed argument namespace from :func:`parse_args`.
    """
    _configure_dataset_env(args)

    os.makedirs(args.out_dir, exist_ok=True)

    n_folds = 5
    folds_to_run = (
        [int(f.strip()) for f in args.folds.split(",")]
        if args.folds is not None
        else list(range(n_folds))
    )

    for idx, fold_idx in enumerate(folds_to_run):
        if not (0 <= fold_idx < n_folds):
            raise ValueError(f"Fold index {fold_idx} is out of range [0, {n_folds - 1}].")
    from IMC.data.duke_dataloader_local import LiverDataset
    dataset_kwargs: dict = dict(
        num_samples=None,
        n_slices=1,
        sampling_type="equidistant",
        augment_conf="NONE2D",
        aggregated_metadata=False,
        use_preselected_features=args.use_selected,
        exclude_contrast_yn=False,
        label_names=DUKE_ORIGINAL_LABEL_NAMES,
    )

    results: list[dict] = []

    for fold_idx in folds_to_run:
        val_fold = (fold_idx + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in {fold_idx, val_fold}]

        print("=" * 80)
        print(f"RF TRAINING  –  fold {fold_idx}  (val={val_fold}, train={train_folds})")
        print("=" * 80)

        train_ds = LiverDataset(
            split=[f"fold_{f}" for f in train_folds],
            **dataset_kwargs,
        )
        val_ds = LiverDataset(
            split=[f"fold_{val_fold}"],
            **dataset_kwargs,
        )

        print(f"  Train samples : {len(train_ds)}")
        print(f"  Val   samples : {len(val_ds)}")

        pipeline, val_acc = train_fold_rf(train_ds, val_ds, batch_size=args.batch_size)

        model_path = os.path.join(args.out_dir, f"duke_metadata_rf_fold_{fold_idx}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"  Saved : {model_path}")
        print(f"  Val accuracy : {val_acc:.4f}")

        results.append({"fold": fold_idx, "val_acc": val_acc, "model_path": model_path})

    print("\nCross-validation RF summary")
    print("-" * 50)
    for r in results:
        print(f"  Fold {r['fold']} : val_acc={r['val_acc']:.4f}  →  {r['model_path']}")


if __name__ == "__main__":
    main(parse_args())
