"""
Train RandomForest metadata classifiers on Duke using the same CV splits and single-slice selection as the pixel model.
- Uses LiverDataset with n_slices=1 and sampling_type="equidistant" to match the pixel model's middle slice.
- Performs 5-fold CV consistent with net4_duke: for fold i, train on all folds except {i, (i+1)%5}, validate on (i+1)%5.
- Saves one RF model per fold.
"""

import os
import argparse
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from IMC.data.duke_dataloader_local import LiverDataset, DUKE_ORIGINAL_LABEL_NAMES



def extract_X_y(dataset: LiverDataset, batch_size: int = 64):
    """Extract (X, y) from a LiverDataset for SequenceType_Code_norm using the first slice's metadata.
    Returns X (N, D), y (N,) as numpy arrays.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Determine index for SequenceType_Code_norm in targets tuple
    label_keys = list(dataset.label_names.keys())
    if "SequenceType_Code_norm" in label_keys:
        seq_idx = label_keys.index("SequenceType_Code_norm")
    else:
        # Fallback: use first label if specific key not present
        seq_idx = 0
    X_rows = []
    y_rows = []
    with torch.no_grad():
        for images, metadata, targets, masks in loader:
            # metadata: (B, N_slices, D) or (B, D) depending on dataset; with n_slices=1 we expect (B, 1, D)
            if metadata.dim() == 3:
                meta_first = metadata[:, 0, :]  # (B, D)
            else:
                meta_first = metadata  # (B, D)
            y = targets[seq_idx]  # tensor of shape (B,)
            X_rows.append(meta_first.numpy())
            y_rows.append(y.numpy())
    X = np.concatenate(X_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    return X, y


def train_fold_rf(train_ds: LiverDataset, val_ds: LiverDataset):
    X_train, y_train = extract_X_y(train_ds)
    X_val, y_val = extract_X_y(val_ds)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestClassifier(n_estimators=373, random_state=42, n_jobs=2, min_samples_leaf=2, min_samples_split=10, bootstrap=False))
    ])
    pipe.fit(X_train, y_train)
    val_acc = pipe.score(X_val, y_val)
    return pipe, float(val_acc)


def main():
    parser = argparse.ArgumentParser(description="Train metadata RF classifiers on Duke with CV and single-slice metadata matching pixel model")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save RF models (one per fold)")
    parser.add_argument("--use_selected", action="store_true", help="Use selected metadata features subset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for metadata extraction")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Duke env paths (scripts use these envs)
    os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
    os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"
    os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet"

    # CV folds consistent with net4_duke/train04_baseline_multi_slices.py
    folds_to_run = [0, 1, 2, 3, 4]
    n_folds = 5

    # Dataset configuration matches pixel model: single slice, equidistant sampling
    num_samples = None
    aggregated_metadata = False
    use_preselected_features = True if args.use_selected else False
    exclude_contrast_yn = False

    results = []
    for fold_idx in folds_to_run:
        val_fold = (fold_idx + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in [fold_idx, val_fold]]
        print("="*80)
        print(f"RF TRAINING - FOLD {fold_idx} (val={val_fold})")
        print("Train folds:", train_folds)
        # Build datasets
        train_ds = LiverDataset(
            split=[f"fold_{f}" for f in train_folds],
            num_samples=num_samples,
            n_slices=1,
            sampling_type="equidistant",
            augment_conf="NONE2D",
            aggregated_metadata=aggregated_metadata,
            use_preselected_features=use_preselected_features,
            exclude_contrast_yn=exclude_contrast_yn,
            label_names=DUKE_ORIGINAL_LABEL_NAMES,
        )
        val_ds = LiverDataset(
            split=[f"fold_{val_fold}"],
            num_samples=num_samples,
            n_slices=1,
            sampling_type="equidistant",
            augment_conf="NONE2D",
            aggregated_metadata=aggregated_metadata,
            use_preselected_features=use_preselected_features,
            exclude_contrast_yn=exclude_contrast_yn,
            label_names=DUKE_ORIGINAL_LABEL_NAMES,
        )
        # Train RF for this fold
        pipe, val_acc = train_fold_rf(train_ds, val_ds)
        model_path = os.path.join(args.out_dir, f"duke_metadata_rf_fold_{fold_idx}.joblib")
        joblib.dump(pipe, model_path)
        print(f"Saved RF model: {model_path}")
        print(f"Fold {fold_idx} validation accuracy: {val_acc:.4f}")
        results.append({"fold": fold_idx, "val_acc": val_acc, "model": model_path})

    # Summary
    print("\nCV RF summary:")
    for r in results:
        print(f"Fold {r['fold']}: val_acc={r['val_acc']:.4f}, model={r['model']}")


if __name__ == "__main__":
    main()
