"""
Inference script for Network 07 (3D Pyramid Pooling Network) on ADNI Dataset.

Loads a trained PyramidPooling3DClassifier checkpoint and runs inference over a
specified data split (or the full dataset when no split is given). Predictions
for all three ADNI tasks are decoded from argmax logits and written to a CSV
file alongside the series filepath.

Optionally computes per-task accuracy and macro-F1 when ground-truth labels are
present in the label CSV.

Usage
-----
Single fold
    python -m IMC.net7_adni.infer07_adni \\
        --ckpt ./logs/net07_adni_cv/fold_0/best_model.pth \\
        --output_dir ./logs/net07_adni_cv/fold_0/infer \\
        --split fold_0

All folds (full dataset)
    python -m IMC.net7_adni.infer07_adni \\
        --ckpt ./logs/net07_adni_cv/fold_0/best_model.pth \\
        --output_dir ./logs/net07_adni_cv/fold_0/infer_all

Authors: Tuan Truong
Date: 2026
"""

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from IMC.helper import normalize_per_sample
from IMC.network07 import PyramidPooling3DClassifier

# ---------------------------------------------------------------------------
# ADNI environment defaults (override via shell / .env)
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "ADNI_LOCAL_DATASET_PATH",
    "/home/tuan.truong/data/ADNI_full",
)
os.environ.setdefault(
    "ADNI_LABEL_CSV_PATH",
    "/home/tuan.truong/codebase/IMC/labels/labels_brain_20250721.csv",
)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_predictions(
    pred_df: pd.DataFrame,
    label_csv_path: str,
    output_dir: str,
    task_cols: List[str],
) -> None:
    """
    Merge predictions with ground-truth labels and print / save per-task metrics.

    Args:
        pred_df:       DataFrame with columns 'Filepath' + pred column per task.
        label_csv_path: Path to the label CSV containing ground-truth.
        output_dir:    Directory where ``evaluation.csv`` will be written.
        task_cols:     List of task column names (must exist in both DataFrames).
    """
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        print("scikit-learn not available — skipping evaluation.")
        return

    gt_df = pd.read_csv(label_csv_path)
    # Normalise filepath column name
    fp_col_gt = next(
        (c for c in gt_df.columns if c.lower() in ("filepath", "file_path")), None
    )
    if fp_col_gt is None:
        print("Ground-truth CSV has no 'Filepath' column — skipping evaluation.")
        return

    gt_df = gt_df.rename(columns={fp_col_gt: "Filepath"})
    merged = pred_df.merge(gt_df, on="Filepath", suffixes=("_pred", "_gt"))

    rows = []
    for col in task_cols:
        pred_col = f"{col}_pred"
        gt_col   = f"{col}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            print(f"  Skipping {col}: columns not found after merge.")
            continue

        valid = merged[[pred_col, gt_col]].dropna()
        valid = valid[valid[gt_col] != "na"]
        if valid.empty:
            print(f"  {col}: no valid samples after dropping 'na'.")
            continue

        acc = accuracy_score(valid[gt_col], valid[pred_col])
        f1  = f1_score(valid[gt_col], valid[pred_col], average="macro", zero_division=0)
        print(f"  {col:30s}  acc={acc:.4f}  macro-F1={f1:.4f}  (n={len(valid)})")
        rows.append({"task": col, "accuracy": acc, "macro_f1": f1, "n_samples": len(valid)})

    if rows:
        eval_path = os.path.join(output_dir, "evaluation.csv")
        pd.DataFrame(rows).to_csv(eval_path, index=False)
        print(f"\nEvaluation saved to {eval_path}")


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def main(
    ckpt: str,
    output_dir: str,
    split: Optional[List[str]],
    n_slices: int,
    img_size: int,
    batch_size: int,
    device: torch.device,
    backbone_type: str,
    backbone_channels: int,
    backbone_blocks: List[int],
    growth_rate: int,
    embedding_dim: int,
    num_samples: Optional[int],
) -> pd.DataFrame:
    """
    Run inference and return a DataFrame of predictions.

    Returns
    -------
    pd.DataFrame with columns:
        Filepath, label_AcquisitionPlane, label_SequenceContrast, label_Localizer
    """
    from IMC.data.adni_dataloader_local import ADNI_LABEL_NAMES, ADNIDataset

    inner = ADNIDataset(
        split=split,
        n_slices=n_slices,
        img_size=img_size,
        augment_conf="NONE2D",
        is_infer=True,
        num_samples=num_samples,
    )
    dataset = ADNI3DDataset(inner)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Inference on {len(dataset)} samples  |  splits={split}")

    # ---- Model ----
    cl_d = inner.get_n_labels()
    model = PyramidPooling3DClassifier(
        num_classes_dict=cl_d,
        backbone_type=backbone_type,
        backbone_channels=backbone_channels,
        backbone_blocks=backbone_blocks,
        growth_rate=growth_rate,
        embedding_dim=embedding_dim,
    ).to(device)

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading checkpoint: {ckpt}")
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Label maps for decoding argmax back to class strings
    label_maps: Dict[str, List[str]] = ADNI_LABEL_NAMES  # {task: [class, …]}
    task_names  = list(label_maps.keys())

    all_filepaths: List[str] = []
    # One list per task: decoded class-name predictions
    all_preds: Dict[str, List[str]] = {t: [] for t in task_names}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring"):
            # Inference mode: (images, metadata, filepath)
            images, _metadata, filepaths = batch

            images = normalize_per_sample(images.to(device))
            # network07 forward: outputs is a list of logits, one per task
            outputs = model(images)

            for task_idx, task_name in enumerate(task_names):
                logits = outputs[task_idx]                       # (B, n_classes)
                preds  = torch.argmax(logits, dim=1).cpu().tolist()
                class_names = label_maps[task_name]
                all_preds[task_name].extend(class_names[p] for p in preds)

            all_filepaths.extend(filepaths)

    pred_df = pd.DataFrame({"Filepath": all_filepaths, **all_preds})

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    return pred_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for Network 07 on ADNI dataset."
    )

    # Required
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a trained model checkpoint (best_model.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where predictions.csv (and evaluation.csv) will be saved.",
    )

    # Data
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Comma-separated fold names to run inference on, e.g. 'fold_0,fold_1'. "
            "Omit to run on the entire dataset."
        ),
    )
    parser.add_argument("--n_slices", type=int, default=16,
                        help="Number of slices per series (depth for network07).")
    parser.add_argument("--img_size", type=int, default=224, help="Spatial resolution.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap dataset size (smoke-tests).")

    # Model architecture (must match checkpoint)
    parser.add_argument("--backbone_type", type=str, default="resnet")
    parser.add_argument("--backbone_channels", type=int, default=32)
    parser.add_argument("--backbone_blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--growth_rate", type=int, default=12)
    parser.add_argument("--embedding_dim", type=int, default=512)

    # Hardware
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    _device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _split  = [s.strip() for s in args.split.split(",")] if args.split else None

    os.makedirs(args.output_dir, exist_ok=True)

    pred_df = main(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        split=_split,
        n_slices=args.n_slices,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=_device,
        backbone_type=args.backbone_type,
        backbone_channels=args.backbone_channels,
        backbone_blocks=args.backbone_blocks,
        growth_rate=args.growth_rate,
        embedding_dim=args.embedding_dim,
        num_samples=args.num_samples,
    )

    # ---- Optional evaluation ----
    label_csv = os.getenv(
        "ADNI_LABEL_CSV_PATH",
        "/home/tuan.truong/codebase/IMC/labels/labels_brain_20250721.csv",
    )
    if os.path.exists(label_csv):
        print("\nRunning evaluation against ground-truth labels …")
        from IMC.data.adni_dataloader_local import ADNI_LABEL_NAMES
        evaluate_predictions(
            pred_df=pred_df,
            label_csv_path=label_csv,
            output_dir=args.output_dir,
            task_cols=list(ADNI_LABEL_NAMES.keys()),
        )
    else:
        print(f"\nSkipping evaluation — label CSV not found: {label_csv}")
