"""Inference entry point for Network v07 (PyramidPooling3DClassifier) on ADNI.

Loads a trained :class:`~IMC.network07.PyramidPooling3DClassifier` checkpoint
and runs batch inference over a specified data split (or the full dataset when
no split is given).  Predictions for all three ADNI tasks are decoded from
argmax logits and written to a CSV file alongside the series filepath.

Optionally computes per-task accuracy and macro-F1 when ground-truth labels
are present in the label CSV.

Typical usage
-------------
Single fold::

    python -m IMC.net7.infer_adni \\
        --ckpt ./logs/net07_adni_cv/fold_0/best_model.pth \\
        --output_dir ./infer_out/fold_0 \\
        --split fold_0

Full dataset::

    python -m IMC.net7.infer_adni \\
        --ckpt ./logs/net07_adni_cv/fold_0/best_model.pth \\
        --output_dir ./infer_out/all

Environment variables
---------------------
::

    ADNI_LOCAL_DATASET_PATH – root folder of the ADNI brain MRI dataset
    ADNI_LABEL_CSV_PATH     – path to the ADNI label CSV
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from IMC.data.constants import ADNI_LABEL_NAMES
from IMC.network07 import PyramidPooling3DClassifier

# ---------------------------------------------------------------------------
# Default dataset paths
# ---------------------------------------------------------------------------
_ADNI_DATASET_PATH = "/home/tuan.truong/data/ADNI_full"
_ADNI_LABEL_CSV_PATH = (
    "/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv"
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
    """Merge predictions with ground-truth labels and report per-task metrics.

    Computes per-task classification accuracy and macro-averaged F1 score,
    prints a summary to stdout, and writes an ``evaluation.csv`` file.

    Args:
        pred_df: DataFrame with columns ``Filepath`` + one predicted column
            per task.
        label_csv_path: Path to the label CSV containing ground-truth columns.
        output_dir: Directory where ``evaluation.csv`` will be written.
        task_cols: List of task column names that must exist in both DataFrames
            (e.g. ``["label_AcquisitionPlane", "label_SequenceContrast",
            "label_Localizer"]``).
    """
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        print("scikit-learn not available – skipping evaluation.")
        return

    gt_df = pd.read_csv(label_csv_path)
    fp_col_gt = next(
        (c for c in gt_df.columns if c.lower() in ("filepath", "file_path")), None
    )
    if fp_col_gt is None:
        print("Ground-truth CSV has no 'Filepath' column – skipping evaluation.")
        return

    gt_df = gt_df.rename(columns={fp_col_gt: "Filepath"})
    merged = pred_df.merge(gt_df, on="Filepath", suffixes=("_pred", "_gt"))

    rows: list[dict] = []
    for col in task_cols:
        pred_col = f"{col}_pred"
        gt_col = f"{col}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            print(f"  Skipping {col}: columns not found after merge.")
            continue

        valid = merged[[pred_col, gt_col]].dropna()
        valid = valid[valid[gt_col] != "na"]
        if valid.empty:
            print(f"  {col}: no valid samples after dropping 'na'.")
            continue

        acc = accuracy_score(valid[gt_col], valid[pred_col])
        f1 = f1_score(valid[gt_col], valid[pred_col], average="macro", zero_division=0)
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
    num_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Run inference and return a DataFrame of predictions.

    Args:
        ckpt: Path to the trained model checkpoint.
        output_dir: Directory where ``predictions.csv`` is written.
        split: List of fold-name strings to infer on (e.g. ``["fold_0"]``).
            Pass ``None`` to use the full dataset.
        n_slices: Number of slices per series (acts as depth for network07).
        img_size: Spatial resolution (H = W = img_size).
        batch_size: Batch size for inference.
        device: Target device.
        backbone_type: 3-D CNN backbone name.
        backbone_channels: Initial backbone channels.
        backbone_blocks: ResNet block counts per stage.
        growth_rate: DenseNet growth rate.
        embedding_dim: MLP projection dimension.
        num_samples: Cap the dataset size (for smoke-tests).

    Returns:
        :class:`~pandas.DataFrame` with columns ``Filepath`` + one column
        per ADNI task.
    """
    from IMC.data.adni_dataloader_local import ADNI3DDataset
    dataset = ADNI3DDataset(
        split=split,
        n_slices=n_slices,
        img_size=img_size,
        augment_conf="NONE3D",
        is_infer=True,
        num_samples=num_samples,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Inference on {len(dataset)} samples  |  splits={split}")

    # ---- Model ----
    cl_d = dataset.get_n_labels()
    model = PyramidPooling3DClassifier(
        num_classes_dict=cl_d,
        backbone_type=backbone_type,
        backbone_channels=backbone_channels,
        backbone_blocks=list(backbone_blocks),
        growth_rate=growth_rate,
        embedding_dim=embedding_dim,
    ).to(device)

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading checkpoint: {ckpt}")
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ---- Label maps for decoding ----
    label_maps: Dict[str, List[str]] = ADNI_LABEL_NAMES
    task_names = list(label_maps.keys())

    all_filepaths: List[str] = []
    all_preds: Dict[str, List[str]] = {t: [] for t in task_names}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring"):
            # Dataset yields (images, metadata, filepath) in inference mode
            images, _metadata, filepaths = batch
            images = images.to(device)

            outputs = model(images)  # list[(B, n_classes_i)]

            for task_idx, task_name in enumerate(task_names):
                logits = outputs[task_idx]
                preds = torch.argmax(logits, dim=1).cpu().tolist()
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for net7 ADNI inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch inference for Network v07 (PyramidPooling3DClassifier) "
            "on the ADNI Brain Dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions.csv.")

    # Data
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Comma-separated fold names to infer on (e.g. 'fold_0,fold_1'). "
            "Omit to infer on the full dataset."
        ),
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=16,
        help="Number of slices per series (depth for network07).",
    )
    parser.add_argument("--img_size", type=int, default=224, help="Spatial resolution (H = W = img_size).")
    parser.add_argument("--num_samples", type=int, default=None, help="Cap dataset size (smoke-tests).")

    # Architecture (must match checkpoint)
    parser.add_argument("--backbone_type", type=str, default="resnet", help="3-D CNN backbone name.")
    parser.add_argument("--backbone_channels", type=int, default=32)
    parser.add_argument(
        "--backbone_blocks",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
    )
    parser.add_argument("--growth_rate", type=int, default=12)
    parser.add_argument("--embedding_dim", type=int, default=512)

    # Hardware
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (-1 for CPU).")

    # Evaluation
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate predictions against ground-truth labels after inference.",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Override ADNI_LOCAL_DATASET_PATH.")
    parser.add_argument("--label_csv_path", type=str, default=None, help="Override ADNI_LABEL_CSV_PATH.")

    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()

    os.environ.setdefault(
        "ADNI_LOCAL_DATASET_PATH",
        _args.dataset_path or _ADNI_DATASET_PATH,
    )
    if _args.dataset_path:
        os.environ["ADNI_LOCAL_DATASET_PATH"] = _args.dataset_path

    label_csv_path = _args.label_csv_path or _ADNI_LABEL_CSV_PATH
    os.environ.setdefault("ADNI_LABEL_CSV_PATH", label_csv_path)
    if _args.label_csv_path:
        os.environ["ADNI_LABEL_CSV_PATH"] = _args.label_csv_path

    _device = (
        torch.device(f"cuda:{_args.gpu}")
        if _args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    _split = (
        [s.strip() for s in _args.split.split(",")]
        if _args.split is not None
        else None
    )

    os.makedirs(_args.output_dir, exist_ok=True)

    _pred_df = main(
        ckpt=_args.ckpt,
        output_dir=_args.output_dir,
        split=_split,
        n_slices=_args.n_slices,
        img_size=_args.img_size,
        batch_size=_args.batch_size,
        device=_device,
        backbone_type=_args.backbone_type,
        backbone_channels=_args.backbone_channels,
        backbone_blocks=_args.backbone_blocks,
        growth_rate=_args.growth_rate,
        embedding_dim=_args.embedding_dim,
        num_samples=_args.num_samples,
    )

    # ---- Optional evaluation ----
    if _args.eval and os.path.exists(label_csv_path):
        print("\nRunning evaluation against ground-truth labels …")
        evaluate_predictions(
            pred_df=_pred_df,
            label_csv_path=label_csv_path,
            output_dir=_args.output_dir,
            task_cols=list(ADNI_LABEL_NAMES.keys()),
        )
    elif _args.eval:
        print(f"\nSkipping evaluation – label CSV not found: {label_csv_path}")
