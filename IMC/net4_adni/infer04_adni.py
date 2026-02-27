"""
Inference script for Network v04 (cross-attention fusion) on ADNI Dataset.

Loads a trained MRISequenceClassifier checkpoint and runs inference over a
specified data split (or the full dataset when no split is given). Predictions
for all three ADNI tasks are decoded from argmax logits and written to a CSV.

Optionally computes per-task accuracy and macro-F1 when ground-truth labels are
present in the label CSV.

Usage
-----
Single fold
    python -m IMC.net4_adni.infer04_adni \\
        --ckpt ./logs/net04_adni_cv/fold_0/best_model.pth \\
        --output_dir ./logs/net04_adni_cv/fold_0/infer \\
        --split fold_0

All folds (full dataset)
    python -m IMC.net4_adni.infer04_adni \\
        --ckpt ./logs/net04_adni_cv/fold_0/best_model.pth \\
        --output_dir ./logs/net04_adni_cv/fold_0/infer_all

Authors: Tuan Truong
Date: 2026
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from IMC.helper import normalize_per_sample
from IMC.network04 import MRISequenceClassifier, ImageBasedClassifier, MetadataBasedClassifier, SingleChannelImageBasedClassifier, MRISequenceClassifierWithSparseMetadata
import torch.nn as nn

# ---------------------------------------------------------------------------
# ADNI environment defaults
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "ADNI_LOCAL_DATASET_PATH",
    "/home/tuan.truong/data/ADNI_full",
)
os.environ.setdefault(
    "ADNI_LABEL_CSV_PATH",
    "/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv",
)

os.environ.setdefault(
    "ADNI_METADATA_PATH",
    "/home/tuan.truong/codebase/IMC/labels/adni_metadata/adni_encoded_metadata_20260224.parquet",
)

from IMC.data.adni_dataloader_local import ADNIDataset

# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_inference_dataloader(
    split: Optional[List[str]],
    n_slices: int,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    aggregated_metadata: bool = False,
    num_samples: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create an ADNI DataLoader in inference mode (no labels returned)."""

    dataset = ADNIDataset(
        split=split,
        n_slices=n_slices,
        img_size=img_size,
        augment_conf="NONE2D",
        is_infer=True,
        aggregated_metadata=aggregated_metadata,
        num_samples=num_samples,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    num_classes_dict: dict,
    metadata_input_dim: int,
    img_enc_backbone: str = "densenet121",
    modality: str = "combined",
    image_naive_approach: bool = False,
    metadata_enc_type: str = "none",
    sparse_enc_version: str = "v1",
    fusion_module_version: str = "v1",
    incl_regression: bool = True,
    device: torch.device = None
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_classes_dict: Dictionary of number of classes per task
        metadata_input_dim: Dimension of metadata input
        img_enc_backbone: Image encoder backbone architecture
        incl_regression: Whether the model includes regression
        modality: str = "combined",
        metadata_enc_type: str = "none",
        sparse_enc_version: str = "v1",
        fusion_module_version: str = "v1",
        image_naive_approach: bool = False,
        device: torch.device = None
    
    Returns:
        Loaded model ready for inference
    """
    # Initialize model architecture
    if modality == "image":
        if image_naive_approach:
            model = SingleChannelImageBasedClassifier(num_classes_dict=num_classes_dict)
        else:
            model = ImageBasedClassifier(num_classes_dict=num_classes_dict, backbone=img_enc_backbone)
    elif modality == "metadata":
        if metadata_enc_type == "none":
            model = MetadataBasedClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_encoder_type=metadata_enc_type,
                imputer_type="contextual",
                incl_regression=incl_regression
            )
        elif metadata_enc_type == "sparse":
            if sparse_enc_version == "v1":
                model = MetadataBasedClassifier(
                    metadata_input_dim=metadata_input_dim,
                    num_classes_dict=num_classes_dict,
                    metadata_encoder_type="sparse",
                    incl_regression=incl_regression
                )
            else:
                model = MetadataBasedClassifier(
                    metadata_input_dim=metadata_input_dim,
                    num_classes_dict=num_classes_dict,
                    metadata_encoder_type="sparse_v2",
                    incl_regression=incl_regression
                )
        else:
            # Imputer is NanIgnorer
            model = MetadataBasedClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_encoder_type="imputer",
                imputer_type="ignore",
                incl_regression=incl_regression
            )
    else:
        # Combined model with cross-attention fusion
        if metadata_enc_type == "none":
            model = MRISequenceClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                img_enc_backbone=img_enc_backbone,
                incl_regression=incl_regression,
                fusion_module_version=fusion_module_version,
                imputer_type="ignore"
            )
        elif metadata_enc_type == "imputer":
            model = MRISequenceClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                img_enc_backbone=img_enc_backbone,
                incl_regression=incl_regression,
                fusion_module_version=fusion_module_version,
                imputer_type="contextual"
            )
        elif metadata_enc_type == "sparse" and sparse_enc_version == "v1":
            model = MRISequenceClassifierWithSparseMetadata(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_embeder_type="sparse",
                img_enc_backbone=img_enc_backbone,
                dropout_metadata=False,
                fusion_module_version=fusion_module_version,
                include_regression=incl_regression
            )
        elif metadata_enc_type == "sparse" and sparse_enc_version == "v2":
            model = MRISequenceClassifierWithSparseMetadata(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_embeder_type="sparse_v2",
                img_enc_backbone=img_enc_backbone,
                dropout_metadata=False,
                fusion_module_version=fusion_module_version,
                include_regression=incl_regression
            )
        elif metadata_enc_type == "sparse" and sparse_enc_version == "v5":
            model = MRISequenceClassifierWithSparseMetadata(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_embeder_type="sparse_v5",
                img_enc_backbone=img_enc_backbone,
                dropout_metadata=False,
                fusion_module_version=fusion_module_version,
                include_regression=incl_regression,
            )
        else:
            raise ValueError(f"Invalid metadata_enc_type '{metadata_enc_type}' or sparse_enc_version '{sparse_enc_version}'")
            
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    return model



# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Run inference and return decoded predictions and filepaths.

    Returns
    -------
    predictions : dict[task_name -> list[class_str]]
    filepaths   : list[str]
    """
    from IMC.data.adni_dataloader_local import ADNI_LABEL_NAMES

    label_maps   = ADNI_LABEL_NAMES          # {task: [class, …]}
    task_names   = list(label_maps.keys())
    predictions  = {t: [] for t in task_names}
    filepaths: List[str] = []

    print(f"Running inference on {len(dataloader.dataset)} samples …")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            # Inference mode: (images, metadata, filepath)
            images, metadata, paths = batch

            images   = normalize_per_sample(images.to(device))
            metadata = metadata.to(device)
            outputs  = model(images, metadata)

            for task_idx, task_name in enumerate(task_names):
                logits      = outputs[task_idx]               # (B, n_classes)
                preds       = torch.argmax(logits, dim=1).cpu().tolist()
                class_names = label_maps[task_name]
                predictions[task_name].extend(class_names[p] for p in preds)

            filepaths.extend(paths)

    return predictions, filepaths


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------

def save_predictions(
    predictions: Dict[str, List[str]],
    filepaths: List[str],
    output_dir: str,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    """Save predictions to CSV and write an inference metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)

    pred_df = pd.DataFrame({"Filepath": filepaths, **predictions})

    pred_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved to {pred_path}")

    if checkpoint_path:
        meta = {
            "checkpoint": checkpoint_path,
            "num_samples": len(filepaths),
            "tasks": list(predictions.keys()),
        }
        meta_path = os.path.join(output_dir, "inference_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"✓ Inference metadata saved to {meta_path}")

    return pred_df


# ---------------------------------------------------------------------------
# Optional evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    pred_df: pd.DataFrame,
    label_csv_path: str,
    output_dir: str,
    task_names: List[str],
) -> None:
    """Compute per-task accuracy and macro-F1; save evaluation.csv and a report."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
        )
    except ImportError:
        print("scikit-learn not available — skipping evaluation.")
        return

    gt_df   = pd.read_csv(label_csv_path)
    fp_col  = next((c for c in gt_df.columns if c.lower() in ("filepath", "file_path")), None)
    if fp_col is None:
        print("Ground-truth CSV has no 'Filepath' column — skipping evaluation.")
        return

    gt_df  = gt_df.rename(columns={fp_col: "Filepath"})
    merged = pred_df.merge(gt_df, on="Filepath", suffixes=("_pred", "_gt"))

    eval_rows = []
    detailed_rows = []
    os.makedirs(output_dir, exist_ok=True)

    for task in task_names:
        pred_col = f"{task}_pred"
        gt_col   = f"{task}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            print(f"  Skipping {task}: columns missing after merge.")
            continue

        valid = merged[[pred_col, gt_col]].dropna()
        valid = valid[valid[gt_col] != "na"]
        if valid.empty:
            print(f"  {task}: no valid samples after dropping 'na'.")
            continue

        acc = accuracy_score(valid[gt_col], valid[pred_col])
        f1  = f1_score(valid[gt_col], valid[pred_col], average="macro", zero_division=0)
        wp  = f1_score(valid[gt_col], valid[pred_col], average="weighted", zero_division=0)
        print(f"  {task:35s}  acc={acc:.4f}  macro-F1={f1:.4f}  (n={len(valid)})")

        eval_rows.append({
            "Task": task,
            "Accuracy": acc,
            "Macro_F1": f1,
            "Weighted_F1": wp,
            "Samples": len(valid),
        })

        # Per-class detail via classification_report
        report = classification_report(
            valid[gt_col],
            valid[pred_col],
            zero_division=0,
            output_dict=True,
        )
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                detailed_rows.append({
                    "Task": task,
                    "Label": label,
                    "Precision": metrics.get("precision", None),
                    "Recall": metrics.get("recall", None),
                    "F1-Score": metrics.get("f1-score", None),
                    "Support": metrics.get("support", None),
                })

    # Save summary
    if eval_rows:
        summary_path = os.path.join(output_dir, "summary_metrics.csv")
        pd.DataFrame(eval_rows).to_csv(summary_path, index=False)
        print(f"✓ Summary metrics saved to {summary_path}")

    # Save detailed
    if detailed_rows:
        detailed_path = os.path.join(output_dir, "detailed_metrics.csv")
        pd.DataFrame(detailed_rows).to_csv(detailed_path, index=False)
        print(f"✓ Detailed metrics saved to {detailed_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from IMC.data.adni_dataloader_local import ADNI_LABEL_NAMES

    parser = argparse.ArgumentParser(
        description="Inference for Network 04 on ADNI dataset."
    )

    # Required
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a trained model checkpoint (best_model.pth).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where predictions.csv will be saved.")

    # Data
    parser.add_argument(
        "--split", type=str, default=None,
        help="Comma-separated fold names, e.g. 'fold_0,fold_1'. Omit for full dataset.",
    )
    parser.add_argument("--n_slices", type=int, default=3,
                        help="Number of slices per series.")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Spatial resolution.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap dataset size (smoke-tests).")

    # Model architecture (must match checkpoint)
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--modality", type=str, default="combined",
                        choices=["combined", "image", "metadata"])
    parser.add_argument("--metadata_enc_type", type=str, default="none",
                        choices=["none", "imputer", "sparse"])
    parser.add_argument("--sparse_enc_version", type=str, default="v1",
                        choices=["v1", "v2", "v5"])
    parser.add_argument("--image_naive_approach", action="store_true",
                        help="Whether the image-based model uses a naive approach (no pretrained backbone, no cross-attention).")
    parser.add_argument("--fusion_module_version", type=str, default="v1")

    # Hardware
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)

    # Evaluation
    parser.add_argument("--eval", action="store_true",
                        help="Compute metrics against ground-truth labels after inference.")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    split  = [s.strip() for s in args.split.split(",")] if args.split else None

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Dataloader ----
    print("Creating dataloader …")
    dataloader = create_inference_dataloader(
        split=split,
        n_slices=args.n_slices,
        img_size=args.img_size,
        batch_size=args.batch_size,
        aggregated_metadata=False,
        num_samples=args.num_samples,
    )
    print(f"✓ Loaded {len(dataloader.dataset)} samples")

    # ---- Model ----
    num_classes_dict   = dataloader.dataset.get_n_labels()
    metadata_input_dim = dataloader.dataset.num_metadata_features

    print(f"\nModel configuration:")
    print(f"  Backbone          : {args.backbone}")
    print(f"  Fusion version    : {args.fusion_module_version}")
    print(f"  Metadata dim      : {metadata_input_dim}")
    print(f"  Tasks             : {list(num_classes_dict.keys())}")

    model = load_model(
        checkpoint_path=args.ckpt,
        num_classes_dict=num_classes_dict,
        metadata_input_dim=metadata_input_dim,
        img_enc_backbone=args.backbone,
        fusion_module_version=args.fusion_module_version,
        metadata_enc_type=args.metadata_enc_type,
        sparse_enc_version=args.sparse_enc_version,
        image_naive_approach=args.image_naive_approach,
        modality=args.modality,
        device=device,
    )

    # ---- Inference ----
    predictions, filepaths = run_inference(model, dataloader, device)

    # ---- Save ----
    pred_df = save_predictions(
        predictions=predictions,
        filepaths=filepaths,
        output_dir=args.output_dir,
        checkpoint_path=args.ckpt,
    )

    # ---- Evaluation ----
    if args.eval:
        label_csv = os.environ.get(
            "ADNI_LABEL_CSV_PATH",
            "/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv",
        )
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)
        if os.path.exists(label_csv):
            eval_dir = os.path.join(args.output_dir, "evaluation")
            run_evaluation(
                pred_df=pred_df,
                label_csv_path=label_csv,
                output_dir=eval_dir,
                task_names=list(ADNI_LABEL_NAMES.keys()),
            )
        else:
            print(f"Label CSV not found: {label_csv} — skipping evaluation.")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Results saved to : {args.output_dir}")
    print(f"Samples processed: {len(filepaths)}")


if __name__ == "__main__":
    main()
