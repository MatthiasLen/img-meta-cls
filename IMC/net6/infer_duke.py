"""Inference entry point for Network v06 (PixelOnlyModel) on the Duke dataset.

Loads a trained :class:`~IMC.network06.PixelOnlyModel` checkpoint and runs
batch inference over a specified fold split (or the full dataset when no split
is given).  Predictions are decoded from argmax logits and written to a CSV.

Heuristic RF gate
-----------------
Optionally a per-fold Random Forest model trained by :mod:`IMC.net6.train_rf`
can be provided via ``--rf_model``.  When supplied, the RF is evaluated on the
tabular metadata for each sample.  For ``SequenceType_Code_norm``, if the RF's
maximum class probability exceeds ``--threshold`` the RF prediction is used;
otherwise the image-model prediction is used.  All other tasks always use the
image-model predictions.

Typical usage
-------------
::

    # Image-model only (no RF gate)
    python -m IMC.net6.infer \\
        --ckpt ./logs/.../fold_0/best_model.pth \\
        --output_dir ./infer_out/fold_0

    # With RF gate for SequenceType_Code_norm
    python -m IMC.net6.infer \\
        --ckpt ./logs/.../fold_0/best_model.pth \\
        --output_dir ./infer_out/fold_0 \\
        --rf_model ./rf_checkpoints/duke_metadata_rf_fold_0.joblib \\
        --threshold 0.7

Environment variables
---------------------
::

    LOCAL_DATASET_PATH – root folder of the Duke MRI image dataset
    METADATA_PATH      – path to the Duke encoded metadata parquet file
    LABEL_CSV_PATH     – path to the Duke label CSV
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from IMC.data.constants import DUKE_ORIGINAL_LABEL_NAMES
from IMC.evaluate_duke import run_evaluation
from IMC.network06 import PixelOnlyModel

# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------


def create_inference_dataloader(
    batch_size: int = 16,
    num_workers: int = 4,
    fold_indices: Optional[list[int]] = None,
) -> torch.utils.data.DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` for Duke inference.

    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader worker processes.
        fold_indices: List of fold indices to include (e.g. ``[0, 1]``).
            Pass ``None`` to use the full dataset.

    Returns:
        Configured :class:`~torch.utils.data.DataLoader` over a single-slice
        :class:`~IMC.data.duke_dataloader_local.LiverDataset`.
    """
    from IMC.data.duke_dataloader_local import LiverDataset

    split = [f"fold_{i}" for i in fold_indices] if fold_indices is not None else None
    dataset = LiverDataset(
        split=split,
        num_samples=None,
        n_slices=1,
        sampling_type="equidistant",
        augment_conf="IMAGENET299_CENTER",
        aggregated_metadata=False,
        is_infer=True,
        label_names=DUKE_ORIGINAL_LABEL_NAMES,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    num_classes_dict: dict,
    img_enc_backbone: str = "densenet121",
    device: Optional[torch.device] = None,
) -> PixelOnlyModel:
    """Load a :class:`~IMC.network06.PixelOnlyModel` from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (``best_model.pth``).
        num_classes_dict: Task → number-of-classes mapping (must match the
            checkpoint).
        img_enc_backbone: CNN backbone name (must match the checkpoint).
        device: Target device.  Defaults to CUDA if available, else CPU.

    Returns:
        :class:`~IMC.network06.PixelOnlyModel` in eval mode on *device*.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PixelOnlyModel(
        num_classes_dict=num_classes_dict,
        image_backbone=img_enc_backbone,
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✓ Model loaded successfully.")
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: PixelOnlyModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.7,
    rf_model_path: Optional[str] = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Run inference and return decoded predictions for all tasks.

    For ``SequenceType_Code_norm``, a heuristic RF gate is applied when an RF
    model is provided: if the RF's max class probability ≥ *threshold* the RF
    prediction is accepted; otherwise the image model's prediction is used.
    All other tasks always use image-model predictions.

    Args:
        model: Loaded :class:`~IMC.network06.PixelOnlyModel` in eval mode.
        dataloader: DataLoader yielding ``(images, metadata, paths)`` batches.
        device: Target device for inference.
        threshold: Confidence threshold for the RF gate.
        rf_model_path: Optional path to a joblib-serialised
            :class:`sklearn.pipeline.Pipeline` (scaler + RF).

    Returns:
        tuple: ``(predictions, filepaths)`` where *predictions* is a dict
        mapping task name to a list of class-name strings, and *filepaths* is
        the corresponding list of file paths.
    """
    num_classes_dict: dict = dataloader.dataset.get_n_labels()
    label_maps: dict = dataloader.dataset.label_names
    task_names = list(num_classes_dict.keys())

    seq_task = "SequenceType_Code_norm"
    seq_idx = task_names.index(seq_task) if seq_task in task_names else 0

    rf_model = None
    if rf_model_path:
        print(f"Loading RF model : {rf_model_path}")
        rf_model = joblib.load(rf_model_path)

    predictions: dict[str, list[str]] = {t: [] for t in task_names}
    filepaths: list[str] = []

    print(f"Running inference on {len(dataloader.dataset)} samples …")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            images, metadata, paths, _ = batch
            images = images.to(device)

            # Image logits: list of (B, n_classes_i) per task
            img_logits = model.get_image_logits(images)

            # ---- Handle SequenceType_Code_norm with optional RF gate ----
            if rf_model is not None:
                meta_np = metadata.numpy()
                rf_probs_batch = np.stack([rf_model.predict_proba(row.reshape(1, -1))[0] for row in meta_np])  # (B, C)
                rf_probs_t = torch.tensor(rf_probs_batch, dtype=torch.float32, device=device)
                max_rf_conf = rf_probs_t.max(dim=1).values  # (B,)
                use_rf = max_rf_conf >= threshold  # (B,)
                # Approximate RF logits as log(p); clamp to avoid log(0)
                rf_logits = torch.log(torch.clamp(rf_probs_t, min=1e-8))
                gated_logits = torch.where(use_rf.unsqueeze(1), rf_logits, img_logits[seq_idx])
            else:
                gated_logits = img_logits[seq_idx]

            # Decode SequenceType_Code_norm
            seq_preds = torch.argmax(gated_logits, dim=1).cpu().tolist()
            seq_classes = label_maps[seq_task]
            predictions[seq_task].extend(seq_classes[p] for p in seq_preds)

            # Decode remaining tasks (image logits only)
            for i, task in enumerate(task_names):
                if task == seq_task:
                    continue
                preds = torch.argmax(img_logits[i], dim=1).cpu().tolist()
                predictions[task].extend(label_maps[task][p] for p in preds)

            filepaths.extend(paths)

    return predictions, filepaths


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_predictions(
    predictions: dict[str, list[str]],
    filepaths: list[str],
    output_dir: str,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    """Write predictions to ``<output_dir>/predictions.csv``.

    Also writes ``inference_metadata.json`` when *checkpoint_path* is provided.

    Args:
        predictions: Dict mapping task name to list of predicted class strings.
        filepaths: List of sample file paths (same order as *predictions*).
        output_dir: Directory where output files are written.
        checkpoint_path: Optional checkpoint path recorded in the metadata JSON.

    Returns:
        :class:`~pandas.DataFrame` with columns ``Filepath`` + one column per task.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_df = pd.DataFrame({"Filepath": filepaths, **predictions})
    pred_path = os.path.join(output_dir, "predictions.csv")
    output_df.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved to {pred_path}")

    if checkpoint_path:
        meta = {
            "checkpoint": checkpoint_path,
            "num_samples": len(filepaths),
            "tasks": list(predictions.keys()),
        }
        with open(os.path.join(output_dir, "inference_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return output_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for net6 Duke inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch inference for IMC Network v06 (PixelOnlyModel) on the Duke dataset, "
            "with optional Random Forest gate for SequenceType_Code_norm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (-1 for CPU).")
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated fold indices to infer on (e.g. '0,1,2'). Defaults to all data.",
    )
    parser.add_argument(
        "--img_enc_backbone",
        type=str,
        default="densenet121",
        help="CNN backbone (must match the checkpoint).",
    )
    parser.add_argument(
        "--rf_model",
        type=str,
        default="",
        help="Path to a joblib RF pipeline for SequenceType_Code_norm gating.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold above which the RF prediction overrides the image model.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation against ground-truth labels after inference.",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Override LOCAL_DATASET_PATH.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Override METADATA_PATH.")
    parser.add_argument("--label_csv_path", type=str, default=None, help="Override LABEL_CSV_PATH.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run net6 inference on the Duke dataset.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.
    """
    os.environ["DEBUG_MODE"] = "0"
    if args.dataset_path:
        os.environ["LOCAL_DATASET_PATH"] = args.dataset_path
    if args.metadata_path:
        os.environ["METADATA_PATH"] = args.metadata_path
    if args.label_csv_path:
        os.environ["LABEL_CSV_PATH"] = args.label_csv_path

    missing = [name for name in ("LOCAL_DATASET_PATH", "METADATA_PATH", "LABEL_CSV_PATH") if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "Missing Duke dataset configuration. Set the environment variables "
            f"{', '.join(missing)} or pass the corresponding CLI overrides."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")
    print(f"Device : {device}")

    fold_indices = [int(f.strip()) for f in args.folds.split(",")] if args.folds is not None else None
    if fold_indices is not None:
        print(f"Folds  : {fold_indices}")
    else:
        print("Folds  : all data")

    print("\nCreating dataloader …")
    dataloader = create_inference_dataloader(
        batch_size=args.batch_size,
        num_workers=4,
        fold_indices=fold_indices,
    )
    print(f"✓ Loaded {len(dataloader.dataset)} samples.")

    num_classes_dict = dataloader.dataset.get_n_labels()
    print(f"\nTasks  : {list(num_classes_dict.keys())}")
    print(f"Backbone: {args.img_enc_backbone}")

    model = load_model(
        checkpoint_path=args.ckpt,
        num_classes_dict=num_classes_dict,
        img_enc_backbone=args.img_enc_backbone,
        device=device,
    )

    predictions, filepaths = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        rf_model_path=args.rf_model if args.rf_model else None,
    )

    pred_df = save_predictions(
        predictions=predictions,
        filepaths=filepaths,
        output_dir=args.output_dir,
        checkpoint_path=args.ckpt,
    )

    if args.eval:
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)
        label_csv_path = os.environ["LABEL_CSV_PATH"]
        label_df = pd.read_csv(label_csv_path)
        label_df["Filepath"] = label_df["Filepath"].map(dataloader.dataset._normalize_dataset_index)
        run_evaluation(pred_df, label_df, args.output_dir)
        print(f"✓ Evaluation results saved to {args.output_dir}")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Results saved to      : {args.output_dir}")
    print(f"Total samples processed: {len(filepaths)}")


if __name__ == "__main__":
    main(parse_args())
