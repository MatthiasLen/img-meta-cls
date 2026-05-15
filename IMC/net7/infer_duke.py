"""Inference entry point for Network v07 (PyramidPooling3DClassifier) on Duke.

Loads a trained :class:`~IMC.network07.PyramidPooling3DClassifier` checkpoint
and runs batch inference over a specified fold split (or the full dataset when
no split is given).  Predictions for all Duke tasks are decoded from argmax
logits and written to a CSV file alongside the series filepath.

Optionally computes per-task accuracy when ground-truth labels are present in
the label CSV.

Typical usage
-------------
Infer on a single fold::

    python -m IMC.net7.infer_duke \\
        --ckpt ./logs/.../fold_0/best_model.pth \\
        --output_dir ./infer_out/fold_0 \\
        --split fold_0

Infer on the full dataset::

    python -m IMC.net7.infer_duke \\
        --ckpt ./logs/.../fold_0/best_model.pth \\
        --output_dir ./infer_out/all

Environment variables
---------------------
::

    LOCAL_DATASET_PATH – root folder of the Duke MRI image dataset
    LABEL_CSV_PATH     – path to the Duke label CSV
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from IMC.data.constants import DUKE_ORIGINAL_LABEL_NAMES
from IMC.evaluate_duke import run_evaluation
from IMC.network07 import PyramidPooling3DClassifier

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    num_classes_dict: dict,
    backbone_type: str = "resnet",
    backbone_channels: int = 32,
    backbone_blocks: List[int] = (2, 2, 2, 2),
    growth_rate: int = 12,
    embedding_dim: int = 512,
    device: Optional[torch.device] = None,
) -> PyramidPooling3DClassifier:
    """Load a :class:`~IMC.network07.PyramidPooling3DClassifier` from a checkpoint.

    All architecture arguments must match the checkpoint that was saved during
    training.

    Args:
        checkpoint_path: Path to the ``best_model.pth`` file.
        num_classes_dict: Task → number-of-classes mapping (must match checkpoint).
        backbone_type: 3-D CNN backbone name.
        backbone_channels: Initial backbone channels.
        backbone_blocks: ResNet block counts per stage.
        growth_rate: DenseNet growth rate.
        embedding_dim: MLP projection dimension.
        device: Target device.  Defaults to CUDA if available, else CPU.

    Returns:
        :class:`~IMC.network07.PyramidPooling3DClassifier` in eval mode.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = PyramidPooling3DClassifier(
        num_classes_dict=num_classes_dict,
        backbone_type=backbone_type,
        backbone_channels=backbone_channels,
        backbone_blocks=list(backbone_blocks),
        growth_rate=growth_rate,
        embedding_dim=embedding_dim,
    )
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
    ckpt: str,
    output_dir: str,
    split: Optional[List[str]],
    target_depth: int,
    batch_size: int,
    device: torch.device,
    backbone_type: str = "resnet",
    backbone_channels: int = 32,
    backbone_blocks: List[int] = (2, 2, 2, 2),
    growth_rate: int = 12,
    embedding_dim: int = 512,
    num_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Run inference and return a DataFrame of predictions.

    Args:
        ckpt: Path to the model checkpoint.
        output_dir: Directory where ``predictions.csv`` is written.
        split: List of fold-name strings to infer on (e.g. ``["fold_0"]``).
            Pass ``None`` to use all data.
        target_depth: Number of z-slices to resample each volume to.
        batch_size: Batch size for inference.
        device: Target device.
        backbone_type: 3-D CNN backbone name.
        backbone_channels: Initial backbone channels.
        backbone_blocks: ResNet block counts per stage.
        growth_rate: DenseNet growth rate.
        embedding_dim: MLP projection dimension.
        num_samples: Cap the dataset size (for smoke-tests).

    Returns:
        :class:`~pandas.DataFrame` with columns ``Filepath`` + one column per task.
    """
    from IMC.data.duke_dataloader_3d import DukeLiverDataset3D
    dataset = DukeLiverDataset3D(
        split=split,
        target_depth=target_depth,
        augment_conf="NONE3D",
        is_infer=True,
        num_samples=num_samples,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"Inference on {len(dataset)} samples  |  splits={split}")

    cl_d = dataset.get_n_labels()
    label_maps: Dict[str, List[str]] = DUKE_ORIGINAL_LABEL_NAMES
    task_names = list(cl_d.keys())

    model = load_model(
        checkpoint_path=ckpt,
        num_classes_dict=cl_d,
        backbone_type=backbone_type,
        backbone_channels=backbone_channels,
        backbone_blocks=backbone_blocks,
        growth_rate=growth_rate,
        embedding_dim=embedding_dim,
        device=device,
    )

    all_filepaths: List[str] = []
    all_preds: Dict[str, List[str]] = {t: [] for t in task_names}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring"):
            images, filepaths = batch
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
    """Parse command-line arguments for net7 Duke inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch inference for Network v07 (PyramidPooling3DClassifier) "
            "on the Duke Liver Dataset."
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
    parser.add_argument("--target_depth", type=int, default=64, help="Target depth for 3-D volumes.")
    parser.add_argument("--num_samples", type=int, default=None, help="Cap dataset size (smoke-tests).")

    # Architecture (must match checkpoint)
    parser.add_argument("--backbone_type", type=str, default="resnet", help="3-D CNN backbone name.")
    parser.add_argument("--backbone_channels", type=int, default=32, help="Initial backbone channels.")
    parser.add_argument(
        "--backbone_blocks",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
        help="ResNet block counts per stage.",
    )
    parser.add_argument("--growth_rate", type=int, default=12, help="DenseNet growth rate.")
    parser.add_argument("--embedding_dim", type=int, default=512, help="MLP projection dimension.")

    # Hardware
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (-1 for CPU).")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Evaluate predictions against ground truth.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override LOCAL_DATASET_PATH.")
    parser.add_argument("--label_csv_path", type=str, default=None, help="Override LABEL_CSV_PATH.")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run net7 inference on the Duke dataset.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.
    """
    if args.dataset_path:
        os.environ["LOCAL_DATASET_PATH"] = args.dataset_path
    if args.label_csv_path:
        os.environ["LABEL_CSV_PATH"] = args.label_csv_path

    missing = [
        name for name in ("LOCAL_DATASET_PATH", "LABEL_CSV_PATH")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(
            "Missing Duke dataset configuration. Set the environment variables "
            f"{', '.join(missing)} or pass the corresponding CLI overrides."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device : {device}")

    split = (
        [s.strip() for s in args.split.split(",")]
        if args.split is not None
        else None
    )

    pred_df = run_inference(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        split=split,
        target_depth=args.target_depth,
        batch_size=args.batch_size,
        device=device,
        backbone_type=args.backbone_type,
        backbone_channels=args.backbone_channels,
        backbone_blocks=args.backbone_blocks,
        growth_rate=args.growth_rate,
        embedding_dim=args.embedding_dim,
        num_samples=args.num_samples,
    )

    if args.eval:
        label_csv_path = os.environ["LABEL_CSV_PATH"]
        if os.path.exists(label_csv_path):
            print("\nRunning evaluation …")
            label_df = pd.read_csv(label_csv_path)
            run_evaluation(pred_df, label_df, args.output_dir)
        else:
            print(f"\nSkipping evaluation – label CSV not found: {label_csv_path}")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print(f"Results saved to : {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
