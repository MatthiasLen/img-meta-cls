"""Inference entry point for Network v04 on the Duke Liver Dataset.

This script runs batch inference using a trained Network-v04 checkpoint on the
Duke-specific :class:`~IMC.data.duke_dataloader_local.LiverDataset`.  The
correct model variant is selected via ``--modality``, which must match the
value used during training.

Supported modalities
--------------------
- ``combined``  : :class:`~IMC.network04.MRISequenceClassifier` (image +
  metadata fusion via cross-attention).
- ``image``     : :class:`~IMC.network04.ImageBasedClassifier` (or
  :class:`~IMC.network04.SimpleImageBasedClassifier` with
  ``--vanilla_image_classifier``).
- ``metadata``  : :class:`~IMC.network04.MetadataBasedClassifier`.

Output
------
``predictions.csv`` is written to ``--output_dir``.  Each row corresponds to
one DICOM series and contains the filepath followed by one column per
classification task.

If ``--run_eval`` is set, evaluation metrics are computed and written to the
same directory using :func:`~IMC.evaluate_duke.run_evaluation`.

Typical usage
-------------
Run inference on all folds of the Duke dataset::

    python -m IMC.net4_duke.infer \\
        --ckpt /path/to/best_model.pth \\
        --output_dir /path/to/results \\
        --modality combined

Run inference on fold 4 only, then evaluate::

    python -m IMC.net4_duke.infer \\
        --ckpt /path/to/best_model.pth \\
        --output_dir /path/to/results \\
        --modality combined \\
        --folds 4 \\
        --run_eval

Image-only inference::

    python -m IMC.net4_duke.infer \\
        --ckpt /path/to/best_model.pth \\
        --output_dir /path/to/results \\
        --modality image \\
        --img_enc_backbone densenet121

Environment variables
---------------------
The following environment variables configure Duke dataset paths.  They can
also be overridden via the corresponding CLI arguments, which always take
precedence::

    LOCAL_DATASET_PATH – root folder of the Duke MRI image dataset
    METADATA_PATH      – path to the Duke encoded metadata parquet file
    LABEL_CSV_PATH     – path to the Duke label CSV
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from IMC.helper import normalize_per_sample
from IMC.evaluate_duke import run_evaluation


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Duke inference script.

    All model-architecture arguments (``--modality``, ``--img_enc_backbone``,
    ``--fusion_module_version``, ``--sparse_enc_version``,
    ``--metadata_enc_type``, ``--metadata_embed_dim``, ``--output_emb_dim``)
    must exactly match the values used when training the checkpoint being
    loaded.

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Inference script for IMC Network v04 on the Duke Liver Dataset.  "
            "Run predictions with a trained checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ I/O
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where predictions.csv (and evaluation results) are saved.",
    )

    # ------------------------------------------------------------------ mode
    parser.add_argument(
        "--modality",
        type=str,
        choices=["image", "metadata", "combined"],
        required=True,
        help=(
            "Model variant (must match training): "
            "'combined' = image + metadata fusion, "
            "'image' = image only, "
            "'metadata' = metadata only."
        ),
    )

    # ------------------------------------------------------- model structure
    parser.add_argument(
        "--img_enc_backbone",
        type=str,
        default="densenet121",
        help="CNN backbone (must match training).",
    )
    parser.add_argument(
        "--vanilla_image_classifier",
        action="store_true",
        help=(
            "Use SimpleImageBasedClassifier instead of ImageBasedClassifier.  "
            "Active only when --modality image; must match training."
        ),
    )
    parser.add_argument(
        "--fusion_module_version",
        type=str,
        default="v1",
        choices=["v1", "v2", "concat"],
        help="Fusion module version (combined mode; must match training).",
    )
    parser.add_argument(
        "--metadata_enc_type",
        type=str,
        default="imputer",
        choices=["imputer", "sparse"],
        help=(
            "Metadata encoder type (must match training).  "
            "'imputer' = contextual imputer, 'sparse' = sparse attention encoder."
        ),
    )
    parser.add_argument(
        "--imputer_type",
        type=str,
        default="contextual",
        choices=["contextual", "ignore"],
        help="Imputer strategy (imputer mode only; must match training).",
    )
    parser.add_argument(
        "--sparse_enc_version",
        type=str,
        default="v1",
        choices=["v1", "v2", "v5"],
        help="Sparse metadata encoder version (sparse mode only; must match training).",
    )
    parser.add_argument(
        "--metadata_embed_dim",
        type=int,
        default=128,
        help="Metadata embedding dimension (must match training).",
    )
    parser.add_argument(
        "--output_emb_dim",
        type=int,
        default=256,
        help="Output projection dimension (must match training).",
    )
    parser.add_argument(
        "--incl_regression",
        action="store_true",
        help="Model was trained with a regression head for the ContrastPhase task.",
    )

    # ------------------------------------------------------- dataset / runtime
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help=(
            "Comma-separated fold indices to infer on "
            "(e.g. '0,1,2').  Defaults to all data (no fold filter)."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override LOCAL_DATASET_PATH environment variable.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Override METADATA_PATH environment variable.",
    )
    parser.add_argument(
        "--label_csv_path",
        type=str,
        default=None,
        help="Override LABEL_CSV_PATH environment variable.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference mini-batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader worker processes."
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="CUDA device index.  Use -1 for CPU."
    )
    parser.add_argument(
        "--n_slices", type=int, default=3,
        help="Number of slices to sample from each MRI volume.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help=(
            "Run evaluation after inference.  Requires a ground-truth label "
            "CSV at LABEL_CSV_PATH / --label_csv_path."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset environment setup
# ---------------------------------------------------------------------------

def _configure_dataset_env(args: argparse.Namespace) -> None:
    """Apply Duke-specific dataset environment variables.

    Applies CLI path overrides for ``LOCAL_DATASET_PATH``, ``METADATA_PATH``,
    and ``LABEL_CSV_PATH`` when provided and validates that all required Duke
    dataset variables are configured before inference starts.

    Args:
        args: Parsed arguments from :func:`parse_args`.
    """
    os.environ["DEBUG_MODE"] = "0"

    # CLI overrides always win
    if args.dataset_path:
        os.environ["LOCAL_DATASET_PATH"] = args.dataset_path
    if args.metadata_path:
        os.environ["METADATA_PATH"] = args.metadata_path
    if args.label_csv_path:
        os.environ["LABEL_CSV_PATH"] = args.label_csv_path

    missing = [
        name for name in ("LOCAL_DATASET_PATH", "METADATA_PATH", "LABEL_CSV_PATH")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(
            "Missing Duke dataset configuration. Set the environment variables "
            f"{', '.join(missing)} or pass the corresponding CLI overrides."
        )


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def create_inference_dataloader(
    fold_indices: Optional[List[int]],
    batch_size: int = 16,
    num_workers: int = 4,
    use_preselected_features: bool = False,
    exclude_contrast_yn: bool = True,
    n_slices: int = 3,
) -> DataLoader:
    """Create a Duke-specific :class:`~torch.utils.data.DataLoader` for inference.

    Constructs a :class:`~IMC.data.duke_dataloader_local.LiverDataset` with
    ``is_infer=True`` (no random augmentation) and wraps it in a
    ``DataLoader`` with deterministic ordering (``shuffle=False``).

    Args:
        fold_indices:             List of integer fold indices to include (e.g.
                                  ``[0, 1, 2]``).  Pass ``None`` to use the
                                  entire dataset without fold filtering.
        batch_size:               Mini-batch size.
        num_workers:              Number of DataLoader worker processes.
        use_preselected_features: Restrict metadata to the pre-selected feature
                                  subset (must match training).
        exclude_contrast_yn:      Exclude the binary ``label_Contrast`` task
                                  from the dataset label configuration.
        n_slices:                 Number of slices to sample from each MRI volume.

    Returns:
        :class:`~torch.utils.data.DataLoader` ready for inference.
    """
    from IMC.data.duke_dataloader_local import LiverDataset
    from IMC.data.constants import DUKE_ORIGINAL_LABEL_NAMES

    split = [f"fold_{i}" for i in fold_indices] if fold_indices is not None else None

    dataset = LiverDataset(
        split=split,
        num_samples=None,
        augment_conf="NONE2D",
        aggregated_metadata=False,
        use_preselected_features=use_preselected_features,
        exclude_contrast_yn=exclude_contrast_yn,
        is_infer=True,
        label_names=DUKE_ORIGINAL_LABEL_NAMES,
        n_slices=n_slices,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(
    args: argparse.Namespace,
    num_classes_dict: dict,
    metadata_input_dim: int,
    device: torch.device,
) -> nn.Module:
    """Load and return a Network-v04 model from a checkpoint.

    The model architecture is constructed via
    :func:`~IMC.net4.helper.build_model` to exactly mirror the variant used
    during training.  The following ``args`` attributes control architecture
    selection and **must match** the training invocation:

    - ``args.modality``
    - ``args.img_enc_backbone``
    - ``args.vanilla_image_classifier``
    - ``args.metadata_enc_type``
    - ``args.imputer_type``
    - ``args.sparse_enc_version``
    - ``args.fusion_module_version``
    - ``args.metadata_embed_dim``
    - ``args.output_emb_dim``
    - ``args.incl_regression``

    Args:
        args:               Parsed arguments from :func:`parse_args`.
        num_classes_dict:   Task-name → number-of-classes mapping from the
                            inference dataloader.
        metadata_input_dim: Number of metadata features provided by the
                            inference dataloader.
        device:             Device on which to place the model.

    Returns:
        Model in ``eval()`` mode, ready for inference.

    Raises:
        FileNotFoundError: If the checkpoint at ``args.ckpt`` does not exist.
    """
    from IMC.net4_duke.helper import build_model

    model = build_model(
        modality=args.modality,
        vanilla_image_classifier=args.vanilla_image_classifier,
        img_enc_backbone=args.img_enc_backbone,
        metadata_enc_type=args.metadata_enc_type,
        imputer_type=args.imputer_type,
        sparse_enc_version=args.sparse_enc_version,
        output_emb_dim=args.output_emb_dim,
        incl_regression=args.incl_regression,
        num_classes_dict=num_classes_dict,
        metadata_input_dim=metadata_input_dim,
        metadata_embed_dim=args.metadata_embed_dim,
        fusion_module_version=args.fusion_module_version,
    )

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    print(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    return model


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    incl_regression: bool = False,
) -> pd.DataFrame:
    """Run batch inference and return a tidy predictions DataFrame.

    For each sample the model produces one prediction per task.  Predictions
    are converted from class indices to human-readable label names using the
    ``label_names`` attribute of ``dataloader.dataset``.

    When ``incl_regression=True``, the ``label_ContrastPhase`` head is treated
    as a regression output: raw scalar values are rounded to the nearest
    integer and clipped to the valid label-index range before mapping to names.

    A derived binary column ``label_Contrast`` is appended:
    ``"post"`` for any phase other than ``"pre"`` or ``"na"``, ``"pre"``
    otherwise.

    Args:
        model:            Trained :class:`~torch.nn.Module` in ``eval()`` mode.
        dataloader:       Inference :class:`~torch.utils.data.DataLoader`.
                          Its ``dataset`` must expose ``get_n_labels()``
                          (``dict[str, int]``) and a ``label_names`` attribute
                          (``dict[str, dict[int, str]]``).
        device:           Device on which to run the forward pass.
        incl_regression:  Treat the ``label_ContrastPhase`` output as a
                          continuous regression value instead of class logits.

    Returns:
        :class:`~pandas.DataFrame` with a ``Filepath`` column followed by one
        column per task (including the derived ``label_Contrast`` column).
    """
    num_classes_dict = dataloader.dataset.get_n_labels()
    label_maps: dict = dataloader.dataset.label_names

    predictions: dict[str, list] = {task: [] for task in num_classes_dict}
    filepaths: list[str] = []

    print(f"Running inference on {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            images, metadata, paths = batch
            images = images.to(device)
            metadata = metadata.to(device)
            images = normalize_per_sample(images)

            outputs = model(images, metadata)

            for i, task in enumerate(num_classes_dict.keys()):
                if incl_regression and task == "label_ContrastPhase":
                    # Regression branch: round scalar → clip to valid range
                    preds = np.rint(outputs[i].cpu().numpy()).astype(int)
                    preds = np.clip(preds, 0, len(label_maps[task]) - 1).flatten()
                else:
                    preds = torch.argmax(outputs[i], dim=1).cpu().numpy()

                predictions[task].extend([label_maps[task][p] for p in preds])

            filepaths.extend(paths)

    # Derive binary contrast label from the phase prediction
    if "label_ContrastPhase" in label_maps:
        predictions["label_Contrast"] = [
            "post" if phase not in ("pre", "na") else "pre"
            for phase in predictions["label_ContrastPhase"]
        ]

    output_df = pd.DataFrame({"Filepath": filepaths})
    for task in predictions:
        output_df[task] = predictions[task]

    return output_df


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Run inference on the Duke Liver Dataset and optionally evaluate.

    Steps
    -----
    1.  Parse CLI arguments via :func:`parse_args`.
    2.  Apply Duke-specific environment variables via
        :func:`_configure_dataset_env`.
    3.  Create the inference :class:`~torch.utils.data.DataLoader` via
        :func:`create_inference_dataloader`, filtered to ``--folds`` if
        provided or covering all data by default.
    4.  Load the trained model from ``--ckpt`` via :func:`load_model`.
    5.  Run :func:`run_inference` to produce a predictions
        :class:`~pandas.DataFrame`.
    6.  Save ``predictions.csv`` and ``inference_metadata.json`` to
        ``--output_dir``.
    7.  If ``--run_eval`` is set, call
        :func:`~IMC.evaluate_duke.run_evaluation` with the ground-truth label
        CSV and write evaluation metrics to ``--output_dir``.
    """
    args = parse_args()
    _configure_dataset_env(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- fold selection ----------------------------------------------------
    fold_indices: Optional[List[int]] = None
    if args.folds is not None:
        fold_indices = [int(f) for f in args.folds.split(",")]
        print(f"Running inference on folds: {fold_indices}")
    else:
        print("Running inference on all data")

    # ---- device ------------------------------------------------------------
    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # ---- dataloader --------------------------------------------------------
    print("\nCreating dataloader...")
    dataloader = create_inference_dataloader(
        fold_indices=fold_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_preselected_features=False,
        exclude_contrast_yn=True,
        n_slices=args.n_slices,
    )
    print(f"✓ Loaded {len(dataloader.dataset)} samples")

    num_classes_dict = dataloader.dataset.get_n_labels()
    metadata_input_dim = dataloader.dataset.num_metadata_features
    print("\nModel configuration:")
    print(f"  Backbone        : {args.img_enc_backbone}")
    print(f"  Metadata dim    : {metadata_input_dim}")
    print(f"  Tasks           : {list(num_classes_dict.keys())}")
    print(f"  Incl regression : {args.incl_regression}")

    # ---- model -------------------------------------------------------------
    model = load_model(
        args=args,
        num_classes_dict=num_classes_dict,
        metadata_input_dim=metadata_input_dim,
        device=device,
    )

    # ---- inference ---------------------------------------------------------
    pred_df = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        incl_regression=args.incl_regression,
    )

    # ---- save predictions --------------------------------------------------
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n✓ Predictions saved to {pred_path}")

    meta_json = {
        "checkpoint": args.ckpt,
        "num_samples": len(pred_df),
        "tasks": list(pred_df.columns[1:]),
        "folds": fold_indices,
    }
    with open(os.path.join(args.output_dir, "inference_metadata.json"), "w") as f:
        json.dump(meta_json, f, indent=2)

    # ---- optional evaluation -----------------------------------------------
    if args.run_eval:
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)
        label_csv_path = os.environ.get("LABEL_CSV_PATH")
        label_df = pd.read_csv(label_csv_path)
        run_evaluation(pred_df, label_df, args.output_dir)
        print(f"\n✓ Evaluation results saved to {args.output_dir}")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"  Results saved to        : {args.output_dir}")
    print(f"  Total samples processed : {len(pred_df)}")


if __name__ == "__main__":
    main()
