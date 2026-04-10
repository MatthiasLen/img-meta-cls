"""
Inference entry point for the two-stage Network v04 experiment (IMC).

This script loads the stage-2 combined model whose image branch was frozen
during training (``FrozenImageCombinedClassifier``).  The architecture is
The fusion module version (``fusion_module_version``) is configurable via
``--fusion_module_version`` or read from ``config.json`` (default ``"v2"``).

Loading architecture arguments from config.json
------------------------------------------------
All architecture arguments can be inferred automatically from the
``config.json`` written by ``net4_two_stage/train.py`` when you pass
``--config /path/to/run_dir/config.json``.  Any argument explicitly provided
on the CLI takes precedence over the config file.

Stage-1 comparison
------------------
Optionally pass ``--stage1_ckpt`` to also run inference with the stage-1
image-only ``ImageBasedClassifier`` checkpoint.  Results are saved alongside
the stage-2 predictions for easy comparison.

Ablation scenarios
------------------
For the stage-2 checkpoint two ablation scenarios are automatically run when
``--run_ablations`` is set:

- **image_zeros**   – replace image tensors with zeros before inference.
- **metadata_nan**  – replace metadata tensors with NaN before inference.

Dataset selection
-----------------
Same convention as ``net4/infer.py``; controlled by ``--dataset`` or the
``IS_DUKE`` / ``IS_PROSTATE_X`` / ``IS_PROSTATE_MRI`` / ``IS_LHIC_MRI``
environment variables.

Output
------
All CSV files are written to ``--output_dir``:

- ``stage2_predictions.csv``            – stage-2 combined model
- ``stage2_predictions_no_image.csv``   – image-zeros ablation (optional)
- ``stage2_predictions_no_metadata.csv``– metadata-NaN ablation (optional)
- ``stage1_predictions.csv``            – stage-1 image-only model (optional)

Typical usage
-------------
# Full inference with auto config::

    python -m IMC.net4_two_stage.infer \\
        --ckpt /path/to/run/stage2/stage2_best_model.pth \\
        --config /path/to/run/config.json \\
        --output_dir /path/to/results \\
        --run_eval

# With explicit architecture args (no config.json)::

    python -m IMC.net4_two_stage.infer \\
        --ckpt /path/to/stage2_best_model.pth \\
        --output_dir /path/to/results \\
        --img_enc_backbone densenet121 \\
        --metadata_enc_type sparse \\
        --sparse_enc_version v1

# Also evaluate stage-1 performance for comparison::

    python -m IMC.net4_two_stage.infer \\
        --ckpt /path/to/stage2/stage2_best_model.pth \\
        --stage1_ckpt /path/to/stage1/stage1_best_model.pth \\
        --config /path/to/run/config.json \\
        --output_dir /path/to/results \\
        --run_eval --run_ablations
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from IMC.data.constants import DUKE_MAP_LABELS
from IMC.evaluate import run_evaluation
from IMC.helper import normalize_per_sample
from IMC.net4_two_stage.train import (
    FrozenImageCombinedClassifier,
    freeze_image_branch,
    load_image_branch_from_checkpoint,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the two-stage inference script.

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Inference for the two-stage IMC Network v04. "
            "Loads the stage-2 combined model (frozen image branch + trained metadata branch)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ I/O
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the stage-2 best-model checkpoint (stage2_best_model.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where prediction CSVs (and evaluation results) are saved.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to config.json from the training run.  "
            "Architecture arguments are read from this file; CLI arguments take precedence."
        ),
    )
    parser.add_argument(
        "--stage1_ckpt",
        type=str,
        default=None,
        help=(
            "Optional path to a stage-1 ImageBasedClassifier checkpoint "
            "(stage1_best_model.pth).  When provided, inference is also run "
            "with the image-only model and results saved to stage1_predictions.csv."
        ),
    )

    # ------------------------------------------------------------------ model
    parser.add_argument(
        "--img_enc_backbone",
        type=str,
        default="densenet121",
        help="CNN backbone identifier (must match training).",
    )
    parser.add_argument(
        "--metadata_enc_type",
        type=str,
        default="sparse",
        choices=["imputer", "sparse"],
        help="Metadata encoder type (must match training).",
    )
    parser.add_argument(
        "--imputer_type",
        type=str,
        default="contextual",
        choices=["contextual", "ignore"],
        help="Imputer strategy (only relevant for --metadata_enc_type imputer; must match training).",
    )
    parser.add_argument(
        "--sparse_enc_version",
        type=str,
        default="v1",
        choices=["v1", "v2", "v5"],
        help="Sparse encoder version (must match training).",
    )
    parser.add_argument(
        "--metadata_embed_dim",
        type=int,
        default=128,
        help="Metadata embedding dimension (must match training).",
    )
    parser.add_argument(
        "--scalar_modulation",
        action="store_true",
        help="Scalar modulation in fusion module (must match training).",
    )
    parser.add_argument(
        "--fusion_module_version",
        type=str,
        default="v2",
        choices=["v1", "v2", "concat"],
        help="Cross-attention fusion variant (must match training; default 'v2').",
    )
    parser.add_argument(
        "--incl_regression",
        action="store_true",
        help="Model was trained with a regression head for ContrastPhase.",
    )

    # ------------------------------------------------------- dataset / runtime
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["pvai", "duke", "prostate_x", "prostate_mri", "lhic"],
        help="Target dataset.  Overrides IS_DUKE / IS_PROSTATE_X / IS_PROSTATE_MRI / IS_LHIC_MRI env vars.",
    )
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Override LOCAL_DATASET_PATH env var.")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Override METADATA_PATH env var.")
    parser.add_argument("--label_csv_path", type=str, default=None,
                        help="Override LABEL_CSV_PATH env var.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference mini-batch size.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index.  Use -1 for CPU.")
    parser.add_argument("--use_preselected_features", action="store_true",
                        help="Use the pre-selected metadata feature subset (must match training).")
    parser.add_argument("--num_slices", type=int, default=3,
                        help="Number of slices per series (must match training).")
    parser.add_argument("--run_eval", action="store_true",
                        help="Compute evaluation metrics after inference.")
    parser.add_argument(
        "--run_ablations",
        action="store_true",
        help=(
            "Run two ablation scenarios for the stage-2 model: "
            "image_zeros (image branch zeroed) and metadata_nan (metadata set to NaN)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config-file argument merging
# ---------------------------------------------------------------------------

def _merge_config(args: argparse.Namespace, config_path: str) -> argparse.Namespace:
    """Overlay ``config.json`` values onto *args*, giving CLI arguments priority.

    Only architecture-relevant keys are merged:
    ``img_enc_backbone``, ``metadata_enc_type``, ``imputer_type``,
    ``sparse_enc_version``, ``metadata_embed_dim``, ``scalar_modulation``,
    ``incl_regression``, ``use_preselected_features``.

    Args:
        args:        Parsed CLI namespace.
        config_path: Path to ``config.json``.

    Returns:
        Updated namespace with config values filling in CLI defaults.
    """
    _ARCH_KEYS = {
        "img_enc_backbone",
        "metadata_enc_type",
        "imputer_type",
        "sparse_enc_version",
        "metadata_embed_dim",
        "scalar_modulation",
        "incl_regression",
        "use_preselected_features",
        "fusion_module_version",
    }

    with open(config_path) as f:
        cfg: dict = json.load(f)

    # Compute which args are still at their parser defaults (were not set by user)
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_enc_backbone", default="densenet121")
    parser.add_argument("--metadata_enc_type", default="sparse")
    parser.add_argument("--imputer_type", default="contextual")
    parser.add_argument("--sparse_enc_version", default="v1")
    parser.add_argument("--metadata_embed_dim", type=int, default=128)
    parser.add_argument("--scalar_modulation", action="store_true")
    parser.add_argument("--incl_regression", action="store_true")
    parser.add_argument("--use_preselected_features", action="store_true")
    parser.add_argument("--fusion_module_version", default="v2")
    defaults = vars(parser.parse_args([]))

    for key in _ARCH_KEYS:
        if key not in cfg:
            continue
        # Only override if the CLI value is still the default (i.e. user did not provide it)
        if getattr(args, key, None) == defaults.get(key):
            setattr(args, key, cfg[key])

    return args


# ---------------------------------------------------------------------------
# Dataset environment setup  (mirrors net4/infer.py)
# ---------------------------------------------------------------------------

_DATASET_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "pvai": (
        "/home/tuan.truong/data/PV.AI",
        "/home/tuan.truong/codebase/IMC/labels/encoded_metadata_20251217.parquet",
        "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20250603_local.csv",
    ),
    "duke": (
        "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2",
        "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet",
        "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv",
    ),
    "prostate_x": (
        "/home/tuan.truong/data/PROSTATEx-v1-doiJNLP",
        "/home/tuan.truong/codebase/IMC/labels/prostate_x_metadata/prostate_x_encoded_metadata_20260223.parquet",
        "/home/tuan.truong/codebase/IMC/labels/prostate_x_dataset_pvai_labels.csv",
    ),
    "prostate_mri": (
        "/home/tuan.truong/data/PROSTATE-MRI-5-18-2018-doiJNLP-dKJJAqnS",
        "/home/tuan.truong/codebase/IMC/labels/prostate_mri_metadata/prostate_mri_encoded_metadata_20260223.parquet",
        "/home/tuan.truong/codebase/IMC/labels/prostate_mri_dataset_pvai_labels.csv",
    ),
    "lhic": (
        "/home/tuan.truong/data/doiJNLP-TCGA-LIHC-01-30-2017",
        "/home/tuan.truong/codebase/IMC/labels/lhic_mri_metadata/lhic_mri_encoded_metadata_20260224.parquet",
        "/home/tuan.truong/codebase/IMC/labels/labels_LHIC_MRI_20260224.csv",
    ),
}


def _configure_dataset_env(args: argparse.Namespace) -> str:
    """Configure dataset environment variables and return the dataset name."""
    os.environ["DEBUG_MODE"] = "0"

    if args.dataset is not None:
        dataset = args.dataset
    elif os.environ.get("IS_DUKE", "0") == "1":
        dataset = "duke"
    elif os.environ.get("IS_PROSTATE_X", "0") == "1":
        dataset = "prostate_x"
    elif os.environ.get("IS_PROSTATE_MRI", "0") == "1":
        dataset = "prostate_mri"
    elif os.environ.get("IS_LHIC_MRI", "0") == "1":
        dataset = "lhic"
    else:
        dataset = "pvai"

    dp, mp, lp = _DATASET_DEFAULTS.get(dataset, _DATASET_DEFAULTS["pvai"])
    os.environ.setdefault("LOCAL_DATASET_PATH", dp)
    os.environ.setdefault("METADATA_PATH", mp)
    os.environ.setdefault("LABEL_CSV_PATH", lp)

    if args.dataset_path:
        os.environ["LOCAL_DATASET_PATH"] = args.dataset_path
    if args.metadata_path:
        os.environ["METADATA_PATH"] = args.metadata_path
    if args.label_csv_path:
        os.environ["LABEL_CSV_PATH"] = args.label_csv_path

    return dataset


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _build_device(args: argparse.Namespace) -> torch.device:
    if args.gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def load_stage2_model(
    args: argparse.Namespace,
    num_classes_dict: dict,
    metadata_input_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    """Build ``FrozenImageCombinedClassifier`` and load the stage-2 checkpoint.

    ``fusion_module_version`` is taken from ``args.fusion_module_version``
    (default ``"v2"``).  Must match the value used during stage-2 training.

    Args:
        args:               Parsed arguments (post config-merge).
        num_classes_dict:   Task -> num-classes mapping from dataloader.
        metadata_input_dim: Number of metadata features from dataloader.
        device:             Inference device.

    Returns:
        Model in ``eval()`` mode.
    """
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {args.ckpt}")

    model = FrozenImageCombinedClassifier(
        num_classes_dict=num_classes_dict,
        metadata_input_dim=metadata_input_dim,
        img_enc_backbone=args.img_enc_backbone,
        incl_regression=args.incl_regression,
        metadata_encoder_type=args.metadata_enc_type,
        imputer_type=args.imputer_type,
        sparse_enc_version=args.sparse_enc_version,
        fusion_module_version=args.fusion_module_version,
        metadata_embed_dim=args.metadata_embed_dim,
        scalar_modulation=args.scalar_modulation,
    )

    print(f"Loading stage-2 checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_stage1_model(
    stage1_ckpt: str,
    args: argparse.Namespace,
    num_classes_dict: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Build ``ImageBasedClassifier`` and load the stage-1 checkpoint.

    Args:
        stage1_ckpt:      Path to ``stage1_best_model.pth``.
        args:             Parsed arguments (used for backbone and regression flag).
        num_classes_dict: Task -> num-classes mapping from dataloader.
        device:           Inference device.

    Returns:
        Model in ``eval()`` mode.
    """
    from IMC.net4.helper import build_model

    if not os.path.exists(stage1_ckpt):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt}")

    model = build_model(
        modality="image",
        vanilla_image_classifier=False,
        img_enc_backbone=args.img_enc_backbone,
        incl_regression=args.incl_regression,
        num_classes_dict=num_classes_dict,
    )

    print(f"Loading stage-1 checkpoint from {stage1_ckpt}")
    ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes_dict: dict,
    device: torch.device,
    is_duke: bool = False,
    include_regression: bool = False,
    ablation: Optional[str] = None,
    desc: str = "Inferring",
) -> pd.DataFrame:
    """Run batch inference and return a DataFrame of per-series predictions.

    Args:
        model:             Model in ``eval()`` mode.
        dataloader:        Returns ``(images, metadata, paths)`` batches.
        num_classes_dict:  Task-name -> number-of-classes dict.
        device:            Inference device.
        is_duke:           Apply Duke-specific label remapping.
        include_regression: ContrastPhase head uses regression output.
        ablation:          ``"image_zeros"`` – zero the image tensor;
                           ``"metadata_nan"`` – fill metadata with NaN;
                           ``None`` (default) – no ablation.
        desc:              Progress-bar description string.

    Returns:
        :class:`~pandas.DataFrame` with ``Filepath`` + one column per task.
    """
    label_maps: dict = dataloader.dataset.label_names

    predictions: dict[str, list] = {task: [] for task in num_classes_dict}
    predictions["label_Contrast"] = []
    filepaths: list[str] = []

    _desc = {
        "image_zeros":  f"{desc} [image→zeros]",
        "metadata_nan": f"{desc} [metadata→NaN]",
    }.get(ablation, desc)  # type: ignore[arg-type]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=_desc):
            images, metadata, paths = batch
            images = images.to(device)
            metadata = metadata.to(device)
            images = normalize_per_sample(images)

            if ablation == "image_zeros":
                images = torch.zeros_like(images)
            elif ablation == "metadata_nan":
                metadata = torch.full_like(metadata, float("nan"))

            outputs = model(images, metadata)

            for i, task in enumerate(num_classes_dict):
                if include_regression and task == "label_ContrastPhase":
                    preds = outputs[i].cpu().numpy()
                    preds = np.clip(np.rint(preds).astype(int), 0, 4).flatten()
                else:
                    preds = torch.argmax(outputs[i], dim=1).cpu().numpy()

                pred_names = [label_maps[task][p] for p in preds]
                if is_duke and task in DUKE_MAP_LABELS:
                    pred_names = [DUKE_MAP_LABELS[task].get(n, n) for n in pred_names]
                predictions[task].extend(pred_names)

            filepaths.extend(paths)

    predictions["label_Contrast"] = [
        "post" if phase != "pre" else "pre"
        for phase in predictions["label_ContrastPhase"]
    ]

    df = pd.DataFrame({"Filepath": filepaths})
    for task in predictions:
        df[task] = predictions[task]
    return df


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _maybe_evaluate(
    pred_df: pd.DataFrame,
    output_dir: str,
    is_duke: bool,
    run_eval: bool,
) -> None:
    """Run evaluation if ``run_eval`` is True and the ground-truth CSV exists."""
    if not run_eval:
        return
    label_csv = os.environ.get("LABEL_CSV_PATH", "")
    if not os.path.isfile(label_csv):
        print(
            f"WARNING: Ground-truth CSV not found at '{label_csv}'. "
            "Skipping evaluation."
        )
        return
    label_df = pd.read_csv(label_csv)
    run_evaluation(pred_df, label_df, output_dir, is_duke=is_duke)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate two-stage inference.

    Steps
    -----
    1. Parse CLI args; optionally merge architecture settings from config.json.
    2. Configure dataset environment variables.
    3. Build inference dataloader.
    4. Load stage-2 model and run inference → ``stage2_predictions.csv``.
    5. Optionally run ablation scenarios (``--run_ablations``).
    6. Optionally run evaluation (``--run_eval``).
    7. Optionally load stage-1 model and run inference for comparison
       (``--stage1_ckpt``).
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Merge config.json (CLI args take priority)
    if args.config is not None:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        args = _merge_config(args, args.config)
        print(f"Architecture args merged from: {args.config}")

    dataset = _configure_dataset_env(args)
    is_duke = dataset == "duke"
    is_pvai = dataset == "pvai"

    device = _build_device(args)

    print(f"Dataset  : {dataset}")
    print(f"Data path: {os.environ['LOCAL_DATASET_PATH']}")
    print(f"Metadata : {os.environ['METADATA_PATH']}")
    print(f"Labels   : {os.environ['LABEL_CSV_PATH']}")
    print(f"Device   : {device}")

    # ---- dataloader --------------------------------------------------------
    from IMC.data.liver_dataloader_local import get_infer_dataloader

    folder_split = ["fold_9"] if is_pvai else None
    dataloader = get_infer_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=None,
        folder_split=folder_split,
        aggregated_metadata=False,
        use_preselected_features=args.use_preselected_features,
        exclude_contrast_yn=True,
        n_slices=args.num_slices,
    )

    num_classes_dict = dataloader.dataset.get_n_labels()
    metadata_input_dim = dataloader.dataset.num_metadata_features
    print(f"Label config      : {num_classes_dict}")
    print(f"Metadata input dim: {metadata_input_dim}")

    # Shared inference kwargs (model will be swapped per stage)
    _infer_kwargs = dict(
        dataloader=dataloader,
        num_classes_dict=num_classes_dict,
        device=device,
        is_duke=is_duke,
        include_regression=args.incl_regression,
    )

    # ======================================================================
    # Stage-2: combined model (frozen image branch + trained metadata branch)
    # ======================================================================
    stage2_model = load_stage2_model(args, num_classes_dict, metadata_input_dim, device)

    print("\n--- Stage-2: combined model ---")
    s2_df = run_inference(model=stage2_model, desc="Stage-2", **_infer_kwargs)

    s2_pred_path = os.path.join(args.output_dir, "stage2_predictions.csv")
    s2_df.to_csv(s2_pred_path, index=False)
    print(f"Stage-2 predictions saved to: {s2_pred_path}")
    _maybe_evaluate(s2_df, args.output_dir, is_duke, args.run_eval)

    # ---- ablation scenarios -----------------------------------------------
    if args.run_ablations:
        _ablation_scenarios = [
            ("image_zeros",   "stage2_predictions_no_image.csv",    "stage2_no_image_eval"),
            ("metadata_nan",  "stage2_predictions_no_metadata.csv", "stage2_no_metadata_eval"),
        ]
        for ablation_tag, csv_name, eval_subdir in _ablation_scenarios:
            print(f"\n--- Stage-2 ablation: {ablation_tag} ---")
            abl_df = run_inference(
                model=stage2_model,
                desc="Stage-2 ablation",
                ablation=ablation_tag,
                **_infer_kwargs,
            )
            abl_path = os.path.join(args.output_dir, csv_name)
            abl_df.to_csv(abl_path, index=False)
            print(f"Ablation predictions saved to: {abl_path}")

            if args.run_eval:
                abl_eval_dir = os.path.join(args.output_dir, eval_subdir)
                os.makedirs(abl_eval_dir, exist_ok=True)
                _maybe_evaluate(abl_df, abl_eval_dir, is_duke, args.run_eval)

    # ======================================================================
    # Stage-1: image-only model (optional, for comparison)
    # ======================================================================
    if args.stage1_ckpt is not None:
        print("\n--- Stage-1: image-only model (comparison) ---")
        stage1_model = load_stage1_model(
            args.stage1_ckpt, args, num_classes_dict, device
        )
        s1_df = run_inference(model=stage1_model, desc="Stage-1", **_infer_kwargs)

        s1_pred_path = os.path.join(args.output_dir, "stage1_predictions.csv")
        s1_df.to_csv(s1_pred_path, index=False)
        print(f"Stage-1 predictions saved to: {s1_pred_path}")

        if args.run_eval:
            s1_eval_dir = os.path.join(args.output_dir, "stage1_eval")
            os.makedirs(s1_eval_dir, exist_ok=True)
            _maybe_evaluate(s1_df, s1_eval_dir, is_duke, args.run_eval)

    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
