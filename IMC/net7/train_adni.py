"""Training entry point for Network v07 (PyramidPooling3DClassifier) on ADNI.

This script trains :class:`~IMC.network07.PyramidPooling3DClassifier` on the
**ADNI Brain MRI** dataset for **one fold per invocation**.  Run five processes
in parallel (``--fold 0 … 4``) to complete a full 5-fold cross-validation.

The fold split strategy mirrors the IMC standard:

- **Test fold**  : ``fold_idx``  (``--fold``)
- **Val fold**   : ``(fold_idx + 1) % 5``
- **Train folds**: all remaining folds

Model
-----
:class:`~IMC.network07.PyramidPooling3DClassifier` – 3-D CNN backbone
(ResNet-3D or DenseNet-3D) + 3-D Pyramid Pooling + optional MLP projection +
:class:`~IMC.nn.multi_task_head.MultiTaskHead`.
Three ADNI tasks are trained simultaneously:
``label_AcquisitionPlane``, ``label_SequenceContrast``, ``label_Localizer``.

Training configuration
----------------------
- AdamW optimiser with per-group weight decay.
- Linear-warmup + cosine-decay LR schedule.
- :class:`~IMC.nn.multi_task_loss.MultiTaskLoss` (label smoothing = 0.1).
- Mixed precision via :class:`torch.amp.GradScaler`.
- Early stopping controlled by ``--patience``.
- Per-fold ``config.json`` and ``model_architecture.txt`` written to the fold
  log directory.
- Per-fold ``fold_results.json`` written after the test pass.

Typical usage
-------------
Run all folds in parallel::

    for i in 0 1 2 3 4; do
        python -m IMC.net7.train_adni \\
            --fold $i --gpu 0 \\
            --base_log_dir ./logs/net07_adni_cv &
    done

Or sequentially::

    python -m IMC.net7.train_adni --fold 0 --gpu 0

Environment variables
---------------------
The following environment variables configure ADNI dataset paths.  They can
also be overridden via CLI arguments which take precedence::

    ADNI_LOCAL_DATASET_PATH – root folder of the ADNI brain MRI dataset
    ADNI_LABEL_CSV_PATH     – path to the ADNI label CSV
    DEBUG_MODE              – set to "1" to enable extra debug output
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from IMC.data.constants import ADNI_LABEL_NAMES
from IMC.helper import (
    capture_console_to_log,
    log_training_end,
    log_training_start,
)
from IMC.network07 import PyramidPooling3DClassifier
from IMC.nn.multi_task_loss import MultiTaskLoss
from IMC.tensorboard_logging import setup_combined_logging
from IMC.trainer import Trainer

# ---------------------------------------------------------------------------
# Default dataset paths
# ---------------------------------------------------------------------------
_ADNI_DATASET_PATH = "/home/tuan.truong/data/ADNI_full"
_ADNI_LABEL_CSV_PATH = (
    "/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv"
)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def init_weights(module: nn.Module) -> None:
    """Initialise weights for selected module types.

    Applies the following rules:

    - ``nn.Linear``    – Xavier (Glorot) uniform initialisation for weights;
      zeros for bias.
    - ``nn.LayerNorm`` – ones for weight; zeros for bias.

    Pass to ``model.apply(init_weights)`` after model construction.

    Args:
        module: A single layer/sub-module returned by ``model.apply()``.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


def create_optimizer(
    model: nn.Module,
    lr: float = 1.0e-6,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
) -> Optimizer:
    """Build an AdamW optimiser with per-parameter-group weight decay.

    Biases and normalisation layer parameters are placed in a separate group
    with ``weight_decay=0.0``.

    Args:
        model: Model whose parameters will be optimised.
        lr: Base learning rate.
        weight_decay: Weight decay for non-normalisation parameters.
        eps: Epsilon for numerical stability in AdamW.

    Returns:
        Configured :class:`~torch.optim.AdamW` optimiser.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ["bias", "norm", "ln", "layernorm", "bn"]):
            no_decay.append(param)
        else:
            decay.append(param)

    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        eps=eps,
    )


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------


def get_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_scale_factor: float = 100.0,
    min_scale_factor: float = 0.1,
) -> LambdaLR:
    """Construct a linear-warmup + cosine-decay LR scheduler.

    The multiplier applied to the base LR is:

    1. **Linear warmup** – ramps from ``1.0`` to ``peak_scale_factor`` over
       ``warmup_steps`` gradient steps.
    2. **Cosine decay** – decays from ``peak_scale_factor`` to
       ``min_scale_factor`` between step ``warmup_steps`` and ``total_steps``.
    3. **Floor** – remains at ``min_scale_factor`` beyond ``total_steps``.

    Args:
        optimizer: Optimiser whose LR will be scaled.
        warmup_steps: Number of warm-up gradient steps.
        total_steps: Total steps at which cosine decay completes.
        peak_scale_factor: Maximum LR multiplier at the end of warm-up.
        min_scale_factor: Minimum LR multiplier after cosine decay.

    Returns:
        :class:`~torch.optim.lr_scheduler.LambdaLR` scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step <= warmup_steps and warmup_steps > 0:
            return max(1.0, peak_scale_factor * float(current_step) / warmup_steps)
        elif warmup_steps < current_step <= total_steps:
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(
                min_scale_factor,
                peak_scale_factor * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )
        else:
            return min_scale_factor

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Backbone validation
# ---------------------------------------------------------------------------

_VALID_BACKBONES: List[str] = [
    "resnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet_custom",
]


def _validate_backbone(value: str) -> str:
    """Validate the backbone_type argument.

    Args:
        value: The backbone type string provided by the user.

    Returns:
        The validated backbone type string.

    Raises:
        argparse.ArgumentTypeError: If *value* is not in the allowed list.
    """
    if value not in _VALID_BACKBONES:
        raise argparse.ArgumentTypeError(
            f"Invalid backbone_type '{value}'. Valid options: {_VALID_BACKBONES}"
        )
    return value


# ---------------------------------------------------------------------------
# Dataset environment configuration
# ---------------------------------------------------------------------------


def _configure_dataset_env(args: argparse.Namespace) -> None:
    """Set ADNI dataset environment variables from CLI args (if provided)."""
    os.environ["DEBUG_MODE"] = "1" if args.debug else "0"
    os.environ.setdefault(
        "ADNI_LOCAL_DATASET_PATH",
        args.dataset_path or _ADNI_DATASET_PATH,
    )
    if args.dataset_path:
        os.environ["ADNI_LOCAL_DATASET_PATH"] = args.dataset_path
    os.environ.setdefault(
        "ADNI_LABEL_CSV_PATH",
        args.label_csv_path or _ADNI_LABEL_CSV_PATH,
    )
    if args.label_csv_path:
        os.environ["ADNI_LABEL_CSV_PATH"] = args.label_csv_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for net7 ADNI single-fold training.

    Returns:
        Populated :class:`argparse.Namespace` with all resolved arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Single-fold training of Network v07 (PyramidPooling3DClassifier) "
            "on the ADNI Brain Dataset.  "
            "Run once per fold (--fold 0 … 4) to complete 5-fold CV.\n"
            "Pass the same --base_log_dir to all fold invocations so outputs "
            "land in a single shared directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Fold selection ----
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Test fold index (0–4).  Val fold = (fold + 1) %% n_folds.",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Total number of folds.")

    # ---- Dataset ----
    parser.add_argument(
        "--n_slices",
        type=int,
        default=16,
        help="Number of slices to sample per series (acts as depth for network07).",
    )
    parser.add_argument("--img_size", type=int, default=224, help="Spatial resolution (H = W = img_size).")
    parser.add_argument(
        "--augment_config",
        type=str,
        default="DEFAULT3D",
        help="3-D augmentation configuration string for training splits.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap dataset size (useful for smoke-tests).",
    )

    # ---- Backbone ----
    parser.add_argument(
        "--backbone_type",
        type=_validate_backbone,
        default="resnet",
        metavar="BACKBONE",
        help=f"3-D CNN backbone: {_VALID_BACKBONES}.",
    )
    parser.add_argument("--backbone_channels", type=int, default=32, help="Initial backbone channels.")
    parser.add_argument(
        "--backbone_blocks",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
        help="ResNet block counts per stage.",
    )
    parser.add_argument("--growth_rate", type=int, default=12, help="DenseNet growth rate k.")
    parser.add_argument("--embedding_dim", type=int, default=512, help="MLP projection dimension.")

    # ---- Training hyper-parameters ----
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Maximum epochs per fold.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Base learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs).")

    # ---- Hardware / misc ----
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (-1 for CPU).")
    parser.add_argument(
        "--base_log_dir",
        type=str,
        default=None,
        help=(
            "Shared root log directory for all folds.  "
            "Defaults to ./logs/<timestamp>_net07_adni_5fold_cv.  "
            "Pass the same value for all folds so their outputs land together."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG_MODE.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override ADNI_LOCAL_DATASET_PATH.")
    parser.add_argument("--label_csv_path", type=str, default=None, help="Override ADNI_LABEL_CSV_PATH.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Train PyramidPooling3DClassifier on ADNI for a single fold.

    1. Constructs train/val/test :class:`~IMC.data.adni_dataloader_local.ADNI3DDataset`
       instances.
    2. Builds a :class:`~IMC.network07.PyramidPooling3DClassifier`.
    3. Trains with AdamW + cosine-decay LR and multi-task cross-entropy loss
       until early stopping.
    4. Tests the best checkpoint and writes results to ``fold_results.json``.
    5. Writes ``config.json`` and ``model_architecture.txt`` to the fold log
       directory.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.
    """
    _configure_dataset_env(args)

    n_folds = args.n_folds
    fold_idx = args.fold
    if not (0 <= fold_idx < n_folds):
        raise ValueError(f"--fold must be in [0, {n_folds - 1}], got {fold_idx}.")

    test_fold = fold_idx
    val_fold = (fold_idx + 1) % n_folds
    train_folds = [f for f in range(n_folds) if f not in {test_fold, val_fold}]

    # ---- Log directory ----
    if args.base_log_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_log_dir = os.path.join("./logs", f"{timestamp}_net07_adni_5fold_cv")
    else:
        base_log_dir = args.base_log_dir

    fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
    os.makedirs(fold_log_dir, exist_ok=True)
    experiment_name = f"fold_{fold_idx}_net07_adni_{args.backbone_type}"

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ---- Logging ----
    logger, log_path, tb_logger = setup_combined_logging(
        experiment_name=experiment_name,
        log_dir=fold_log_dir,
        tb_log_dir=fold_log_dir,
    )

    config = {
        "fold": fold_idx,
        "n_folds": n_folds,
        "train_folds": train_folds,
        "val_fold": val_fold,
        "test_fold": test_fold,
        "device": str(device),
        "backbone_type": args.backbone_type,
        "backbone_channels": args.backbone_channels,
        "backbone_blocks": args.backbone_blocks,
        "growth_rate": args.growth_rate,
        "embedding_dim": args.embedding_dim,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "n_slices": args.n_slices,
        "img_size": args.img_size,
        "augment_config": args.augment_config,
        "num_samples": args.num_samples,
        "patience": args.patience,
    }
    with open(os.path.join(fold_log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("=" * 80)
    print(f"Network07 ADNI  –  Fold {fold_idx} / {n_folds - 1}")
    print(f"  Train folds : {train_folds}")
    print(f"  Val fold    : {val_fold}")
    print(f"  Test fold   : {test_fold}")
    print(f"  Device      : {device}")
    print(f"  Backbone    : {args.backbone_type}")
    print(f"  Log dir     : {fold_log_dir}")
    print("=" * 80)

    log_training_start(logger, config=config)

    with capture_console_to_log(logger):

        # ---- Datasets ----
        print("Creating dataloaders …")
        from IMC.data.adni_dataloader_local import ADNI3DDataset
        train_dataset = ADNI3DDataset(
            split=[f"fold_{f}" for f in train_folds],
            n_slices=args.n_slices,
            img_size=args.img_size,
            augment_conf=args.augment_config,
            num_samples=args.num_samples,
        )
        val_dataset = ADNI3DDataset(
            split=[f"fold_{val_fold}"],
            n_slices=args.n_slices,
            img_size=args.img_size,
            augment_conf="NONE3D",
            num_samples=args.num_samples,
        )
        test_dataset = ADNI3DDataset(
            split=[f"fold_{test_fold}"],
            n_slices=args.n_slices,
            img_size=args.img_size,
            augment_conf="NONE3D",
            num_samples=args.num_samples,
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        print(f"Train samples : {len(train_dataset)}")
        print(f"Val   samples : {len(val_dataset)}")
        print(f"Test  samples : {len(test_dataset)}")

        # ---- Label config ----
        cl_d = train_dataset.get_n_labels()
        print(f"Label config  : {cl_d}")

        # ---- Model ----
        model = PyramidPooling3DClassifier(
            num_classes_dict=cl_d,
            backbone_type=args.backbone_type,
            backbone_channels=args.backbone_channels,
            backbone_blocks=args.backbone_blocks,
            growth_rate=args.growth_rate,
            embedding_dim=args.embedding_dim,
        )
        with open(os.path.join(fold_log_dir, "model_architecture.txt"), "w") as f:
            f.write(str(model))

        model.to(device)
        model.apply(init_weights)

        # ---- Optimiser & scheduler ----
        steps_per_epoch = len(train_loader)
        total_steps = args.num_epochs * steps_per_epoch
        warmup_steps = max(1, int(0.1 * total_steps))

        optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
        scheduler = get_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
        criterion = MultiTaskLoss(
            label_smoothing=0.1,
            task_names=list(ADNI_LABEL_NAMES.keys()),
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 8)
        task_weights = [1.0] * len(cl_d)

        # ---- Trainer ----
        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            tb_logger=tb_logger,
            logger=logger,
            patience=args.patience,
            task_weights=task_weights,
            use_mixed_precision=True,
        )

        # ---- Train ----
        print(f"\nStarting training for fold {fold_idx} …")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_path=os.path.join(fold_log_dir, "best_model.pth"),
        )

        # ---- Test ----
        print(f"\nTesting fold {fold_idx} …")
        trainer.load_checkpoint(os.path.join(fold_log_dir, "best_model.pth"))
        test_results = trainer.test(test_loader)

        fold_summary = {
            "fold": fold_idx,
            "train_folds": train_folds,
            "val_fold": val_fold,
            "test_fold": test_fold,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "test_results": test_results,
            "log_dir": fold_log_dir,
        }
        with open(os.path.join(fold_log_dir, "fold_results.json"), "w") as f:
            json.dump(fold_summary, f, indent=4, default=str)

        print(f"\nFold {fold_idx} completed.")
        print(f"Test results : {test_results}")
        print(f"Outputs in   : {fold_log_dir}")

    log_training_end(logger)
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
