"""Unified training entry point for Network v06 (PixelOnlyModel) on Duke.

This script runs 5-fold cross-validation (CV) training of
:class:`~IMC.network06.PixelOnlyModel` on the Duke Liver MRI dataset.  Only
the ``SequenceType_Code_norm`` task is trained, using focal loss to handle
class imbalance.

The fold split strategy mirrors a standard leave-one-fold-out scheme:

- **Test fold**  : ``fold_idx``
- **Val fold**   : ``(fold_idx + 1) % 5``
- **Train folds**: all remaining folds

Model
-----
:class:`~IMC.network06.PixelOnlyModel` – 2-D image-only encoder with a
:class:`~IMC.nn.image_encoder.MultiSliceImageEncoder` backbone.
Single-slice inputs (``n_slices=1``) are used; the model operates on the
equidistantly sampled middle slice.

Training configuration
----------------------
- AdamW optimiser with per-group weight decay (biases/norms excluded).
- Linear-warmup + cosine-decay LR schedule.
- Focal loss (α = 1.0, γ = 2.0) for ``SequenceType_Code_norm``.
- Mixed precision via :class:`torch.amp.GradScaler`.
- Early stopping controlled by ``--patience``.
- Per-fold ``config.json`` and ``model_architecture.txt`` written inside the
  fold log directory.
- Cross-validation summary CSV written to ``<log_dir>/cv_summary.csv``.

Typical usage
-------------
::

    python -m IMC.net6.train --gpu 0

Train only folds 0 and 1::

    python -m IMC.net6.train --folds 0,1

Environment variables
---------------------
The following environment variables configure Duke dataset paths.  They can
also be overridden via the corresponding CLI arguments, which take precedence::

    LOCAL_DATASET_PATH – root folder of the Duke MRI image dataset
    METADATA_PATH      – path to the Duke encoded metadata parquet file
    LABEL_CSV_PATH     – path to the Duke label CSV
    DEBUG_MODE         – set to "1" to enable extra debug output
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from IMC.data.constants import DUKE_ORIGINAL_LABEL_NAMES
from IMC.helper import (
    capture_console_to_log,
    log_training_end,
    log_training_start,
)
from IMC.network06 import PixelOnlyModel
from IMC.tensorboard_logging import setup_combined_logging
from IMC.trainer import Trainer

# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def init_weights(module: nn.Module) -> None:
    """Initialise weights for selected module types.

    Applies the following rules:

    - ``nn.Linear``    – Xavier (Glorot) uniform initialisation for weights;
      zeros for bias.
    - ``nn.LayerNorm`` – ones for weight; zeros for bias.

    Pass to ``model.apply(init_weights)`` after model construction to
    re-initialise only the non-pretrained fusion heads and projection layers.

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
    with ``weight_decay=0.0`` to avoid over-regularising scale/shift params.

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
# Focal loss (single-task wrapper for Trainer compatibility)
# ---------------------------------------------------------------------------


class _SingleTaskFocalLoss(nn.Module):
    """Focal loss wrapper compatible with the IMC :class:`~IMC.trainer.Trainer`.

    The Trainer passes ``(preds, targets, masks, task_weights)`` to the
    criterion, where each is a list with one element per task.  This wrapper
    extracts the first (and only) task's logits/targets and computes focal loss.

    Args:
        alpha: Weighting factor applied to the per-sample focal weights.
        gamma: Focusing parameter.  ``gamma=0`` reduces to cross-entropy.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self._ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        masks: list[torch.Tensor],
        task_weights: list[float] | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Compute focal loss for the single ``SequenceType_Code_norm`` task.

        Args:
            preds: List of logit tensors ``[(B, C)]``.
            targets: List of integer label tensors ``[(B,)]``.
            masks: List of boolean-like tensors ``[(B,)]`` indicating valid samples.
            task_weights: Unused; present for interface compatibility.

        Returns:
            tuple: ``(scalar_loss, [item_loss])`` where the second element is a
            one-element Python list containing the loss value as a float.
        """
        logits = preds[0]
        tgt = targets[0]
        msk = masks[0].bool()

        if not msk.any():
            zero = torch.tensor(0.0, device=logits.device)
            return zero, [0.0]

        logits = logits[msk]
        tgt = tgt[msk]

        ce = self._ce(logits, tgt)
        pt = torch.exp(-ce)
        focal = self.alpha * ((1.0 - pt) ** self.gamma) * ce
        loss = focal.mean()
        return loss, [loss.item()]


# ---------------------------------------------------------------------------
# Dataset environment configuration
# ---------------------------------------------------------------------------


def _configure_dataset_env(args: argparse.Namespace) -> None:
    """Apply CLI path overrides and validate Duke dataset configuration."""
    os.environ["DEBUG_MODE"] = "1" if args.debug else "0"
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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Duke net6 5-fold CV training.

    Returns:
        Populated :class:`argparse.Namespace` with all resolved arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "5-fold cross-validation training for IMC Network v06 (PixelOnlyModel) "
            "on the Duke Liver Dataset (single-task: SequenceType_Code_norm)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------- model
    parser.add_argument(
        "--img_enc_backbone",
        type=str,
        default="densenet121",
        help="CNN backbone for the image encoder (e.g. 'densenet121', 'resnet50').",
    )

    # ---------------------------------------- training
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=25,
        help="Maximum number of training epochs per fold.",
    )
    parser.add_argument("--lr", type=float, default=1e-6, help="Base learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing parameter γ (0 = standard cross-entropy).",
    )

    # ---------------------------------------- hardware / misc
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (-1 for CPU).")
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help=("Comma-separated list of fold indices to run (e.g. '0,1,2'). Defaults to all 5 folds."),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap dataset size (useful for smoke-tests).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Root directory for logs and checkpoints.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG_MODE.")

    # ---------------------------------------- dataset path overrides
    parser.add_argument("--dataset_path", type=str, default=None, help="Override LOCAL_DATASET_PATH.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Override METADATA_PATH.")
    parser.add_argument("--label_csv_path", type=str, default=None, help="Override LABEL_CSV_PATH.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Execute 5-fold (or partial) CV training of PixelOnlyModel on Duke.

    For each fold the function:

    1. Constructs train/val/test :class:`~IMC.data.duke_dataloader_local.LiverDataset`
       instances with ``n_slices=1``.
    2. Builds a :class:`~IMC.network06.PixelOnlyModel` with the chosen backbone.
    3. Trains with AdamW + cosine-decay LR and focal loss until early stopping.
    4. Tests the best checkpoint and appends results to *all_fold_results*.
    5. Writes ``config.json`` and ``model_architecture.txt`` to the fold log
       directory.

    A ``cv_summary.csv`` is written to ``<log_dir>/<timestamp>_net6_duke_5fold/``
    after all requested folds complete.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.
    """
    _configure_dataset_env(args)

    n_folds = 5
    folds_to_run = [int(f.strip()) for f in args.folds.split(",")] if args.folds is not None else list(range(n_folds))
    for fold_idx in folds_to_run:
        if not (0 <= fold_idx < n_folds):
            raise ValueError(f"Fold index {fold_idx} is out of range [0, {n_folds - 1}].")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(args.log_dir, f"{timestamp}_net6_duke_5fold")
    os.makedirs(base_log_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")

    print("=" * 80)
    print("Training PixelOnlyModel (net6) on Duke  –  5-fold CV")
    print(f"Backbone : {args.img_enc_backbone}")
    print(f"Device   : {device}")
    print(f"Folds    : {folds_to_run}")
    print("=" * 80)

    all_fold_results: list[dict] = []

    for fold_idx in folds_to_run:
        test_fold = fold_idx
        val_fold = (fold_idx + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in {test_fold, val_fold}]

        fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
        os.makedirs(fold_log_dir, exist_ok=True)
        experiment_name = f"fold_{fold_idx}_net6_duke_{args.img_enc_backbone}"

        logger, log_path, tb_logger = setup_combined_logging(
            experiment_name=experiment_name,
            log_dir=fold_log_dir,
            tb_log_dir=fold_log_dir,
        )

        config = {
            "fold": fold_idx,
            "train_folds": train_folds,
            "val_fold": val_fold,
            "test_fold": test_fold,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "img_enc_backbone": args.img_enc_backbone,
            "focal_gamma": args.focal_gamma,
            "patience": args.patience,
            "n_slices": 1,
            "task": "SequenceType_Code_norm",
            "net": "06",
            "num_samples": args.num_samples,
        }
        with open(os.path.join(fold_log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        log_training_start(logger, config=config)

        with capture_console_to_log(logger):
            print(f"\n{'=' * 80}")
            print(f"FOLD {fold_idx} / {n_folds - 1}")
            print(f"  Train folds : {train_folds}")
            print(f"  Val fold    : {val_fold}")
            print(f"  Test fold   : {test_fold}")
            print(f"{'=' * 80}")

            # ---- Datasets ----------------------------------------
            _ds_kwargs: dict = dict(
                num_samples=args.num_samples,
                n_slices=1,
                sampling_type="equidistant",
                augment_conf="IMAGENET299_CENTER",
                aggregated_metadata=False,
                use_preselected_features=False,
                exclude_contrast_yn=True,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
            )
            from IMC.data.duke_dataloader_local import LiverDataset

            train_ds = LiverDataset(split=[f"fold_{f}" for f in train_folds], **_ds_kwargs)
            val_ds = LiverDataset(split=[f"fold_{val_fold}"], **_ds_kwargs)
            test_ds = LiverDataset(split=[f"fold_{test_fold}"], **_ds_kwargs)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

            print(f"Train samples : {len(train_ds)}")
            print(f"Val   samples : {len(val_ds)}")
            print(f"Test  samples : {len(test_ds)}")

            # ---- Single-task label config -------------------------
            full_label_map = train_ds.get_n_labels()
            if "SequenceType_Code_norm" not in full_label_map:
                raise RuntimeError("SequenceType_Code_norm not found in dataset label map.")
            cl_d = {"SequenceType_Code_norm": full_label_map["SequenceType_Code_norm"]}
            print(f"Label config  : {cl_d}")

            # ---- Model -------------------------------------------
            model = PixelOnlyModel(
                num_classes_dict=cl_d,
                image_backbone=args.img_enc_backbone,
            )
            with open(os.path.join(fold_log_dir, "model_architecture.txt"), "w") as f:
                f.write(str(model))
            model.to(device)
            model.apply(init_weights)

            # ---- Optimiser & scheduler ---------------------------
            steps_per_epoch = len(train_loader)
            total_steps = args.num_epochs * steps_per_epoch
            warmup_steps = max(1, int(0.1 * total_steps))

            optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = get_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
            criterion = _SingleTaskFocalLoss(alpha=1.0, gamma=args.focal_gamma)
            scaler = torch.amp.GradScaler("cuda", init_scale=2**8)

            # ---- Trainer -----------------------------------------
            trainer = Trainer(
                model=model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                task_names=list(DUKE_ORIGINAL_LABEL_NAMES.keys()),
                scaler=scaler,
                tb_logger=tb_logger,
                logger=logger,
                patience=args.patience,
                task_weights=[1.0],
                incl_regression=False,
                use_mixed_precision=True,
            )

            print("\nStarting training …")
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.num_epochs,
                save_path=os.path.join(fold_log_dir, "best_model.pth"),
            )
            trainer.load_checkpoint(os.path.join(fold_log_dir, "best_model.pth"))

            print("\nTesting …")
            test_results = trainer.test(test_loader)
            print(f"Test results : {test_results}")

            all_fold_results.append(
                {
                    "fold": fold_idx,
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                    "test_samples": len(test_ds),
                    "test_results": str(test_results),
                    "log_dir": fold_log_dir,
                }
            )

        log_training_end(logger)

    # ---- CV summary --------------------------------------------------
    summary_path = os.path.join(base_log_dir, "cv_summary.csv")
    results_df = pd.DataFrame(
        [
            {
                "fold": r["fold"],
                "train_samples": r["train_samples"],
                "val_samples": r["val_samples"],
                "test_samples": r["test_samples"],
                "test_results": r["test_results"],
            }
            for r in all_fold_results
        ]
    )
    results_df.to_csv(summary_path, index=False)
    print(f"\nAll results saved to : {base_log_dir}")
    print(f"Summary saved to     : {summary_path}")


if __name__ == "__main__":
    main(parse_args())
