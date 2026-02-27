"""
Training script for Network 07 (3D Pyramid Pooling Network) on ADNI Dataset.

This script implements per-fold training for the PyramidPooling3DClassifier model
on the ADNI brain MRI dataset. Each run trains exactly ONE fold, enabling multiple
folds to be trained in parallel to reduce total wall-clock time.

Fold assignment (5-fold CV):
    test_fold  = --fold
    val_fold   = (--fold + 1) % n_folds
    train_folds = remaining folds

Usage (run each fold in a separate process / tmux pane):
    python -m IMC.net7_adni.train07_adni --fold 0
    python -m IMC.net7_adni.train07_adni --fold 1
    python -m IMC.net7_adni.train07_adni --fold 2
    python -m IMC.net7_adni.train07_adni --fold 3
    python -m IMC.net7_adni.train07_adni --fold 4

Key features:
- Single-fold execution designed for parallel runs.
- ADNIDataset wrapped to expose images as (1, N, H, W) for network07.
- Multi-task classification: AcquisitionPlane, SequenceContrast, Localizer.
- PyramidPooling3DClassifier (network07) backbone.
- Mixed precision training with gradient scaling.
- Integrated logging with TensorBoard and JSON config saving.

Authors: Tuan Truong
Date: 2026
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from IMC.helper import (
    capture_console_to_log,
    log_training_end,
    log_training_start,
    setup_experiment_logging,
)
from IMC.network07 import PyramidPooling3DClassifier
from IMC.nn.multi_task_loss import MultiTaskLoss
from IMC.tensorboard_logging import setup_combined_logging
from IMC.trainer import Trainer

# ---------------------------------------------------------------------------
# ADNI environment variables – override via shell or .env before running
# ---------------------------------------------------------------------------
os.environ["ADNI_LOCAL_DATASET_PATH"] = os.getenv(
    "ADNI_LOCAL_DATASET_PATH",
    "/home/tuan.truong/data/ADNI_full",
)
os.environ["ADNI_LABEL_CSV_PATH"] = os.getenv(
    "ADNI_LABEL_CSV_PATH",
    "/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv",
)
# Leave ADNI_METADATA_PATH unset to encode metadata on-the-fly, or set it:
os.environ["ADNI_METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/adni_metadata/adni_encoded_metadata_20260224.parquet"

# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module) -> None:
    """Xavier init for Linear layers; ones/zeros for LayerNorm."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def get_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_scale_factor: float = 100,
    min_scale_factor: float = 0.1,
) -> LambdaLR:
    """
    Warmup + cosine decay LR scheduler.

    The base LR is multiplied by a factor that:
    - Linearly ramps to ``peak_scale_factor`` over ``warmup_steps``.
    - Cosine-decays to ``min_scale_factor`` until ``total_steps``.
    - Stays at ``min_scale_factor`` thereafter.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step <= warmup_steps and warmup_steps > 0:
            return max(1.0, peak_scale_factor * float(current_step) / warmup_steps)
        elif warmup_steps < current_step <= total_steps:
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(
                min_scale_factor,
                peak_scale_factor * 0.5 * (1 + math.cos(math.pi * progress)),
            )
        else:
            return min_scale_factor

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def create_optimizer(
    model: nn.Module,
    lr: float = 1.0e-6,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
) -> Optimizer:
    """
    AdamW with two parameter groups:
    - Norm/bias parameters → no weight decay.
    - All other parameters → standard weight decay.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []

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
# CLI argument validation
# ---------------------------------------------------------------------------

def validate_str_argument(value: str, options: List[str], name: str) -> str:
    if value not in options:
        raise ValueError(f"Invalid {name} '{value}'. Valid options: {options}")
    return value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from IMC.data.adni_dataloader_local import ADNI_LABEL_NAMES, ADNI3DDataset

    parser = argparse.ArgumentParser(
        description=(
            "Train Network 07 (3D Pyramid Pooling) on ADNI — ONE fold per invocation.\n"
            "Run five processes in parallel (--fold 0 … 4) to complete full 5-fold CV."
        )
    )

    # ---- Fold selection ----
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Index of the *test* fold to hold out (0–4). Val = (fold+1) %% n_folds.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Total number of folds (default: 5).",
    )

    # ---- Dataset ----
    parser.add_argument(
        "--n_slices",
        type=int,
        default=16,
        help="Number of slices to sample per DICOM series (acts as depth for network07).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Spatial resolution of each slice (H = W = img_size).",
    )
    parser.add_argument(
        "--augment_config",
        type=str,
        default="DEFAULT3D",
        help="Augmentation config string for training splits.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap dataset size (useful for smoke-tests).",
    )

    # ---- Model ----
    parser.add_argument(
        "--backbone_type",
        type=str,
        default="resnet",
        help="Backbone: resnet | densenet121 | densenet169 | densenet201 | densenet_custom",
    )
    parser.add_argument("--backbone_channels", type=int, default=32)
    parser.add_argument(
        "--backbone_blocks",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
        help="Block counts for ResNet (e.g., 2 2 2 2).",
    )
    parser.add_argument("--growth_rate", type=int, default=12, help="DenseNet growth rate.")
    parser.add_argument("--embedding_dim", type=int, default=512, help="MLP projection dim.")

    # ---- Training hyper-parameters ----
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience.")

    # ---- Hardware ----
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")

    # ---- Output ----
    parser.add_argument(
        "--base_log_dir",
        type=str,
        default=None,
        help=(
            "Root log directory shared across all folds. "
            "Defaults to ./logs/<timestamp>_net07_adni_5fold_cv. "
            "Pass the same value when launching all folds so outputs land together."
        ),
    )

    args = parser.parse_args()

    # ---- Validate fold index ----
    n_folds = args.n_folds
    fold_idx = args.fold
    if not (0 <= fold_idx < n_folds):
        raise ValueError(f"--fold must be in [0, {n_folds - 1}], got {fold_idx}.")

    # ---- Validate backbone type ----
    backbone_type = validate_str_argument(
        args.backbone_type,
        ["resnet", "densenet121", "densenet169", "densenet201", "densenet_custom"],
        "backbone_type",
    )

    # ---- Fold split ----
    test_fold = fold_idx
    val_fold = (fold_idx + 1) % n_folds
    train_folds = [f for f in range(n_folds) if f not in {test_fold, val_fold}]

    # ---- Logging directories ----
    if args.base_log_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_log_dir = os.path.join("./logs", f"{timestamp}_net07_adni_5fold_cv")
    else:
        base_log_dir = args.base_log_dir

    fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
    os.makedirs(fold_log_dir, exist_ok=True)

    experiment_name = f"fold_{fold_idx}_net07_adni_{backbone_type}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- Setup logging ----
    logger, log_path, tb_logger = setup_combined_logging(
        experiment_name=experiment_name,
        log_dir=fold_log_dir,
        tb_log_dir=fold_log_dir,
    )

    # ---- Persist config ----
    config = {
        "fold": fold_idx,
        "n_folds": n_folds,
        "train_folds": train_folds,
        "val_fold": val_fold,
        "test_fold": test_fold,
        "device": str(device),
        "backbone_type": backbone_type,
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
    print(f"Network07 ADNI — Fold {fold_idx}/{n_folds - 1}")
    print(f"  Train folds : {train_folds}")
    print(f"  Val fold    : {val_fold}")
    print(f"  Test fold   : {test_fold}")
    print(f"  Device      : {device}")
    print(f"  Backbone    : {backbone_type}")
    print(f"  Log dir     : {fold_log_dir}")
    print("=" * 80)

    log_training_start(logger, config=config)

    with capture_console_to_log(logger):

        # ---- Datasets ----
        print("Creating dataloaders …")

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

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        print(f"Train samples : {len(train_dataset)}")
        print(f"Val   samples : {len(val_dataset)}")
        print(f"Test  samples : {len(test_dataset)}")

        # ---- Label config ----
        cl_d = train_dataset.get_n_labels()
        print(f"Label config  : {cl_d}")

        # ---- Model ----
        model = PyramidPooling3DClassifier(
            num_classes_dict=cl_d,
            backbone_type=backbone_type,
            backbone_channels=args.backbone_channels,
            backbone_blocks=args.backbone_blocks,
            growth_rate=args.growth_rate,
            embedding_dim=args.embedding_dim,
        )
        with open(os.path.join(fold_log_dir, "model_architecture.txt"), "w") as f:
            f.write(str(model))

        model.to(device)
        model.apply(init_weights)

        # ---- Training components ----
        steps_per_epoch = len(train_loader)
        total_steps = args.num_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)

        optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
        scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
        criterion = MultiTaskLoss(
            label_smoothing=0.1,
            task_names=list(ADNI_LABEL_NAMES.keys()),
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=2**8)

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
            img_ft_only=True,
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
