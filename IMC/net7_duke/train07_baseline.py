"""
Training script for Network 07 (3D Pyramid Pooling Network) on Duke Dataset.

This script implements a 5-fold cross-validation training workflow for the
PyramidPooling3DClassifier model on the Duke liver MRI dataset. It is adapted
from the net4_duke template and customized for 3D volumetric data.

Key features:
- 5-fold cross-validation setup with flexible backbone selection (ResNet/DenseNet).
- Uses DukeLiverDataset3D for volumetric data loading.
- Implements the PyramidPooling3DClassifier (network07).
- Single-task classification for SequenceType_Code_norm.
- Integrated logging with TensorBoard and JSON config saving.
- Mixed precision training with gradient scaling.

Authors: Claude Code
Date: 2026
"""

import argparse
import json
import math
import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from IMC.helper import (
    capture_console_to_log, log_training_end, log_training_start, setup_experiment_logging
)
from IMC.network07 import PyramidPooling3DClassifier
from IMC.nn.multi_task_loss import MultiTaskLoss
from IMC.tensorboard_logging import setup_combined_logging
from IMC.trainer import Trainer

# Set environment variables for dataset paths
os.environ["DEBUG_MODE"] = "1"
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"


def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for selected modules.

    - Linear layers: Xavier (Glorot) uniform for weights, zeros for bias
    - LayerNorm: ones for weights, zeros for bias

    Use: model.apply(init_weights)
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def get_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_scale_factor: float = 100,
    min_scale_factor: float = 0.1,
) -> LambdaLR:
    """
    Construct a warmup + cosine decay LR scheduler.

    The scheduler scales the base LR by a factor computed as:
    - Linear warmup to `peak_scale_factor` over `warmup_steps`
    - Cosine decay from `peak_scale_factor` down to `min_scale_factor` until `total_steps`
    - Constant `min_scale_factor` afterwards

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        warmup_steps: Number of steps for the warmup phase.
        total_steps: Total number of steps for the schedule.
        peak_scale_factor: Maximum LR scale during warmup.
        min_scale_factor: Minimum LR scale during decay and steady phases.

    Returns:
        LambdaLR: The configured learning rate scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step <= warmup_steps and warmup_steps > 0:
            return max(1.0, peak_scale_factor * float(current_step) / warmup_steps)
        # Cosine decay
        elif current_step > warmup_steps and current_step <= total_steps:
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_scale_factor, peak_scale_factor * 0.5 * (1 + math.cos(math.pi * progress)))
        # Minimum scale
        else:
            return min_scale_factor

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer(
    model: nn.Module,
    lr: float = 1.0e-6,
    weight_decay: float = 1e-2,
    eps: float = 1e-8
) -> Optimizer:
    """
    Build an AdamW optimizer with parameter groups.

    - Parameters in normalization layers and biases get no weight decay.
    - All other parameters get standard weight decay.

    Args:
        model: Model to optimize.
        lr: Base learning rate.
        weight_decay: Weight decay for non-normalization parameters.
        eps: Epsilon for numerical stability.

    Returns:
        Optimizer: Configured AdamW optimizer.
    """
    decay = []
    no_decay = []

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


def validate_str_arguments(value: str, options: List[str], name: str) -> str:
    """
    Validate command-line arguments for training configuration.
    
    Args:
        value: The argument value to validate.
        options: List of valid options for the argument.
        name: The name of the argument (for error messages).
        
    Raises:
        ValueError: If any argument value is invalid.
    
    Returns:
        str: The validated value.
    """
    if value not in options:
        raise ValueError(f"Invalid {name} '{value}'. Valid options are: {options}")
    return value


if __name__ == "__main__":
    from IMC.data.duke_dataloader_3d import DUKE_ORIGINAL_LABEL_NAMES, DukeLiverDataset3D

    parser = argparse.ArgumentParser(description="Train 3D Pyramid Pooling Network with 5-Fold CV")
    parser.add_argument("--backbone_type", type=str, default="resnet",
                        help="Backbone type: resnet, densenet121, densenet169, densenet201, densenet_custom")
    parser.add_argument("--backbone_channels", type=int, default=32, help="Initial channels in backbone")
    parser.add_argument("--backbone_blocks", type=int, nargs="+", default=[2, 2, 2, 2],
                        help="Block counts for ResNet (e.g., 2 2 2 2)")
    parser.add_argument("--growth_rate", type=int, default=12, help="Growth rate for DenseNet")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-6, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--target_depth", type=int, default=64, help="Target depth for 3D volumes")
    parser.add_argument("--augment_config", type=str, default="DEFAULT3D", help="3D augmentation configuration")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit dataset size for testing")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")

    args = parser.parse_args()

    # Validate backbone type
    backbone_type = validate_str_arguments(
        args.backbone_type,
        ["resnet", "densenet121", "densenet169", "densenet201", "densenet_custom"],
        "backbone_type"
    )

    # Cross-validation configuration
    folds_to_run = [0, 1, 2, 3, 4]
    n_folds = 5
    
    # Validate fold indices
    for fold in folds_to_run:
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"Invalid fold index {fold}. Must be in range [0, {n_folds-1}]")

    # Create base log directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join("./logs", f"{timestamp}_net07_5fold_cv")
    os.makedirs(base_log_dir, exist_ok=True)

    # Training configuration
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Store results for all folds
    all_fold_results = []

    print("=" * 80)
    print("Starting 5-Fold Cross-Validation for Network07")
    print(f"Folds to train: {folds_to_run}")
    print(f"Device: {device}")
    print(f"Backbone: {backbone_type}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs per fold: {num_epochs}")
    print(f"Learning rate: {lr}")
    print("=" * 80)

    for fold_idx in folds_to_run:
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{n_folds-1}")
        print(f"{'='*80}")

        # Define fold splits
        test_fold = fold_idx
        val_fold = (fold_idx + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in [test_fold, val_fold]]

        print(f"Train folds: {train_folds}")
        print(f"Val fold: {val_fold}")
        print(f"Test fold: {test_fold}")

        # Create fold-specific log directory
        fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
        os.makedirs(fold_log_dir, exist_ok=True)
        experiment_name = f"fold_{fold_idx}_net07_{backbone_type}"

        # Setup combined logging (file + TensorBoard)
        logger, log_path, tb_logger = setup_combined_logging(
            experiment_name=experiment_name,
            log_dir=fold_log_dir,
            tb_log_dir=fold_log_dir
        )

        # Log configuration
        config = {
            "fold": fold_idx,
            "train_folds": train_folds,
            "val_fold": val_fold,
            "test_fold": test_fold,
            "device": str(device),
            "backbone_type": backbone_type,
            "backbone_channels": args.backbone_channels,
            "backbone_blocks": args.backbone_blocks,
            "growth_rate": args.growth_rate,
            "embedding_dim": args.embedding_dim,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "target_depth": args.target_depth,
            "augment_config": args.augment_config,
            "num_samples": args.num_samples,
            "patience": args.patience
        }
        
        # Save config to JSON for reproducibility
        with open(os.path.join(fold_log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        log_training_start(logger, config=config)

        with capture_console_to_log(logger):
            # Create datasets and dataloaders
            print("Creating dataloaders...")
            
            train_dataset = DukeLiverDataset3D(
                split=[f"fold_{f}" for f in train_folds],
                target_depth=args.target_depth,
                augment_conf=args.augment_config,
                num_samples=args.num_samples
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )

            val_dataset = DukeLiverDataset3D(
                split=[f"fold_{val_fold}"],
                target_depth=args.target_depth,
                augment_conf="NONE3D",
                num_samples=args.num_samples
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

            test_dataset = DukeLiverDataset3D(
                split=[f"fold_{test_fold}"],
                target_depth=args.target_depth,
                augment_conf="NONE3D",
                num_samples=args.num_samples
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")

            # Get label configuration
            cl_d = train_dataset.get_n_labels()
            print(f"Label config: {cl_d}")

            # Create model with specified backbone
            model = PyramidPooling3DClassifier(
                num_classes_dict=cl_d,
                backbone_type=backbone_type,
                backbone_channels=args.backbone_channels,
                backbone_blocks=args.backbone_blocks,
                growth_rate=args.growth_rate,
                embedding_dim=args.embedding_dim
            )
            
            # Save the model architecture to the fold log directory for reference
            with open(os.path.join(fold_log_dir, "model_architecture.txt"), "w") as f:
                f.write(str(model))
            
            model.to(device)
            
            # Initialize weights
            model.apply(init_weights)

            # Setup training components
            steps_per_epoch = len(train_loader)
            total_steps = num_epochs * steps_per_epoch
            warmup_steps = int(0.1 * total_steps)
            
            eps = 1e-7
            optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay, eps=eps)
            scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
            criterion = MultiTaskLoss(label_smoothing=0.1, task_names=list(DUKE_ORIGINAL_LABEL_NAMES.keys()))
            scaler = torch.amp.GradScaler("cuda", init_scale=2**8)

            # Task weights
            task_weights = [1.0] * len(cl_d)

            # Create trainer
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
                img_ft_only=True
            )

            # Train
            print(f"\nStarting training for fold {fold_idx}...")
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                save_path=os.path.join(fold_log_dir, "best_model.pth")
            )

            # Test with best model
            print(f"\nTesting fold {fold_idx}...")
            trainer.load_checkpoint(os.path.join(fold_log_dir, "best_model.pth"))
            test_results = trainer.test(test_loader)

            # Store results
            fold_results = {
                "fold": fold_idx,
                "train_folds": train_folds,
                "val_fold": val_fold,
                "test_fold": test_fold,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "test_results": test_results,
                "log_dir": fold_log_dir
            }
            all_fold_results.append(fold_results)
            
            print(f"\nFold {fold_idx} completed!")
            print(f"Test results: {test_results}")

        # Log training end
        log_training_end(logger)

    # Save summary of all folds
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    summary_path = os.path.join(base_log_dir, "cv_summary.csv")
    results_df = pd.DataFrame([
        {
            "fold": r["fold"],
            "train_folds": str(r["train_folds"]),
            "val_fold": r["val_fold"],
            "test_fold": r["test_fold"],
            "train_samples": r["train_samples"],
            "val_samples": r["val_samples"],
            "test_samples": r["test_samples"],
            "test_results": str(r["test_results"])
        }
        for r in all_fold_results
    ])
    results_df.to_csv(summary_path, index=False)
    
    print(f"\nResults for {len(all_fold_results)} folds:")
    for result in all_fold_results:
        print(f"\nFold {result['fold']}:")
        print(f"  Train samples: {result['train_samples']}")
        print(f"  Val samples: {result['val_samples']}")
        print(f"  Test samples: {result['test_samples']}")
        print(f"  Test results: {result['test_results']}")
    
    print(f"\nAll results saved to: {base_log_dir}")
    print(f"Summary saved to: {summary_path}")
    print("="*80)
