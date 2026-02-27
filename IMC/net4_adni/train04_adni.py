"""
Training script for Network v04 (cross-attention fusion) on ADNI Dataset.

This script trains ONE fold per invocation, allowing all 5 folds to be launched
in parallel across GPU workers to reduce total wall-clock time.

Fold assignment (5-fold CV):
    test_fold   = --fold
    val_fold    = (--fold + 1) % n_folds
    train_folds = remaining folds

Usage (run each fold in a separate process / tmux pane):
    python -m IMC.net4_adni.train04_adni --fold 0 --gpu 0
    python -m IMC.net4_adni.train04_adni --fold 1 --gpu 1
    python -m IMC.net4_adni.train04_adni --fold 2 --gpu 2
    python -m IMC.net4_adni.train04_adni --fold 3 --gpu 3
    python -m IMC.net4_adni.train04_adni --fold 4 --gpu 4

Pass the same --base_log_dir to all processes so outputs land in one tree:
    python -m IMC.net4_adni.train04_adni --fold 0 \\
        --base_log_dir ./logs/net04_adni_cv

Key features:
- Single-fold execution designed for parallel runs.
- MRISequenceClassifier (network04) with image + metadata cross-attention.
- Multi-task classification for 3 ADNI tasks (no regression).
- Mixed precision training with gradient scaling.
- Integrated TensorBoard and file logging.
- config.json saved per fold for reproducibility.

Authors: Tuan Truong
Date: 2026
"""

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
import pandas as pd

from IMC.helper import (
    capture_console_to_log,
    log_training_end,
    log_training_start,
)
from IMC.network04 import MRISequenceClassifier, MetadataBasedClassifier, ImageBasedClassifier, SingleChannelImageBasedClassifier, MRISequenceClassifierWithSparseMetadata
from IMC.nn.multi_task_loss import MultiTaskLoss
from IMC.tensorboard_logging import setup_combined_logging
from IMC.trainer import Trainer

# ---------------------------------------------------------------------------
# ADNI environment defaults (override via shell / .env before running)
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "ADNI_LOCAL_DATASET_PATH",
    "/home/tuan.truong/data/ADNI_full",
)
os.environ.setdefault(
    "ADNI_LABEL_CSV_PATH",
    "/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv",
)
# Uncomment to use a pre-encoded metadata parquet (faster startup):
os.environ.setdefault(
    "ADNI_METADATA_PATH",
    "/home/tuan.truong/codebase/IMC/labels/adni_metadata/adni_encoded_metadata_20260224.parquet",
)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module) -> None:
    """Xavier init for Linear; ones/zeros for LayerNorm."""
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
    """Warmup + cosine decay LR scheduler."""

    def lr_lambda(current_step: int) -> float:
        if current_step <= warmup_steps and warmup_steps > 0:
            return max(1.0, peak_scale_factor * float(current_step) / warmup_steps)
        elif warmup_steps < current_step <= total_steps:
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(
                min_scale_factor,
                peak_scale_factor * 0.5 * (1 + math.cos(math.pi * progress)),
            )
        return min_scale_factor

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def create_optimizer(
    model: nn.Module,
    lr: float = 1e-6,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
) -> Optimizer:
    """AdamW with no weight decay on norms/biases."""
    decay, no_decay = [], []
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
    from IMC.data.adni_dataloader_local import ADNI_LABEL_NAMES, ADNIDataset

    VALID_BACKBONES = ["densenet121", "resnet50", "resnet18", "efficientnet"]
    VALID_FUSION    = ["v1", "v2", "concat"]

    parser = argparse.ArgumentParser(
        description=(
            "Train Network 04 (cross-attention fusion) on ADNI — ONE fold per invocation.\n"
            "Run five processes in parallel (--fold 0 … 4) to complete full 5-fold CV."
        )
    )

    # ---- Dataset ----
    parser.add_argument("--n_slices", type=int, default=3,
                        help="Number of slices sampled per DICOM series.")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Spatial resolution of each slice (H = W = img_size).")
    parser.add_argument("--augment_config", type=str, default="NONE2D",
                        help="Augmentation config for the training split.")
    parser.add_argument("--aggregated_metadata", action="store_true", default=True,
                        help="Use aggregated (series-level) metadata.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap dataset size (smoke-tests).")

    # ---- Model ----
    parser.add_argument("--backbone", type=str, default="densenet121",
                        help=f"Image encoder backbone: {VALID_BACKBONES}")
    parser.add_argument("--modality", type=str, default="combined", help="Modality to use: 'combined', 'image', or 'metadata'")
    parser.add_argument("--image_naive_approach", action="store_true", help="Whether to use the naive approach for image-only model (single slice with 3 channels)")
    parser.add_argument("--metadata_enc_type", type=str, default='none', help="Which metadata encoder to use: 'none', 'imputer', 'sparse'")
    parser.add_argument("--sparse_enc_version", type=str, default="v1", help="Sparse encoder version: v1 or v2")
    parser.add_argument("--fusion_module_version", type=str, default="v1", help="Fusion module version: v1 or v2")
    
    # ---- Training hyper-parameters ----
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=30)

    # ---- Hardware ----
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    # ---- Validate args ----
    # Parse which folds to run
    folds_to_run = [0, 1, 2, 3, 4]
    n_folds = 5
    backbone = validate_str_argument(args.backbone, VALID_BACKBONES, "backbone")
    validate_str_argument(args.fusion_module_version, VALID_FUSION, "fusion_module_version")

    # Validate fold indices
    for fold in folds_to_run:
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"Invalid fold index {fold}. Must be in range [0, {n_folds-1}]")

    # Directory for all experiments
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join("./logs", f"{timestamp}_5fold_cv")
    os.makedirs(base_log_dir, exist_ok=True)

    # Store results for all folds
    all_fold_results = []
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("="*80)
    print(f"Starting 5-Fold Cross-Validation Training")
    print(f"Folds to train: {folds_to_run}")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs per fold: {args.num_epochs}")
    print("="*80)

    # Iterate through each fold
    for fold_idx in folds_to_run:
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{n_folds-1}")
        print(f"{'='*80}")

        # Define fold splits
        # Test fold: current fold
        # Val fold: next fold (cyclically)
        # Train folds: all remaining folds
        test_fold = fold_idx
        val_fold = (fold_idx + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in [test_fold, val_fold]]
        
        print(f"Train folds: {train_folds}")
        print(f"Val fold: {val_fold}")
        print(f"Test fold: {test_fold}")
        
        # Create fold-specific log directory
        fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
        os.makedirs(fold_log_dir, exist_ok=True)
        experiment_name = f"fold_{fold_idx}_baseline_multi_slices_04_{args.backbone}_sparse_encoder_{args.sparse_enc_version}_fusion_{args.fusion_module_version}"
        

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
            "backbone": backbone,
            "fusion_module_version": args.fusion_module_version,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "n_slices": args.n_slices,
            "img_size": args.img_size,
            "augment_config": args.augment_config,
            "aggregated_metadata": args.aggregated_metadata,
            "num_samples": args.num_samples,
            "patience": args.patience,
            "modality": args.modality,
            "image_naive_approach": args.image_naive_approach,
            "metadata_enc_type": args.metadata_enc_type,
            "sparse_enc_version": args.sparse_enc_version,
        }
        with open(os.path.join(fold_log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        print("=" * 80)
        print(f"Network04 ADNI — Fold {fold_idx}/{n_folds - 1}")
        print(f"  Train folds : {train_folds}")
        print(f"  Val fold    : {val_fold}")
        print(f"  Test fold   : {test_fold}")
        print(f"  Device      : {device}")
        print(f"  Backbone    : {backbone}")
        print(f"  Log dir     : {fold_log_dir}")
        print("=" * 80)

        log_training_start(logger, config=config)

        with capture_console_to_log(logger):

            # ---- Datasets ----
            print("Creating dataloaders …")

            train_dataset = ADNIDataset(
                split=[f"fold_{f}" for f in train_folds],
                n_slices=args.n_slices,
                img_size=args.img_size,
                augment_conf=args.augment_config,
                aggregated_metadata=False,
                num_samples=args.num_samples,
                sampling_type="random",
                label_names=ADNI_LABEL_NAMES

            )
            val_dataset = ADNIDataset(
                split=[f"fold_{val_fold}"],
                n_slices=args.n_slices,
                img_size=args.img_size,
                augment_conf="NONE2D",
                aggregated_metadata=False,
                num_samples=args.num_samples,
                sampling_type="equidistant",
                label_names=ADNI_LABEL_NAMES
            )
            test_dataset = ADNIDataset(
                split=[f"fold_{test_fold}"],
                n_slices=args.n_slices,
                img_size=args.img_size,
                augment_conf="NONE2D",
                aggregated_metadata=False,
                num_samples=args.num_samples,
                sampling_type="equidistant",
                label_names=ADNI_LABEL_NAMES
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

            # ---- Label / metadata config ----
            cl_d                = train_dataset.get_n_labels()
            metadata_input_dim  = train_dataset.num_metadata_features
            print(f"Label config  : {cl_d}")
            print(f"Metadata dim  : {metadata_input_dim}")

            # ---- Model ----
        # Create model
            if args.modality == "image":
                if args.image_naive_approach:
                    model = SingleChannelImageBasedClassifier(num_classes_dict=cl_d)
                else:
                    model = ImageBasedClassifier(num_classes_dict=cl_d, backbone=args.backbone)
            elif args.modality == "metadata":
                if args.metadata_enc_type == "none":
                    model = MetadataBasedClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_encoder_type=args.metadata_enc_type,
                        imputer_type="contextual",
                        incl_regression=False
                    )
                elif args.metadata_enc_type == "sparse":
                    if args.sparse_enc_version == "v1":
                        model = MetadataBasedClassifier(
                            metadata_input_dim=train_dataset.num_metadata_features,
                            num_classes_dict=cl_d,
                            metadata_encoder_type="sparse",
                            incl_regression=False
                        )
                    else:
                        model = MetadataBasedClassifier(
                            metadata_input_dim=train_dataset.num_metadata_features,
                            num_classes_dict=cl_d,
                            metadata_encoder_type="sparse_v2",
                            incl_regression=False
                        )
                else:
                    # Imputer is NanIgnorer
                    model = MetadataBasedClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_encoder_type="imputer",
                        imputer_type="ignore",
                        incl_regression=False
                    )
            else:
                # Combined model with cross-attention fusion
                if args.metadata_enc_type == "none":
                    model = MRISequenceClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        img_enc_backbone=args.backbone,
                        incl_regression=False,
                        fusion_module_version=args.fusion_module_version,
                        imputer_type="ignore"
                    )
                elif args.metadata_enc_type == "imputer":
                    model = MRISequenceClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        img_enc_backbone=args.backbone,
                        incl_regression=False,
                        fusion_module_version=args.fusion_module_version,
                        imputer_type="contextual"
                    )
                elif args.metadata_enc_type == "sparse" and args.sparse_enc_version == "v1":
                    model = MRISequenceClassifierWithSparseMetadata(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_embeder_type="sparse",
                        img_enc_backbone=args.backbone,
                        dropout_metadata=False,
                        fusion_module_version=args.fusion_module_version,
                        include_regression=False
                    )
                elif args.metadata_enc_type == "sparse" and args.sparse_enc_version == "v2":
                    model = MRISequenceClassifierWithSparseMetadata(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_embeder_type="sparse_v2",
                        img_enc_backbone=args.backbone,
                        dropout_metadata=False,
                        fusion_module_version=args.fusion_module_version,
                        include_regression=False
                    )
                elif args.metadata_enc_type == "sparse" and args.sparse_enc_version == "v5":
                    model = MRISequenceClassifierWithSparseMetadata(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_embeder_type="sparse_v5",
                        img_enc_backbone=args.backbone,
                        dropout_metadata=False,
                        fusion_module_version=args.fusion_module_version,
                        include_regression=False
                    )
                else:
                    raise ValueError(f"Invalid metadata_enc_type '{args.metadata_enc_type}' or sparse_enc_version '{args.sparse_enc_version}'")
            
            with open(os.path.join(fold_log_dir, "model_architecture.txt"), "w") as f:
                f.write(str(model))

            model.to(device)
            model.apply(init_weights)

            # ---- Training components ----
            steps_per_epoch = len(train_loader)
            total_steps     = args.num_epochs * steps_per_epoch
            warmup_steps    = int(0.1 * total_steps)

            optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
            scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
            criterion = MultiTaskLoss(
                label_smoothing=0.1,
                incl_regression=False,
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
                incl_regression=False,
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
