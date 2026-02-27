"""
Training script for Network v04 (cross-attention fusion).

This module provides a runnable training entry point with:
- Mixed precision training via torch.amp.GradScaler
- Checkpointing that preserves scaler state for seamless resume
- Early stopping and learning rate scheduling (warmup + cosine decay)
- Integrated TensorBoard and file logging

It orchestrates the training loop, model optimization, and checkpoint management
for the v04 architecture. Environment variables are used to configure local
dataset and label paths.
"""

from typing import Union
import math
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from IMC.helper import plot_batch_per_sample, normalize_per_sample, count_parameters
from IMC.helper import setup_experiment_logging, capture_console_to_log, log_training_start, log_training_end
from IMC.nn.multi_task_loss import MultiTaskLoss

import os 

# Set environment variables or paths for local dataset
os.environ["DEBUG_MODE"] = "1"  # Enable debug mode
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet"

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


def create_optimizer(model: torch.nn.Module, lr: float = 1.0e-6, weight_decay: float = 1e-2, eps: float = 1e-8) -> Optimizer:
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


# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from IMC.network04 import MRISequenceClassifierWithSparseMetadata, MRISequenceClassifier, ImageBasedClassifier, MetadataBasedClassifier, SingleChannelImageBasedClassifier
    from IMC.data.duke_dataloader_local import LiverDataset, DUKE_ORIGINAL_LABEL_NAMES
    from IMC.tensorboard_logging import setup_combined_logging
    import time 
    import argparse
    import pandas as pd
    import json
    from typing import List

    parser = argparse.ArgumentParser(description="Train MRI Sequence Classifier with 5-Fold CV")
    parser.add_argument("--backbone", type=str, default="densenet121", help="Image encoder backbone")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id to use")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs per fold")
    parser.add_argument("--modality", type=str, default="combined", help="Modality to use: 'combined', 'image', or 'metadata'")
    parser.add_argument("--image_naive_approach", action="store_true", help="Whether to use the naive approach for image-only model (single slice with 3 channels)")
    parser.add_argument("--metadata_enc_type", type=str, default='none', help="Which metadata encoder to use: 'none', 'imputer', 'sparse'")
    parser.add_argument("--sparse_enc_version", type=str, default="v1", help="Sparse encoder version: v1 or v2")
    parser.add_argument("--fusion_module_version", type=str, default="v1", help="Fusion module version: v1 or v2")
    parser.add_argument("--n_slices", type=int, default=3, help="Number of slices to use from the MRI sequence")
    args = parser.parse_args()

    # Check the validity of command-line arguments
    def validate_str_arguments(value: str, options: List[str], name: str) -> None:
        """
        Validate command-line arguments for training configuration.
        
        Args:
            value: The argument value to validate.
            options: List of valid options for the argument.
            name: The name of the argument (for error messages).
            
        Raises:
            ValueError: If any argument value is invalid.
        """
        
        if value not in options:
            raise ValueError(f"Invalid {name} '{value}'. Valid options are: {options}")
        return value


    modality = validate_str_arguments(args.modality, ["combined", "image", "metadata"], "modality")
    metadata_enc_type = validate_str_arguments(args.metadata_enc_type, ["none", "imputer", "sparse"], "metadata_enc_type")
    sparse_enc_version = validate_str_arguments(args.sparse_enc_version, ["v1", "v2", "v5"], "sparse_enc_version")
    fusion_module_version = validate_str_arguments(args.fusion_module_version, ["v1", "v2", "concat"], "fusion_module_version")

    # Parse which folds to run
    folds_to_run = [0, 1, 2, 3, 4]
    n_folds = 5
    
    # Validate fold indices
    for fold in folds_to_run:
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"Invalid fold index {fold}. Must be in range [0, {n_folds-1}]")

    # Directory for all experiments
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join("./logs", f"{timestamp}_5fold_cv")
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Configuration
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = 1e-6
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    incl_regression = False  # Whether to include regression task in the multi-task head
    
    # Dataset configuration
    num_samples = None  # Set to None to use full dataset
    aggregated_metadata = False
    use_preselected_features = False
    exclude_contrast_yn = False
    
    # Store results for all folds
    all_fold_results = []

    print("="*80)
    print(f"Starting 5-Fold Cross-Validation Training")
    print(f"Folds to train: {folds_to_run}")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs per fold: {num_epochs}")
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
        experiment_name = f"fold_{fold_idx}_baseline_multi_slices_04_{args.backbone}_sparse_encoder_{sparse_enc_version}_fusion_{fusion_module_version}"
        
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
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "incl_regression": incl_regression,
            "modality": args.modality,
            "metadata_enc_type": metadata_enc_type,
            "backbone": args.backbone,
            "sparse_encoder_version": sparse_enc_version,
            "fusion_module_version": fusion_module_version,
            "n_slices": args.n_slices
        }
        # Save config to JSON for reproducibility
        with open(os.path.join(fold_log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        log_training_start(logger, config=config)

        with capture_console_to_log(logger):
            # Create datasets and dataloaders using fold column
            print("Creating dataloaders...")
            
            # Train loader
            train_dataset = LiverDataset(
                split=[f"fold_{f}" for f in train_folds],  # Use fold naming convention
                num_samples=num_samples,
                augment_conf="NONE2D",
                aggregated_metadata=aggregated_metadata,
                use_preselected_features=use_preselected_features,
                exclude_contrast_yn=exclude_contrast_yn,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
                sampling_type="random",
                n_slices=args.n_slices
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            
            # Validation loader
            val_dataset = LiverDataset(
                split=[f"fold_{val_fold}"],
                num_samples=num_samples,
                augment_conf="NONE2D",
                aggregated_metadata=aggregated_metadata,
                use_preselected_features=use_preselected_features,
                exclude_contrast_yn=exclude_contrast_yn,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
                sampling_type="equidistant",
                n_slices=args.n_slices
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Test loader
            test_dataset = LiverDataset(
                split=[f"fold_{test_fold}"],
                num_samples=num_samples,
                augment_conf="NONE2D",
                aggregated_metadata=aggregated_metadata,
                use_preselected_features=use_preselected_features,
                exclude_contrast_yn=exclude_contrast_yn,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
                sampling_type="equidistant",
                n_slices=args.n_slices
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
            
            cl_d = train_dataset.get_n_labels()
            print("Label config:", cl_d)

            # Create model
            if modality == "image":
                if args.image_naive_approach:
                    model = SingleChannelImageBasedClassifier(num_classes_dict=cl_d)
                else:
                    model = ImageBasedClassifier(num_classes_dict=cl_d, backbone=args.backbone)
            elif modality == "metadata":
                if metadata_enc_type == "none":
                    model = MetadataBasedClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_encoder_type=metadata_enc_type,
                        imputer_type="contextual",
                        incl_regression=incl_regression
                    )
                elif metadata_enc_type == "sparse":
                    if sparse_enc_version == "v1":
                        model = MetadataBasedClassifier(
                            metadata_input_dim=train_dataset.num_metadata_features,
                            num_classes_dict=cl_d,
                            metadata_encoder_type="sparse",
                            incl_regression=incl_regression
                        )
                    else:
                        model = MetadataBasedClassifier(
                            metadata_input_dim=train_dataset.num_metadata_features,
                            num_classes_dict=cl_d,
                            metadata_encoder_type="sparse_v2",
                            incl_regression=incl_regression
                        )
                else:
                    # Imputer is NanIgnorer
                    model = MetadataBasedClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_encoder_type="imputer",
                        imputer_type="ignore",
                        incl_regression=incl_regression
                    )
            else:
                # Combined model with cross-attention fusion
                if metadata_enc_type == "none":
                    model = MRISequenceClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        img_enc_backbone=args.backbone,
                        incl_regression=incl_regression,
                        fusion_module_version=fusion_module_version,
                        imputer_type="ignore"
                    )
                elif metadata_enc_type == "imputer":
                    model = MRISequenceClassifier(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        img_enc_backbone=args.backbone,
                        incl_regression=incl_regression,
                        fusion_module_version=fusion_module_version,
                        imputer_type="contextual"
                    )
                elif metadata_enc_type == "sparse" and sparse_enc_version == "v1":
                    model = MRISequenceClassifierWithSparseMetadata(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_embeder_type="sparse",
                        img_enc_backbone=args.backbone,
                        dropout_metadata=False,
                        fusion_module_version=fusion_module_version,
                        include_regression=incl_regression
                    )
                elif metadata_enc_type == "sparse" and sparse_enc_version == "v2":
                    model = MRISequenceClassifierWithSparseMetadata(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_embeder_type="sparse_v2",
                        img_enc_backbone=args.backbone,
                        dropout_metadata=False,
                        fusion_module_version=fusion_module_version,
                        include_regression=incl_regression
                    )
                elif metadata_enc_type == "sparse" and sparse_enc_version == "v5":
                    model = MRISequenceClassifierWithSparseMetadata(
                        metadata_input_dim=train_dataset.num_metadata_features,
                        num_classes_dict=cl_d,
                        metadata_embeder_type="sparse_v5",
                        img_enc_backbone=args.backbone,
                        dropout_metadata=False,
                        fusion_module_version=fusion_module_version,
                        include_regression=incl_regression
                    )
                else:
                    raise ValueError(f"Invalid metadata_enc_type '{metadata_enc_type}' or sparse_enc_version '{sparse_enc_version}'")
            
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
            optimizer = create_optimizer(model, lr=lr, eps=eps)  
            scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
            criterion = MultiTaskLoss(label_smoothing=0.1, incl_regression=incl_regression, task_names=list(cl_d.keys()))
            scaler = torch.amp.GradScaler("cuda", init_scale=2**8)

            # Task weights
            task_weights = [1.0] * len(cl_d)

            # Create trainer
            from IMC.trainer import Trainer
            trainer = Trainer(
                model=model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                scaler=scaler,
                tb_logger=tb_logger,
                logger=logger,
                patience=30,
                task_weights=task_weights,
                incl_regression=incl_regression,
                use_mixed_precision=True
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