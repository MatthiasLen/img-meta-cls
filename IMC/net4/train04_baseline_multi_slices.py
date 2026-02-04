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
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/PV.AI"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/encoded_metadata_20251217.parquet"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20250603_local.csv"

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
    from IMC.network04 import MRISequenceClassifier
    from IMC.data.liver_dataloader_local import get_train_dataloader, get_valid_dataloader, get_test_dataloader
    from IMC.tensorboard_logging import setup_combined_logging
    import time 
    import argparse

    parser = argparse.ArgumentParser(description="Train MRI Sequence Classifier")
    parser.add_argument("--backbone", type=str, default="densenet", help="Image encoder backbone")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id to use")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    # Directory for profiling
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("./logs", timestamp)
    profiler_dir = os.path.join(log_dir, "profiler")
    os.makedirs(profiler_dir, exist_ok=True)
    experiment_name = f"baseline_model_multi_slices_04_{args.backbone}"

    # Setup combined logging (file + TensorBoard)
    logger, log_path, tb_logger = setup_combined_logging(
        experiment_name=experiment_name,
        log_dir=log_dir,
        tb_log_dir=log_dir
    )
    # Log training start
    batch_size = args.batch_size
    num_epochs = 15
    lr = 1e-6
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    incl_regression = True

    log_training_start(logger, config={"device": str(device), "batch_size": batch_size, "num_epochs": num_epochs, "dataset_version": "local", "model_version": "04", "impute": "yes", "learning_rate": lr, "checkpoint": args.ckpt})

    with capture_console_to_log(logger):
        num_samples = None # Set to None to use full dataset
        aggregated_metadata = False
        use_preselected_features = True
        exclude_contrast_yn = True
        train_loader = get_train_dataloader(batch_size=batch_size, num_samples=num_samples, num_workers=4, aggregated_metadata=aggregated_metadata, use_preselected_features=use_preselected_features, exclude_contrast_yn=exclude_contrast_yn)
        val_loader = get_valid_dataloader(batch_size=batch_size, num_samples=num_samples, num_workers=4, aggregated_metadata=aggregated_metadata, use_preselected_features=use_preselected_features, exclude_contrast_yn=exclude_contrast_yn)
        test_loader = get_test_dataloader(batch_size=batch_size, num_samples=num_samples, num_workers=4, aggregated_metadata=aggregated_metadata, use_preselected_features=use_preselected_features, exclude_contrast_yn=exclude_contrast_yn)
        cl_d = train_loader.dataset.get_n_labels()
        print("Label config", cl_d)

        metadata_input_dim = 32 if use_preselected_features else 119
        model = MRISequenceClassifier(metadata_input_dim=metadata_input_dim, num_classes_dict=cl_d, img_enc_backbone=args.backbone, incl_regression=incl_regression)
        if args.ckpt is not None:
            print(f"Loading checkpoint from {args.ckpt}")
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    
        # Initialize weights once before training, do NOT overwrite pretrained weights inside backbone
        model.apply(init_weights)

        # lr scheduler
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # warmup for 10% of total steps
        
        # optimizer
        eps = 1e-7
        optimizer = create_optimizer(model, lr=lr, eps=eps)  
        scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

        # loss
        criterion = MultiTaskLoss(label_smoothing=0.1, incl_regression=incl_regression, task_names=list(cl_d.keys()))

        # gradient scaler
        scaler = torch.amp.GradScaler("cuda", init_scale=2**8)

        # task weights 
        task_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weights for 7 tasks
        # task_weights[4] = 1.2
        # task_weights[5] = 1.1

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
            patience=5,
            task_weights=task_weights,
            incl_regression=incl_regression,
            use_mixed_precision=True
        )
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            save_path=os.path.join(log_dir, "best_model.pth")
        )
        # Load best model for testing
        trainer.load_checkpoint(os.path.join(log_dir, "best_model.pth"))
        test_acc = trainer.test(test_loader)
        print(f"Profiler results saved to {profiler_dir}. Run: tensorboard --logdir {profiler_dir}")

    # Log training end
    log_training_end(logger)