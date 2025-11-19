"""
    TRAINING SCRIPT FOR MODEL VERSION 0.5
    2025/10/01

    Class to manage training with mixed precision and gradient scaling (via torch.amp.GradScaler),
    including saving and loading of checkpoints with scaler state. Early stopping and learning rate
    scheduling is included.

    This class handles the training loop, model optimization, and checkpointing,
    ensuring that the GradScaler's internal state is saved correctly.
    This enables seamless resumption of mixed precision training without
    disrupting the dynamic loss scaling process.

"""

from typing import Union
import math
import os
from tqdm import tqdm
import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from IMC.helper import capture_console_to_log, log_training_start, log_training_end
from IMC.tensorboard_logging import setup_combined_logging
from IMC.nn.multi_task_loss import MultiTaskLoss

# Set environment variables or paths for local dataset
os.environ["DEBUG_MODE"] = "1"  # Enable debug mode
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/PV.AI"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20251114_encoded_local.csv"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20250603_local.csv"

from IMC.data.liver_dataloader_local import get_train_dataloader, get_valid_dataloader, get_test_dataloader


# Directory for profiling
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("./logs", timestamp)
profiler_dir = os.path.join(log_dir, "profiler")
os.makedirs(profiler_dir, exist_ok=True)
experiment_name = "baseline_model_05"

# Setup combined logging (file + TensorBoard)
logger, log_path, tb_logger = setup_combined_logging(
    experiment_name=experiment_name,
    log_dir=log_dir,
    tb_log_dir=log_dir
)

def init_weights(module: nn.Module) -> None:
    """
    Custom weight initialization for neural network layers.

    Applies Xavier (Glorot) uniform initialization to Linear layers.
    Sets LayerNorm weights to 1 and biases to 0.
    Useful for consistent initialization when using custom models.

    Args:
        module (nn.Module): A layer or submodule of the model to initialize.
    
    Usage:
        model.apply(init_weights)
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
    peak_scale_factor : float = 100,
    min_scale_factor : float = 0.1
) -> LambdaLR:
    """
    Warmup + cosine decay scheduler with learning rates expressed as ratios of initial LR.

    Args:
        optimizer: Optimizer whose lr will be scheduled.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of steps.
    Returns:
        LambdaLR scheduler.
    """
    
    def lr_lambda(current_step: int) -> float:
        if (current_step <= warmup_steps) and (warmup_steps > 0):
            return max(1, peak_scale_factor * float(current_step) / warmup_steps)
        elif  (current_step > warmup_steps) and (current_step <= total_steps):
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            a = math.cos(math.pi * progress)
            return max(min_scale_factor, peak_scale_factor * 0.5 * (1 + a))
        else:
            return min_scale_factor

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer(model: torch.nn.Module, lr: float = 1.0e-6, weight_decay: float = 1e-2) -> Optimizer:
    """
    AdamW optimizer with separate weight decay for bias and norm layers.

    Args:
        model: Model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay.
    Returns:
        AdamW optimizer.
        
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

    return AdamW([{"params": decay, "weight_decay": weight_decay},{"params": no_decay, "weight_decay": 0.0}],lr=lr)

# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from IMC.network05 import UnifiedTransformerModel
    import torch.profiler
    from torch.profiler import profile, tensorboard_trace_handler
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING ON DEVICE: {device} ===")

    # Log training start
    batch_size = 16
    num_epochs = 15
    log_training_start(logger, config={"device": str(device), "batch_size": batch_size, "num_epochs": num_epochs, "dataset_version": "local", "model_version": "05", "impute": "no"})

    with capture_console_to_log(logger):
        train_loader = get_train_dataloader(batch_size=16, num_samples=None, num_workers=4)
        val_loader = get_valid_dataloader(batch_size=16, num_samples=None, num_workers=4)
        test_loader = get_test_dataloader(batch_size=16, num_samples=None, num_workers=4)
        cl_d = train_loader.dataset.get_n_labels()
        print("Label config", cl_d)

        # initialize model
        model = UnifiedTransformerModel(
            metadata_input_dim=88,
            metadata_embed_dim=128,
            transformer_dim=256,
            num_transformer_layers=4,
            num_heads=8,
            num_classes_dict=cl_d
        )

        # Build optimizer/scheduler/criterion/scaler
        optimizer = create_optimizer(model, lr=1.0e-7)
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
        criterion = MultiTaskLoss(label_smoothing=0.1)
        scaler = torch.amp.GradScaler("cuda", init_scale=2**16)

        # Use reusable Trainer
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

