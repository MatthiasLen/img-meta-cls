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

from helper import plot_batch_per_sample, normalize_per_sample, count_parameters
from nn.multi_task_loss import MultiTaskLoss

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


def classification_losses(outputs: list , targets: list) -> list:
    """
    Calculates and prints the accuracy for each classification task.
    
    Args:
        outputs (list of torch.Tensor): List of model outputs for each task, where each output is a tensor of class scores.
        targets (list of torch.Tensor): List of ground truth labels for each task.
    Raises:
        AssertionError: If the number of outputs and targets do not match.
    Returns:
        list with task accuracies
        
    """    
    assert len(outputs) == len(targets)
    accu_list = []
    
    # iterate over tasks
    for i, (output, target) in enumerate(zip(outputs, targets)):
        pred = output.clone().detach().cpu().numpy()
        target_cl = target.clone().detach().cpu().numpy()
        pred_cl = np.argmax(pred, axis=1)
        accuracy = np.mean(pred_cl == target_cl)
        accu_list.append(accuracy)
        print(f"Task {i} Accuracy:", accuracy)
       
    return accu_list


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    save_path: Union[str, os.PathLike],
) -> None:
    """
    Trains a PyTorch model with mixed precision, multi-task loss, and learning rate scheduling.

    This function performs training and validation over multiple epochs, using:
    - Mixed precision training with torch.cuda.amp.GradScaler for efficiency.
    - AdamW optimizer with separate weight decay handling.
    - Cosine learning rate scheduler with warmup.
    - Multi-task loss combining cross-entropy and binary cross-entropy losses.
    - Gradient clipping to stabilize training.
    - Early stopping based on validation loss.
    - Checkpointing to save the best model state.

    Args:
        model (torch.nn.Module): The model to train. Expected to output a tuple of logits per task.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Maximum number of training epochs.
        device (torch.device): Device to run training on (CPU or CUDA).
        save_path (Union[str, os.PathLike]): File path to save the best model checkpoint.

    Outputs and result handling:
        Saves the best model state dict to `save_path`.
        Prints training and validation loss per epoch.
        Stops training early if validation loss does not improve for 5 consecutive epochs.
    """

    model.to(device)
    
    # Initialize weights once before training, do NOT overwrite pretrained weights inside backbone
    model.apply(init_weights)

    # lr scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # warmup for 10% of total steps
    
    # optimizer
    optimizer = create_optimizer(model, lr=1.0e-6)  
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    # loss
    criterion = MultiTaskLoss(label_smoothing=0.1)

    # gradient scaler
    scaler = torch.amp.GradScaler("cuda", init_scale=2**16)

    best_val_loss = float("inf")
    last_scale = float("inf")
    patience = 5
    epochs_no_improve = 0

    print(f"Model parameters (total, req. gradient): {count_parameters(model)}")

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0

        # Track individual losses for logging
        train_losses = []

        batch_id = 0
        
        # --- Training loop for current epoch ---
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            batch_id += 1
            
            # unpack batch
            images, metadata, targets = batch
            images = images.to(device, non_blocking=True) # TODO: profile
            metadata = metadata.to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in targets]

            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):     
                # Normalize per sample 
                images = normalize_per_sample(images)

                # Plot batch
                #plot_batch_per_sample(images, title="Each Row = One Sample (5 Images)", id = f"ep{epoch}_b{batch_id}")

                # Forward
                outputs = model(images, metadata)

                # Compute losses
                loss, indiv_losses = criterion(outputs, targets)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients
            scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Scaler.step() first unscales the gradients of the optimizer's assigned params. If already unscaled this is skipped.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called. Otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            scaler.update()

            current_scale = scaler.get_scale()
            
            # only do scheduler step if scaling is stable
            #if last_scale <= (current_scale * 1.001):
            #    scheduler.step()
            last_scale  = current_scale
            scheduler.step()

            # Optionally log LR and scale
            print(f"Current Scale: {last_scale:.6e}")
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print(f"Current Learning Rate: {current_lr:.2e}")
            print("Sched. LR", scheduler.get_last_lr()[0])
            
            # epoch training loss
            train_loss_accum += loss.item()
            
            # Unpack individual losses
            train_losses.append(indiv_losses)

            classification_losses(outputs, targets)

            #print(train_losses)
            #print(f"Current task-specific average losses: {np.average(train_losses, axis=0)}")

        # average losses over all batches from training loop
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_ind = np.average(train_losses, axis=0)

        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} "
              f"(Individual task losses: {avg_ind})")

        # --- Validation step ---
        model.eval()
        val_loss_accum = 0.0
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                images, metadata, targets = batch
                images = images.to(device)
                metadata = metadata.to(device)
                targets = [t.to(device) for t in targets]

                with torch.autocast("cuda"):
                    # Normalize per sample 
                    images = normalize_per_sample(images)
                
                    outputs = model(images, metadata)
                    loss, indiv_losses = criterion(outputs, targets)

                val_loss_accum += loss.item()
                val_losses.append(indiv_losses)

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_ind = np.average(val_losses, axis=0)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f} "
              f"(Individual losses: {avg_ind})")

        # --- Early stopping & checkpointing ---
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model.")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if (epochs_no_improve >= patience) and (epoch >= 0.7 * num_epochs):
            print("Early stopping triggered.")
            break

    print("Training complete.")


# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from network05 import UnifiedTransformerModel
    from data.liver_dataloader01 import LiverDataset
    import torch.profiler
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING ON DEVICE: {device} ===")

    # liver dataset
    dummy_dataset = LiverDataset(num_samples=1024, n_slices=5, label_path="~/pvai_labels_20250603.csv")
    dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)
    cl_d = dummy_dataset.get_n_labels()
    print("Label config", cl_d)
    
    # initialize model
    model = UnifiedTransformerModel(
        metadata_input_dim=89,
        metadata_embed_dim=128,
        transformer_dim=256,
        num_transformer_layers=4,
        num_heads=8,
        num_classes_dict=cl_d
    )



    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # run training
        train_loop(
            model=model,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            num_epochs=50,
            device=device,
            save_path="best_model.pth"
        )
