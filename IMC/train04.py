"""
    TRAINIGN SCRIPT FOR MODEL VERSION 0.4
    2025/09/18

    Class to manage training with mixed precision and gradient scaling (via torch.amp.GradScaler),
    including saving and loading of checkpoints with scaler state. Early stopping and learning rate
    scheduling is included.

    This class handles the training loop, model optimization, and checkpointing,
    ensuring that the GradScaler's internal state is saved and restored correctly.
    This enables seamless resumption of mixed precision training without
    disrupting the dynamic loss scaling process.


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


class MultiTaskLoss(nn.Module):
    """
    Computes a combined multi-task loss for classification and binary tasks.

    This loss module calculates the sum of:
    - CrossEntropyLoss with label smoothing for multiple classification heads.
    - BCEWithLogitsLoss for binary classification head.

    Args:
        label_smoothing (float, optional): Label smoothing factor for CrossEntropyLoss. Defaults to 0.1.

    Inputs:
        preds (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
            Tuple containing logits for sequence, plane, body, and contrast predictions.
        targets (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
            Tuple containing ground-truth labels for sequence, plane, body, and contrast.

    Returns:
        Tuple[torch.Tensor, Tuple[float, float, float, float]]:
            - total_loss: combined scalar loss tensor.
            - individual_losses: tuple with individual losses (sequence, plane, body, contrast) as floats.
    """
    
    def __init__(self, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
  
        seq_logits, plane_logits, body_logits, contrast_logits = preds
        seq_t, plane_t, body_t, contrast_t = targets

        loss_seq = self.ce_loss(seq_logits, seq_t)
        loss_plane = self.ce_loss(plane_logits, plane_t)
        loss_body = self.ce_loss(body_logits, body_t)
        loss_contrast = self.bce_loss(contrast_logits.flatten(), contrast_t.float())

        total_loss = loss_seq + loss_plane + loss_body + loss_contrast
        return total_loss, (loss_seq.item(), loss_plane.item(), loss_body.item(), loss_contrast.item())


def get_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """
    Creates a learning rate scheduler with a linear warmup followed by cosine decay.

    The learning rate increases linearly from 0 to the optimizer's initial learning rate
    during the warmup phase. After warmup, it follows a cosine decay schedule until total_steps.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        warmup_steps (int): Number of steps for the linear warmup phase.
        total_steps (int): Total number of training steps for the schedule.

    Returns:
        LambdaLR: A PyTorch LambdaLR scheduler applying the warmup + cosine decay schedule.
    """

    def lr_lambda(current_step : int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))   

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4) -> Optimizer:
    """
    Creates an AdamW optimizer with parameter groups separating parameters that
    should and should not have weight decay applied.

    Typically, biases and normalization layer parameters (e.g., LayerNorm weights) are excluded
    from weight decay.

    Args:
        model (nn.Module): The model containing parameters to optimize.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        weight_decay (float, optional): Weight decay coefficient. Defaults to 1e-4.

    Returns:
        Optimizer: AdamW optimizer with appropriately grouped parameters.
    """

    decay, no_decay = [], []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        # Exclude bias, LayerNorm, BatchNorm from weight decay
        if any(nd in name.lower() for nd in ["bias", "norm", "ln", "layernorm", "bn"]):
            no_decay.append(param)
        else:
            decay.append(param)

    return AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    save_path: Union[str, os.PathLike],
    warmup_steps: int = 1000,
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
        warmup_steps (int, optional): Number of steps to linearly warm up the learning rate. Default is 1000.

    Outputs and result handling:
        Saves the best model state dict to `save_path`.
        Prints training and validation loss per epoch.
        Stops training early if validation loss does not improve for 5 consecutive epochs.
    """

    model.to(device)
    
    # Initialize weights once before training, do NOT overwrite pretrained weights inside backbone
    model.apply(init_weights)

    optimizer = create_optimizer(model)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    criterion = MultiTaskLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler("cuda", init_scale=2**16)
    best_val_loss = float("inf")
    last_scale = float("inf")
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0

        # Track individual losses for logging
        train_losses_seq, train_losses_plane, train_losses_body, train_losses_contrast = [], [], [], []

        # Training loop for current epoch
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            
            # unpack batch
            images, metadata, targets = batch
            images = images.to(device)
            metadata = metadata.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = model(images, metadata)
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
            if last_scale <= (current_scale * 1.001):
                scheduler.step()
            last_scale  = current_scale
            
            # epoch training loss
            train_loss_accum += loss.item()
            
            # Unpack individual losses
            loss_seq, loss_plane, loss_body, loss_contrast = indiv_losses
            train_losses_seq.append(loss_seq)
            train_losses_plane.append(loss_plane)
            train_losses_body.append(loss_body)
            train_losses_contrast.append(loss_contrast)

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_seq = sum(train_losses_seq) / len(train_losses_seq)
        avg_plane = sum(train_losses_plane) / len(train_losses_plane)
        avg_body = sum(train_losses_body) / len(train_losses_body)
        avg_contrast = sum(train_losses_contrast) / len(train_losses_contrast)

        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} "
              f"(Seq: {avg_seq:.4f}, Plane: {avg_plane:.4f}, Body: {avg_body:.4f}, Contrast: {avg_contrast:.4f})")

        # Validation step
        model.eval()
        val_loss_accum = 0.0
        val_losses_seq, val_losses_plane, val_losses_body, val_losses_contrast = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                images, metadata, targets = batch
                images = images.to(device)
                metadata = metadata.to(device)
                targets = [t.to(device) for t in targets]

                with torch.autocast("cuda"):
                    outputs = model(images, metadata)
                    loss, indiv_losses = criterion(outputs, targets)

                val_loss_accum += loss.item()
                loss_seq, loss_plane, loss_body, loss_contrast = indiv_losses
                val_losses_seq.append(loss_seq)
                val_losses_plane.append(loss_plane)
                val_losses_body.append(loss_body)
                val_losses_contrast.append(loss_contrast)

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_seq = sum(val_losses_seq) / len(val_losses_seq)
        avg_val_plane = sum(val_losses_plane) / len(val_losses_plane)
        avg_val_body = sum(val_losses_body) / len(val_losses_body)
        avg_val_contrast = sum(val_losses_contrast) / len(val_losses_contrast)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f} "
              f"(Seq: {avg_val_seq:.4f}, Plane: {avg_val_plane:.4f}, Body: {avg_val_body:.4f}, Contrast: {avg_val_contrast:.4f})")

        # Early stopping & checkpointing
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

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        # Optionally log LR
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current LR: {current_lr:.6e} | Current Scale: {last_scale:.6e}")

    print("Training complete.")


# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from network04 import MRISequenceClassifier
    from dummy_dataloader import DummyMRIDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    cl_d = {"sequence": 5, "plane": 3, "body": 5, "contrast": 1}

    dummy_dataset_train = DummyMRIDataset(img_size = 300, num_samples=200,  n_slices = 5, metadata_dim = 3*256, num_classes_dict=cl_d)
    dummy_loader_train = DataLoader(dummy_dataset_train, batch_size=8,shuffle=True)

    dummy_dataset_val = DummyMRIDataset(img_size = 300, num_samples=50,  n_slices = 5, metadata_dim = 3*256,  num_classes_dict=cl_d)
    dummy_loader_val = DataLoader(dummy_dataset_val, batch_size=8,shuffle=True)

    # Assume you have your dataset and dataloaders ready: train_loader, val_loader
    # Each batch from dataloader returns: images (B,N,C,H,W), metadata (B,meta_dim), targets = (seq_t, plane_t, body_t, contrast_t)

    model = MRISequenceClassifier(metadata_input_dim=256*3, num_classes_dict=cl_d)

    train_loop(
        model=model,
        train_loader=dummy_loader_train,
        val_loader=dummy_loader_val,
        num_epochs=50,
        device=device,
        save_path="best_model.pth",
        warmup_steps=500,
    )
