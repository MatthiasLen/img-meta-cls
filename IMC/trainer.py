import time
import math
import numpy as np
import torch
from typing import Union
from torch.utils.data import DataLoader

from IMC.helper import normalize_per_sample
from IMC.tensorboard_logging import log_batch_metrics, log_training_metrics

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

class Trainer:
    """
    Reusable Trainer for IMC experiments with mixed precision, multi-task loss,
    LR scheduling, gradient clipping, checkpointing, early stopping, and TensorBoard logging.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 criterion,
                 scaler: torch.amp.GradScaler,
                 tb_logger=None,
                 logger=None,
                 patience: int = 5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = scaler
        self.tb_logger = tb_logger
        self.logger = logger
        self.patience = patience

    def _classification_accuracies(self, outputs, targets):
        """Compute per-task accuracies from logits and targets."""
        assert len(outputs) == len(targets)
        accu_list = []
        for output, target in zip(outputs, targets):
            pred = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            pred_cls = np.argmax(pred, axis=1)
            accuracy = np.mean(pred_cls == target_np)
            accu_list.append(float(accuracy))
        return accu_list

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        train_loss_accum = 0.0
        train_losses = []

        batch_id = 0
        last_scale = self.scaler.get_scale()

        for batch in train_loader:
            batch_id += 1

            timings = []
            timings.append((time.time(), "start"))

            images, metadata, targets = batch
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            targets = [t.to(self.device) for t in targets]

            timings.append((time.time(), "to(device)"))

            self.optimizer.zero_grad()
            timings.append((time.time(), "zero_grad"))

            with torch.amp.autocast("cuda"):
                images = normalize_per_sample(images)
                timings.append((time.time(), "normalize_per_sample"))

                outputs = self.model(images, metadata)

                loss, indiv_losses = self.criterion(outputs, targets)
                timings.append((time.time(), "forward+loss"))

            self.scaler.scale(loss).backward()
            timings.append((time.time(), "backward"))

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            timings.append((time.time(), "step+update"))

            current_scale = self.scaler.get_scale()
            last_scale = current_scale
            self.scheduler.step()

            # Optional prints
            if self.logger:
                self.logger.info(f"Current Scale: {last_scale:.6e}")
                for pg in self.optimizer.param_groups:
                    self.logger.info(f"Current LR: {pg['lr']:.2e}")
                self.logger.info(f"Sched. LR {self.scheduler.get_last_lr()[0]:.2e}")

            # Accumulate
            train_loss_accum += loss.item()
            train_losses.append(indiv_losses)

            # Batch TB logging
            if self.tb_logger and getattr(self.tb_logger, 'enabled', False):
                # Prepare individual losses list
                indiv_list = []
                if isinstance(indiv_losses, (list, tuple)):
                    indiv_list = [float(x) for x in indiv_losses]
                elif hasattr(indiv_losses, 'detach'):
                    try:
                        indiv_list = [float(x) for x in indiv_losses.detach().cpu().numpy().tolist()]
                    except Exception:
                        pass

                current_lr_tb = self.optimizer.param_groups[0]['lr']
                amp_scale_tb = current_scale

                log_batch_metrics(
                    self.tb_logger,
                    batch_idx=batch_id,
                    total_batches=len(train_loader),
                    epoch=epoch,
                    loss=float(loss.item()),
                    individual_losses=indiv_list,
                    accuracies=None,
                    lr=current_lr_tb,
                    amp_scale=amp_scale_tb
                )

        avg_train_loss = train_loss_accum / max(1, len(train_loader))
        avg_ind_train = np.average(train_losses, axis=0)
        return avg_train_loss, avg_ind_train

    def validate_epoch(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        val_loss_accum = 0.0
        val_losses = []
        val_acc = []

        with torch.no_grad():
            for batch in val_loader:
                images, metadata, targets = batch
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = [t.to(self.device) for t in targets]

                with torch.autocast("cuda"):
                    images = normalize_per_sample(images)
                    outputs = self.model(images, metadata)
                    loss, indiv_losses = self.criterion(outputs, targets)
                    val_acc.append(self._classification_accuracies(outputs, targets))

                val_loss_accum += loss.item()
                val_losses.append(indiv_losses)

        avg_val_loss = val_loss_accum / max(1, len(val_loader))
        avg_ind_val = np.average(val_losses, axis=0)
        avg_val_acc = np.mean(val_acc, axis=0)

        return avg_val_loss, avg_ind_val, avg_val_acc

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
            save_path: Union[str, bytes]):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            avg_train_loss, avg_ind_train = self.train_epoch(train_loader, epoch)
            avg_val_loss, avg_ind_val, avg_val_acc = self.validate_epoch(val_loader, epoch)

            if self.logger:
                self.logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} (Task losses: {avg_ind_train})")
                self.logger.info(f"Epoch {epoch+1} Val   Loss: {avg_val_loss:.4f} (Task losses: {avg_ind_val})")

            # TB epoch logging
            if self.tb_logger and getattr(self.tb_logger, 'enabled', False):
                indiv_epoch = {
                    'train': [float(x) for x in avg_ind_train],
                    'val': [float(x) for x in avg_ind_val]
                }
                log_training_metrics(
                    self.tb_logger,
                    epoch=epoch,
                    train_loss=float(avg_train_loss),
                    val_loss=float(avg_val_loss),
                    train_accuracies=None,
                    val_accuracies=avg_val_acc,
                    individual_losses=indiv_epoch,
                    optimizer=self.optimizer,
                    model=self.model,
                    scaler=self.scaler
                )

            # Early stopping + checkpoint
            if avg_val_loss < best_val_loss:
                if self.logger:
                    self.logger.info(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model.")
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict()
                }, save_path)
            else:
                epochs_no_improve += 1
                if self.logger:
                    self.logger.info(f"No improvement for {epochs_no_improve} epochs.")

            if (epochs_no_improve >= self.patience) and (epoch >= 0.7 * num_epochs):
                if self.logger:
                    self.logger.info("Early stopping triggered.")
                break

        if self.logger:
            self.logger.info("Training complete.")
