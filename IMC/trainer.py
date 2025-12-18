import time
import math
import numpy as np
from sklearn import logger
import torch
from typing import Union
from torch.utils.data import DataLoader

from IMC.helper import normalize_per_sample
from IMC.tensorboard_logging import log_batch_metrics, log_training_metrics
from pathlib import Path

import os

DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

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
                 patience: int = 5,
                 img_ft_only: bool = False,
                 task_weights: Union[list, None] = None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = scaler
        self.tb_logger = tb_logger
        self.logger = logger
        self.patience = patience
        self.img_ft_only = img_ft_only
        self.task_weights = task_weights if task_weights is not None else [1.0] * 7

    def _classification_accuracies(self, outputs, targets, masks):
        """Compute per-task accuracies from logits and targets."""
        assert len(outputs) == len(targets)
        accu_list = []
        for output, target, mask in zip(outputs, targets, masks):
            pred = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            pred_cls = np.argmax(pred, axis=1)
            # Only compute accuracy for masked (valid) targets
            valid_indices = mask_np.astype(bool)
            if valid_indices.sum() > 0:  # Avoid division by zero
                accuracy = np.mean(pred_cls[valid_indices] == target_np[valid_indices])
            else:
                accuracy = 0.0
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

            images, metadata, targets, masks = batch

            images = images.to(self.device)
            if not self.img_ft_only:
                metadata = metadata.to(self.device)
            targets = [t.to(self.device) for t in targets]
            masks = [m.to(self.device) for m in masks]


            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                images = normalize_per_sample(images)

                if not self.img_ft_only:
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                
            loss, indiv_losses = self.criterion(outputs, targets, masks, self.task_weights)

            # Check for NaN in loss and outputs

            if torch.isnan(loss).any():
                self.logger.error(f"NaN detected in loss: {loss}")
                self.logger.error(f"Outputs contain NaN: {[torch.isnan(output).any().item() for output in outputs]}")
                raise RuntimeError("NaN loss detected - stopping training")
            
            if any(torch.isnan(output).any() for output in outputs):
                self.logger.error("NaN detected in model outputs")
                self.logger.error(f"Input images stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")
                self.logger.error(f"Input metadata stats: min={metadata.min():.3f}, max={metadata.max():.3f}, mean={metadata.mean():.3f}")
                raise RuntimeError("NaN outputs detected - stopping training")    
            
            self.scaler.scale(loss).backward()
            
            # Check for NaN in gradients
            if DEBUG_MODE:
                self.logger.info("Check gradient after backward")
                has_nan_grads = False
                max_grad_norm = 0.0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            logger.error(f"NaN gradient detected in parameter: {name}")
                            has_nan_grads = True
                        if torch.isinf(param.grad).any():
                            logger.error(f"Inf gradient detected in parameter: {name}")
                            has_nan_grads = True
                        max_grad_norm = max(max_grad_norm, param.grad.norm().item())
                        
                if has_nan_grads:
                    logger.error("Invalid gradients detected - skipping step")

                # Log gradient norms periodically
                if batch_id % 50 == 0:
                    logger.info(f"Max gradient norm: {max_grad_norm:.4f}")

            self.scaler.unscale_(self.optimizer)
                
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            current_scale = self.scaler.get_scale()
            last_scale = current_scale

            # Optional prints (reduced frequency to avoid log spam)
            if self.logger and batch_id % 100 == 0:  # Log every 100 batches instead of every batch
                self.logger.info(f"Batch {batch_id}: Scale: {last_scale:.6e}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
            # Log scale issues
            if last_scale < 1.0:
                self.logger.warning(f"Gradient scale very low: {last_scale:.6e} - possible gradient overflow")
            elif last_scale > 1e6:
                self.logger.warning(f"Gradient scale very high: {last_scale:.6e} - possible gradient underflow")

            # Accumulate
            train_loss_accum += loss.item()
            train_losses.append(indiv_losses)

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
                images, metadata, targets, masks = batch
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = [t.to(self.device) for t in targets]
                masks = [m.to(self.device) for m in masks]

                with torch.autocast("cuda"):
                    images = normalize_per_sample(images)
                    if not self.img_ft_only:
                        outputs = self.model(images, metadata)
                    else:
                        outputs = self.model(images)
                    loss, indiv_losses = self.criterion(outputs, targets, masks, self.task_weights)
                    val_acc.append(self._classification_accuracies(outputs, targets, masks))

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
                    self.logger.info(f"Saving latest model checkpoint.")
                # Save the current model as the latest checkpoint
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict()
                }, Path(save_path).with_name("latest_model.pth"))

            if (epochs_no_improve >= self.patience) and (epoch >= 0.7 * num_epochs):
                if self.logger:
                    self.logger.info("Early stopping triggered.")
                break

        if self.logger:
            self.logger.info("Training complete.")

    def test(self, test_loader: DataLoader):
        self.model.eval()
        test_acc = []

        with torch.no_grad():
            for batch in test_loader:
                images, metadata, targets, masks = batch
                images = images.to(self.device)
                if not self.img_ft_only:
                    metadata = metadata.to(self.device)
                targets = [t.to(self.device) for t in targets]
                masks = [m.to(self.device) for m in masks]

                with torch.amp.autocast("cuda"):
                    images = normalize_per_sample(images)
                    if not self.img_ft_only:
                        outputs = self.model(images, metadata)
                    else:
                        outputs = self.model(images)
                    test_acc.append(self._classification_accuracies(outputs, targets, masks))


        avg_test_acc = np.mean(test_acc, axis=0)

        if self.logger:
            self.logger.info(f"Test Accuracy: {avg_test_acc}")

        if self.tb_logger and getattr(self.tb_logger, 'enabled', False):
            self.tb_logger.log_scalar('Test/Accuracy', float(np.mean(avg_test_acc)), 0)
            for i, acc in enumerate(avg_test_acc):
                self.tb_logger.log_scalar(f'Test/Task_{i}_Accuracy', float(acc), 0)
            self.tb_logger.flush()
        return avg_test_acc
    
    def load_checkpoint(self, checkpoint_path: Union[str, bytes]):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

