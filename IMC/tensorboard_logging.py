"""
TensorBoard logging utilities for IMC training.
These functions integrate with the existing logging system in helper.py.
"""

import os
import warnings
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


class TensorBoardLogger:
    """
    TensorBoard logging utility for training progress tracking.
    Integrates with the existing file logging system.
    """

    def __init__(self, log_dir=None, experiment_name="training", use_timestamp=True):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir (str): Directory for TensorBoard logs
            experiment_name (str): Name of the experiment
            use_timestamp (bool): Whether to add timestamp to log directory
        """
        self.writer = None
        self.enabled = TENSORBOARD_AVAILABLE

        if not self.enabled:
            warnings.warn(
                "TensorBoard not available. Install with: pip install tensorboard",
                stacklevel=2,
            )
            return

        if log_dir is None:
            log_dir = "./tb_logs"

        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        else:
            tb_log_dir = os.path.join(log_dir, experiment_name)

        self.log_dir = tb_log_dir
        os.makedirs(tb_log_dir, exist_ok=True)

        try:
            self.writer = SummaryWriter(tb_log_dir)
        except Exception as e:
            warnings.warn(f"Failed to initialize TensorBoard: {e}", stacklevel=2)
            self.enabled = False

    def log_scalar(self, tag, value, step):
        """Log a scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, scalar_dict, step):
        """Log multiple scalars in one plot."""
        if self.enabled and self.writer:
            self.writer.add_scalars(tag, scalar_dict, step)

    def log_histogram(self, tag, values, step, bins=50):
        """Log histogram of values."""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step, bins=bins)

    def log_image(self, tag, image, step):
        """Log an image tensor."""
        if self.enabled and self.writer:
            self.writer.add_image(tag, image, step)

    def log_images(self, tag, images, step):
        """Log a batch of images."""
        if self.enabled and self.writer:
            self.writer.add_images(tag, images, step)

    def log_model_graph(self, model, input_to_model):
        """Log the model graph."""
        if self.enabled and self.writer:
            try:
                self.writer.add_graph(model, input_to_model)
            except Exception as e:
                warnings.warn(f"Failed to log model graph: {e}", stacklevel=2)

    def log_hyperparameters(self, hparam_dict, metric_dict=None):
        """Log hyperparameters."""
        if self.enabled and self.writer:
            self.writer.add_hparams(hparam_dict, metric_dict or {})

    def log_text(self, tag, text, step):
        """Log text data."""
        if self.enabled and self.writer:
            self.writer.add_text(tag, text, step)

    def log_learning_rate(self, optimizer, step):
        """Log current learning rate."""
        if self.enabled and self.writer:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                self.writer.add_scalar(f"Learning_Rate/group_{i}", lr, step)

    def log_gradient_norms(self, model, step):
        """Log gradient norms for model parameters."""
        if self.enabled and self.writer:
            total_norm = 0.0
            param_count = 0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

                    # Log individual layer gradient norms (only for key layers to avoid clutter)
                    if any(key in name for key in ["encoder", "decoder", "attention", "fc", "classifier"]):
                        self.writer.add_scalar(f"Gradients/{name}", param_norm, step)

            if param_count > 0:
                total_norm = total_norm ** (1.0 / 2)
                self.writer.add_scalar("Gradients/total_norm", total_norm, step)

    def log_model_weights(self, model, step):
        """Log model weight histograms."""
        if self.enabled and self.writer:
            for name, param in model.named_parameters():
                if param.requires_grad and any(key in name for key in ["weight", "bias"]):
                    self.writer.add_histogram(f"Weights/{name}", param, step)

    def log_amp_scaler_info(self, scaler, step):
        """Log automatic mixed precision scaler information."""
        if self.enabled and self.writer:
            self.writer.add_scalar("Training/AMP_Scale", scaler.get_scale(), step)

    def flush(self):
        """Flush pending logs to disk."""
        if self.enabled and self.writer:
            self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_combined_logging(experiment_name="training", log_dir="./logs", tb_log_dir="./tb_logs", use_timestamp=True):
    """
    Setup both file logging and TensorBoard logging together.

    Args:
        experiment_name (str): Name of the experiment
        log_dir (str): Directory for text log files
        tb_log_dir (str): Directory for TensorBoard logs
        use_timestamp (bool): Whether to add timestamps to filenames/directories

    Returns:
        tuple: (logger, log_path, tb_logger) - File logger, log path, and TensorBoard logger
    """
    from IMC.helper import setup_experiment_logging

    # Setup file logging
    logger, log_path = setup_experiment_logging(experiment_name=experiment_name, log_dir=log_dir)

    # Setup TensorBoard logging
    tb_logger = TensorBoardLogger(log_dir=tb_log_dir, experiment_name=experiment_name, use_timestamp=use_timestamp)

    return logger, log_path, tb_logger


def log_training_metrics(
    tb_logger,
    epoch,
    train_loss,
    val_loss,
    train_accuracies=None,
    val_accuracies=None,
    individual_losses=None,
    optimizer=None,
    model=None,
    scaler=None,
):
    """
    Log training metrics to TensorBoard.

    Args:
        tb_logger: TensorBoard logger instance
        epoch (int): Current epoch number
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_accuracies (list, optional): Training accuracies per task
        val_accuracies (list, optional): Validation accuracies per task
        individual_losses (dict, optional): Individual task losses
        optimizer (optional): Optimizer to log learning rate
        model (optional): Model to log gradient norms
        scaler (optional): AMP scaler for mixed precision training
    """
    if not tb_logger.enabled:
        return

    # Log main losses
    tb_logger.log_scalars("Loss/Epoch", {"Train": train_loss, "Validation": val_loss}, epoch)

    # Log individual task accuracies
    if train_accuracies is not None:
        accuracy_dict = {}
        for i, acc in enumerate(train_accuracies):
            accuracy_dict[f"Task_{i}"] = acc
        tb_logger.log_scalars("Accuracy/Train", accuracy_dict, epoch)

    if val_accuracies is not None:
        accuracy_dict = {}
        for i, acc in enumerate(val_accuracies):
            accuracy_dict[f"Task_{i}"] = acc
        tb_logger.log_scalars("Accuracy/Validation", accuracy_dict, epoch)

    # Log individual task losses
    if individual_losses is not None:
        if "train" in individual_losses:
            loss_dict = {}
            for i, loss in enumerate(individual_losses["train"]):
                loss_dict[f"Task_{i}"] = loss
            tb_logger.log_scalars("Loss/Train_Tasks", loss_dict, epoch)

        if "val" in individual_losses:
            loss_dict = {}
            for i, loss in enumerate(individual_losses["val"]):
                loss_dict[f"Task_{i}"] = loss
            tb_logger.log_scalars("Loss/Val_Tasks", loss_dict, epoch)

    # Log learning rate
    if optimizer is not None:
        tb_logger.log_learning_rate(optimizer, epoch)

    # Log AMP scaler info
    if scaler is not None:
        tb_logger.log_amp_scaler_info(scaler, epoch)

    # Log gradient norms (only occasionally to avoid overhead)
    if model is not None and epoch % 5 == 0:
        tb_logger.log_gradient_norms(model, epoch)

    # Log model weights (only occasionally)
    if model is not None and epoch % 10 == 0:
        tb_logger.log_model_weights(model, epoch)

    tb_logger.flush()


def log_batch_metrics(
    tb_logger, batch_idx, total_batches, epoch, loss, individual_losses=None, accuracies=None, lr=None, amp_scale=None
):
    """
    Log batch-level metrics during training.

    Args:
        tb_logger: TensorBoard logger instance
        batch_idx (int): Current batch index
        total_batches (int): Total number of batches per epoch
        epoch (int): Current epoch
        loss (float): Current batch loss
        individual_losses (list, optional): Individual task losses for this batch
        accuracies (list, optional): Accuracies for this batch
        lr (float, optional): Current learning rate
        amp_scale (float, optional): AMP scaler value
    """
    if not tb_logger.enabled:
        return

    # Calculate global step
    global_step = epoch * total_batches + batch_idx

    # Log batch loss
    tb_logger.log_scalar("Loss/Batch", loss, global_step)

    # Log individual task losses
    if individual_losses is not None:
        for i, task_loss in enumerate(individual_losses):
            tb_logger.log_scalar(f"Loss/Batch_Task_{i}", task_loss, global_step)

    # Log batch accuracies
    if accuracies is not None:
        for i, acc in enumerate(accuracies):
            tb_logger.log_scalar(f"Accuracy/Batch_Task_{i}", acc, global_step)

    # Log learning rate
    if lr is not None:
        tb_logger.log_scalar("Learning_Rate/Batch", lr, global_step)

    # Log AMP scale
    if amp_scale is not None:
        tb_logger.log_scalar("Training/AMP_Scale_Batch", amp_scale, global_step)

    # Flush every few batches to see updates
    if batch_idx % 10 == 0:
        tb_logger.flush()
