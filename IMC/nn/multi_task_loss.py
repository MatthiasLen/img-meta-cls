import torch
import torch.nn as nn
import logging
import os
from typing import List

logger = logging.getLogger('IMC')
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
class MultiTaskLoss(torch.nn.Module):
    """
    A flexible multi-task loss function for medical image classification.

    This module computes a combined loss for a model with multiple output heads,
    supporting a mix of classification, binary, and regression tasks. It is designed
    to work seamlessly with the `MultiTaskHead`.

    Key Features:
    - **Handles Mixed Task Types**:
      - **Multi-class Classification**: Uses `CrossEntropyLoss` with optional label smoothing.
      - **Binary Classification**: Uses `BCEWithLogitsLoss` for heads with a single output logit.
      - **Regression**: Uses `MSELoss` for designated regression tasks.
    - **Label Masking**: Accepts a `masks` tuple to selectively exclude samples from the
      loss calculation for each task. This is crucial for handling missing or invalid
      labels (e.g., "na" values) in a multi-task setting.
    - **Task Weighting**: Allows individual task losses to be weighted, providing control
      over the contribution of each task to the final loss.
    - **Debug Mode**: Includes checks for NaN or infinity values in the loss, which can
      help diagnose training instability.

    The total loss is the weighted sum of the individual task losses.
    """
    
    def __init__(self, label_smoothing: float = 0.1, incl_regression: bool = True, task_names: List[str] = None) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = nn.MSELoss()
        self.incl_regression = incl_regression
        self.task_names = task_names

    def forward(self, preds: tuple, targets: tuple, masks: tuple, task_weights: List[float] | None = None) -> tuple[torch.Tensor, list]:
        """
        Calculates the multi-task loss.

        Args:
            preds (tuple): A tuple of tensors, where each tensor contains the model's
                           predictions (logits) for a specific task.
            targets (tuple): A tuple of tensors, where each tensor contains the ground-truth
                             labels for a specific task.
            masks (tuple): A tuple of boolean tensors. A `True` value indicates that the
                           corresponding label is valid and should be included in the loss.
            task_weights (list, optional): A list of weights to apply to each task's loss.
                                           If None, all tasks are weighted equally. Defaults to None.

        Returns:
            tuple:
                - total_loss (torch.Tensor): The final, combined loss as a scalar tensor.
                - individual_losses (list): A list of the individual loss values for each task.
        """
        individual_losses = []
        total_loss = 0.0

        for i, (pred, target, mask, task_name) in enumerate(zip(preds, targets, masks, self.task_names)):
            # Determine if there are any valid samples for this task in the batch
            valid_samples_mask = mask.bool()
            if not valid_samples_mask.any():
                # No valid samples for this task, so loss is 0
                task_loss = torch.tensor(0.0, device=pred.device)
            else:
                # Filter predictions and targets to only include valid samples
                valid_preds = pred[valid_samples_mask]
                valid_targets = target[valid_samples_mask]

                # Check if this is the regression task
                is_regression = self.incl_regression and task_name == "label_ContrastPhase"

                if is_regression:
                    task_loss = self.regression_loss(valid_preds.squeeze(1), valid_targets.to(valid_preds.dtype))
                elif pred.shape[1] == 1:
                    # Binary classification task
                    task_loss = self.bce_loss(valid_preds.squeeze(1), valid_targets.float())
                else:
                    # Multi-class classification task
                    task_loss = self.ce_loss(valid_preds, valid_targets.long())

            # Apply task-specific weight if provided
            if task_weights is not None:
                task_loss = task_loss * task_weights[i]

            # Debugging: check for invalid loss values
            if DEBUG_MODE:
                if torch.isnan(task_loss).any() or torch.isinf(task_loss).any():
                    logger.error(f"Invalid loss for task {i}: {task_loss.item()}")

            total_loss += task_loss
            individual_losses.append(task_loss.item())

        if DEBUG_MODE:
            logger.info(f"Total loss: {total_loss.item()}")

        return total_loss, individual_losses
