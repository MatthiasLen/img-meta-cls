import torch
import torch.nn as nn
class MultiTaskLoss(torch.nn.Module):
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

    def forward(self, preds: tuple, targets: tuple) -> tuple[float, list]:
        losses = []
        total_loss = 0.
          
        for i in range(len(preds)):
            if preds[i].shape[1] == 1:  # Binary
                l = self.bce_loss(preds[i].squeeze(1), targets[i])
            else:
                l = self.ce_loss(preds[i], targets[i])
            total_loss += l
            losses.append(l.item())
    
        return total_loss, losses
