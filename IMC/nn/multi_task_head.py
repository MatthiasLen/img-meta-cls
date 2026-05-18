import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """
    A multi-task learning head for medical image classification.

    This module takes a shared feature embedding from a backbone model and passes it
    through multiple, independent task-specific heads. Each head is a small MLP
    (Multi-Layer Perceptron) responsible for a single classification or regression task.
    This architecture allows the model to learn shared representations while also
    specializing for individual tasks.

    The supported tasks are defined by the `num_classes_dict`, which maps task names
    to the number of classes for that task.
    """

    def __init__(self, input_dim: int, num_classes_dict: dict, dropout: float = 0.1):
        super().__init__()

        self.tasks_heads = nn.ModuleDict()
        for task_name, n_classes in num_classes_dict.items():
            self.tasks_heads[task_name] = self.make_task_head(input_dim, n_classes, dropout)

    def make_task_head(self, in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        """
        Creates a task-specific MLP head.

        The head consists of two linear layers with LayerNorm, GELU activation, and Dropout.

        Args:
            in_dim (int): Input dimension from the shared backbone.
            out_dim (int): Output dimension (number of classes for the task).
            dropout (float): Dropout probability.

        Returns:
            nn.Sequential: The task-specific head.
        """
        hidden_dim = (in_dim + out_dim) // 2

        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass for the multi-task head.

        Args:
            x (torch.Tensor): A shared feature embedding of shape (B, input_dim).

        Returns:
            list: A list of tensors, where each tensor contains the output logits
                  for a specific task.
        """

        return [head(x) for _, head in self.tasks_heads.items()]
