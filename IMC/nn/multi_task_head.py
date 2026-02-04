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

    The module can handle both classification and regression tasks. By setting
    `incl_regression` to True, the head for the task named "label_ContrastPhase"
    is configured as a regression head with a single output neuron.
    """

    def __init__(self, input_dim: int, num_classes_dict: dict, dropout: float = 0.1, incl_regression: bool = True):
        super().__init__()

        self.tasks_heads = nn.ModuleDict()
        for task_name, n_classes in num_classes_dict.items():
            if incl_regression and task_name == "label_ContrastPhase":  # regression task
                self.tasks_heads[task_name] = self.make_task_head(input_dim, 1, dropout)
            else:
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
    

class MultiTaskHeadV0(nn.Module):
    """
    A simplified multi-task head with a single linear layer.

    This is an older, less flexible version of the `MultiTaskHead`. It uses a single
    linear layer to predict the logits for all tasks simultaneously. The output is
    then manually split into per-task logits.

    This approach is less powerful than using separate heads for each task, as it
    forces all tasks to share the same final layer.
    """
    def __init__(self, input_dim: int, num_classes_dict: dict, dropout: float = 0.1):
        super().__init__()
        self.tasks_heads = nn.Linear(input_dim, sum(num_classes_dict.values()))
        self.num_classes_dict = num_classes_dict

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass for the simplified multi-task head.

        Args:
            x (torch.Tensor): A shared feature embedding of shape (B, input_dim).

        Returns:
            list: A list of tensors, where each tensor contains the output logits
                  for a specific task.
        """
        logits = self.tasks_heads(x)
        outputs = []
        start_idx = 0
        for n_classes in self.num_classes_dict.values():
            end_idx = start_idx + n_classes
            outputs.append(logits[:, start_idx:end_idx])
            start_idx = end_idx
        return outputs