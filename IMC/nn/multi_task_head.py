import torch
import torch.nn as nn
  

class MultiTaskHead(nn.Module):
    """
    Multi-task output heads for sequence, plane, body region, and contrast classification.
    Shared feature extractor followed by separate heads for each task.
    """

    def __init__(self, input_dim: int, num_classes_dict: dict, dropout: float = 0.1):
        super().__init__()
        
        self.tasks_heads = nn.ModuleDict()
        for task_name, n_classes in num_classes_dict.items():
            self.tasks_heads[task_name] = self.make_task_head(input_dim, n_classes, dropout)

        print("Task head configuration", self.tasks_heads)

    def make_task_head(self, in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        """
        Creates a task-specific head (MLP) for classification.
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension (number of classes)
            
        Returns:
            nn.Sequential: Task head module
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
        Args:
            x (torch.Tensor): Joint feature embedding (B, input_dim)
            
        Returns:
            list: head logits
        """
        
        return [head(x) for _, head in self.tasks_heads.items()]