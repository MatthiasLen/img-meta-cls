"""
3D Pyramid Pooling module for volumetric feature aggregation.

This module implements Spatial Pyramid Pooling in 3D, which aggregates features
from a convolutional feature map at multiple scales. This allows the network to
handle inputs of varying sizes and capture context at different levels.
"""

import torch
import torch.nn as nn


class PyramidPooling3D(nn.Module):
    """
    3D Spatial Pyramid Pooling layer.

    Aggregates features from a 3D feature map at multiple pooling levels.
    The outputs are concatenated to form a fixed-length feature vector.

    Args:
        in_channels: Number of channels in the input feature map.
        levels: A list of tuples, where each tuple specifies the grid size
                for one level of adaptive average pooling (e.g., (1, 1, 1)).
    """

    def __init__(self, in_channels: int, levels: list = [(1, 1, 1), (2, 2, 2), (4, 4, 4)]):
        super().__init__()
        self.levels = levels
        self.poolers = nn.ModuleList([nn.AdaptiveAvgPool3d(output_size=level) for level in levels])

        # Each pooled level is flattened, so the total output dimension is the sum of all flattened pool outputs
        # C * D * H * W for each level
        self.output_dim = in_channels * sum([l[0] * l[1] * l[2] for l in levels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pyramid pooling and concatenate the results.

        Args:
            x: Input feature map of shape (B, C, D, H, W).

        Returns:
            A flattened feature vector of shape (B, output_dim).
        """
        batch_size = x.size(0)
        pooled_outputs = []
        for pooler in self.poolers:
            pooled = pooler(x)
            pooled_outputs.append(pooled.view(batch_size, -1))

        return torch.cat(pooled_outputs, dim=1)


if __name__ == "__main__":
    # Test the PyramidPooling3D module
    in_channels = 256
    model = PyramidPooling3D(in_channels=in_channels, levels=[(1, 1, 1), (2, 2, 2), (4, 4, 4)])
    print(model)
    print(f"Output dimension: {model.output_dim}")

    # Create a dummy input feature map
    dummy_input = torch.randn(2, in_channels, 8, 16, 16)  # (B, C, D, H, W)
    print(f"Input shape: {dummy_input.shape}")

    # Get the output from the model
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape[1] == model.output_dim
