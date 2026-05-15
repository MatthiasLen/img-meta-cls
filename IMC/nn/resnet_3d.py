"""
Lightweight 3D ResNet-like backbone for volumetric medical image processing.

This module provides a simple 3D CNN based on ResNet principles. It is designed
to be a lightweight feature extractor for 3D volumes, suitable for tasks like
volumetric classification.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block with Conv3D -> BatchNorm -> GELU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.bn(self.conv(x)))


class ResidualBlock3D(nn.Module):
    """3D Residual Block with a skip connection."""

    def __init__(self, channels, stride=1):
        super().__init__()
        self.block1 = ConvBlock3D(channels, channels, stride=stride)
        self.block2 = ConvBlock3D(channels, channels)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block1(x)
        out = self.block2(out)
        out += residual
        return out


class ResNet3D(nn.Module):
    """
    Lightweight 3D ResNet-like backbone for volumetric feature extraction.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale).
        initial_channels: Number of channels in the first convolutional layer.
        block_counts: List of integers specifying the number of residual blocks in each stage.
    """

    def __init__(self, in_channels=1, initial_channels=32, block_counts=[2, 2, 2, 2]):
        super().__init__()
        self.in_channels = initial_channels

        self.initial_layer = ConvBlock3D(in_channels, initial_channels, stride=2)  # Downsample initially

        self.layers = nn.ModuleList()
        channels = initial_channels
        for count in block_counts:
            self.layers.append(self._make_layer(channels, count, stride=2))
            channels *= 2

    def _make_layer(self, channels, block_count, stride):
        strides = [stride] + [1] * (block_count - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock3D(channels, stride=s))
        # The layer for the next stage will need more input channels
        next_stage_in_channels = channels * 2
        layers.append(ConvBlock3D(channels, next_stage_in_channels, kernel_size=1, padding=0))
        self.in_channels = next_stage_in_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features from a 3D volume."""
        x = self.initial_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Test the ResNet3D model
    model = ResNet3D(in_channels=1, initial_channels=16, block_counts=[1, 1, 1])
    print(model)

    # Create a dummy input volume
    dummy_input = torch.randn(2, 1, 64, 128, 128)  # (B, C, D, H, W)
    print(f"Input shape: {dummy_input.shape}")

    # Get the output from the model
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
