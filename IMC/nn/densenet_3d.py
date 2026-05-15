"""
Lightweight 3D DenseNet-like backbone for volumetric medical image processing.

This module provides a 3D CNN based on DenseNet principles. It is designed
to be a lightweight feature extractor for 3D volumes, suitable for tasks like
volumetric classification. DenseNet uses dense connectivity where each layer
receives feature maps from all preceding layers, promoting feature reuse and
gradient flow.

Authors: Claude Code
Date: 2026
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block with BatchNorm -> GELU -> Conv3D."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.gelu = nn.GELU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        return self.conv(self.gelu(self.bn(x)))


class DenseLayer3D(nn.Module):
    """
    A single dense layer with bottleneck design.

    Each layer follows: BN -> GELU -> Conv1x1 -> BN -> GELU -> Conv3x3
    The bottleneck (1x1 conv) reduces the number of input channels before the 3x3 conv.

    Args:
        in_channels: Number of input channels.
        growth_rate: Number of output channels (added to the input).
        bottleneck_factor: Multiplier for bottleneck channels (typically 4).
    """

    def __init__(self, in_channels, growth_rate, bottleneck_factor=4):
        super().__init__()
        bottleneck_channels = bottleneck_factor * growth_rate

        # Bottleneck layer (1x1 conv)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.gelu1 = nn.GELU()
        self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False)

        # Main conv layer (3x3 conv)
        self.bn2 = nn.BatchNorm3d(bottleneck_channels)
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv3d(bottleneck_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Bottleneck
        out = self.conv1(self.gelu1(self.bn1(x)))
        # Main conv
        out = self.conv2(self.gelu2(self.bn2(out)))
        # Concatenate input and output (dense connection)
        return torch.cat([x, out], dim=1)


class DenseBlock3D(nn.Module):
    """
    Dense Block consisting of multiple dense layers.

    Each layer in the block receives all previous feature maps as input,
    creating dense connectivity.

    Args:
        in_channels: Number of input channels.
        num_layers: Number of dense layers in the block.
        growth_rate: Number of channels each layer adds.
        bottleneck_factor: Multiplier for bottleneck channels.
    """

    def __init__(self, in_channels, num_layers, growth_rate, bottleneck_factor=4):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(DenseLayer3D(layer_in_channels, growth_rate, bottleneck_factor))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer3D(nn.Module):
    """
    Transition layer between dense blocks.

    Reduces the number of channels and spatial dimensions using 1x1 conv
    followed by average pooling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for pooling (default 2 for downsampling).
    """

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.gelu = nn.GELU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, x):
        out = self.conv(self.gelu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet3D(nn.Module):
    """
    Lightweight 3D DenseNet-like backbone for volumetric feature extraction.

    This implementation is based on DenseNet121 architecture adapted for 3D volumes.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale).
        initial_channels: Number of channels in the first convolutional layer.
        growth_rate: Number of channels each dense layer adds (k in the paper).
        block_config: Tuple of integers specifying the number of layers in each dense block.
        compression: Compression factor for transition layers (0 < compression <= 1).
        bottleneck_factor: Multiplier for bottleneck channels in dense layers.
    """

    def __init__(
        self,
        in_channels=1,
        initial_channels=32,
        growth_rate=12,
        block_config=(6, 12, 24, 16),  # DenseNet121 configuration
        compression=0.5,
        bottleneck_factor=4,
    ):
        super().__init__()

        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(initial_channels),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Build dense blocks and transition layers
        num_channels = initial_channels
        self.features = nn.Sequential()

        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock3D(
                in_channels=num_channels,
                num_layers=num_layers,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_channels += num_layers * growth_rate

            # Add transition layer (except after the last dense block)
            if i != len(block_config) - 1:
                out_channels = int(num_channels * compression)
                trans = TransitionLayer3D(num_channels, out_channels)
                self.features.add_module(f"transition{i + 1}", trans)
                num_channels = out_channels

        # Final batch normalization
        self.final_bn = nn.BatchNorm3d(num_channels)
        self.final_gelu = nn.GELU()

        self.out_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from a 3D volume.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Feature tensor of shape (B, out_channels, D', H', W').
        """
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.final_gelu(self.final_bn(x))
        return x


def densenet121_3d(in_channels=1, initial_channels=32, growth_rate=12):
    """
    Constructs a 3D DenseNet-121 model.

    Args:
        in_channels: Number of input channels.
        initial_channels: Number of channels after initial convolution.
        growth_rate: Growth rate (k) of the network.

    Returns:
        A DenseNet3D model with DenseNet121 configuration.
    """
    return DenseNet3D(
        in_channels=in_channels,
        initial_channels=initial_channels,
        growth_rate=growth_rate,
        block_config=(6, 12, 24, 16),
        compression=0.5,
        bottleneck_factor=4,
    )


def densenet169_3d(in_channels=1, initial_channels=32, growth_rate=12):
    """
    Constructs a 3D DenseNet-169 model.

    Args:
        in_channels: Number of input channels.
        initial_channels: Number of channels after initial convolution.
        growth_rate: Growth rate (k) of the network.

    Returns:
        A DenseNet3D model with DenseNet169 configuration.
    """
    return DenseNet3D(
        in_channels=in_channels,
        initial_channels=initial_channels,
        growth_rate=growth_rate,
        block_config=(6, 12, 32, 32),
        compression=0.5,
        bottleneck_factor=4,
    )


def densenet201_3d(in_channels=1, initial_channels=32, growth_rate=12):
    """
    Constructs a 3D DenseNet-201 model.

    Args:
        in_channels: Number of input channels.
        initial_channels: Number of channels after initial convolution.
        growth_rate: Growth rate (k) of the network.

    Returns:
        A DenseNet3D model with DenseNet201 configuration.
    """
    return DenseNet3D(
        in_channels=in_channels,
        initial_channels=initial_channels,
        growth_rate=growth_rate,
        block_config=(6, 12, 48, 32),
        compression=0.5,
        bottleneck_factor=4,
    )


if __name__ == "__main__":
    # Test the DenseNet3D model
    print("=" * 80)
    print("Testing DenseNet121-3D")
    print("=" * 80)

    model = densenet121_3d(in_channels=1, initial_channels=16, growth_rate=8)
    print(model)
    print(f"\nOutput channels: {model.out_channels}")

    # Create a dummy input volume
    dummy_input = torch.randn(2, 1, 64, 128, 128)  # (B, C, D, H, W)
    print(f"\nInput shape: {dummy_input.shape}")

    # Get the output from the model
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test with PyramidPooling3D compatibility
    print("\n" + "=" * 80)
    print("Testing compatibility with PyramidPooling3D")
    print("=" * 80)

    from pyramid_pooling_3d import PyramidPooling3D

    pyramid = PyramidPooling3D(in_channels=model.out_channels, levels=[(1, 1, 1), (2, 2, 2), (4, 4, 4)])
    print(f"\nPyramid pooling output dimension: {pyramid.output_dim}")

    with torch.no_grad():
        pooled_output = pyramid(output)
    print(f"Pooled output shape: {pooled_output.shape}")

    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)
