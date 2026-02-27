"""
Network 07: 3D Pyramid Pooling Network for Volumetric Classification.

This module implements the complete 3D Pyramid Pooling Network architecture
(PyramidPooling3DClassifier) for medical image series classification. It combines
a 3D ResNet-like backbone with a 3D Pyramid Pooling layer to create a powerful
volumetric classifier.

This architecture is designed to be image-only, following the approach described
in the plan.

Authors: Claude Code
Date: 2026
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from IMC.nn.densenet_3d import DenseNet3D, densenet121_3d, densenet169_3d, densenet201_3d
from IMC.nn.multi_task_head import MultiTaskHead
from IMC.nn.pyramid_pooling_3d import PyramidPooling3D
from IMC.nn.resnet_3d import ResNet3D

logger = logging.getLogger('IMC')

class PyramidPooling3DClassifier(nn.Module):
    """
    A 3D CNN with pyramid pooling for volumetric classification.

    This model consists of:
    1. A 3D backbone (ResNet or DenseNet) to extract features from the input volume.
    2. A 3D Pyramid Pooling layer to aggregate features at multiple scales.
    3. An optional MLP projection layer.
    4. A multi-task head for classification (used here for a single task).

    Args:
        num_classes_dict: A dictionary mapping task names to the number of classes.
                          e.g., {"SequenceType_Code_norm": 13}
        backbone_type: Type of backbone ('resnet', 'densenet121', 'densenet169', 'densenet201', 'densenet_custom').
        backbone_channels: Number of initial channels in the backbone.
        backbone_blocks: List of block counts for ResNet backbone.
        growth_rate: Growth rate (k) for DenseNet backbone.
        densenet_block_config: Configuration for custom DenseNet (tuple of layer counts per block).
        compression: Compression factor for DenseNet transition layers.
        pyramid_pooling_levels: List of grid sizes for the pyramid pooling layer.
        embedding_dim: Dimension of the MLP projection layer. If 0, no MLP is used.
    """
    def __init__(
        self,
        num_classes_dict: Dict[str, int],
        backbone_type: str = 'resnet',
        backbone_channels: int = 32,
        backbone_blocks: List[int] = [2, 2, 2, 2],
        growth_rate: int = 12,
        densenet_block_config: tuple = (6, 12, 24, 16),
        compression: float = 0.5,
        pyramid_pooling_levels: List = [(1, 1, 1), (2, 2, 2), (4, 4, 4)],
        embedding_dim: int = 512
    ):
        super().__init__()
        self.num_classes_dict = num_classes_dict
        self.backbone_type = backbone_type

        # 1. 3D CNN Backbone
        if backbone_type == 'resnet':
            self.backbone = ResNet3D(
                in_channels=1,
                initial_channels=backbone_channels,
                block_counts=backbone_blocks
            )
            # For ResNet: output channels = initial_channels * (2 ** len(backbone_blocks))
            pyramid_in_channels = backbone_channels * (2 ** len(backbone_blocks))
        elif backbone_type == 'densenet121':
            self.backbone = densenet121_3d(
                in_channels=1,
                initial_channels=backbone_channels,
                growth_rate=growth_rate
            )
            pyramid_in_channels = self.backbone.out_channels
        elif backbone_type == 'densenet169':
            self.backbone = densenet169_3d(
                in_channels=1,
                initial_channels=backbone_channels,
                growth_rate=growth_rate
            )
            pyramid_in_channels = self.backbone.out_channels
        elif backbone_type == 'densenet201':
            self.backbone = densenet201_3d(
                in_channels=1,
                initial_channels=backbone_channels,
                growth_rate=growth_rate
            )
            pyramid_in_channels = self.backbone.out_channels
        elif backbone_type == 'densenet_custom':
            self.backbone = DenseNet3D(
                in_channels=1,
                initial_channels=backbone_channels,
                growth_rate=growth_rate,
                block_config=densenet_block_config,
                compression=compression
            )
            pyramid_in_channels = self.backbone.out_channels
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. "
                           f"Choose from 'resnet', 'densenet121', 'densenet169', 'densenet201', or 'densenet_custom'.")

        # 2. 3D Pyramid Pooling
        self.pyramid_pooling = PyramidPooling3D(
            in_channels=pyramid_in_channels,
            levels=pyramid_pooling_levels
        )

        # 3. MLP Projection (optional)
        self.embedding_dim = embedding_dim
        if self.embedding_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(self.pyramid_pooling.output_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU(),
                nn.Dropout(0.5)
            )
            head_input_dim = embedding_dim
        else:
            self.projection = nn.Identity()
            head_input_dim = self.pyramid_pooling.output_dim

        # 4. Classification Head
        self.head = MultiTaskHead(head_input_dim, num_classes_dict)

        logger.info(f"PyramidPooling3DClassifier initialized with {backbone_type} backbone.")
        logger.info(f"Backbone out channels: {pyramid_in_channels}")
        logger.info(f"Pyramid pooling out dim: {self.pyramid_pooling.output_dim}")
        logger.info(f"Head in dim: {head_input_dim}")

    def forward(self, images: torch.Tensor, metadata: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the 3D classifier.

        Args:
            images: Input 3D volume of shape (B, 1, D, H, W).
            metadata: Unused placeholder for trainer compatibility.

        Returns:
            A dictionary of logits for each classification task.
        """
        # Log input shape if in debug mode
        if logging.getLogger('IMC').isEnabledFor(logging.DEBUG):
            logger.debug(f"images shape: {images.shape}")

        # 1. Backbone feature extraction
        features = self.backbone(images) # -> (B, C_f, D', H', W')
        if logging.getLogger('IMC').isEnabledFor(logging.DEBUG):
            logger.debug(f"backbone features shape: {features.shape}")

        # 2. Pyramid pooling
        pooled_features = self.pyramid_pooling(features) # -> (B, C_pp)
        if logging.getLogger('IMC').isEnabledFor(logging.DEBUG):
            logger.debug(f"pooled features shape: {pooled_features.shape}")

        # 3. MLP projection
        embedding = self.projection(pooled_features) # -> (B, E)
        if logging.getLogger('IMC').isEnabledFor(logging.DEBUG):
            logger.debug(f"embedding shape: {embedding.shape}")

        # 4. Classification head
        logits_list = self.head(embedding)
        
        return logits_list


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Configuration
    num_classes = {"SequenceType_Code_norm": 13}
    batch_size = 2
    depth, height, width = 64, 128, 128

    print("=" * 80)
    print("Testing ResNet Backbone")
    print("=" * 80)

    # Create model with ResNet
    model_resnet = PyramidPooling3DClassifier(
        num_classes_dict=num_classes,
        backbone_type='resnet'
    )
    print(model_resnet)

    # Create a dummy input volume
    dummy_input = torch.randn(batch_size, 1, depth, height, width)
    print(f"\nInput shape: {dummy_input.shape}")

    # Get model output
    output_logits = model_resnet(dummy_input)

    print("\nOutput logits:")
    for task, logits in output_logits.items():
        print(f"  Task: {task}, Logits shape: {logits.shape}")
        assert logits.shape == (batch_size, num_classes[task])

    print("\n" + "=" * 80)
    print("Testing DenseNet121 Backbone")
    print("=" * 80)

    # Create model with DenseNet121
    model_densenet = PyramidPooling3DClassifier(
        num_classes_dict=num_classes,
        backbone_type='densenet121',
        backbone_channels=16,
        growth_rate=8
    )

    # Get model output
    with torch.no_grad():
        output_logits = model_densenet(dummy_input)

    print("\nOutput logits:")
    for task, logits in output_logits.items():
        print(f"  Task: {task}, Logits shape: {logits.shape}")
        assert logits.shape == (batch_size, num_classes[task])

    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)
