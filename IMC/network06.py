"""
    VERSION 0.6
    2026/02/10

    Fusion model based on concatenating probabilities from separate image and metadata models.
    This architecture is inspired by the method in the Abdominal_MRI_series_classification repository.

        Architecture Overview:

        +-------------------+       +---------------------+
        |  Image Slices     |       |    Metadata Input   |
        | (B, N_slices,     |       | (B, D_meta)         |
        |  C, H, W)         |       +---------------------+
        +---------+---------+                 |
                  |                           |
                  v                           v
        +-------------------+       +---------------------+
        | MultiSliceImage   |       | MetadataEncoder MLP |
        | Encoder           |       |                     |
        +---------+---------+       +----------+----------+
                  |                            |
        (B, N_slices, feat_dim)         (B, meta_embed_dim)
                  |                            |
        Mean pooling over slices      Final representation
        (B, feat_dim)                   (B, meta_embed_dim)
                  |                            |
                  v                            v
        +-------------------+       +---------------------+
        | MultiTaskHead     |       | MultiTaskHead       |
        | (Image)           |       | (Metadata)          |
        +---------+---------+       +----------+----------+
                  |                            |
        (Logits per task)             (Logits per task)
                  |                            |
                  v                            v
              Softmax                      Softmax
                  |                            |
        (Probs per task)              (Probs per task)
                  \____________  ____________/
                               \/
                 Concatenate along feature dim
           (B, sum of all class counts for all tasks * 2)
                                |
                  +-------------+--------------+
                  |         Fusion Head        |
                  |       (MultiTaskHead)      |
                  +-------------+--------------+
                                |
                      MultiTaskHead classifiers
                      (final fused logits per task)

"""

from IMC.nn.image_encoder import MultiSliceImageEncoder
from IMC.nn.multi_task_head import MultiTaskHead

import torch
import torch.nn as nn
import logging
import os

logger = logging.getLogger('IMC')
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"


class PixelOnlyModel(nn.Module):
    """
    Pixel (image) model for Duke single-slice classification.

    No fusion. This model encodes a single slice and predicts multi-task logits.
    """

    def __init__(
        self,
        num_classes_dict: dict,
        dropout: float = 0.1,
        image_backbone: str = "densenet121",
    ):
        super().__init__()
        self.num_classes_dict = num_classes_dict
        self.image_encoder = MultiSliceImageEncoder(backbone=image_backbone)
        image_feat_dim = self.image_encoder.get_feature_dimension()
        self.image_head = MultiTaskHead(image_feat_dim, num_classes_dict, dropout=dropout, incl_regression=False)
        self.softmax = nn.Softmax(dim=-1)


    def get_image_logits(self, image_slices: torch.Tensor):
        """Return per-task logits from image stream only."""
        image_feats = self.image_encoder(image_slices)
        image_feats_pooled = image_feats.mean(dim=1)
        return self.image_head(image_feats_pooled)

    def get_image_probs(self, image_slices: torch.Tensor):
        """Return per-task probabilities from image stream only."""
        image_logits = self.get_image_logits(image_slices)
        return [self.softmax(logits) for logits in image_logits]

    def forward(self, image_slices: torch.Tensor):
        """Forward pass of pixel-only model returning list of logits per task."""
        return self.get_image_logits(image_slices)
