"""Network 06: Pixel-Only (Image-Only) Classifier for 2-D Single-Slice Inputs.

This module implements :class:`PixelOnlyModel`, an image-only MRI series
classifier that operates on **single slices** or small stacks of 2-D slices.
It is designed for the Duke Liver MRI dataset and performs single-task
classification of ``SequenceType_Code_norm``, although the multi-task head
supports extending to multiple targets.

Architecture overview
---------------------
::

    Input: (B, N_slices, C, H, W)
              |
              v
    MultiSliceImageEncoder          ← 2-D CNN backbone (e.g. DenseNet-121)
    (B, N_slices, feat_dim)
              |
         Mean-pool over slices
    (B, feat_dim)
              |
              v
    MultiTaskHead                   ← one linear classifier per task
    list[(B, n_classes_i), …]       ← logits per task

For optional RF-gated inference (SequenceType_Code_norm), see
``net6/infer_duke.py`` which combines these image logits with a Random Forest
trained on tabular metadata (``net6/train_rf_duke.py``).

Version: 0.6
"""

from IMC.nn.image_encoder import MultiSliceImageEncoder
from IMC.nn.multi_task_head import MultiTaskHead

import torch
import torch.nn as nn
import logging
import os

logger = logging.getLogger("IMC")
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"


class PixelOnlyModel(nn.Module):
    """Image-only classifier for Duke single-slice (or few-slice) MRI series.

    This model encodes one or more 2-D image slices through a shared CNN
    backbone, mean-pools the resulting per-slice feature vectors, and passes
    the pooled representation to a multi-task classification head.

    No metadata or tabular features are used.  For RF-gated inference that
    combines these image predictions with a separately-trained Random Forest,
    see ``IMC.net6.infer``.

    Args:
        num_classes_dict: Mapping from task name to number of output classes,
            e.g. ``{"SequenceType_Code_norm": 13}``.
        dropout: Dropout probability applied inside the :class:`MultiTaskHead`.
        image_backbone: Name of the 2-D CNN backbone passed to
            :class:`~IMC.nn.image_encoder.MultiSliceImageEncoder`
            (e.g. ``"densenet121"``, ``"resnet50"``).

    Example::

        model = PixelOnlyModel(
            num_classes_dict={"SequenceType_Code_norm": 13},
            image_backbone="densenet121",
        )
        images = torch.randn(4, 1, 3, 224, 224)  # (B, N_slices, C, H, W)
        logits = model(images)  # list of tensors [(4, 13)]
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
        self.image_head = MultiTaskHead(image_feat_dim, num_classes_dict, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def get_image_logits(self, image_slices: torch.Tensor):
        """Compute per-task classification logits from image slices.

        Args:
            image_slices: Float tensor of shape ``(B, N_slices, C, H, W)``.

        Returns:
            list[torch.Tensor]: One tensor of shape ``(B, n_classes_i)`` per
            task, in the same order as *num_classes_dict*.
        """
        image_feats = self.image_encoder(image_slices)
        image_feats_pooled = image_feats.mean(dim=1)
        return self.image_head(image_feats_pooled)

    def get_image_probs(self, image_slices: torch.Tensor):
        """Compute per-task softmax probabilities from image slices.

        Args:
            image_slices: Float tensor of shape ``(B, N_slices, C, H, W)``.

        Returns:
            list[torch.Tensor]: One probability tensor of shape
            ``(B, n_classes_i)`` per task.
        """
        image_logits = self.get_image_logits(image_slices)
        return [self.softmax(logits) for logits in image_logits]

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor | None = None):
        """Run the full forward pass.

        Args:
            image_slices: Float tensor of shape ``(B, N_slices, C, H, W)``.
            metadata: Unused placeholder for :class:`~IMC.trainer.Trainer`
                compatibility.  Pass ``None`` or omit.

        Returns:
            list[torch.Tensor]: One logit tensor of shape ``(B, n_classes_i)``
            per task, in the same order as *num_classes_dict*.
        """
        return self.get_image_logits(image_slices)
