"""
    VERSION 0.4
    2025/09/18

    Updates / Change-log:

    1. Implemented Bi-Directional Cross-Attention embedding fusion. This is how it works:
    Both modalities are projected to the same dimension.
    Image embedding queries metadata embedding to get info from metadata relevant to image features.
    Metadata embedding queries image embedding similarly.
    Each attention output passes through a Transformer-style feedforward block with residuals and normalization.
    Finally, concatenate both outputs and project down to a fixed output dimension.

    2. Added a residual connection and layer normalization in SliceFeatureFusion to improve training stability.

    3. Improved initialization of CNN backbone. RGB filters from pretrained weights are averaged to obtain a 
    grayscale image filter that can be used for the 1-channel convolution.

    4. MetadataEncoder was extended a bit to incorporate a residual block , dropout and layer normalization.


    High-level architecture Diagram:

    A) Multiple Slices (N x 2D images) ---> Shared CNN Backbone
                                                |
                                        Slice Embeddings
                                                |
                        Slice Embeddings Fusion with Multi-Head Self-Attention
                                                |
                                Fused Image Feature Vector (f_img)


    B)                          DICOM Metadata Vector 
                        (assuming single vector per volumetric image)
                (concatenate slice embeddings if needed and feed concatednated vector) 
                                            |
                                    Metadata Encoder 
                                            |
                                    Metadata Embedding (f_meta)


    C) Bi-Directional Cross-Modal Attention Fusion ---> Multi-task Output Heads
                                                               |
                _______________________________________________|_____________
                |                  |                     |                   |
            Sequence Classifier   Plane Classifier   Body Region Classifier   Contrast Classifier
            (Softmax)            (Softmax)            (Softmax)               (Sigmoid)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from IMC.nn.image_encoder import MultiSliceImageEncoder
from IMC.nn.metadata_encoder import MetadataEncoder
from IMC.nn.sparse_metadata_encoder import SparseMetadataEncoder as SparseEncoderV1
from IMC.nn.sparse_metadata_encoder_v2 import SparseMetadataEncoder as SparseEncoderV2
from IMC.nn.sparse_metadata_encoder_v5 import SparseMetadataEncoder as SparseEncoderV5
from IMC.nn.multi_task_head import MultiTaskHead

from torchvision.models.densenet import DenseNet121_Weights
from torchvision import models
import logging 
import os

logger = logging.getLogger('IMC')
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

class SliceFeatureFusion(nn.Module):
    """
    Fuse per-slice embeddings with multi-head self-attention.

    - Projects Q, K, V from slice features and applies scaled dot-product attention
    - Residual connection + LayerNorm for stability
    - Optional reduction over the slice dimension
    - Final MLP projection to the fused feature dimension
    """

    def __init__(
        self,
        slice_feat_dim: int = 512,
        fused_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        reduce: bool = False    
    ):
        super().__init__()
        self.slice_feat_dim = slice_feat_dim
        self.num_heads = num_heads

        # Check that slice_feat_dim is divisible by num_heads
        if not slice_feat_dim % num_heads == 0:
            raise ValueError("slice_feat_dim must be divisible by num_heads")

        # Linear projections for multi-head attention
        self.qkv_proj = nn.Linear(slice_feat_dim, slice_feat_dim * 3)
        self.out_proj = nn.Sequential(nn.Linear(slice_feat_dim, fused_dim), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        
        self.layer_norm =  nn.LayerNorm(self.slice_feat_dim)

        # Scaling factor for attention scores
        self.scale = (slice_feat_dim // num_heads) ** -0.5

        self.reduce = reduce

    def forward(self, slice_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_feats (torch.Tensor): Input tensor of shape (B, N_slices, slice_feat_dim)
        Returns:
            torch.Tensor: Fused image feature vector of shape (B, fused_dim)
        """
        B, N, C = slice_feats.shape

        # Compute Q, K, V in one projection for efficiency
        qkv = self.qkv_proj(slice_feats)  # (B, N, 3*C)

        # Split qkv into separate Q, K, V tensors and reshape for multi-head attention
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)

        # Compute scaled dot-product attention scores
        attn_scores = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # (B, heads, N, N)
        # Softmax over last dimension (keys)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply dropout to attention probabilities
        attn_probs = self.dropout(attn_probs)

        # Attention weighted sum of values
        attn_output = torch.matmul(attn_probs, v)  # (B, heads, N, head_dim)

        # Concatenate heads back to (B, N, C)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        
        # residual connection with layer norm
        x = slice_feats + attn_output
        x = self.layer_norm(x)
 
        # Pool over slice dimension (tokens) by averaging
        if self.reduce:
            x = x.mean(dim=1)  # (B, C)

        # Final MLP projection to fused_dim
        fused = self.out_proj(x)  # (B, fused_dim)
        return fused


class BiDirectionalCrossModalAttentionFusion(nn.Module):
    """
    Bi-directional cross-attention fusion between image and metadata embeddings.

    Both modalities are projected to a shared embedding size and attend to each other:
    - Image queries metadata (img→meta)
    - Metadata queries image (meta→img)

    Each output passes through FFN + residual + LayerNorm for stability.
    The two fused embeddings are concatenated and projected to the final output dimension.
    """
    def __init__(
        self,
        image_emb_dim: int,
        metadata_emb_dim: int,
        embed_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_hidden_mult: int = 4,
    ):
        super().__init__()

        # Project input embeddings to common dimension
        self.img_proj = nn.Linear(image_emb_dim, embed_dim)
        self.meta_proj = nn.Linear(metadata_emb_dim, embed_dim)

        # Multi-head attention modules for both directions
        self.img_to_meta_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.meta_to_img_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # LayerNorms and Feedforward blocks for both outputs
        self.norm_img1 = nn.LayerNorm(embed_dim)
        self.norm_img2 = nn.LayerNorm(embed_dim)
        self.norm_meta1 = nn.LayerNorm(embed_dim)
        self.norm_meta2 = nn.LayerNorm(embed_dim)

        self.ff_img = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.ff_meta = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Final projection from concatenated fused embeddings
        self.output_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, img_feat: torch.Tensor, meta_feat: torch.Tensor):
        """
        Args:
            img_feat: (B, image_emb_dim)
            meta_feat: (B, metadata_emb_dim)
        Returns:
            fused embedding: (B, output_dim)
        """

        # Project to embed_dim
        img_emb = self.img_proj(img_feat)  # (B, embed_dim)
        meta_emb = self.meta_proj(meta_feat)  # (B, embed_dim)

        # Unsqueeze to sequence length 1 for multihead attention (B, S=1, E)
        img_emb_seq = img_emb.unsqueeze(1)
        meta_emb_seq = meta_emb.unsqueeze(1)

        # Image queries metadata
        img_attn_out, _ = self.img_to_meta_attn(query=img_emb_seq, key=meta_emb_seq, value=meta_emb_seq)  # (B,1,embed_dim)
        img_out = self.norm_img1(img_attn_out.squeeze(1) + img_emb)  # Residual + Norm

        img_out_ff = self.ff_img(img_out)
        img_out = self.norm_img2(img_out + img_out_ff)  # FFN + Residual + Norm

        # Metadata queries image
        meta_attn_out, _ = self.meta_to_img_attn(query=meta_emb_seq, key=img_emb_seq, value=img_emb_seq)  # (B,1,embed_dim)
        meta_out = self.norm_meta1(meta_attn_out.squeeze(1) + meta_emb)  # Residual + Norm

        meta_out_ff = self.ff_meta(meta_out)
        meta_out = self.norm_meta2(meta_out + meta_out_ff)  # FFN + Residual + Norm

        # Concatenate bi-directional fused embeddings
        fused = torch.cat([img_out, meta_out], dim=1)  # (B, 2*embed_dim)

        # Final projection
        output = self.output_proj(fused)  # (B, output_dim)

        return output
    
class BiDirectionalCrossModalAttentionFusionV2(nn.Module):
    """
    Cross-attention fusion (v2) using sequence-style inputs and learned weighted pooling.

    Differences vs v1:
    - Operates directly on embeddings without adding singleton sequence length
    - Adds a learned weighted pooling over the fused output to emphasize salient features
    """
    def __init__(
        self,
        image_emb_dim: int,
        metadata_emb_dim: int,
        embed_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_hidden_mult: int = 4,
    ):
        super().__init__()

        # Project input embeddings to common dimension
        self.img_proj = nn.Linear(image_emb_dim, embed_dim)
        self.meta_proj = nn.Linear(metadata_emb_dim, embed_dim)

        # Multi-head attention modules for both directions
        self.img_to_meta_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.meta_to_img_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # LayerNorms and Feedforward blocks for both outputs
        self.norm_img1 = nn.LayerNorm(embed_dim)
        self.norm_img2 = nn.LayerNorm(embed_dim)
        self.norm_meta1 = nn.LayerNorm(embed_dim)
        self.norm_meta2 = nn.LayerNorm(embed_dim)

        self.ff_img = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.ff_meta = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Final projection from concatenated fused embeddings
        self.output_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        # Add weighted pooling to fuse at the slice level
        self.weighted_pooling = nn.Linear(embed_dim, 1)

    def forward(self, img_feat: torch.Tensor, meta_feat: torch.Tensor):
        """
        Args:
            img_feat: (B, image_emb_dim)
            meta_feat: (B, metadata_emb_dim)
        Returns:
            fused embedding: (B, output_dim)
        """

        # Project to embed_dim
        img_emb_seq = self.img_proj(img_feat)  # (B, embed_dim)
        meta_emb_seq = self.meta_proj(meta_feat)  # (B, embed_dim)

        # Unsqueeze to sequence length 1 for multihead attention (B, S=1, E)
        # img_emb_seq = img_emb.unsqueeze(1)
        # meta_emb_seq = meta_emb.unsqueeze(1)

        # Image queries metadata
        img_attn_out, _ = self.img_to_meta_attn(query=img_emb_seq, key=meta_emb_seq, value=meta_emb_seq)  # (B,1,embed_dim)
        img_out = self.norm_img1(img_attn_out + img_emb_seq)  # Residual + Norm

        img_out_ff = self.ff_img(img_out)
        img_out = self.norm_img2(img_out + img_out_ff)  # FFN + Residual + Norm

        # Metadata queries image
        meta_attn_out, _ = self.meta_to_img_attn(query=meta_emb_seq, key=img_emb_seq, value=img_emb_seq)  # (B,1,embed_dim)
        meta_out = self.norm_meta1(meta_attn_out + meta_emb_seq)  # Residual + Norm

        meta_out_ff = self.ff_meta(meta_out)
        meta_out = self.norm_meta2(meta_out + meta_out_ff)  # FFN + Residual + Norm

        # Concatenate bi-directional fused embeddings
        fused = torch.cat([img_out, meta_out], dim=-1)  # (B, 2*embed_dim)

        # Final projection
        output = self.output_proj(fused)  # (B, output_dim)

        # Weighted pooling 
        weights = F.softmax(self.weighted_pooling(output), dim=1)  # (B, 1)
        output = (output * weights).sum(dim=1)  # (B, output_dim)

        return output


class SimpleConcatFusion(nn.Module):
    """
    Simple concatenation fusion of image and metadata features.

    This is a baseline fusion method that just concatenates the image and metadata embeddings
    and projects them to the output dimension. No attention or interaction between modalities.
    """

    def __init__(self, image_emb_dim: int, metadata_emb_dim: int, output_dim: int = 128, reduce: bool = True):
        super().__init__()
        self.reduce = reduce
        self.output_proj = nn.Sequential(
            nn.Linear(image_emb_dim + metadata_emb_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, img_feat: torch.Tensor, meta_feat: torch.Tensor):
        # Concatenate image and metadata features
        fused = torch.cat([img_feat, meta_feat], dim=2)  # (B, N, image_emb_dim + metadata_emb_dim)

        # Project to output dimension
        output = self.output_proj(fused)  # (B, N, output_dim)

        # Average over the sequence dimension if needed (if fused is still a sequence)
        if self.reduce:
            output = output.mean(dim=1)  # (B, output_dim)
        return output


class MetadataGatedFusion(nn.Module):
    """Metadata-gated fusion (v3) for multi-modal MRI classification.

    Metadata produces a **per-dimension soft gate** that controls how much the
    image vs. metadata contribution flows into the final joint embedding.  When
    metadata carries strong evidence for a class, the gate can suppress the image
    branch entirely (gate → 1); when metadata is ambiguous the image branch
    contributes more (gate → 0).

    Both encoders use ``reduce=True`` (2D tensors, like v1 fusion), so this
    module is a drop-in replacement for v1 when ``fusion_module_version='v3'``.

    Architecture::

        g = σ(gate_fc(meta_feat))           ∈ (0, 1)^output_dim
        output = g ⊙ meta_proj(meta_feat)
               + (1 - g) ⊙ img_proj(img_feat)

    The gate weights are initialised to zero so the gate starts at 0.5
    (balanced between modalities) and is learned end-to-end.

    Args:
        image_emb_dim:    Dimension of fused image features (``fused_feat_dim``).
        metadata_emb_dim: Dimension of metadata embedding.
        output_dim:       Output dimension, must match ``MultiTaskHead.input_dim``.
        dropout:          Dropout probability in projection MLPs.
    """

    def __init__(
        self,
        image_emb_dim: int,
        metadata_emb_dim: int,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.img_proj = nn.Sequential(
            nn.Linear(image_emb_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(metadata_emb_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Gate: projects metadata to a per-dimension weight in (0,1).
        # Weights initialised to 0 → gate starts at sigmoid(0) = 0.5.
        self.gate_fc = nn.Linear(metadata_emb_dim, output_dim, bias=True)
        nn.init.zeros_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, img_feat: torch.Tensor, meta_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat:   ``(B, image_emb_dim)`` — output of SliceFeatureFusion (reduce=True).
            meta_feat:  ``(B, metadata_emb_dim)`` — output of metadata encoder (reduce=True).

        Returns:
            Fused embedding ``(B, output_dim)``.
        """
        g = torch.sigmoid(self.gate_fc(meta_feat))          # (B, output_dim)
        return g * self.meta_proj(meta_feat) + (1 - g) * self.img_proj(img_feat)


class MRISequenceClassifier(nn.Module):
    """
    Multi-modal MRI classifier with cross-attention fusion.

    - Encodes image slices and metadata
    - Fuses them via bi-directional cross-attention (v1 or v2)
    - Predicts multiple tasks via a shared backbone and task-specific heads

    Supports optional metadata dropout during training to improve robustness
    to missing metadata.

    Args:
        num_classes_dict: Mapping of task name to number of classes.
        metadata_input_dim: Dimension of raw metadata input.
        metadata_embed_dim: Dimension of metadata embedding.
        fused_feat_dim: Dimension of fused slice features.
        output_emb_dim: Dimension of fused image+metadata embedding.
        imputer_type: Strategy for metadata imputation.
        img_enc_backbone: Image encoder backbone name (e.g., 'swin', 'densenet').
        incl_regression: Whether to include regression task in the head.
        dropout_metadata: Enable random metadata masking during training.
        fusion_module_version: 'v1' or 'v2' fusion module selection.
    """

    def __init__(
        self,
        num_classes_dict: dict,
        metadata_input_dim: int,
        metadata_embed_dim: int = 128,
        fused_feat_dim: int = 256,
        output_emb_dim: int = 128,
        metadata_encoder_type: str = "imputer", # "imputer" or "sparse"
        imputer_type: str = "contextual", # if metadata_encoder_type is "imputer", which type to use ("contextual" or "ignore")
        sparse_enc_version: str = "v1", # if metadata_encoder_type is "sparse", which version to use ("v1", "v2", or "v5")
        img_enc_backbone: str | None = "densenet121", # "densenet" or "swin", None for resnet50 as default
        incl_regression: bool = True,
        dropout_metadata: bool = False,
        fusion_module_version: str = "v1", #v1 or v2 or concat or v3
        scalar_modulation: bool = False,
        n_channels: int = 1,  # 1 = single-window grayscale; 3 = multi-window (e.g. soft_tissue/angio/bone),
        learn_missing_embed: bool = False,  # Whether to learn a special embedding for missing metadata instead of using 0 tensors in the sparse metadata encoder V1
        pre_processors=None,   # Optional list of PreProcessor callables; applied to metadata before encoding (eval only).
        post_processors=None,  # Optional list of PostProcessor callables; applied to logits after the head (eval only).
    ):
        super().__init__()
        self.image_encoder = MultiSliceImageEncoder(backbone=img_enc_backbone, n_channels=n_channels)
        slice_feat_dim = self.image_encoder.get_feature_dimension()

        # Check fusion module version
        assert fusion_module_version in ["v1", "v2", "concat", "v3"], "fusion_module_version must be 'v1', 'v2', 'concat', or 'v3'"
        # Check metadata encoder type
        assert metadata_encoder_type in ["imputer", "sparse"], "metadata_encoder_type must be 'imputer' or 'sparse'"
        # Check imputer type
        assert imputer_type in ["contextual", "ignore"], "imputer_type must be 'contextual' or 'ignore'"
        # Check sparse encoder version
        assert sparse_enc_version in ["v1", "v2", "v5"], "sparse_enc_version must be 'v1', 'v2', or 'v5'"
        
        # v3 uses 2D (reduced) tensors like v1; v2/concat use 3D sequence tensors
        _use_reduce = fusion_module_version in ("v1", "v3")

        # Prefill for slice_fusion and metadata encoder based on fusion module version
        self.slice_fusion = SliceFeatureFusion(
            slice_feat_dim=slice_feat_dim, 
            fused_dim=fused_feat_dim, 
            reduce=_use_reduce,
        )
        if metadata_encoder_type == "imputer":
            self.metadata_encoder = MetadataEncoder(
                metadata_input_dim, 
                embed_dim=metadata_embed_dim, 
                imputer=imputer_type, 
                reduce=_use_reduce,
            )
        elif metadata_encoder_type == "sparse":
            if sparse_enc_version == "v1":
                self.metadata_encoder = SparseEncoderV1(
                    metadata_input_dim,
                    out_dim=metadata_embed_dim,
                    reduce=_use_reduce,
                    scalar_modulation=scalar_modulation,
                    learn_missing_emb=learn_missing_embed
                )
            elif sparse_enc_version == "v2":
                self.metadata_encoder = SparseEncoderV2(
                    metadata_input_dim,
                    out_dim=metadata_embed_dim,
                    reduce=_use_reduce,
                )
            elif sparse_enc_version == "v5":
                self.metadata_encoder = SparseEncoderV5(
                    metadata_input_dim,
                    out_dim=metadata_embed_dim,
                    reduce=_use_reduce,
                )

        # Create fusion module based on version
        if fusion_module_version == "v1":
            self.embedding_fusion = BiDirectionalCrossModalAttentionFusion(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
            )
        elif fusion_module_version == "v2":
            self.embedding_fusion = BiDirectionalCrossModalAttentionFusionV2(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
            )
        elif fusion_module_version == "concat":
            self.embedding_fusion = SimpleConcatFusion(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
                reduce=True
            )
        elif fusion_module_version == "v3":
            self.embedding_fusion = MetadataGatedFusion(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
            )
        # Initialize primary multi-task head
        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict, incl_regression=incl_regression)
        # Optional: metadata dropout during training
        self.dropout_metadata = dropout_metadata
        self._task_names = list(num_classes_dict.keys())
        # Plain Python callables (not nn.Module); only invoked during eval.
        # Assign after construction if needed: model.pre_processors = [...] / model.post_processors = [...]
        self.pre_processors  = list(pre_processors)  if pre_processors  else []
        self.post_processors = list(post_processors) if post_processors else []

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor) -> tuple:
        """
        Args:
            image_slices (torch.Tensor): MRI slices (B, N_slices, C, H, W)
            metadata (torch.Tensor): DICOM metadata (B, metadata_input_dim)
        Returns:
            tuple: (seq_logits, plane_logits, body_logits, contrast_logits)
        """
        if DEBUG_MODE:
            logger.debug(f"Input image_slices shape: {image_slices.shape}")
            logger.debug(f"Input metadata shape: {metadata.shape}")
            logger.debug(f"Input stats - Images: min={image_slices.min():.3f}, max={image_slices.max():.3f}, mean={image_slices.mean():.3f}")
            logger.debug(f"Input stats - Metadata: min={metadata.min():.3f}, max={metadata.max():.3f}, mean={metadata.mean():.3f}")
            has_nan = torch.isnan(metadata).any()
            has_inf = torch.isinf(metadata).any()
            if has_nan or has_inf:
                logger.error(f"Metadata input contains invalid values - NaN: {has_nan}, Inf: {has_inf}")

        # Optional metadata dropout: randomly mask features to simulate missing values
        if self.training and self.dropout_metadata:
            prob = 0.3
            mask = torch.rand(metadata.shape, device=metadata.device) < prob
            metadata = metadata.masked_fill(mask, float('nan'))
        
        # Encode image slices
        slice_feats = self.image_encoder(image_slices)  # (B, N_slices, slice_feat_dim)
        fused_img_feat = self.slice_fusion(slice_feats)  # (B, fused_feat_dim)
        
        if DEBUG_MODE:
            logger.debug(f"Fused image feature shape: {fused_img_feat.shape}")
            logger.debug(f"Fused image feature stats: min={fused_img_feat.min():.3f}, max={fused_img_feat.max():.3f}, mean={fused_img_feat.mean():.3f}")
            has_nan = torch.isnan(fused_img_feat).any()
            has_inf = torch.isinf(fused_img_feat).any()
            if has_nan or has_inf:
                logger.debug(f"Fused image feature contains invalid values - NaN: {has_nan}, Inf: {has_inf}")

        # Inference-time metadata pre-processing (outlier cleaning, etc.)
        if not self.training and self.pre_processors:
            for pre_proc in self.pre_processors:
                metadata = pre_proc(metadata)

        # Encode metadata
        metadata_feat = self.metadata_encoder(metadata)  # (B, metadata_embed_dim) or (B, N, metadata_embed_dim)

        if DEBUG_MODE:
            logger.debug(f"Metadata feature shape: {metadata_feat.shape}")
            logger.debug(f"Metadata feature stats: min={metadata_feat.min():.3f}, max={metadata_feat.max():.3f}, mean={metadata_feat.mean():.3f}")
            has_nan = torch.isnan(metadata_feat).any()
            has_inf = torch.isinf(metadata_feat).any()
            if has_nan or has_inf:
                logger.debug(f"Metadata feature contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
        
        # Fuse features from image embedding and meta data embedding
        joint_feat = self.embedding_fusion(fused_img_feat, metadata_feat)

        if DEBUG_MODE:
            logger.debug(f"Joint feature shape: {joint_feat.shape}")
            logger.debug(f"Joint feature stats: min={joint_feat.min():.3f}, max={joint_feat.max():.3f}, mean={joint_feat.mean():.3f}")
            has_nan = torch.isnan(joint_feat).any()
            has_inf = torch.isinf(joint_feat).any()
            if has_nan or has_inf:
                logger.debug(f"Joint feature contains invalid values - NaN: {has_nan}, Inf: {has_inf}")

        # feed combined embedding to primary multi-task head
        res = self.multi_task_head(joint_feat)

        # Inference-time post-processing (rule-based overrides, etc.)
        if not self.training and self.post_processors:
            for post_proc in self.post_processors:
                res = post_proc(res, metadata, self._task_names)

        if DEBUG_MODE:
            for i, r in enumerate(res):
                logger.debug(f"Output logits for task {i} shape: {r.shape}")
                logger.debug(f"Output logits for task {i} stats: min={r.min():.3f}, max={r.max():.3f}, mean={r.mean():.3f}")
                has_nan = torch.isnan(r).any()
                has_inf = torch.isinf(r).any()
                if has_nan or has_inf:
                    logger.debug(f"Output logits for task {i} contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
                    
        return res
    
    @torch.no_grad()
    def get_metadata_features(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Utility function to extract metadata features alone.
        Args:
            metadata (torch.Tensor): DICOM metadata (B, metadata_input_dim)
        Returns:
            torch.Tensor: Metadata features (B, metadata_embed_dim)
        """
        return self.metadata_encoder(metadata)
class ImageBasedClassifier(nn.Module):
    """
    Image-only ablation model.

    Encodes and self-attention-fuses image slices, projects to an embedding,
    then predicts tasks via the multi-task head.
    """

    def __init__(
        self,
        num_classes_dict: dict,
        fused_feat_dim: int = 256,
        output_emb_dim: int = 128,
        img_enc_backbone: str | None = "swin", # "densenet" or "swin", None for resnet50 as default
        incl_regression: bool = False,
        n_channels: int = 1,  # 1 = single-window grayscale; 3 = multi-window (e.g. soft_tissue/angio/bone)
    ):
        super().__init__()
        self.image_encoder = MultiSliceImageEncoder(backbone=img_enc_backbone, n_channels=n_channels)
        
        slice_feat_dim = self.image_encoder.get_feature_dimension()

        self.slice_fusion = SliceFeatureFusion(
            slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim, reduce=True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(fused_feat_dim, output_emb_dim),
            nn.LayerNorm(output_emb_dim),
            nn.GELU(),
        )

        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict, incl_regression=incl_regression)

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor) -> tuple:
        """
        Args:
            image_slices (torch.Tensor): MRI slices (B, N_slices, C, H, W)
        Returns:
            tuple: (seq_logits, plane_logits, body_logits, contrast_logits)
        """
        
        # Encode image slices
        slice_feats = self.image_encoder(image_slices)  # (B, N_slices, slice_feat_dim)
        fused_img_feat = self.slice_fusion(slice_feats)  # (B, fused_feat_dim)
        
        joint_feat = self.output_proj(fused_img_feat)

        # feed combined embedding to multi-task head
        res = self.multi_task_head(joint_feat)
        return res

class SimpleImageBasedClassifier(nn.Module):

    def __init__(
        self,
        num_classes_dict: dict,
        incl_regression: bool = False,
    ):
        super().__init__()
        self.image_encoder = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        feature_dim = self.image_encoder.classifier.in_features
        self.image_encoder.classifier = nn.Identity()  # Remove the original classifier
        self.multi_task_head = MultiTaskHead(feature_dim, num_classes_dict, incl_regression=incl_regression)

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor) -> tuple:
        """
        Args:
            image_slices (torch.Tensor): MRI slices (B, N_slices, C, H, W)
        Returns:
            tuple: (seq_logits, plane_logits, body_logits, contrast_logits)
        """
        B, N, H, W = image_slices.shape
        if N == 1:
            # If only one slice, just repeat it to create a batch of size 3 for the DenseNet
            image_slices = image_slices.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        # Encode image slices
        slice_feats = self.image_encoder(image_slices)  # (B, N_slices, slice_feat_dim)
        res = self.multi_task_head(slice_feats)
        return res
    
class MetadataBasedClassifier(nn.Module):
    """
    Metadata-only ablation model.

    Encodes DICOM metadata using either an imputer-based encoder or a sparse encoder,
    projects to an embedding vector, and predicts tasks via the multi-task head.
    """

    def __init__(
        self,
        num_classes_dict: dict,
        metadata_input_dim: int,
        metadata_embed_dim: int = 128,
        output_emb_dim: int = 256,
        metadata_encoder_type: str = "imputer", # "imputer" or "sparse"
        imputer_type: str = "contextual", # "contextual" or "ignore" (only relevant if metadata_encoder_type is "imputer"),
        sparse_enc_version: str = "v1", # if metadata_encoder_type is "sparse", which version to use ("v1", "v2", or "v5")
        incl_regression: bool = False,
    ):
        super().__init__()
        logger.info(f"Initializing MetadataBasedClassifier with metadata_encoder_type={metadata_encoder_type}")
        if metadata_encoder_type == "imputer":
            self.metadata_encoder = MetadataEncoder(
                metadata_input_dim, 
                embed_dim=metadata_embed_dim, 
                imputer=imputer_type, 
                reduce=True
            )
            logger.info(f"Using MetadataEncoder with imputer type: {imputer_type}")
        elif metadata_encoder_type == "sparse":
            if sparse_enc_version == "v1":
                self.metadata_encoder = SparseEncoderV1(
                    num_features=metadata_input_dim,
                    out_dim=metadata_embed_dim,
                    reduce=True,
                )
            elif sparse_enc_version == "v2":
                self.metadata_encoder = SparseEncoderV2(
                    num_features=metadata_input_dim,
                    out_dim=metadata_embed_dim,
                    reduce=True,
                )
            elif sparse_enc_version == "v5":
                self.metadata_encoder = SparseEncoderV5(
                    num_features=metadata_input_dim,
                    out_dim=metadata_embed_dim,
                    reduce=True,
                )
            logger.info(f"Using SparseMetadataEncoder version: {sparse_enc_version}")
        else:
            raise ValueError("metadata_encoder_type must be 'imputer' or 'sparse'")

        self.output_proj = nn.Sequential(
            nn.Linear(metadata_embed_dim, output_emb_dim),
            nn.LayerNorm(output_emb_dim),
            nn.GELU(),
        )

        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict, incl_regression=incl_regression)

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor) -> tuple:
        """
        Args:
            image_slices (torch.Tensor): MRI slices (B, N_slices, C, H, W)
        Returns:
            tuple: (seq_logits, plane_logits, body_logits, contrast_logits)
        """
        metadata_feat = self.metadata_encoder(metadata)  # (B, metadata_embed_dim)
        proj_feat = self.output_proj(metadata_feat)

        # feed combined embedding to multi-task head
        res = self.multi_task_head(proj_feat)
        return res

if __name__ == "__main__":
    # Quick test for MRISequenceClassifier
    # Dummy input sizes
    batch_size = 4
    num_slices = 3
    channels = 1  # greyscale
    height, width = 224, 224  # spatial input tensor size (h w)
    metadata_dim = 512  # example number of DICOM features per slice (or volume)

    # Number of classes for each task
    num_classes_tasks = {
        "sequence": 5,  # e.g. T1, T2, DWI, FLAIR, Other
        "plane": 3,  # axial, coronal, sagittal
        "body": 5,  # head, thorax, abdomen, extremities, other
        "contrast": 1,  # with / without
    }

    # Instantiate model
    model = MRISequenceClassifier(
        metadata_input_dim=metadata_dim,  # assuming 1 emb per slice
        num_classes_dict=num_classes_tasks,
        img_enc_backbone="densenet121",
        incl_regression=True,
        # slice_feat_dim=512,  # internal dimension of slice feature space
        # fused_feat_dim=256,  # internal dimension of fused slice feature space
        # metadata_embed_dim=128,  # internal dimension of meta data feature space
        # output_emb_dim=128,  # internal dimension of fusing image and meta data feature space
    )

    # Generate random input tensors
    images = torch.randn(batch_size, num_slices, channels, height, width)
    metadata = torch.randn(batch_size, num_slices, metadata_dim)

    # Forward pass
    res = model(images, metadata)

    # Print output shapes for verification
    for r in res:
        print(r.shape)

        
