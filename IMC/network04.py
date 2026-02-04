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
from IMC.nn.emb_metadata_encoder import FTTransformerLikeMetadataEncoder
from IMC.nn.multi_task_head import MultiTaskHead
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
        imputer_type: str = "contextual",
        img_enc_backbone: str | None = "swin", # "densenet" or "swin", None for resnet50 as default
        incl_regression: bool = True,
        dropout_metadata: bool = False,
        fusion_module_version: str = "v1" #v1 or v2
    ):
        super().__init__()
        self.image_encoder = MultiSliceImageEncoder(backbone=img_enc_backbone) 
        slice_feat_dim = self.image_encoder.get_feature_dimension()

        if fusion_module_version == "v1":
            self.embedding_fusion = BiDirectionalCrossModalAttentionFusion(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
            )
            self.slice_fusion = SliceFeatureFusion(
                slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim, reduce=True
            )

            self.metadata_encoder = MetadataEncoder(
                metadata_input_dim, embed_dim=metadata_embed_dim, imputer=imputer_type, reduce='none'
            )
        else:
            self.embedding_fusion = BiDirectionalCrossModalAttentionFusionV2(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
            )
            self.slice_fusion = SliceFeatureFusion(
                slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim, reduce=False
            )

            self.metadata_encoder = MetadataEncoder(
                metadata_input_dim, embed_dim=metadata_embed_dim, imputer=imputer_type, reduce='none'
            )

        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict, incl_regression=incl_regression)

        self.dropout_metadata = dropout_metadata

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
        # The 8th feature (enc_ScanOptions_fatsat) is masked more frequently to stress-test robustness
        if self.training and self.dropout_metadata:
            prob = 0.3
            mask = torch.rand(metadata.shape, device=metadata.device) < prob
            mask = mask | (torch.rand(metadata.shape, device=metadata.device) < prob) * (
                torch.arange(metadata.shape[1], device=metadata.device) == 8
            )
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
        
        # Encode metadata
        metadata_feat = self.metadata_encoder(metadata)  # (B, metadata_embed_dim)

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

        # feed combined embedding to multi-task head
        res = self.multi_task_head(joint_feat)

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

class MRISequenceClassifierWithSparseMetadata(nn.Module):
    """
    Multi-modal classifier using sparse metadata encoding.

    Uses SparseMetadataEncoder (v1 or v2) or an FT-like encoder to embed metadata with
    NaN handling, fuses with image features via cross-attention, and predicts multiple tasks.

    Args:
        num_classes_dict: Mapping of task name to number of classes.
        metadata_input_dim: Number of metadata features.
        metadata_embed_dim: Dimension of metadata embedding.
        fused_feat_dim: Dimension of fused slice features.
        output_emb_dim: Dimension of fused image+metadata embedding.
        metadata_embeder_type: 'sparse', 'sparse_v2', or 'ft'.
        include_regression: Whether to include regression task in the head.
        img_enc_backbone: Image encoder backbone name.
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
        metadata_embeder_type: str = "sparse", # "ft" or "sparse" or "sparse_v2",
        include_regression: bool = True,
        img_enc_backbone: str | None = "swin", # "densenet" or "swin", None for resnet50 as default
        dropout_metadata: bool = False,
        fusion_module_version: str = "v1" #v1 or v2
    ):
        super().__init__()
        self.image_encoder = MultiSliceImageEncoder(backbone=img_enc_backbone)
        slice_feat_dim = self.image_encoder.get_feature_dimension()

        if metadata_embeder_type == "sparse_v2":
            self.metadata_encoder = SparseEncoderV2(
                num_features=metadata_input_dim,
                out_dim=metadata_embed_dim,
                reduce=True if fusion_module_version == "v1" else False
            ) 
        elif metadata_embeder_type == "sparse":
            self.metadata_encoder = SparseEncoderV1(
                num_features=metadata_input_dim,
                out_dim=metadata_embed_dim,
                reduce=True if fusion_module_version == "v1" else False
            )           
        elif metadata_embeder_type == "ft":
            self.metadata_encoder = FTTransformerLikeMetadataEncoder(
                out_dim=metadata_embed_dim,
            )
        if fusion_module_version == "v1":
            self.embedding_fusion = BiDirectionalCrossModalAttentionFusion(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
            )
            self.slice_fusion = SliceFeatureFusion(
                slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim, reduce=True
            )
        else:
            self.embedding_fusion = BiDirectionalCrossModalAttentionFusionV2(
                image_emb_dim=fused_feat_dim,
                metadata_emb_dim=metadata_embed_dim,
                output_dim=output_emb_dim,
            )
            self.slice_fusion = SliceFeatureFusion(
                slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim, reduce=False
            )
        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict, incl_regression=include_regression)
        self.dropout_metadata = dropout_metadata

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor) -> tuple:
        """
        Args:
            image_slices (torch.Tensor): MRI slices (B, N_slices, C, H, W)
            metadata (torch.Tensor): Sparse DICOM metadata (B, metadata_input_dim) with NaNs for missing values
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
        # Randomly set metadata to nan during training
        if self.training and self.dropout_metadata:
            prob = 0.3
            # Mask mostly the 8-th column (enc_ScanOptions_fatsat) which is the most important feature, but also randomly mask other features to make model more robust to missing values
            mask = torch.rand(metadata.shape, device=metadata.device) < prob
            mask = mask | (torch.rand(metadata.shape, device=metadata.device) < prob) * (torch.arange(metadata.shape[2], device=metadata.device) == 8) # Ensure the 8-th column has higher chance to be masked
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
        
        # Encode sparse metadata
        # Metadata has shape of (B, metadata_input_dim) with NaNs for missing values
        # metadata = metadata.unsqueeze(1)    # (B, 1, metadata_input_dim) N_slices = 1 for SparseMetadataEncoder
        metadata_feat = self.metadata_encoder(metadata)  # (B, metadata_embed_dim)

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

        # feed combined embedding to multi-task head
        res = self.multi_task_head(joint_feat)

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
            metadata (torch.Tensor): Sparse DICOM metadata (B, metadata_input_dim) with NaNs for missing values
        Returns:
            torch.Tensor: Metadata features (B, metadata_embed_dim)
        """
        metadata = metadata.unsqueeze(1)    # (B, 1, metadata_input_dim) N_slices = 1 for SparseMetadataEncoder
        return self.metadata_encoder(metadata)

class ImageFusionClassifier(nn.Module):
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
        backbone: str | None = "swin", # "densenet" or "swin", None for resnet50 as default
    ):
        super().__init__()
        self.image_encoder = MultiSliceImageEncoder(backbone=backbone)
        
        slice_feat_dim = self.image_encoder.get_feature_dimension()

        self.slice_fusion = SliceFeatureFusion(
            slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim
        )

        self.output_proj = nn.Sequential(
            nn.Linear(fused_feat_dim, output_emb_dim),
            nn.LayerNorm(output_emb_dim),
            nn.GELU(),
        )

        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict)

    def forward(self, image_slices: torch.Tensor) -> tuple:
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

class MetadataFusionClassifier(nn.Module):
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
        imputer_type: str = "contextual", # "contextual" or "ignorer"
    ):
        super().__init__()
        logger.info(f"Initializing MetadataFusionClassifier with metadata_encoder_type={metadata_encoder_type}")
        if metadata_encoder_type == "imputer":
            self.metadata_encoder = MetadataEncoder(
                metadata_input_dim, embed_dim=metadata_embed_dim, imputer=imputer_type
            )
            logger.info(f"Using MetadataEncoder with imputer type: {imputer_type}")
        elif metadata_encoder_type == "sparse":
            self.metadata_encoder = SparseMetadataEncoder(
                num_features=metadata_input_dim,
                out_dim=metadata_embed_dim,
                reduce=True,
            )
        else:
            raise ValueError("metadata_encoder_type must be 'imputer' or 'sparse'")

        self.output_proj = nn.Sequential(
            nn.Linear(metadata_embed_dim, output_emb_dim),
            nn.LayerNorm(output_emb_dim),
            nn.GELU(),
        )

        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict)

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor) -> tuple:
        """
        Args:
            image_slices (torch.Tensor): MRI slices (B, N_slices, C, H, W)
        Returns:
            tuple: (seq_logits, plane_logits, body_logits, contrast_logits)
        """
        
        # Encode metadata
        if self.metadata_encoder.__class__ == SparseMetadataEncoder:
            metadata = metadata.unsqueeze(1)    # (B, 1, metadata_input_dim) N_slices = 1 for SparseMetadataEncoder
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

        
