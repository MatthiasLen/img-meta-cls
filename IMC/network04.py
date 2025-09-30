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
from nn.image_encoder import MultiSliceImageEncoder
from nn.metadata_encoder import MetadataEncoder
from nn.multi_task_head import MultiTaskHead

class SliceFeatureFusion(nn.Module):
    """
    Fuses slice embeddings using transformer-style multi-head self-attention.
    Models inter-slice relationships for richer feature representation.
    """

    def __init__(
        self,
        slice_feat_dim: int = 512,
        fused_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
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
        pooled = x.mean(dim=1)  # (B, C)

        # Final MLP projection to fused_dim
        fused = self.out_proj(pooled)  # (B, fused_dim)
        return fused


class BiDirectionalCrossModalAttentionFusion(nn.Module):
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



class MRISequenceClassifier(nn.Module):
    """
    Main model for multi-task MRI classification.
    Combines image slice features and metadata using cross-attention fusion,
    then predicts sequence type, viewplane, body region, and contrast.
    """

    def __init__(
        self,
        metadata_input_dim: int,
        num_classes_dict: dict,
        fused_feat_dim: int = 256,
        metadata_embed_dim: int = 128,
        output_emb_dim: int = 128,
    ):
        super().__init__()
        self.image_encoder = MultiSliceImageEncoder()
        
        slice_feat_dim = self.image_encoder.get_feature_dimension()

        self.slice_fusion = SliceFeatureFusion(
            slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim
        )

        self.metadata_encoder = MetadataEncoder(
            metadata_input_dim, embed_dim=metadata_embed_dim
        )

        self.embedding_fusion = BiDirectionalCrossModalAttentionFusion(
            image_emb_dim=fused_feat_dim,
            metadata_emb_dim=metadata_embed_dim,
            output_dim=output_emb_dim,
        )

        self.multi_task_head = MultiTaskHead(output_emb_dim, num_classes_dict)

    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor) -> tuple:
        """
        Args:
            image_slices (torch.Tensor): MRI slices (B, N_slices, C, H, W)
            metadata (torch.Tensor): DICOM metadata (B, metadata_input_dim)
        Returns:
            tuple: (seq_logits, plane_logits, body_logits, contrast_logits)
        """
        
        # Encode image slices
        slice_feats = self.image_encoder(image_slices)  # (B, N_slices, slice_feat_dim)
        fused_img_feat = self.slice_fusion(slice_feats)  # (B, fused_feat_dim)
        
        # Encode metadata
        metadata_feat = self.metadata_encoder(metadata)  # (B, metadata_embed_dim)
        
        # Fuse features from image embedding and meta data embedding
        joint_feat = self.embedding_fusion(fused_img_feat, metadata_feat)

        # feed combined embedding to multi-task head
        res = self.multi_task_head(joint_feat)
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
        metadata_input_dim=num_slices * metadata_dim,  # assuming 1 emb per slice
        num_classes_dict=num_classes_tasks,
        # slice_feat_dim=512,  # internal dimension of slice feature space
        # fused_feat_dim=256,  # internal dimension of fused slice feature space
        # metadata_embed_dim=128,  # internal dimension of meta data feature space
        # output_emb_dim=128,  # internal dimension of fusing image and meta data feature space
    )

    # Generate random input tensors
    images = torch.randn(batch_size, num_slices, channels, height, width)
    metadata = torch.randn(batch_size, num_slices * metadata_dim)

    # Forward pass
    res = model(images, metadata)

    # Print output shapes for verification
    for r in res:
        print(r.shape)

        
