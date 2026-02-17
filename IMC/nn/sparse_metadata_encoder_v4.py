import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """
    Two-layer residual MLP block with pre-norm stabilization.

    Architecture:
        x -> LayerNorm -> Linear -> GELU -> Linear -> Dropout -> + x

    This pre-norm design is more stable for deeper stacking
    compared to post-norm residual blocks.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class SparseMetadataEncoder(nn.Module):
    """
    Sparse metadata encoder using gated FiLM modulation and
    attention-based feature aggregation.

    This module treats metadata as a set of (feature_id, value) pairs.

    Improvements over naive FiLM + mean pooling:
        - Fully vectorized masked computation (no torch.nonzero)
        - Stabilized FiLM scaling (tanh)
        - Learnable feature importance gating
        - Attention-based aggregation across features
        - Learned empty metadata token
        - Pre-norm residual refinement

    Args:
        num_features: Total number of metadata features.
        embed_dim: Feature index embedding dimension.
        value_hidden_dim: Hidden size for value MLP.
        out_dim: Output embedding dimension.
        dropout: Dropout probability.
        reduce: Whether to average over sequence dimension S.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        value_hidden_dim: int = 64,
        out_dim: int = 128,
        dropout: float = 0.1,
        reduce: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.reduce = reduce

        # Feature identity embeddings
        self.index_embedding = nn.Embedding(num_features, embed_dim)

        # Value modulation network
        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, value_hidden_dim),
            nn.GELU(),
            nn.Linear(value_hidden_dim, embed_dim * 2),
        )

        # Feature importance scoring (attention weights)
        self.feature_attn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Post aggregation refinement
        self.post = nn.Sequential(
            ResidualMLP(embed_dim, embed_dim * 2, dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        # Learned representation for completely missing metadata
        self.empty_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, S, F)
               NaN indicates missing values.

        Returns:
            Tensor:
                (B, out_dim) if reduce=True
                (B, S, out_dim) otherwise
        """
        B, S, F = x.shape
        device = x.device

        # Mask of observed features
        mask = ~torch.isnan(x)  # (B, S, F)

        # Replace NaNs with zero for safe arithmetic
        x_clean = torch.nan_to_num(x, nan=0.0)

        # Get feature embeddings
        feature_ids = torch.arange(F, device=device)
        feat_emb = self.index_embedding(feature_ids)  # (F, D)
        feat_emb = feat_emb.view(1, 1, F, self.embed_dim)

        # Expand to batch
        feat_emb = feat_emb.expand(B, S, F, self.embed_dim)

        # Concatenate value + embedding
        val_input = torch.cat(
            [x_clean.unsqueeze(-1), feat_emb], dim=-1
        )  # (B, S, F, D+1)

        # Predict FiLM parameters
        alpha_beta = self.value_mlp(val_input)
        alpha, beta = alpha_beta.chunk(2, dim=-1)

        # Stabilized scaling
        alpha = torch.tanh(alpha)

        # Apply FiLM
        modulated = feat_emb * (1 + alpha) + beta

        # Zero-out missing features
        modulated = modulated * mask.unsqueeze(-1)

        # Attention scores per feature
        attn_logits = self.feature_attn(modulated).squeeze(-1)  # (B, S, F)

        # Mask attention
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, S, F)

        # Weighted sum
        aggregated = torch.sum(
            modulated * attn_weights.unsqueeze(-1), dim=2
        )  # (B, S, D)

        # Detect completely empty metadata
        empty_mask = (mask.sum(dim=2) == 0)  # (B, S)

        out = self.post(aggregated)  # (B, S, out_dim)

        # Replace empty positions with learned token
        if empty_mask.any():
            out = torch.where(
                empty_mask.unsqueeze(-1),
                self.empty_token.expand(B, S, -1),
                out,
            )

        if self.reduce:
            return out.mean(dim=1)

        return out
