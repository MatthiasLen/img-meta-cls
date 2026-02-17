"""
Version 5
Date: 2026-02-17

Key Changes

- reworked FiLM generator
- Self-attention block over features
- Pre-norm transformer-style blocks
- Controlled scaling
- changed dropout placement
- Deeper post network
- Cleaner normalization ordering

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Transformer-style feedforward block:
        x -> LN -> Linear -> GELU -> Dropout -> Linear -> Dropout -> + x
    """

    def __init__(self, dim: int, expansion: int, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        return x + self.dropout(h)


class MetadataSelfAttention(nn.Module):
    """
    Pre-norm multi-head self-attention block with residual connection.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        h = self.norm(x)
        h, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return x + self.dropout(h)


class SparseMetadataEncoder(nn.Module):
    """
    Advanced sparse metadata encoder using:

        - Stabilized FiLM modulation
        - Multi-head self-attention across features
        - Transformer-style feedforward refinement
        - Proper normalization and residual design

    Designed for medical metadata fusion.
    """

    def __init__(
        self,
        num_features: int,
        # embed_dim: int = 128,
        # value_hidden_expansion: int = 2,
        # num_heads: int = 4,
        # depth: int = 2,
        out_dim: int = 256, # e.g. out_dim =  image_feature_dim or  out_dim = image_feature_dim // 2
        dropout: float = 0.1,
        reduce: bool = True,
    ):
        super().__init__()

        # Auto scaling hyperparameters.
        # 
        # IMPORTANT: Assumptions of these scaling laws will require some experimentation.
        # The true driver for selecting network capacity ENTROPY of the metadata.
        # This we do not adress here and use a rather weak estimate based on num_features.
        # Can be investigated later.
        
        # 1) embed_dim scales sublinearly
        embed_dim = int(32 * round(math.sqrt(num_features) / 32))
        embed_dim = max(64, min(embed_dim, 256))
    
        # 2) depth capped small
        if num_features <= 128:
            depth = 2
        else:
            depth = 3
    
        # 3) heads ~ 32-dim per head
        num_heads = max(2, min(embed_dim // 32, 8))
        
        # 4) value expansion modest
        value_hidden_expansion = 2 if num_features < 64 else 3

        # ---------------------------------------------------------------------
        
        self.embed_dim = embed_dim
        self.value_hidden_expansion = value_hidden_expansion
        self.num_heads = num_heads
        self.depth = depth
        
        self.reduce = reduce

        # Feature identity embeddings
        self.index_embedding = nn.Embedding(num_features, embed_dim)

        # estimate FiLM values 
        hidden_dim = embed_dim * value_hidden_expansion

        self.value_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim + 1),
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2),
        )

        # Transformer blocks over feature set
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MetadataSelfAttention(embed_dim, num_heads, dropout),
                FeedForward(embed_dim, expansion=value_hidden_expansion, dropout=dropout),
            ])
            for _ in range(depth)
        ])

        # Final projection
        self.final_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        self.empty_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B = batch size, S = sequence length / number of slices, F = number of features
        B, S, F = x.shape
        device = x.device

        # True where values are valid
        mask = ~torch.isnan(x)
        x_clean = torch.nan_to_num(x, nan=0.0) # (B, S, F)

        # Feature embeddings
        feat_ids = torch.arange(F, device=device)
        feat_emb = self.index_embedding(feat_ids) # (F, embed_dim)

        # Each feature at each timestep in each batch gets its feature embedding.
        feat_emb = feat_emb.view(1, 1, F, self.embed_dim)
        feat_emb = feat_emb.expand(B, S, F, self.embed_dim) # (B, S, F, embed_dim)

        # Feature-wise Linear Modulation (FiLM) modulation
        # Each feature embedding is combined with its numeric value to predict scaling (alpha) and shifting (beta) factors.
        val_input = torch.cat([x_clean.unsqueeze(-1), feat_emb], dim=-1) # (B, S, F, embed_dim + 1)
        alpha_beta = self.value_mlp(val_input)
        alpha, beta = alpha_beta.chunk(2, dim=-1)

        alpha = torch.tanh(alpha)  # stable scaling

        # So now each feature embedding is adjusted based on its numeric value and context
        modulated = feat_emb * (1 + alpha) + beta

        # If a feature was originally NaN → its embedding becomes zero.
        modulated = modulated * mask.unsqueeze(-1)

        # Flatten B,S for feature-level attention
        modulated = modulated.view(B * S, F, self.embed_dim)

        # True where feature is missing. Passed to attention to ignore padded features.
        feature_mask = ~(mask.view(B * S, F))

        # Self-attention blocks
        for attn, ff in self.blocks:
            modulated = attn(modulated, key_padding_mask=feature_mask)
            modulated = ff(modulated)

        # Masked mean pooling to collapse features into a single vector per sequence / slice.
        
        # Count valid features
        valid_counts = (~feature_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)

        # Sum only valid features. Missing features contribute 0.
        pooled = modulated.masked_fill(feature_mask.unsqueeze(-1), 0.0).sum(dim=1)
        pooled = pooled / valid_counts # (B*S, embed_dim)

        pooled = self.final_norm(pooled)
        out = self.out_proj(pooled) 
        out = out.view(B, S, -1) # (B, S, output_dim)

        # Handle fully empty cases
        # Replace output with a learned empty_token. This prevents garbage output when no data exists.
        empty_mask = (mask.sum(dim=2) == 0)
        if empty_mask.any():
            out = torch.where(
                empty_mask.unsqueeze(-1),
                self.empty_token.expand(B, S, -1),
                out
            )

        # Optional Sequence Reduction
        # (B, S, D) -> (B, D)
        if self.reduce:
            return out.mean(dim=1)

        return out
