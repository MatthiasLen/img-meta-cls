"""
Version 5
Date: 2026-02-20

Key Changes

- reworked FiLM generator
- stable scaling with alpha value 
- Self-attention blocks over modulated features T(X) := x + MHA(x)+ FF(x + MHA(x))
- Pre-norm transformer-style blocks
- changed dropout placement
- Deeper post network
- Cleaner normalization ordering
- Automatic selection of parameters

"""


import torch
import torch.nn as nn
import torch.nn.functional as func


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
        h = func.gelu(self.fc1(h))
        # h = func.gelu(self.fc2(h))
        h = self.dropout(h)
        h = self.fc2(h)
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
        - Reworked normalization and residual design
        
    """

    def __init__(
        self,
        num_features: int,
        out_dim: int = 128, # e.g. out_dim =  image_feature_dim or  out_dim = image_feature_dim // 2
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
        # This is a VERY Heuristic choice and 
        embed_dim = 32 + int(32 * round(num_features**(7/10) / 32))
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

        # 5) FiLM hidden dim
        hidden_dim = embed_dim * value_hidden_expansion

        print("Selected parameters:")
        print(f"embed_dim  = {embed_dim}")
        print(f"depth      = {depth}")
        print(f"num_heads  = {num_heads}")
        print(f"hidden_dim = {hidden_dim}")

        # ---------------------------------------------------------------------
        
        self.embed_dim = embed_dim
        self.reduce = reduce
        self.out_dim = out_dim

        # Feature identity embeddings
        self.index_embedding = nn.Embedding(num_features, embed_dim)

        # estimate FiLM values 
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
        mask = ~torch.isnan(x)  # (B, S, F)        
        x_clean = torch.nan_to_num(x, nan=0.0) # (B, S, F)

        # Feature embeddings
        feat_ids = torch.arange(F, device=device)
        feat_emb = self.index_embedding(feat_ids) # (F, embed_dim)

        # Each feature at each timestep in each batch gets its feature embedding.
        feat_emb = feat_emb.view(1, 1, F, self.embed_dim)
        feat_emb = feat_emb.expand(B, S, F, self.embed_dim) # (B, S, F, embed_dim)

        # Feature-wise Linear Modulation (FiLM) modulation
        # Each feature embedding is combined with its numeric value to predict 
        # scaling (alpha) and shifting (beta) factors.
        val_input = torch.cat([x_clean.unsqueeze(-1), feat_emb], dim=-1) # (B, S, F, embed_dim + 1)        
        alpha_beta = self.value_mlp(val_input) # (B, S, F, 2*embed_dim)
        
        assert alpha_beta.shape[-1] == 2 * self.embed_dim
        alpha, beta = alpha_beta.chunk(2, dim=-1) # 2* (B, S, F, embed_dim)
        alpha = torch.tanh(alpha)  # stable scaling

        # So now each feature embedding is adjusted based on its numeric value and context
        modulated = feat_emb * (1 + alpha) + beta # (B, S, F, embed_dim)

        # If a feature was originally NaN → its embedding becomes zero.
        modulated = modulated * mask.unsqueeze(-1) # (B, S, F, embed_dim)

        # Flatten B,S for feature-level attention
        modulated = modulated.reshape(B * S, F, self.embed_dim) # (B*S, F, embed_dim)
 
        # True where feature is missing. Passed to attention to ignore padded features.
        feature_mask = ~(mask.reshape(B * S, F)) # (B*S, F)

        # Self-attention blocks
        # Modulated features are (more or less) subjected to multiple tranformations 
        # x -> x + MHA(x) + FF(x + MHA(x))
        for attn, ff in self.blocks:
            modulated = attn(modulated, key_padding_mask=feature_mask)
            modulated = ff(modulated)

        # Masked mean pooling to collapse features into a single vector per sequence / slice.
        # Count valid features
        valid_counts = (~feature_mask).sum(dim=1).clamp(min=1).unsqueeze(-1) # (B*S, 1)
        
        # Sum only valid features. Missing features contribute 0.
        pooled = modulated.masked_fill(feature_mask.unsqueeze(-1), 0.0)  # (B*S, F, embed_dim)
        pooled = pooled.sum(dim=1) / valid_counts # (B*S, embed_dim)

        pooled = self.final_norm(pooled)
        out = self.out_proj(pooled) 
        out = out.view(B, S, self.out_dim) # (B, S, output_dim)

        # Handle slices with fully empty meta data.
        # Replace output with a learned empty_token.
        empty_mask = (mask.sum(dim=2) == 0) # (B, S)
        
        if empty_mask.any():
            out = torch.where(
                empty_mask.unsqueeze(-1),
                self.empty_token.expand(B, S, -1),
                out
            )

        # Optional Sequence Reduction (B, S, output_dim) -> (B, output_dim)
        # IMPORTANT: This averages across timesteps/slices including fully empty ones replaced with empty_token.
        # If many such empty_token exist, they influence mean. Always check that this matches your modeling assumption.
        if self.reduce:
            return out.mean(dim=1) #  (B, output_dim)

        return out


if __name__ == "__main__":
    B, S, F = 4, 8, 10
    x = torch.randn(B, S, F)
    x[torch.rand_like(x) < 0.5] = float('nan')  # 50% missing
    x[:,:, F//2] = 0.0 
    x[1,3,:] = float('nan')

    print("Input tensor:")
    print(x.shape)
    
    encoder = SparseMetadataEncoder(num_features=F, out_dim=128)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')

    z = encoder(x)

    print("\n\nEncoder output:")
    print("output shape:", z.shape) 