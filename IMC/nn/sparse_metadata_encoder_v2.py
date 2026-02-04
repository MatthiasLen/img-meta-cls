
import torch
import torch.nn as nn
import logging

class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h =  torch.nn.functional.gelu(self.lin1(x))
        h = self.lin2(h)
        return self.norm(x + h)

class SparseMetadataEncoder(nn.Module):
    """
    Metadata encoder that handles variable-length metadata.
    - Missing features (NaN) are represented by a specific learnable embedding.
    - A CLS token is used for aggregation, processed by a transformer layer.
    """

    def __init__(
        self,
        num_features: int,
        index_embed_dim: int = 64,
        value_mlp_dim: int = 32,
        out_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 4,
        reduce: bool = True
    ):
        super().__init__()

        self.out_dim = out_dim
        self.index_emb = nn.Embedding(num_features, index_embed_dim)

        # Learnable embedding for NaN values
        self.nan_embedding = nn.Parameter(torch.randn(index_embed_dim))

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, index_embed_dim))

        # FiLM generator
        self.value_mlp = nn.Sequential(
            nn.Linear(1 + index_embed_dim, value_mlp_dim),
            nn.GELU(),
            nn.Linear(value_mlp_dim, index_embed_dim * 2) # alpha and beta
        )

        # Transformer Encoder for aggregation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=index_embed_dim,
            nhead=transformer_heads,
            dim_feedforward=index_embed_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.post = nn.Sequential(
            nn.LayerNorm(index_embed_dim),
            nn.Linear(index_embed_dim, out_dim),
            nn.GELU()
        )
        self.reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, F) tensor with NaN for missing features.
        Returns:
            (B, S, out_dim) dense embedding after CLS token aggregation.
        """
        B, S, F = x.shape
        device = x.device

        x_flat = x.reshape(B * S, F)

        # ------------------------------------------------------------------
        # Feature index embeddings
        # ------------------------------------------------------------------
        all_feats = torch.arange(F, device=device).unsqueeze(0).expand(B * S, F)
        idx_emb = self.index_emb(all_feats)  # (B*S, F, D)

        # ------------------------------------------------------------------
        # Masks
        # ------------------------------------------------------------------
        nan_mask = torch.isnan(x_flat)                # (B*S, F)
        observed_mask = ~nan_mask                     # (B*S, F)
        observed_mask_f = observed_mask.unsqueeze(-1) # (B*S, F, 1)

        # ------------------------------------------------------------------
        # Observed value modulation (fully dense, masked)
        # ------------------------------------------------------------------
        # Replace NaNs with zero for safe concatenation
        safe_vals = torch.where(nan_mask, torch.zeros_like(x_flat), x_flat)
        vals = safe_vals.unsqueeze(-1)                # (B*S, F, 1)

        # Prepare value MLP input
        val_input = torch.cat([vals, idx_emb], dim=-1)   # (B*S, F, 1+D)
        val_params = self.value_mlp(val_input)           # (B*S, F, 2D)

        alpha, beta = val_params.chunk(2, dim=-1)

        modulated = idx_emb * (1.0 + alpha) + beta

        # Apply modulation only where observed
        idx_emb = torch.where(observed_mask_f, modulated, idx_emb)

        # ------------------------------------------------------------------
        # NaN embedding for missing values
        # ------------------------------------------------------------------
        nan_emb = self.nan_embedding.view(1, 1, -1)
        idx_emb = torch.where(nan_mask.unsqueeze(-1), nan_emb, idx_emb)

        # ------------------------------------------------------------------
        # CLS token
        # ------------------------------------------------------------------
        cls_tokens = self.cls_token.expand(B * S, 1, -1)
        full_sequence = torch.cat([cls_tokens, idx_emb], dim=1)

        # ------------------------------------------------------------------
        # Transformer
        # ------------------------------------------------------------------
        transformer_out = self.transformer_encoder(full_sequence)
        cls_output = transformer_out[:, 0]

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------
        out_flat = self.post(cls_output)
        out = out_flat.reshape(B, S, self.out_dim)

        if self.reduce:
            out = out.mean(dim=1)

        return out

if __name__ == "__main__":
    B, S, F = 4, 8, 10
    x = torch.randn(B, S, F)
    x[torch.rand_like(x) < 0.5] = float('nan')

    print("Input tensor (sample from batch):")
    print(x[0,0,:])

    encoder = SparseMetadataEncoder(num_features=F, out_dim=128)
    z = encoder(x)

    print("\nEncoder output shape:", z.shape)
    # assert z.shape == (B, 128)
    print("Done.")
