
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

        x_flat = x.view(B * S, F)

        # Create a full feature representation
        all_feats = torch.arange(F, device=device).long().expand(B * S, -1) # (B*S, F)
        idx_emb = self.index_emb(all_feats) # (B*S, F, D)

        # Identify observed and missing values
        nan_mask = torch.isnan(x_flat) # (B*S, F)

        # Create modulated features for observed values
        observed_mask = ~nan_mask
        sample_idx, feat_idx = torch.nonzero(observed_mask, as_tuple=True)

        if sample_idx.numel() > 0:
            vals = x_flat[sample_idx, feat_idx].unsqueeze(1)

            observed_idx_emb = idx_emb[sample_idx, feat_idx]

            val_input = torch.cat([vals, observed_idx_emb], dim=1)
            val_params = self.value_mlp(val_input)
            alpha, beta = val_params.chunk(2, dim=1)

            modulated_feat = observed_idx_emb * (1 + alpha) + beta
            idx_emb[sample_idx, feat_idx] = modulated_feat

        # Apply NaN embedding for missing values
        nan_sample_idx, nan_feat_idx = torch.nonzero(nan_mask, as_tuple=True)
        if nan_sample_idx.numel() > 0:
            idx_emb[nan_sample_idx, nan_feat_idx] = self.nan_embedding

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B * S, -1, -1)
        full_sequence = torch.cat([cls_tokens, idx_emb], dim=1)

        # Process through transformer
        transformer_out = self.transformer_encoder(full_sequence)

        # Get CLS token output
        cls_output = transformer_out[:, 0]

        # Post-processing
        out_flat = self.post(cls_output)

        out = out_flat.view(B, S, self.out_dim)

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
