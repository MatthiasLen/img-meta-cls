class SparseMetadataEncoder(nn.Module):
    """
    Metadata encoder that naturally handles variable-length metadata and missing entries (NaN simply omitted).
    
    Encodes sparse metadata vectors (with NaNs as missing, zeros as valid).
    Each feature i has a learned embedding e_i. For each observed value v_i,
    we compute f(v_i) = small MLP(value) that produces a modulation vector.
    The final feature representation is e_i * f(v_i), summed (or averaged) over all observed features.
    """

    def __init__(self,
                 num_features: int,
                 index_embed_dim: int = 64,
                 value_mlp_dim: int = 32,
                 out_dim: int = 128,
                 aggregation: str = "mean"):
        super().__init__()

        self.index_emb = nn.Embedding(num_features, index_embed_dim)

        # Map scalar value -> modulation vector of same dim as index_emb
        self.value_mlp = nn.Sequential(
            nn.Linear(1, value_mlp_dim),
            nn.ReLU(),
            nn.Linear(value_mlp_dim, index_embed_dim)
        )

        self.post = nn.Sequential(
            nn.Linear(index_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

        assert aggregation in ("sum", "mean"), "aggregation must be 'sum' or 'mean'"
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F) tensor with NaN for missing features.
               Zeros are considered valid values.

        Returns:
            (B, out_dim) dense embedding.
        """
        B, F = x.shape
        device = x.device

        # mask of observed (non-NaN) entries
        mask = ~torch.isnan(x)
        idxs = torch.nonzero(mask, as_tuple=False)  # shape (N, 2)
        if idxs.numel() == 0:
            # no observed entries in batch (unlikely) -> zero vector
            return torch.zeros(B, self.post[0].out_features, device=device)

        batch_idx = idxs[:, 0]
        feat_idx = idxs[:, 1]
        vals = x[batch_idx, feat_idx].unsqueeze(1)  # (N,1)

        # embeddings and modulation
        idx_emb = self.index_emb(feat_idx)         # (N, d)
        val_emb = self.value_mlp(vals)             # (N, d)

        item = idx_emb * val_emb                   # (N, d)

        # aggregate per batch
        out_dim = idx_emb.shape[1]
        agg = torch.zeros(B, out_dim, device=device)
        agg = agg.index_add(0, batch_idx, item)    # sum along batch

        if self.aggregation == "mean":
            # divide by number of observed entries per sample
            counts = mask.sum(dim=1).clamp(min=1).float().unsqueeze(1)
            agg = agg / counts

        out = self.post(agg)                       # (B, out_dim)
        return out
