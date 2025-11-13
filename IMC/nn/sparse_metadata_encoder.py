import torch
import torch.nn as nn

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
    Metadata encoder that naturally handles variable-length metadata and missing entries (NaN simply omitted).
    
    Encodes sparse metadata vectors (with NaNs as missing, zeros as valid).
    Each feature i has a learned embedding e_i. For each observed value v_i,
    we compute f(v_i) = small MLP(value) that produces a scaling (alpha) and shift (beta) vector .
    The final feature representation is obtained by e_i * (1 + alpha) + beta and summed (or averaged) over all observed features.
    """

    def __init__(
        self,
        num_features: int,
        index_embed_dim: int = 64,
        value_mlp_dim: int = 32,
        out_dim: int = 128,
        aggregation: str = "mean",
        learnable_norm: bool = False,
        p_post_dropout : float = 0.05
    ):
        super().__init__()

        self.index_emb = nn.Embedding(num_features, index_embed_dim)

        # Map scalar value to shift and scale vectors
        self.value_mlp = nn.Sequential(
            nn.Linear(1, value_mlp_dim),
            nn.GELU(),
            nn.Linear(value_mlp_dim, index_embed_dim * 2)
        )

        self.post = nn.Sequential(
            ResidualMLP(index_embed_dim, index_embed_dim * 2),
            nn.Dropout(p_post_dropout),
            nn.Linear(index_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

        assert aggregation in ("sum", "mean"), "aggregation must be 'sum' or 'mean'"
        self.aggregation = aggregation

        # TODO (???) Feature gate to be applied before summation  --> maybe later! 
        # self.feature_gate = nn.Parameter(torch.ones(num_features))
        
        # Learnable affine normalization
        self.learnable_norm = learnable_norm
        if self.learnable_norm:
            self.value_scale = nn.Parameter(torch.ones(num_features))
            self.value_shift = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("value_scale", None)
            self.register_parameter("value_shift", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, F) tensor with NaN for missing features.
               Zeros are considered valid values.

        Returns:
            (B, S,  out_dim) dense embedding.
        """

        device = x.device
        
        B, S, F = x.shape
        x_flat = x.view(B*S, F)        

        # mask of observed (non-NaN) entries
        mask = ~torch.isnan(x_flat)
        idxs = torch.nonzero(mask, as_tuple=False)  # shape (N, 2) where N is num of non-zero
        
        if idxs.numel() == 0:
            # no observed entries in batch -> zero vector
            return torch.zeros(B*S, self.post[0].out_features, device=device)

        sample_idx = idxs[:, 0]
        feat_idx = idxs[:, 1]
        vals = x_flat[sample_idx, feat_idx].unsqueeze(1)  # (N,1)
        
        # Apply learnable normalization if enabled
        if self.learnable_norm:
            vals = vals * self.value_scale[feat_idx] + self.value_shift[feat_idx]

        """ 
        TODO:
        1) Keep a small dictionary of feature types "cathegorical" vs "continuous" (need to modify this in melanies feature encoder).
        2) For categorical features: ignore value_mlp, just use index_emb (optionally with one-hot presence, i.e. put NaN instead of 0s and a single 1; or small learned embedding for each category value).
        3) For numeric: use FiLM/value MLP.
        """
        
        # embeddings and modulation
        idx_emb = self.index_emb(feat_idx)         # (N, index_embed_dim)

        val_params = self.value_mlp(vals)          # (N, 2D)
        alpha, beta = val_params.chunk(2, dim=1)   # each (N, D)
        item = idx_emb * (1 + alpha) + beta        # small residual scaling and shift, FiLM style

        # TODO (???): maybe later! 
        # gate = self.feature_gate[feat_idx].unsqueeze(1)
        # item = item * gate


        # aggregate per sample
        out_dim = idx_emb.shape[1]
        agg = torch.zeros(B*S, out_dim, device=device)
        agg = agg.index_add(0, sample_idx, item)    # sum along batch

        if self.aggregation == "mean":
            # divide by number of observed entries per sample
            counts = mask.sum(dim=1).clamp(min=1).float().unsqueeze(1)
            agg = agg / counts

        out_flat = self.post(agg)                       # (B * S, out_dim)
        out = out_flat.view(B, S, -1)                   # (B, S, out_dim)
        
        return out

if __name__ == "__main__":
    B, S, F = 4, 8, 10
    x = torch.randn(B, S, F)
    x[torch.rand_like(x) < 0.8] = float('nan')  # 80% missing
    x[:,:, F//2] = 0.0 

    print("Input tensor:")
    print(x)
    print(x.shape)
    
    encoder = SparseMetadataEncoder(num_features=F, out_dim=128)
    z = encoder(x)

    print("\n\nEncoder output:")
    print(z)
    print("output shape:", z.shape) 
