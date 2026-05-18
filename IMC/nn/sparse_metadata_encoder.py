import torch
import torch.nn as nn
import logging

logger = logging.getLogger("IMC")


class ResidualMLP(nn.Module):
    """
    Residual Multi-Layer Perceptron (MLP) with Layer Normalization.

    Implements a residual connection around a two-layer MLP with GELU activation.
    The residual connection helps with gradient flow and enables deeper networks.

    Architecture:
        Input -> Linear -> GELU -> Linear -> Add residual -> LayerNorm -> Output

    Args:
        dim (int): Input and output dimension.
        hidden (int): Hidden layer dimension.
    """

    def __init__(self, dim, hidden):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Output tensor of shape (..., dim) after residual connection and normalization.
        """
        h = torch.nn.functional.gelu(self.lin1(x))
        h = self.lin2(h)
        return self.norm(x + h)


class ValueNetwork(nn.Module):
    """
    Value MLP: Maps scalar feature value contextualized by feature embedding to FiLM parameters
    Input: [value (1D), feature_embedding (index_embed_dim)]
    Output: [alpha, beta]
    """

    def __init__(self, index_embed_dim: int, value_mlp_dim: int, scalar_modulation: bool = False):
        super().__init__()
        value_mlp_out_dim = 2 if scalar_modulation else (2 * index_embed_dim)
        print("Value Network output dim:", value_mlp_out_dim)

        self.lin1 = nn.Linear(1 + index_embed_dim, value_mlp_dim)
        self.lin2 = nn.Linear(value_mlp_dim, value_mlp_out_dim)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.lin1(x))
        h = self.lin2(h)
        return h


class SparseMetadataEncoder(nn.Module):
    """
    Sparse Metadata Encoder with Feature-wise Linear Modulation (FiLM).

    This encoder is designed to handle variable-length metadata with missing entries (represented as NaN).
    It uses a FiLM-based architecture to combine learned feature embeddings with their numeric values.

    Key Features:
    - **Sparse Input Handling**: NaN values indicate missing features and are excluded from processing.
      Zero values are treated as valid observations.
    - **Feature Embeddings**: Each feature index has a learnable embedding vector that captures
      feature-specific semantic information.
    - **FiLM Modulation**: For each observed feature value, a small MLP predicts alpha (scaling) and
      beta (shift) parameters to modulate the feature embedding: e_i * (1 + alpha) + beta.
    - **Flexible Aggregation**: Observed features are aggregated using sum or mean pooling.
    - **Optional Learnable Normalization**: Per-feature affine normalization can be applied to
      standardize input values based on training data statistics.

    Architecture Overview:
        1. Extract non-NaN features and their indices
        2. (Optional) Apply learnable per-feature normalization
        3. Generate feature embeddings from indices
        4. Use value MLP to predict FiLM parameters (alpha, beta) from [value, embedding]
        5. Apply FiLM modulation: embedding * (1 + alpha) + beta
        6. Aggregate modulated features per sample (sum or mean)
        7. Apply post-processing MLP with residual connections

    This approach is particularly effective for medical metadata where:
    - Features may be missing (NaN) due to variations in acquisition protocols
    - Different features have different semantic meanings (captured by embeddings)
    - Numeric values need to be contextualized by which feature they belong to
    """

    def __init__(
        self,
        num_features: int,
        index_embed_dim: int = 64,
        value_mlp_dim: int = 32,
        out_dim: int = 128,
        aggregation: str = "mean",
        learnable_norm: bool = False,
        p_post_dropout: float = 0.05,
        reduce: bool = True,
        scalar_modulation: bool = False,
        learn_missing_emb: bool = False,
    ):
        """
        Initialize the Sparse Metadata Encoder.

        Args:
            num_features (int): Total number of features in the metadata vector. This defines
                the vocabulary size for feature embeddings.
            index_embed_dim (int, optional): Dimension of the learned feature index embeddings.
                Default: 64.
            value_mlp_dim (int, optional): Hidden dimension of the MLP that predicts FiLM parameters.
                Default: 32.
            out_dim (int, optional): Output embedding dimension after post-processing. Default: 128.
            aggregation (str, optional): Method to aggregate features across each sample.
                Options: 'sum' or 'mean'. Default: 'mean'.
            learnable_norm (bool, optional): If True, applies learnable per-feature affine
                normalization (scale and shift) to input values. Default: False.
            p_post_dropout (float, optional): Dropout probability in the post-processing network.
                Default: 0.05.
            reduce (bool, optional): If True, applies mean pooling over the sequence dimension
                to produce a single vector per batch. If False, returns one vector per sequence
                position. Default: True.
            scalar_modulation (bool, optional) : If True use scalar instead of vector alpha and
                beta modulation parameter. Default: False.
        """
        super().__init__()

        self.out_dim = out_dim
        self.scalar_modulation = scalar_modulation

        # Learnable embedding for each feature index
        self.index_emb = nn.Embedding(num_features, index_embed_dim)

        # Learnable token substituted for the aggregated representation of samples
        # where ALL features are NaN (e.g. NIfTI slices that carry no DICOM metadata).
        # When learn_missing_emb=True a dedicated parameter is trained; otherwise a
        # fixed 0 vector is used (backward compatibility with checkpoints trained before learn_missing_emb was introduced).
        if learn_missing_emb:
            logger.info("Using learnable missing embedding")
            self.missing_emb = nn.Parameter(torch.randn(index_embed_dim))
        else:
            # Fixed zero vector for backward compatibility with checkpoints trained
            # before learn_missing_emb was introduced.  All-NaN samples will produce
            # post(zeros); this is stable as long as the model was trained that way.
            self.register_buffer("missing_emb", torch.zeros(index_embed_dim), persistent=False)
        self.learn_missing_emb = learn_missing_emb

        # Value MLP: Maps scalar feature value contextualized by feature embedding to FiLM parameters
        # Input: [value (1D), feature_embedding (index_embed_dim)] -> Output: [alpha, beta]

        if not scalar_modulation:
            value_mlp_out_dim = index_embed_dim * 2
            self.value_mlp = nn.Sequential(
                nn.Linear(1 + index_embed_dim, value_mlp_dim),
                nn.GELU(),
                nn.Linear(value_mlp_dim, value_mlp_out_dim),  # FiLM params: alpha and beta
            )
        else:
            self.value_mlp = ValueNetwork(
                index_embed_dim=index_embed_dim, value_mlp_dim=value_mlp_dim, scalar_modulation=scalar_modulation
            )

        # Post-processing network: Refines aggregated features and projects to output dimension
        self.post = nn.Sequential(
            ResidualMLP(index_embed_dim, index_embed_dim * 2),
            nn.Dropout(p_post_dropout),
            nn.Linear(index_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        assert aggregation in ("sum", "mean"), "aggregation must be 'sum' or 'mean'"
        self.aggregation = aggregation

        # Optional learnable affine normalization per feature
        # This normalization is learned based on training dataset statistics and applied
        # before the FiLM modulation. It's independent and not contextualized like FiLM.
        self.learnable_norm = learnable_norm
        if self.learnable_norm:
            self.value_scale = nn.Parameter(torch.ones(num_features))
            self.value_shift = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("value_scale", None)
            self.register_parameter("value_shift", None)

        self.reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode sparse metadata.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, F) where:
                - B is the batch size
                - S is the sequence length (e.g., number of slices or time steps)
                - F is the number of features
                NaN values indicate missing features. Zeros are considered valid values.

        Returns:
            torch.Tensor: Encoded metadata of shape:
                - (B, out_dim) if reduce=True (mean pooling over sequence)
                - (B, S, out_dim) if reduce=False (one embedding per sequence position)
        """

        device = x.device

        B, S, F = x.shape

        # Flatten batch and sequence dimensions for processing
        x_flat = x.view(B * S, F)  # (B*S, F)

        # Create mask of observed (non-NaN) entries
        mask = ~torch.isnan(x_flat)  # (B*S, F)

        # Get indices of all non-NaN values:
        # Returns (N, 2) where N is number of OBSERVED (non NaN) values
        # Each row is [sample_index, feature_index]
        idxs = torch.nonzero(mask, as_tuple=False)

        # Edge case: entire batch is all-NaN.
        # When learn_missing_emb=False, return zeros of the correct output shape
        # for backward compatibility with checkpoints trained before this feature.
        # When learn_missing_emb=True, route through post(missing_emb) so the
        # learnable token is trained and output is consistent with per-sample handling.
        if idxs.numel() == 0:
            if not self.learn_missing_emb:
                if self.reduce:
                    return torch.zeros(B, self.out_dim, device=device, dtype=x.dtype)
                return torch.zeros(B, S, self.out_dim, device=device, dtype=x.dtype)
            else:
                mt = self.missing_emb.to(dtype=x.dtype)
                agg_full = mt.unsqueeze(0).expand(B * S, -1).contiguous()  # (B*S, index_embed_dim)
                out_flat = self.post(agg_full)  # (B*S, out_dim)
                out = out_flat.view(B, S, -1)  # (B, S, out_dim)
                if self.reduce:
                    return out.mean(dim=1)  # (B, out_dim)
                return out

        # Extract sample and feature indices for all observed values
        sample_idx = idxs[:, 0]  # (N,), values in [0, B*S) indicating which sample
        feat_idx = idxs[:, 1]  # (N,), values in [0, F) indicating which feature

        # Extract the actual observed values from x_flat and add dimension for MLP input
        vals = x_flat[sample_idx, feat_idx].unsqueeze(1)  # (N, 1)
        if logger.isEnabledFor(logging.DEBUG):
            has_nan = torch.isnan(vals).any()
            has_inf = torch.isinf(vals).any()
            if has_nan or has_inf:
                logger.debug(
                    "[SparseEncoder] vals has NaN=%s Inf=%s  (N=%d observed tokens)",
                    has_nan.item(),
                    has_inf.item(),
                    vals.shape[0],
                )

        # Optional: Apply learnable per-feature normalization
        # This affine transformation is learned ONLY based on training dataset statistics
        # for each feature dimension individually. This is NOT contextualized like the
        # FiLM modulation below - it's a simple per-feature scaling and shifting.
        if self.learnable_norm:
            norm_sc = self.value_scale[feat_idx].unsqueeze(1)
            norm_shift = self.value_shift[feat_idx].unsqueeze(1)
            vals = vals * norm_sc + norm_shift

        """
        TODO for the future:
        1) Keep a small dictionary of feature types "categorical" vs "continuous".
        2) For categorical features: ignore value_mlp, just use index_emb (optionally with one-hot presence, i.e. put NaN instead of 0s and a single 1; or small learned embedding for each category value).
        3) For numeric: use FiLM/value MLP.
        """

        # Get feature embeddings for all observed features
        idx_emb = self.index_emb(feat_idx)  # (N, index_embed_dim)

        # Contextualize the numeric value with its feature embedding
        # The feature embedding provides semantic context for interpreting the numeric value
        val_input = torch.cat([vals, idx_emb], dim=1)  # (N, 1 + index_embed_dim)

        # Predict FiLM parameters (alpha for scaling, beta for shifting)
        val_params = self.value_mlp(val_input)  # (N, 2 * index_embed_dim)

        # Split into alpha (scale) and beta (shift) parameters
        alpha, beta = val_params.chunk(2, dim=1)  # 2 * (N, index_embed_dim)
        # print(alpha.shape, beta.shape)

        # Apply FiLM modulation: slight residual scaling and shifting
        modulated_feat = idx_emb * (1 + alpha) + beta  # (N, index_embed_dim)

        # Aggregate modulated features per sample using scatter-add.
        # Use the same dtype as modulated_feat to ensure compatibility under
        # mixed-precision (AMP) training where modulated_feat may be float16.
        agg_dim = idx_emb.shape[1]
        agg = torch.zeros(B * S, agg_dim, device=device, dtype=modulated_feat.dtype)  # (B*S, index_embed_dim)

        # Sum all modulated features belonging to the same sample
        agg = agg.index_add(0, sample_idx, modulated_feat)

        # For mean aggregation, normalize by count of observed features per sample
        if self.aggregation == "mean":
            # Count non-NaN features per sample, clamp to avoid division by zero
            counts = mask.sum(dim=1).clamp(min=1).to(agg.dtype).unsqueeze(1)  # (B*S, 1)
            agg = agg / counts

        # Replace fully-missing (all-NaN) sample rows with the missing_emb token.
        # These rows were untouched by index_add (no observed features) and still
        # hold zeros.  Feeding zeros into ResidualMLP's LayerNorm causes near-zero
        # variance -> ~316x gradient amplification -> NaN parameters.
        # Note: only all-NaN rows are replaced, not samples that merely have some NaNs.
        per_sample_obs = mask.sum(dim=1)  # (B*S,)
        all_nan_mask = per_sample_obs == 0
        if all_nan_mask.any():
            agg = agg.clone()  # avoid in-place on a view that autograd may track
            agg[all_nan_mask] = self.missing_emb.to(agg.dtype)

        # Apply post-processing network to refine and project to output dimension
        out_flat = self.post(agg)  # (B*S, out_dim)

        out = out_flat.view(B, S, -1)  # (B, S, out_dim)

        # Optionally reduce sequence dimension by averaging
        if self.reduce:
            out = out.mean(dim=1)  # (B, out_dim) - average pooling over S
        return out


if __name__ == "__main__":
    B, S, F = 4, 8, 10
    x = torch.randn(B, S, F)
    x[torch.rand_like(x) < 0.8] = float("nan")  # 80% missing
    x[:, :, F // 2] = 0.0

    print("Input tensor:")
    print(x.shape)

    encoder = SparseMetadataEncoder(num_features=F, out_dim=128, scalar_modulation=False)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    z = encoder(x)

    print("\n\nEncoder output:")
    print("output shape:", z.shape)
