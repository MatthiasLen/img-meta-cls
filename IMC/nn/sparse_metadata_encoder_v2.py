
import torch
import torch.nn as nn
import logging

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
        h =  torch.nn.functional.gelu(self.lin1(x))
        h = self.lin2(h)
        return self.norm(x + h)

class SparseMetadataEncoder(nn.Module):
    """
    Sparse Metadata Encoder V2 with Transformer Aggregation and CLS Token.
    
    This is an advanced version of the sparse metadata encoder that uses a transformer-based
    architecture for aggregating features. Unlike V1 which uses simple sum/mean pooling,
    V2 employs self-attention to learn complex relationships between features.
    
    Key Features:
    - **Dense Processing**: Unlike V1, this processes all features (both observed and missing)
      in a dense manner, making it more suitable for transformer processing.
    - **Learnable NaN Embedding**: Missing features are represented by a specific learnable
      embedding vector, allowing the model to learn how to handle missingness.
    - **FiLM Modulation**: Similar to V1, uses Feature-wise Linear Modulation to contextualize
      observed values with their feature embeddings.
    - **CLS Token**: Adds a learnable classification token that aggregates information from
      all features through self-attention.
    - **Transformer Encoder**: Multi-layer transformer processes the full sequence of features
      (including CLS token) to capture complex feature interactions.
    - **Attention-based Aggregation**: The final CLS token embedding serves as the aggregated
      representation, learned through self-attention rather than simple pooling.
    
    Architecture Overview:
        1. Generate feature index embeddings for all features
        2. For observed features: Apply FiLM modulation based on their values
        3. For missing features: Replace with learnable NaN embedding
        4. Prepend CLS token to the feature sequence
        5. Process through multi-layer transformer
        6. Extract CLS token output as aggregated representation
        7. Apply post-processing MLP
    
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
        """
        Initialize the Sparse Metadata Encoder V2.
        
        Args:
            num_features (int): Total number of features in the metadata vector.
            index_embed_dim (int, optional): Dimension of feature embeddings and transformer
                hidden states. Must be divisible by transformer_heads. Default: 64.
            value_mlp_dim (int, optional): Hidden dimension of the MLP that predicts FiLM
                parameters for observed values. Default: 32.
            out_dim (int, optional): Output embedding dimension after post-processing. Default: 128.
            transformer_heads (int, optional): Number of attention heads in the transformer.
                Default: 4.
            transformer_layers (int, optional): Number of transformer encoder layers. Default: 4.
            reduce (bool, optional): If True, applies mean pooling over the sequence dimension.
                If False, returns embeddings for all sequence positions. Default: True.
        """
        super().__init__()

        self.out_dim = out_dim
        # Learnable embedding for each feature index
        self.index_emb = nn.Embedding(num_features, index_embed_dim)

        # Learnable embedding to represent missing (NaN) values
        # This allows the model to learn a specific representation for missingness
        self.nan_embedding = nn.Parameter(torch.randn(index_embed_dim))

        # CLS token: A learnable token prepended to the sequence for aggregation
        # The transformer will use self-attention to aggregate information into this token
        self.cls_token = nn.Parameter(torch.randn(1, 1, index_embed_dim))

        # FiLM generator: Maps [value, feature_embedding] -> [alpha, beta] for modulation
        self.value_mlp = nn.Sequential(
            nn.Linear(1 + index_embed_dim, value_mlp_dim),
            nn.GELU(),
            nn.Linear(value_mlp_dim, index_embed_dim * 2) # alpha and beta parameters
        )

        # Transformer Encoder: Processes the full sequence with self-attention
        # This captures complex interactions between features and learns to aggregate
        # information into the CLS token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=index_embed_dim,
            nhead=transformer_heads,
            dim_feedforward=index_embed_dim * 4,
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Post-processing: Refines the CLS token and projects to output dimension
        self.post = nn.Sequential(
            nn.LayerNorm(index_embed_dim),
            nn.Linear(index_embed_dim, out_dim),
            nn.GELU()
        )
        self.reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode sparse metadata using transformer aggregation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, S, F) where:
                - B is the batch size
                - S is the sequence length (e.g., number of slices)
                - F is the number of features
                NaN values indicate missing features.
                
        Returns:
            torch.Tensor: Encoded metadata of shape:
                - (B, out_dim) if reduce=True (mean pooling over sequence)
                - (B, S, out_dim) if reduce=False (one embedding per sequence position)
        """
        B, S, F = x.shape
        device = x.device

        # Flatten batch and sequence dimensions for processing
        x_flat = x.reshape(B * S, F)

        # ------------------------------------------------------------------
        # Generate feature index embeddings for all features
        # ------------------------------------------------------------------
        # Create tensor of feature indices [0, 1, 2, ..., F-1] for each sample
        all_feats = torch.arange(F, device=device).unsqueeze(0).expand(B * S, F)
        idx_emb = self.index_emb(all_feats)  # (B*S, F, index_embed_dim)

        # ------------------------------------------------------------------
        # Create masks for observed vs missing features
        # ------------------------------------------------------------------
        nan_mask = torch.isnan(x_flat)                # (B*S, F) - True where missing
        observed_mask = ~nan_mask                     # (B*S, F) - True where observed
        observed_mask_f = observed_mask.unsqueeze(-1) # (B*S, F, 1) - for broadcasting

        # ------------------------------------------------------------------
        # Apply FiLM modulation to observed values (dense computation with masking)
        # ------------------------------------------------------------------
        # Replace NaNs with zeros for safe processing (will be masked out later)
        safe_vals = torch.where(nan_mask, torch.zeros_like(x_flat), x_flat)
        vals = safe_vals.unsqueeze(-1)                # (B*S, F, 1)

        # Prepare input for value MLP: concatenate [value, feature_embedding]
        val_input = torch.cat([vals, idx_emb], dim=-1)   # (B*S, F, 1+index_embed_dim)
        # Predict FiLM parameters for all features (even missing ones, but we'll mask them)
        val_params = self.value_mlp(val_input)           # (B*S, F, 2*index_embed_dim)

        # Split into alpha (scaling) and beta (shift) parameters
        alpha, beta = val_params.chunk(2, dim=-1)

        # Apply FiLM modulation: embedding * (1 + alpha) + beta
        modulated = idx_emb * (1.0 + alpha) + beta

        # Only apply modulation where features are observed, keep original embedding for missing
        idx_emb = torch.where(observed_mask_f, modulated, idx_emb)

        # ------------------------------------------------------------------
        # Replace embeddings of missing features with learnable NaN embedding
        # ------------------------------------------------------------------
        nan_emb = self.nan_embedding.view(1, 1, -1)  # (1, 1, index_embed_dim)
        # Where nan_mask is True, use nan_embedding; otherwise use the (possibly modulated) idx_emb
        idx_emb = torch.where(nan_mask.unsqueeze(-1), nan_emb, idx_emb)

        # ------------------------------------------------------------------
        # Prepend CLS token to the sequence
        # ------------------------------------------------------------------
        # The CLS token will aggregate information from all features through self-attention
        cls_tokens = self.cls_token.expand(B * S, 1, -1)  # (B*S, 1, index_embed_dim)
        full_sequence = torch.cat([cls_tokens, idx_emb], dim=1)  # (B*S, 1+F, index_embed_dim)

        # ------------------------------------------------------------------
        # Process through transformer encoder
        # ------------------------------------------------------------------
        # The transformer uses self-attention to capture feature interactions
        # and aggregate information into the CLS token
        transformer_out = self.transformer_encoder(full_sequence)  # (B*S, 1+F, index_embed_dim)
        # Extract the CLS token output (first position) as the aggregated representation
        cls_output = transformer_out[:, 0]  # (B*S, index_embed_dim)

        # ------------------------------------------------------------------
        # Post-process and reshape output
        # ------------------------------------------------------------------
        out_flat = self.post(cls_output)  # (B*S, out_dim)
        out = out_flat.reshape(B, S, self.out_dim)  # (B, S, out_dim)

        # Optionally reduce sequence dimension by mean pooling
        if self.reduce:
            out = out.mean(dim=1)  # (B, out_dim)

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
