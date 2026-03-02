import torch
import torch.nn as nn

class ContextualImputer(nn.Module):
    """
    Neural Network-based Contextual Imputer for Missing Tabular Data.
    
    This module learns to impute missing values (NaNs) in tabular metadata by using a
    context-aware MLP. Unlike simple strategies (mean/median imputation), this approach
    learns to predict missing values based on the pattern of observed features.
    
    Key Features:
    - **Learnable Initial Fill**: Each feature has a learnable parameter for initial imputation,
      which can be thought of as a learned per-feature default value.
    - **Context-Aware Imputation**: The MLP takes as input both the filled values and a binary
      mask indicating which features are observed. This allows it to learn which features
      predict others and to generate context-specific imputations.
    - **Preserves Observed Values**: Only missing values are replaced; observed values pass
      through unchanged.
    
    Architecture:
        1. Fill NaNs with learnable per-feature parameters
        2. Create binary mask (1=observed, 0=missing)
        3. Concatenate filled values and mask
        4. Pass through MLP to predict improved imputations
        5. Replace only the missing values with predictions
    
    This is particularly useful for medical metadata where:
    - Missing patterns contain information (e.g., certain tests not performed)
    - Features are correlated (e.g., imaging parameters are related)
    - The model can learn domain-specific imputation strategies
    
    Args:
        num_features (int): Number of features in the input data.
        hidden_dim (int): Number of hidden units in the imputation MLP.

    Attributes:
        learnable_fill (nn.Parameter): Learnable parameter vector (num_features,) used to
            fill missing entries initially. These serve as learned per-feature defaults.
        imputer_net (nn.Sequential): MLP network that predicts imputed values based on the
            concatenation of filled values and the observation mask.

    Input Shape:
        (batch_size, num_slices, num_features) with NaNs representing missing values.
        Note: num_slices can be 1 for single-slice data.

    Output Shape:
        (batch_size, num_features) with all NaNs replaced by model predictions.
    """

    def __init__(self, num_features: int, hidden_dim: int) -> None:
        """
        Initialize the Contextual Imputer.
        
        Args:
            num_features (int): Number of features in the input metadata vector.
            hidden_dim (int): Hidden dimension of the imputation MLP. Larger values
                increase model capacity but also computational cost.
        """
        super().__init__()
        self.num_features = num_features
        
        # Learnable parameter to fill missing values (one scalar per feature)
        # These serve as learned per-feature defaults and are refined by the MLP
        self.learnable_fill = nn.Parameter(torch.zeros(num_features))
        
        # MLP to predict imputations from concatenated [filled_values, observation_mask]
        # Input dimension: num_features (filled values) + num_features (binary mask) = 2 * num_features
        # Output dimension: num_features (predicted values for all features)
        self.imputer_net = nn.Sequential(
            nn.Linear(num_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to impute missing values in the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, F) where:
                - B is batch size
                - N is number of slices/sequences (can be 1)
                - F is number of features
                NaNs indicate missing values.

        Returns:
            torch.Tensor: Imputed tensor of shape (B, F) with all NaNs replaced.
                If N > 1, applies mean aggregation over the slice dimension.
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor but got {type(x)}")

        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("Input tensor x must be a floating-point type")

        if x.shape[-1] != self.num_features:
            raise ValueError(f"Expected input with {self.num_features} features, but got {x.shape[-1]}")
        
        # Create mask: True for observed (non-NaN) values, False for missing
        mask = ~torch.isnan(x)  # (*, num_features)
        
        # Step 1: Fill missing values with learnable parameters
        # Where mask is True (observed), keep original value
        # Where mask is False (missing), use learnable_fill
        x_filled = torch.where(mask, x, self.learnable_fill.unsqueeze(0).expand_as(x))
        
        # Step 2: Prepare input for imputation network
        # Concatenate filled values with observation mask (converted to float)
        # The mask tells the network which values are real vs filled
        imputer_input = torch.cat([x_filled, mask.to(x.dtype)], dim=-1).to(x.dtype)
        
        # Step 3: Predict refined imputed values for all features
        imputed_values = self.imputer_net(imputer_input)  # (*, num_features)
        
        # Step 4: Keep observed values, replace only missing values with predictions
        # This ensures we never modify real data
        x_imputed = torch.where(mask, x_filled, imputed_values)

        return x_imputed




class NanIgnorer(nn.Module):
    """
    Simple NaN Handler that Replaces Missing Values with Zeros.
    
    This module provides a non-learnable, deterministic approach to handling NaNs
    by simply replacing them with zeros. This is useful as a baseline or when
    you want to avoid the complexity of learned imputation.
    
    Unlike ContextualImputer, this module:
    - Has no learnable parameters
    - Is computationally very cheap
    - Provides deterministic behavior
    - May be appropriate when missing values are rare or when zero is a reasonable default
    
    Use Cases:
    - Baseline comparisons against learned imputation
    - When missing data is rare and doesn't carry much information
    - When computational efficiency is critical
    - When you want interpretable, deterministic preprocessing
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Replace all NaN values in the input tensor with zeros.

        Args:
            x (torch.Tensor): Input tensor that may contain NaNs of any shape.

        Returns:
            torch.Tensor: Tensor with the same shape as input, with NaNs replaced by 0.0.
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor but got {type(x)}")

        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("Input tensor x must be a floating-point type")

        # Replace NaNs with 0.0 using PyTorch's built-in function
        # This is more efficient than manual masking for this simple operation
        x_zeroed = torch.nan_to_num(x, nan=0.0)

        return x_zeroed



class MetadataEncoder(nn.Module):
    """
    DICOM Metadata Encoder with Imputation and Multi-Layer Projection.
    
    This is a dense encoder for tabular DICOM metadata. It handles
    missing values and transforms high-dimensional metadata into a compact embedding suitable
    for fusion with image features.
    
    Architecture Overview:
        1. Imputation: Handle missing values (NaNs) using either:
           - ContextualImputer: Learnable neural network-based imputation
           - NanIgnorer: Simple replacement with zeros
        2. Projection: Two-layer MLP with residual connection
           - First layer: input_dim -> hidden_dim with LayerNorm + ReLU + Dropout
           - Second layer: hidden_dim -> embed_dim with LayerNorm + ReLU + Dropout
           - Residual connection from input directly to output
        3. Reduction: Optional aggregation over sequence dimension (mean or max)
    
    
    Args:
        input_dim (int): Dimensionality of the input metadata vector (number of features).
        embed_dim (int, optional): Desired dimensionality of the output embedding. Default: 128.
        dropout (float, optional): Dropout probability for regularization. Applied after
            each layer to prevent overfitting. Default: 0.1.
        imputer (str, optional): Type of imputation strategy. Options:
            - 'contextual': Uses ContextualImputer (learned, context-aware imputation)
            - 'ignore': Uses NanIgnorer (simple zero-filling)
            Default: 'contextual'.
        reduce (bool, optional): Whether to reduce the sequence dimension. Options:
            - True: Apply mean pooling over sequence dimension
            - False: No reduction, return full sequence
            Default: False.

    Input Shape:
        (batch_size, num_slices, input_dim) or (batch_size, input_dim)
        NaNs indicate missing values.

    Output Shape:
        - (batch_size, embed_dim) if reduce=True
        - (batch_size, num_slices, embed_dim) if reduce=False
    """

    def __init__(self, input_dim: int, embed_dim: int = 128, dropout: float = 0.1, imputer: str = 'contextual', reduce: bool = False):
        """
        Initialize the Metadata Encoder.
        
        Args:
            input_dim (int): Dimensionality of the input metadata vector.
            embed_dim (int, optional): Output embedding dimension. Default: 128.
            dropout (float, optional): Dropout probability. Default: 0.1.
            imputer (str, optional): Imputation strategy ('contextual' or 'ignore'). Default: 'contextual'.
            reduce (bool, optional): Whether to reduce the sequence dimension. Default: False.
        """
        super().__init__()

        # Select imputation strategy
        if imputer == 'contextual':
            self.imputer = ContextualImputer(input_dim, hidden_dim=input_dim)
        elif imputer == 'ignore':
            self.imputer = NanIgnorer()
        else:
            raise ValueError(f"Unknown imputer type: {imputer}. Choose from 'contextual' or 'ignore'.")

        # Calculate hidden dimension as midpoint between input and output
        hidden_dim = (embed_dim + input_dim) // 2

        # Two-layer MLP with normalization and dropout
        # This learns complex non-linear transformations of the metadata
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual connection: allows the model to learn identity mappings or simple projections
        # This is particularly useful when some metadata features should pass through with
        # minimal transformation
        if input_dim == embed_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, embed_dim)
        
        # Validate and store reduction method
        self.reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode metadata.
        
        Args:
            x (torch.Tensor): Input metadata tensor. Can be:
                - (batch_size, input_dim): Single metadata vector per sample
                - (batch_size, num_slices, input_dim): Multiple metadata vectors per sample
                NaNs indicate missing values.
        
        Returns:
            torch.Tensor: Encoded metadata embedding:
                - (batch_size, embed_dim) if reduce='mean' or 'max'
                - (batch_size, num_slices, embed_dim) if reduce='none'
        """
        # Step 1: Handle missing values through imputation
        x = self.imputer(x)

        # Step 2: Embed the imputed vector using projection MLP + residual
        # The residual connection helps preserve important input features
        x = self.residual(x) + self.proj(x)

        # Step 3: Optionally reduce over sequence dimension
        if self.reduce:
            x = x.mean(dim=1)  # Average over slices
        return x


if __name__ == "__main__":

    batch_size = 4
    channels = 3
    num_features = 5
    hidden_dim = 10

    model = ContextualImputer(num_features=num_features, hidden_dim=hidden_dim)

    # Create input with NaNs in both 2D and 3D cases
    # 2D input: (batch_size, num_features)
    x_2d = torch.randn(batch_size, num_features)
    x_2d[0, 1] = float('nan')  # Introduce NaN in first sample
    x_2d[2, 3] = float('nan')  # Another NaN

    output_2d = model(x_2d)
    assert output_2d.shape == x_2d.shape, f"Expected output shape {x_2d.shape}, got {output_2d.shape}"
    # Check no NaNs remain
    assert not torch.isnan(output_2d).any(), "NaNs remain in output for 2D input"

    # Check observed values are unchanged
    observed_mask_2d = ~torch.isnan(x_2d)
    assert torch.allclose(output_2d[observed_mask_2d], x_2d[observed_mask_2d]), "Observed values changed in 2D"
