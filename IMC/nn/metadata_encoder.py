import torch
import torch.nn as nn

class ContextualImputer(nn.Module):
    """
    Neural network-based imputer for missing values in tabular data.

    This model imputes missing entries (NaNs) in a tensor of shape
    (batch_size, num_features) by using a context-aware multi-layer perceptron (MLP).
    It takes both observed values and a binary mask indicating missing features as input.
    Missing values are initially filled with a learnable parameter vector before
    imputation.

    Args:
        num_features (int): Number of features in the input data.
        hidden_dim (int): Number of hidden units in the MLP.

    Attributes:
        learnable_fill (nn.Parameter): Learnable parameter vector used to fill missing entries initially.
        imputer_net (nn.Sequential): MLP network predicting imputed values.

    Inputs:
        x (Tensor): Input tensor of shape (batch_size, num_features) with NaNs representing missing values.

    Outputs:
        Tensor: Imputed tensor of shape (batch_size, num_features) with missing entries replaced by model predictions.
    """

    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_features = num_features
        
        # Learnable parameter to fill missing values (one scalar per feature)
        self.learnable_fill = nn.Parameter(torch.zeros(num_features))
        
        # MLP to predict imputations from concatenated [values + mask]
        self.imputer_net = nn.Sequential(
            nn.Linear(num_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to impute missing values in input tensor x.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_features) with NaNs indicating missing values.

        Returns:
            Tensor: Output tensor of same shape as x with missing entries replaced by imputed values.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor but got {type(x)}")

        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("Input tensor x must be a floating-point type")

        if x.shape[-1] != self.num_features:
            raise ValueError(f"Expected input with {self.num_features} features, but got {x.shape[-1]}")
        
        mask = ~torch.isnan(x)  # mask indicating observed (non-NaN) values
        
        # Fill missing values with learnable parameters (broadcasted across batch)
        x_filled = torch.where(mask, x, self.learnable_fill.unsqueeze(0).expand_as(x))
        
        # Concatenate filled values with mask as input features
        imputer_input = torch.cat([x_filled, mask.float()], dim=1)
        
        # Predict imputed values for all features
        imputed_values = self.imputer_net(imputer_input)
        
        # Replace missing entries with predicted imputed values
        x_imputed = torch.where(mask, x_filled, imputed_values)
        
        return x_imputed




class MetadataEncoder(nn.Module):
    """
    Encodes DICOM metadata vectors into a compact, dense embedding suitable for fusion with image features.

    This module applies a two-layer fully connected neural network with ReLU activations to transform
    high-dimensional metadata inputs into a lower-dimensional embedding space.

    Args:
        input_dim (int): Dimensionality of the input metadata vector.
        embed_dim (int, optional): Desired dimensionality of the output embedding. Default is 128.
        dropout (float, optional): Dropout rate

    Inputs:
        x (torch.Tensor): A tensor of shape (B, input_dim) representing the batch of metadata vectors.

    Outputs:
        torch.Tensor: A tensor of shape (B, embed_dim) representing the encoded metadata embeddings.
    """
    
    def __init__(self, input_dim : int , embed_dim : int = 128, dropout : float = 0.1):
        super().__init__()
        
        self.imputer = ContextualImputer(input_dim, hidden_dim=input_dim)

        hidden_dim = (embed_dim + input_dim) // 2

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

        self.residual = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Impute missing values contextually
        x = self.imputer(x)

        # Embed imputed vector
        x = self.residual(x) + self.proj(x)
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
