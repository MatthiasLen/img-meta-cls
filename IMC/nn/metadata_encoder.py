import torch
import torch.nn as nn


class ContextualImputer(nn.Module):
    def __init__(self, num_features : int, hidden_dim  :int):
        super().__init__()
        self.num_features = num_features
        
        # MLP to predict imputations
        self.imputer_net = nn.Sequential(
            nn.Linear(num_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_features)
        mask = ~torch.isnan(x)  # True where not NaN
        
        # Replace NaNs with zeros (or any constant)
        x_filled = torch.where(mask, x, torch.zeros_like(x))
        
        # Concatenate input + mask as features
        imputer_input = torch.cat([x_filled, mask.float()], dim=1)
        
        # Predict imputed values for all features
        imputed_values = self.imputer_net(imputer_input)
        
        # For missing entries, replace with imputed values
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





