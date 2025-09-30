import torch
import torch.nn as nn



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

        hidden_dim = (embed_dim + input_dim)//2

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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.residual(x) + self.proj(x)
        return x
