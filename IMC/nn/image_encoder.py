import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
from torchvision.models.densenet import DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights
import torch.nn.init as init

class MultiSliceImageEncoder(nn.Module):
    """
    Multi-Slice Image Encoder for Medical Image Classification.

    This module provides a flexible and powerful image encoder that processes multiple 2D slices
    from a 3D medical scan (e.g., MRI, CT). It employs a shared-weight CNN backbone to generate
    a sequence of feature embeddings, one for each slice. This approach is central to the IMC
    (Image-Metadata-Classifier) project's multi-modal architecture.

    Key Features:
    - **Multiple Backbone Support**: Easily configurable to use various state-of-the-art CNN
      architectures, including DenseNet, ResNet, EfficientNet, Swin Transformer, and DINOv3.
      This allows for experimentation and selection of the best-performing model for the task.
    - **Pretrained Weights**: Leverages transfer learning by initializing backbones with weights
      pretrained on large-scale datasets like ImageNet, which can significantly improve
      performance and reduce training time.
    - **Input Channel Adaptation**: Automatically adapts the first convolutional layer to handle
      medical images with a different number of input channels (e.g., single-channel grayscale)
      while preserving the pretrained knowledge.
    - **Feature Extraction**: The final classification layer of the backbone is removed to expose
      the rich, high-dimensional feature maps from the convolutional layers.
    - **Global Pooling**: Applies global average pooling to the feature maps to produce a fixed-size
      feature vector for each slice, regardless of the input image size.
    - **Intermediate Output Extraction**: Includes a utility to extract feature maps from
      intermediate layers, which is useful for more advanced fusion techniques or for debugging.

    The output is a sequence of embeddings, which can be fed into a subsequent fusion model
    (e.g., a Transformer or RNN) to be combined with other modalities like DICOM metadata.
    """

    def __init__(self, pretrained: bool = True, n_channels: int = 1, backbone: str = "densenet"):
        """
        Initializes the MultiSliceImageEncoder.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
            n_channels (int): Number of input channels for the medical images (e.g., 1 for grayscale).
            backbone (str): The CNN backbone to use. Supported options include:
                - "densenet121", "densenet161", "densenet169", "densenet201"
                - "resnet50"
                - "swinv2_b", "swinv2_s", "swinv2_t"
                - "efficientnet_v2_l", "efficientnet_v2_m", "efficientnet_v2_s"
                - "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"
                - "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"
        """
        super().__init__()
        self.backbone = backbone
        self._get_backbone(backbone, pretrained)

        # Adapt the first convolutional layer for the given number of input channels
        self._adapt_input_channels(n_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _get_backbone(self, backbone_name: str, pretrained: bool):
        """
        Loads and configures the specified CNN backbone architecture.
        
        This method initializes the backbone network and extracts the feature dimension.
        It removes the final classification layer to expose the convolutional feature maps.
        
        Supported Backbones:
        - DenseNet family: densenet121, densenet161, densenet169, densenet201
          Uses dense connections between layers for improved gradient flow.
        - Swin Transformer V2: swinv2_b, swinv2_s, swinv2_t
          Vision transformer with shifted windows for efficient self-attention.
        - EfficientNet V2: efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s
          Efficient scaling of CNN architectures with improved training speed.
        - EfficientNet B: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
          Compound scaling approach balancing depth, width, and resolution.
        - DINOv3: dinov3_vits16, dinov3_vitb16, dinov3_vitl16
          Self-supervised vision transformers with strong feature representations.
        - ResNet50 (default): Classic residual network architecture.

        Args:
            backbone_name (str): Name of the backbone architecture to load.
            pretrained (bool): If True, loads weights pretrained on ImageNet or other large datasets.
        
        Side Effects:
            - Sets self.cnn to the loaded backbone (without final classification layer)
            - Sets self.slice_feat_dim to the feature dimension of the backbone
        """
        if backbone_name.startswith("densenet"):
            # DenseNet: Dense connections between layers
            # Each layer receives feature maps from all preceding layers
            if backbone_name == "densenet121":
                self.cnn = models.densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "densenet161":
                self.cnn = models.densenet161(weights=DenseNet161_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "densenet169":
                self.cnn = models.densenet169(weights=DenseNet169_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "densenet201":
                self.cnn = models.densenet201(weights=DenseNet201_Weights.DEFAULT if pretrained else None)
            else:
                raise ValueError(f"Unsupported densenet backbone: {backbone_name}")
            # Extract feature dimension from classifier input
            self.slice_feat_dim = self.cnn.classifier.in_features
            # Remove classifier to expose feature maps
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        elif backbone_name.startswith("swinv2"):
            # Swin Transformer V2: Hierarchical vision transformer with shifted windows
            # Achieves efficiency by limiting self-attention to local windows
            if backbone_name == "swinv2_b":
                self.cnn = models.swin_v2_b(weights="IMAGENET1K_V1" if pretrained else None)
                self.slice_feat_dim = self.cnn.head.in_features
            elif backbone_name == "swinv2_s":
                self.cnn = models.swin_v2_s(weights="IMAGENET1K_V1" if pretrained else None)
                self.slice_feat_dim = self.cnn.head.in_features
            elif backbone_name == "swinv2_t":
                self.cnn = models.swin_v2_t(weights="IMAGENET1K_V1" if pretrained else None)
                self.slice_feat_dim = self.cnn.head.in_features
            else:
                raise ValueError(f"Unsupported swin transformer backbone: {backbone_name}")
            # Replace classification head with identity to get features
            self.cnn.head = nn.Identity()

        elif backbone_name.startswith("efficientnet_v2"):
            # EfficientNet V2: Improved training efficiency and parameter efficiency
            # Uses Fused-MBConv blocks for faster training
            if backbone_name == "efficientnet_v2_l":
                self.cnn = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "efficientnet_v2_m":
                self.cnn = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "efficientnet_v2_s":
                self.cnn = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
            else:
                raise ValueError(f"Unsupported efficientnet_v2 backbone: {backbone_name}")
            self.slice_feat_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()

        elif backbone_name.startswith("efficientnet_b"):
            # EfficientNet B-series: Compound scaling of depth, width, and resolution
            # Balances network dimensions for optimal performance
            if backbone_name == "efficientnet_b0":
                self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "efficientnet_b1":
                self.cnn = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "efficientnet_b2":
                self.cnn = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            elif backbone_name == "efficientnet_b3":
                self.cnn = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
            else:
                raise ValueError(f"Unsupported efficientnet_b backbone: {backbone_name}")
            self.slice_feat_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()

        elif backbone_name.startswith("dinov3"):
            # DINOv3: Self-supervised vision transformers
            # Trained with self-distillation without labels
            # NOTE: These use local file paths specific to the original development environment
            import torch.hub
            if backbone_name == "dinov3_vits16":
                self.cnn = torch.hub.load("/home/tuan.truong/codebase/dinov3", 'dinov3_vits16', source='local', weights="/home/tuan.truong/codebase/pretrained_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" if pretrained else None)
                self.slice_feat_dim = 384
            elif backbone_name == "dinov3_vitb16":
                self.cnn = torch.hub.load("/home/tuan.truong/codebase/dinov3", 'dinov3_vitb16', source='local', weights="/home/tuan.truong/codebase/pretrained_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" if pretrained else None)
                self.slice_feat_dim = 768
            elif backbone_name == "dinov3_vitl16":
                self.cnn = torch.hub.load("/home/tuan.truong/codebase/dinov3", 'dinov3_vitl16', source='local', weights="/home/tuan.truong/codebase/pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" if pretrained else None)
                self.slice_feat_dim = 1024
            else:
                raise ValueError(f"Unsupported DINOv3 backbone: {backbone_name}")

        else:
            # Default to ResNet50 if no specific backbone is matched
            # ResNet: Deep residual learning with skip connections
            self.cnn = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            self.slice_feat_dim = self.cnn.fc.in_features
            # Remove fully connected layer and average pooling
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

    def _adapt_input_channels(self, n_channels: int):
        """
        Adapts the first convolutional layer to handle a different number of input channels.
        
        This is crucial for medical imaging where:
        - Grayscale images have 1 channel (unlike RGB with 3 channels)
        - Some modalities may have multiple channels (e.g., multi-echo, multi-contrast)
        
        The adaptation preserves pretrained knowledge by:
        - For 1-channel input: Averaging the RGB weights across the channel dimension
        - For other inputs: Initializing with Xavier uniform (random but scaled appropriately)
        
        This approach allows transfer learning from ImageNet (RGB) to medical images (often grayscale).

        Args:
            n_channels (int): The desired number of input channels (e.g., 1 for grayscale).
        
        Side Effects:
            Modifies the first convolutional layer of self.cnn to accept n_channels inputs.
        """
        if self.backbone.startswith("densenet"):
            conv_layer = self.cnn[0].conv0
            if conv_layer.in_channels != n_channels:
                # Save pretrained weights before modifying the layer
                old_weights = conv_layer.weight.data.clone()
                # Create new conv layer with desired number of input channels
                self.cnn[0].conv0 = nn.Conv2d(n_channels, conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=False)
                if n_channels == 1:
                    # For grayscale: average the RGB weights to preserve learned features
                    self.cnn[0].conv0.weight.data = old_weights.mean(dim=1, keepdim=True)
                else:
                    # For other channel counts: random initialization
                    init.xavier_uniform_(self.cnn[0].conv0.weight)
                    
        elif self.backbone.startswith("resnet"):
            conv_layer = self.cnn[0]
            if conv_layer.in_channels != n_channels:
                old_weights = conv_layer.weight.data.clone()
                self.cnn[0] = nn.Conv2d(n_channels, conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=False)
                if n_channels == 1:
                    # For grayscale: average the RGB weights
                    self.cnn[0].weight.data = old_weights.mean(dim=1, keepdim=True)
                else:
                    init.xavier_uniform_(self.cnn[0].weight)
        else:
            # TODO : 
            print("WARNING: Currently selected backbone type not supported by _adapt_input_channels!")
        
    def get_feature_dimension(self) -> int:
        """
        Get the dimensionality of the feature vectors produced by this encoder.
        
        Returns:
            int: Feature dimension (e.g., 1024 for DenseNet121, 2048 for ResNet50)
        """
        return self.slice_feat_dim
    
    def get_intermediate_outputs(self, x: torch.Tensor, layer_names: list) -> dict:
        """
        Extract feature maps from intermediate layers of the backbone.
        
        This is useful for:
        - Multi-scale feature fusion (combining features from different depths)
        - Feature visualization and analysis
        - Debugging and understanding what the network learns
        - Skip connections in encoder-decoder architectures
        
        Note: Currently only implemented for DenseNet backbones.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N_slices, H, W)
            layer_names (list): List of layer names to extract outputs from.
                These should match the named children of self.cnn[0].
                
        Returns:
            dict: Dictionary mapping layer names to their outputs, with each output
                having shape (B, N_slices, feature_dim) after global average pooling.
        """
        outputs = {}
        B, N, H, W = x.shape
        # Treat each slice as a separate sample in the batch
        x = x.view(B * N, 1, H, W)
        x_temp = x 
        # Forward through layers, extracting specified intermediate outputs
        for name, module in self.cnn[0].named_children():
            x_temp = module(x_temp)
            if name in layer_names:
                # Save a copy of the output
                outputs[name] = x_temp.clone()
        # Post-process outputs: apply global average pooling and reshape
        for name in outputs:
            out = outputs[name]
            # If output is a 4D feature map, apply spatial pooling
            # Otherwise just reshape
            outputs[name] = out.mean(dim=[2, 3]).view(B, N, -1) if out.dim() == 4 else out.view(B, N, -1)

        return outputs


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode multi-slice medical images.
        
        Processing Pipeline:
        1. Reshape: Flatten batch and slice dimensions (B, N, H, W) -> (B*N, 1, H, W)
        2. Channel Expansion: For certain backbones, repeat grayscale to RGB (1 -> 3 channels)
        3. Backbone Forward: Extract convolutional features
        4. Global Pooling: If output is a feature map (4D), apply adaptive average pooling
        5. Reshape: Reorganize to (B, N, feature_dim) to maintain slice structure
        
        The slice structure is preserved in the output, allowing downstream models to:
        - Apply temporal/sequential modeling (e.g., with transformers or RNNs)
        - Perform attention-based fusion across slices
        - Aggregate slice information in a learned manner

        Args:
            x (torch.Tensor): Input tensor of shape (B, N_slices, H, W) where:
                - B is batch size
                - N_slices is the number of 2D slices per 3D volume
                - H, W are the height and width of each slice

        Returns:
            torch.Tensor: Encoded features of shape (B, N_slices, slice_feat_dim).
                Each slice is independently encoded into a feature vector, maintaining
                the sequential structure for downstream processing.
        """
        B, N, H, W = x.shape
        # Flatten batch and slice dimensions: treat each slice as a separate sample
        x = x.view(B * N, 1, H, W)

        # Some backbones (Swin, EfficientNet, DINOv3) require 3-channel input
        # Repeat the grayscale channel to create pseudo-RGB
        if not self.backbone.startswith(("densenet", "resnet")):
            x = x.repeat(1, 3, 1, 1)  # (B*N, 3, H, W)

        # Pass through the CNN backbone to extract features
        features = self.cnn(x)

        # Apply global average pooling if the output is a 4D feature map
        # This converts spatial feature maps to a single feature vector per slice
        if features.dim() == 4:
            features = self.global_avg_pool(features)  # (B*N, feat_dim, 1, 1)

        # Reshape to maintain the batch and slice structure
        # This allows downstream models to process slices sequentially or with attention
        return features.view(B, N, self.slice_feat_dim)
