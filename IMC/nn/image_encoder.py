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
        self._get_backbone(backbone, pretrained)

        # Adapt the first convolutional layer for the given number of input channels
        self._adapt_input_channels(n_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = backbone

    def _get_backbone(self, backbone_name: str, pretrained: bool):
        """
        Loads the specified CNN backbone and sets the feature dimension.

        Args:
            backbone_name (str): The name of the backbone to load.
            pretrained (bool): If True, loads pretrained weights.
        """
        if backbone_name.startswith("densenet"):
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
            self.slice_feat_dim = self.cnn.classifier.in_features
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        elif backbone_name.startswith("swinv2"):
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
            self.cnn.head = nn.Identity()

        elif backbone_name.startswith("efficientnet_v2"):
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
            # Default to ResNet50
            self.cnn = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            self.slice_feat_dim = self.cnn.fc.in_features
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

    def _adapt_input_channels(self, n_channels: int):
        """
        Adapts the first convolutional layer of the backbone to the specified number of input channels.

        Args:
            n_channels (int): The desired number of input channels.
        """
        if self.backbone.startswith("densenet"):
            conv_layer = self.cnn[0].conv0
            if conv_layer.in_channels != n_channels:
                old_weights = conv_layer.weight.data.clone()
                self.cnn[0].conv0 = nn.Conv2d(n_channels, conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=False)
                if n_channels == 1:
                    self.cnn[0].conv0.weight.data = old_weights.mean(dim=1, keepdim=True)
                else:
                    init.xavier_uniform_(self.cnn[0].conv0.weight)
        elif self.backbone.startswith("resnet"):
            conv_layer = self.cnn[0]
            if conv_layer.in_channels != n_channels:
                old_weights = conv_layer.weight.data.clone()
                self.cnn[0] = nn.Conv2d(n_channels, conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=False)
                if n_channels == 1:
                    self.cnn[0].weight.data = old_weights.mean(dim=1, keepdim=True)
                else:
                    init.xavier_uniform_(self.cnn[0].weight)
        
    def get_feature_dimension(self) -> int:
        """Return dimension of feature vector generated by embedding.

        Returns:
            int: number of features
        """
        return self.slice_feat_dim
    
    def get_intermediate_outputs(self, x: torch.Tensor, layer_names: list) -> dict:
        """
        Get intermediate outputs from specified layers.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N_slices, H, W)
            layer_names (list): List of layer names to extract outputs from.
        Returns:
            dict: Dictionary of intermediate outputs.
        """
        outputs = {}
        B, N, H, W = x.shape
        x = x.view(B * N, 1, H, W)  # treat slices as batch
        x_temp = x 
        for name, module in self.cnn[0].named_children():
            x_temp = module(x_temp)
            if name in layer_names:
                outputs[name] = x_temp.clone()
        for name in outputs:
            out = outputs[name]
            outputs[name] = out.mean(dim=[2, 3]).view(B, N, -1) if out.dim() == 4 else out.view(B, N, -1)

        return outputs


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the image encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N_slices, H, W).

        Returns:
            torch.Tensor: A sequence of slice embeddings of shape (B, N_slices, slice_feat_dim).
        """
        B, N, H, W = x.shape
        x = x.view(B * N, 1, H, W)  # Treat slices as a single batch

        # Some backbones require 3-channel input
        if self.backbone not in ["densenet121", "densenet161", "densenet169", "densenet201", "resnet50"]:
            x = x.repeat(1, 3, 1, 1)  # (B*N, 3, H, W)

        # Pass through the CNN backbone
        features = self.cnn(x)

        # Apply global average pooling if the output is a feature map
        if features.dim() == 4:
            features = self.global_avg_pool(features)

        # Reshape to (B, N, slice_feat_dim)
        return features.view(B, N, self.slice_feat_dim)