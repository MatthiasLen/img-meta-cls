import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
from torchvision.models.densenet import DenseNet121_Weights
import torch.nn.init as init

class MultiSliceImageEncoder(nn.Module):
    """
    Encodes multiple MRI slices using a shared CNN backbone (ResNet18).
    Each slice is processed independently, and the output is a sequence of slice embeddings.
    """

    def __init__(self, pretrained: bool = True, n_channels: int = 1, backbone: str = "densenet"):
        super().__init__()

        if backbone == "densenet":   
            # Load pretrained DenseNet121 backbone and adapt for single channel if needed
            print("DenseNet121 backbone")
            self.cnn = (
                models.densenet121(weights=DenseNet121_Weights.DEFAULT)
                if pretrained
                else models.densenet121()
            )

            # Adjust first conv layer if input channel != 3
            if self.cnn.features.conv0.in_channels != n_channels:
                old_weights = self.cnn.features.conv0.weight.data.clone()  # shape (64, 3, 7, 7)
                self.cnn.features.conv0 = nn.Conv2d(
                    n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                if n_channels == 1:
                    # Initialize conv1 weights by averaging pretrained weights across RGB channels
                    new_weights = old_weights.mean(dim=1, keepdim=True)  # shape (64, 1, 7, 7)
                    self.cnn.features.conv0.weight.data = new_weights
                else:
                    # For n_channels != 3 or 1, use Xavier initialization
                    init.xavier_uniform_(self.cnn.conv1.weight)

            # Remove final fc layer
            # output shape: (B, 1024, H', W') since last conv layer of DenseNet has 1024 feature maps
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

            # DenseNet121 will create featuremaps with 1024 channels
            self.slice_feat_dim = 1024
        
        elif backbone == "swinv2":
            print("SwinV2 backbone")
            self.cnn = models.swin_v2_b(weights="DEFAULT") if pretrained else models.swin_v2_b(weights="IMAGENET1K_V1")
            if self.cnn.features[0][0].in_channels != n_channels:
                old_weights = self.cnn.features[0][0].weight.data.clone()  # shape (128, 3, 4, 4)
                self.cnn.features[0][0] = nn.Conv2d(
                    n_channels, 128, kernel_size=4, stride=4, bias=False
                )
                if n_channels == 1:
                    new_weights = old_weights.mean(dim=1, keepdim=True)  # shape (128, 1, 4, 4)
                    self.cnn.features[0][0].weight.data = new_weights
                else:
                    init.xavier_uniform_(self.cnn.features[0][0].weight)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            self.slice_feat_dim = 1024  # Swin V2 base has 1024 feature dimensions
        elif backbone == "dinov3_vits16":
            print("DINOv3 ViT-S/16 backbone")
            import torch.hub
            self.cnn = torch.hub.load("/home/tuan.truong/codebase/dinov3", 'dinov3_vits16', source='local', weights="/home/tuan.truong/codebase/pretrained_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
            self.slice_feat_dim = 384  # DINOv3 ViT-S/16 has 384 feature dimensions
        elif backbone == "dinov3_vitb16":
            print("DINOv3 ViT-B/16 backbone")
            import torch.hub
            self.cnn = torch.hub.load("/home/tuan.truong/codebase/dinov3", 'dinov3_vitb16', source='local', weights="/home/tuan.truong/codebase/pretrained_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
            self.slice_feat_dim = 768  # DINOv3 ViT-B/16 has 768 feature dimensions
        elif backbone == "dinov3_vitl16":
            print("DINOv3 ViT-L/16 backbone")
            import torch.hub
            self.cnn = torch.hub.load("/home/tuan.truong/codebase/dinov3", 'dinov3_vitl16', source='local', weights="/home/tuan.truong/codebase/pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
            self.slice_feat_dim = 1024  # DINOv3 ViT-L/16 has 1024 feature dimensions
        else:
            # Load pretrained ResNet50 backbone and adapt for single channel if needed
            print("ResNet50 backbone")
            self.cnn = (
                models.resnet50(weights=ResNet50_Weights.DEFAULT)
                if pretrained
                else models.resnet50()
            )

            # Adjust first conv layer if input channel != 3
            if self.cnn.conv1.in_channels != n_channels:
                old_weights = self.cnn.conv1.weight.data.clone()  # shape (64, 3, 7, 7)
                self.cnn.conv1 = nn.Conv2d(
                    n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                if n_channels == 1:
                    # Initialize conv1 weights by averaging pretrained weights across RGB channels
                    new_weights = old_weights.mean(dim=1, keepdim=True)  # shape (64, 1, 7, 7)
                    self.cnn.conv1.weight.data = new_weights
                else:
                    # For n_channels != 3 or 1, use Xavier initialization
                    init.xavier_uniform_(self.cnn.conv1.weight)

            # Remove final fc layer and avgpool - since we handle pooling later
            # output shape: (B, 2048, H', W') since last conv layer of ResNet has 2048 feature maps
            self.cnn = nn.Sequential(
                *list(self.cnn.children())[:-2]
            )  

            # ResNet50 will create feature maps with 2048 channels
            self.slice_feat_dim = 2048

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = backbone
        
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
            x (torch.Tensor): Input tensor of shape (B, N_slices, C, H, W)
            layer_names (list): List of layer names to extract outputs from.
        Returns:
            dict: Dictionary of intermediate outputs.
        """
        outputs = {}
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)  # treat slices as batch
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
        Args:
            x (torch.Tensor): Input tensor of shape (B, N_slices, C, H, W)
        Returns:
            torch.Tensor: Slice embeddings of shape (B, N_slices, slice_feat_dim)
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)  # treat slices as batch
        if not self.backbone.startswith("dense") and C == 1:
            # Repeat to have 3 channels
            x = x.repeat(1, 3, 1, 1)  # (B*N, 3, H, W)
        
        features = self.cnn(x)  # For ResNet18 (B*N, 512, H', W'), for DenseNet121 (B*N, 1024, H', W')
        if features.dim() == 4:
            features = self.global_avg_pool(features)
  
        return features.view(B, N, self.slice_feat_dim)  # per slice embeddings, for Resnet18 (B, N, 512), for DenseNet121->(B,N,1024)