import torch 
import torch.nn as nn
from IMC.nn.image_encoder import MultiSliceImageEncoder
from IMC.nn.metadata_encoder import MetadataEncoder
from IMC.nn.multi_task_head import MultiTaskHead
from IMC.helper import normalize_per_sample
from IMC.network04 import SliceFeatureFusion, BiDirectionalCrossModalAttentionFusion
import logging 
import os 

logger = logging.getLogger('IMC')
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

class MultiTaskHeadFusion(nn.Module):
    def __init__(self, image_emb_dim: int, metadata_emb_dim: int, fused_emb_dim: int, num_classes_dict: dict, dropout: float = 0.1):
        super(MultiTaskHeadFusion, self).__init__()
        self.fuser = nn.ModuleDict()
        self.classifier = nn.ModuleDict()

        for task_name, n_classes in num_classes_dict.items():
            self.fuser[task_name] = BiDirectionalCrossModalAttentionFusion(
                image_emb_dim=image_emb_dim,
                metadata_emb_dim=metadata_emb_dim,
                output_dim=fused_emb_dim
            )
            self.classifier[task_name] = self.make_task_head(
                in_dim=fused_emb_dim,
                out_dim=n_classes,
                dropout=dropout
            )
            
    def make_task_head(self, in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        """
        Creates a task-specific head (MLP) for classification.
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension (number of classes)
            
        Returns:
            nn.Sequential: Task head module
        """
        hidden_dim = (in_dim + out_dim) // 2

        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, image_emb: torch.Tensor, metadata_emb: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Joint feature embedding (B, input_dim)
            
        Returns:
            list: head logits
        """
        outputs = []
        for task_name in self.fuser.keys():
            fused_emb = self.fuser[task_name](image_emb, metadata_emb)
            logits = self.classifier[task_name](fused_emb)
            outputs.append(logits)
        return outputs        
            
class LateStageMRISequenceClassifier(nn.Module):
    def __init__(self, num_classes_dict: dict, metadata_input_dim: int, metadata_embed_dim: int = 128, img_enc_backbone: str = "densenet", fused_feat_dim: int = 256, output_emb_dim: int = 128):
        super(LateStageMRISequenceClassifier, self).__init__()
        self.image_encoder = MultiSliceImageEncoder(backbone=img_enc_backbone)
        slice_feat_dim = self.image_encoder.get_feature_dimension()
        self.slice_fusion = SliceFeatureFusion(
            slice_feat_dim=slice_feat_dim, fused_dim=fused_feat_dim
        )

        self.metadata_encoder = MetadataEncoder(
            metadata_input_dim, embed_dim=metadata_embed_dim, imputer='contextual'
        )

        self.multi_task_head = MultiTaskHeadFusion(
            image_emb_dim=fused_feat_dim,
            metadata_emb_dim=metadata_embed_dim,
            fused_emb_dim=output_emb_dim,
            num_classes_dict=num_classes_dict,
            dropout=0.1
        )
        
    def forward(self, image_slices: torch.Tensor, metadata: torch.Tensor):
        if DEBUG_MODE:
            logger.debug(f"Input image_slices shape: {image_slices.shape}")
            logger.debug(f"Input metadata shape: {metadata.shape}")
            logger.debug(f"Input stats - Images: min={image_slices.min():.3f}, max={image_slices.max():.3f}, mean={image_slices.mean():.3f}")
            logger.debug(f"Input stats - Metadata: min={metadata.min():.3f}, max={metadata.max():.3f}, mean={metadata.mean():.3f}")
            has_nan = torch.isnan(metadata).any()
            has_inf = torch.isinf(metadata).any()
            if has_nan or has_inf:
                logger.error(f"Metadata input contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
        
        slice_feats = self.image_encoder(image_slices)
        fused_image_feat = self.slice_fusion(slice_feats)

        if DEBUG_MODE:
            logger.debug(f"Fused image feature shape: {fused_image_feat.shape}")
            logger.debug(f"Fused image feature stats: min={fused_image_feat.min():.3f}, max={fused_image_feat.max():.3f}, mean={fused_image_feat.mean():.3f}")
            has_nan = torch.isnan(fused_image_feat).any()
            has_inf = torch.isinf(fused_image_feat).any()
            if has_nan or has_inf:
                logger.debug(f"Fused image feature contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
        

        metadata_feat = self.metadata_encoder(metadata)
        metadata_feat = normalize_per_sample(metadata_feat)

        if DEBUG_MODE:
            logger.debug(f"Metadata feature shape: {metadata_feat.shape}")
            logger.debug(f"Metadata feature stats: min={metadata_feat.min():.3f}, max={metadata_feat.max():.3f}, mean={metadata_feat.mean():.3f}")
            has_nan = torch.isnan(metadata_feat).any()
            has_inf = torch.isinf(metadata_feat).any()
            if has_nan or has_inf:
                logger.debug(f"Metadata feature contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
        
        outputs = self.multi_task_head(fused_image_feat, metadata_feat)
        if DEBUG_MODE:
            for i, r in enumerate(outputs):
                logger.debug(f"Output logits for task {i} shape: {r.shape}")
                logger.debug(f"Output logits for task {i} stats: min={r.min():.3f}, max={r.max():.3f}, mean={r.mean():.3f}")
                has_nan = torch.isnan(r).any()
                has_inf = torch.isinf(r).any()
                if has_nan or has_inf:
                    logger.debug(f"Output logits for task {i} contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
                    
        return outputs
    
if __name__ == "__main__":
    # Test the model with dummy data
    batch_size = 4
    num_slices = 3
    channels = 1
    height = 128
    width = 128
    metadata_input_dim = 50

    dummy_images = torch.randn(batch_size, num_slices, channels, height, width)
    dummy_metadata = torch.randn(batch_size, num_slices, metadata_input_dim)

    num_classes_dict = {
        "label_Sequence": 7,
        "label_Plane": 4,
        "label_BodyRegion": 5,
    }
    model = LateStageMRISequenceClassifier(
        num_classes_dict=num_classes_dict,
        metadata_input_dim=metadata_input_dim,
        metadata_embed_dim=64,
        img_enc_backbone="densenet",
        fused_feat_dim=128,
        output_emb_dim=64
    )
    outputs = model(dummy_images, dummy_metadata)
    for task_name, output in zip(num_classes_dict.keys(), outputs):
        print(f"Task: {task_name}, Output shape: {output.shape}")