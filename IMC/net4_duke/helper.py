from IMC.network04 import (
    ImageBasedClassifier,
    SimpleImageBasedClassifier,
    MetadataBasedClassifier,
    MRISequenceClassifier,
)
import torch.nn as nn


def build_model(
    modality="combined",
    vanilla_image_classifier=False,
    img_enc_backbone="densenet121",
    metadata_enc_type="sparse",
    imputer_type="contextual",
    sparse_enc_version="v1",
    output_emb_dim=128,
    num_classes_dict: dict = None,
    metadata_input_dim: int = None,
    metadata_embed_dim: int = 128,
    fusion_module_version: str = "v2",
    metadata_dropout: bool = False,
    pre_processors=None,
    post_processors=None,
    **kwargs,
) -> nn.Module:
    """Instantiate the correct Network-v04 variant based on ``modality``.

    Args:
        modality: One of "image", "metadata", or "combined" to specify the
                  model variant.
        vanilla_image_classifier: If True and modality is "image", use the
                                  simpler SimpleImageBasedClassifier variant.
        img_enc_backbone: Backbone architecture for image encoder (e.g. "resnet18").
        metadata_enc_type: Type of metadata encoder ("contextual", "sparse", or "ignore").
        imputer_type: If using imputer encoder, which type to use ("contextual" or "ignore").
        sparse_enc_version: If using sparse metadata encoder, which version to use ("v1", "v2", or "v5").
        output_emb_dim: Output embedding dimension for metadata encoder (used in combined model).
        num_classes_dict:   Mapping from task name to number of classes
                            (or ``1`` for regression), e.g.
                            ``{"label_SequenceType": 5, ...}``.
        metadata_input_dim: Number of metadata features provided by the
                            dataloader.
        metadata_embed_dim: Embedding dimension for metadata features.
        fusion_module_version: Which fusion module version to use in the combined model ("v1" or "v2").
        metadata_dropout: Whether to apply dropout to metadata features in the combined model.

    Returns:
        Uninitialised (random weights) ``nn.Module`` for the selected mode.

    Raises:
        ValueError: If ``modality`` is not one of the supported values.
    """

    if modality == "image":
        image_classifier_type = "vanilla" if vanilla_image_classifier else "default"
        if image_classifier_type == "vanilla":
            model = SimpleImageBasedClassifier(
                num_classes_dict=num_classes_dict,
            )
        else:
            model = ImageBasedClassifier(
                num_classes_dict=num_classes_dict,
                img_enc_backbone=img_enc_backbone,
                n_channels=kwargs.get("n_channels", 1),
            )
    elif modality == "metadata":
        if metadata_enc_type == "imputer":
            model = MetadataBasedClassifier(
                num_classes_dict=num_classes_dict,
                metadata_input_dim=metadata_input_dim,
                metadata_embed_dim=metadata_embed_dim,
                output_emb_dim=output_emb_dim,
                metadata_encoder_type="imputer",
                imputer_type=imputer_type,
            )
        elif metadata_enc_type == "sparse":
            assert sparse_enc_version in ["v1", "v2", "v5"], "Invalid sparse encoder version"
            model = MetadataBasedClassifier(
                num_classes_dict=num_classes_dict,
                metadata_input_dim=metadata_input_dim,
                metadata_embed_dim=metadata_embed_dim,
                output_emb_dim=output_emb_dim,
                metadata_encoder_type="sparse",
                sparse_enc_version=sparse_enc_version,
            )
        else:
            raise ValueError(f"Unknown metadata_enc_type '{metadata_enc_type}'.")
    elif modality == "combined":
        model = MRISequenceClassifier(
            metadata_input_dim=metadata_input_dim,
            num_classes_dict=num_classes_dict,
            img_enc_backbone=img_enc_backbone,
            metadata_encoder_type=metadata_enc_type,
            imputer_type=imputer_type,
            sparse_enc_version=sparse_enc_version,
            metadata_embed_dim=metadata_embed_dim,
            output_emb_dim=output_emb_dim,
            fusion_module_version=fusion_module_version,
            dropout_metadata=metadata_dropout,
            scalar_modulation=kwargs.get("scalar_modulation", False),
            n_channels=kwargs.get("n_channels", 1),
            learn_missing_embed=kwargs.get("learn_missing_embed", False),
            pre_processors=pre_processors,
            post_processors=post_processors,
        )
    else:
        raise ValueError(f"Unknown modality '{modality}'. Choose one of: combined, image, metadata.")

    return model
