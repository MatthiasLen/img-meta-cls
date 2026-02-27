"""
Inference script for Duke dataset using Network v04.

This script performs inference on the Duke MRI dataset using a trained model.
It supports:
- Loading trained model checkpoints
- Running inference on test data or specific folds
- Generating predictions CSV
- Evaluating predictions against ground truth

The script is designed to work with the Duke-specific dataloader and handles
the label mappings and stratified fold splits.
"""

import argparse
import os
import json

# Set environment variables for Duke dataset
os.environ["DEBUG_MODE"] = "0"
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet"

import torch.nn as nn
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
import numpy as np

from IMC.network04 import MRISequenceClassifier, MRISequenceClassifierWithSparseMetadata, ImageBasedClassifier, MetadataBasedClassifier, SingleChannelImageBasedClassifier
from IMC.data.duke_dataloader_local import LiverDataset, DUKE_ORIGINAL_LABEL_NAMES
from torch.utils.data import DataLoader
from IMC.helper import normalize_per_sample
from IMC.evaluate_duke import run_evaluation


def create_inference_dataloader(
    batch_size: int = 16,
    num_workers: int = 4,
    fold_indices: Optional[List[int]] = None,
    aggregated_metadata: bool = False,
    use_preselected_features: bool = True,
    exclude_contrast_yn: bool = True,
    n_slices: int = 3
) -> DataLoader:
    """
    Create a dataloader for inference.
    
    Args:
        batch_size: Batch size for inference
        num_workers: Number of worker processes
        fold_indices: List of fold indices to include (e.g., [0, 1, 2] or None for all)
        aggregated_metadata: Whether to use aggregated metadata
        use_preselected_features: Whether to use preselected features
        exclude_contrast_yn: Whether to exclude label_Contrast
        n_slices: Number of slices to sample from each MRI volume
    Returns:
        DataLoader for inference
    """
    # Convert fold indices to split format
    if fold_indices is not None:
        split = [f"fold_{i}" for i in fold_indices]
    else:
        split = None  # Use all data
    
    dataset = LiverDataset(
        split=split,
        num_samples=None,  # Use all samples
        augment_conf="NONE2D",
        aggregated_metadata=aggregated_metadata,
        use_preselected_features=use_preselected_features,
        exclude_contrast_yn=exclude_contrast_yn,
        is_infer=True,
        label_names=DUKE_ORIGINAL_LABEL_NAMES,
        n_slices=n_slices
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def load_model(
    checkpoint_path: str,
    num_classes_dict: dict,
    metadata_input_dim: int,
    img_enc_backbone: str = "densenet121",
    modality: str = "combined",
    image_naive_approach: bool = False,
    metadata_enc_type: str = "none",
    sparse_enc_version: str = "v1",
    fusion_module_version: str = "v1",
    incl_regression: bool = True,
    device: torch.device = None
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_classes_dict: Dictionary of number of classes per task
        metadata_input_dim: Dimension of metadata input
        img_enc_backbone: Image encoder backbone architecture
        incl_regression: Whether the model includes regression
        modality: str = "combined",
        metadata_enc_type: str = "none",
        sparse_enc_version: str = "v1",
        fusion_module_version: str = "v1",
        image_naive_approach: bool = False,
        device: torch.device = None
    
    Returns:
        Loaded model ready for inference
    """
    # Initialize model architecture
    if modality == "image":
        if image_naive_approach:
            model = SingleChannelImageBasedClassifier(num_classes_dict=num_classes_dict)
        else:
            model = ImageBasedClassifier(num_classes_dict=num_classes_dict, backbone=img_enc_backbone)
    elif modality == "metadata":
        if metadata_enc_type == "none":
            model = MetadataBasedClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_encoder_type=metadata_enc_type,
                imputer_type="contextual",
                incl_regression=incl_regression
            )
        elif metadata_enc_type == "sparse":
            if sparse_enc_version == "v1":
                model = MetadataBasedClassifier(
                    metadata_input_dim=metadata_input_dim,
                    num_classes_dict=num_classes_dict,
                    metadata_encoder_type="sparse",
                    incl_regression=incl_regression
                )
            else:
                model = MetadataBasedClassifier(
                    metadata_input_dim=metadata_input_dim,
                    num_classes_dict=num_classes_dict,
                    metadata_encoder_type="sparse_v2",
                    incl_regression=incl_regression
                )
        else:
            # Imputer is NanIgnorer
            model = MetadataBasedClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_encoder_type="imputer",
                imputer_type="ignore",
                incl_regression=incl_regression
            )
    else:
        # Combined model with cross-attention fusion
        if metadata_enc_type == "none":
            model = MRISequenceClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                img_enc_backbone=img_enc_backbone,
                incl_regression=incl_regression,
                fusion_module_version=fusion_module_version,
                imputer_type="ignore"
            )
        elif metadata_enc_type == "imputer":
            model = MRISequenceClassifier(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                img_enc_backbone=img_enc_backbone,
                incl_regression=incl_regression,
                fusion_module_version=fusion_module_version,
                imputer_type="contextual"
            )
        elif metadata_enc_type == "sparse" and sparse_enc_version == "v1":
            model = MRISequenceClassifierWithSparseMetadata(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_embeder_type="sparse",
                img_enc_backbone=img_enc_backbone,
                dropout_metadata=False,
                fusion_module_version=fusion_module_version,
                include_regression=incl_regression
            )
        elif metadata_enc_type == "sparse" and sparse_enc_version == "v2":
            model = MRISequenceClassifierWithSparseMetadata(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_embeder_type="sparse_v2",
                img_enc_backbone=img_enc_backbone,
                dropout_metadata=False,
                fusion_module_version=fusion_module_version,
                include_regression=incl_regression
            )
        elif metadata_enc_type == "sparse" and sparse_enc_version == "v5":
            model = MRISequenceClassifierWithSparseMetadata(
                metadata_input_dim=metadata_input_dim,
                num_classes_dict=num_classes_dict,
                metadata_embeder_type="sparse_v5",
                img_enc_backbone=img_enc_backbone,
                dropout_metadata=False,
                fusion_module_version=fusion_module_version,
                include_regression=incl_regression,
            )
        else:
            raise ValueError(f"Invalid metadata_enc_type '{metadata_enc_type}' or sparse_enc_version '{sparse_enc_version}'")
            
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    return model


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    incl_regression: bool = True
) -> tuple[dict, list]:
    """
    Run inference on a dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader for inference
        device: Device to run inference on
        incl_regression: Whether model includes regression
        
    Returns:
        Tuple of (predictions_dict, filepaths)
    """
    num_classes_dict = dataloader.dataset.get_n_labels()
    label_maps = dataloader.dataset.label_names
    
    predictions = {task: [] for task in num_classes_dict.keys()}
    if "label_ContrastPhase" in label_maps:
        predictions["label_Contrast"] = []  # For contrast yes/no task
    filepaths = []
    
    print(f"Running inference on {len(dataloader.dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            images, metadata, paths = batch
            images = images.to(device)
            metadata = metadata.to(device)
            
            images = normalize_per_sample(images)
            outputs = model(images, metadata)
            
            for i, task in enumerate(num_classes_dict.keys()):
                if incl_regression and task == "label_ContrastPhase":
                    # Regression output
                    preds = outputs[i].cpu().numpy()
                    preds = np.rint(preds).astype(int)
                    preds = np.clip(preds, 0, len(label_maps[task]) - 1)
                    preds = preds.flatten()
                else:
                    # Classification output
                    preds = torch.argmax(outputs[i], dim=1).cpu().numpy()
                
                # Convert indices to label names
                preds_in_names = [label_maps[task][p] for p in preds]
                predictions[task].extend(preds_in_names)
            
            filepaths.extend(paths)
    
    # Handle contrast yes/no task
    # If label_ContrastPhase has a phase that is not PRE, then label_Contrast is POST, else PRE
    if "label_ContrastPhase" in label_maps:
        predictions["label_Contrast"].extend([
            "post" if phase not in ["pre", "na"] else "pre" 
            for phase in predictions["label_ContrastPhase"]
        ])
        
    return predictions, filepaths


def save_predictions(
    predictions: dict,
    filepaths: list,
    output_dir: str,
    checkpoint_path: str = None
) -> pd.DataFrame:
    """
    Save predictions to CSV.
    
    Args:
        predictions: Dictionary of predictions per task
        filepaths: List of file paths
        output_dir: Directory to save predictions
        checkpoint_path: Path to checkpoint (for metadata)
        
    Returns:
        DataFrame of predictions
    """
    output_df = pd.DataFrame({"Filepath": filepaths})
    
    for task in predictions.keys():
        output_df[task] = predictions[task]
    
    # Save predictions
    pred_path = os.path.join(output_dir, "predictions.csv")
    output_df.to_csv(pred_path, index=False)
    print(f"\n✓ Predictions saved to {pred_path}")
    
    # Save metadata
    if checkpoint_path:
        metadata = {
            "checkpoint": checkpoint_path,
            "num_samples": len(filepaths),
            "tasks": list(predictions.keys())
        }
        metadata_path = os.path.join(output_dir, "inference_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
    
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Run inference on Duke dataset")
    parser.add_argument('--ckpt', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for inference')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU ID to use')
    parser.add_argument('--folds', type=str, default=None, 
                       help='Comma-separated fold indices to infer on (e.g., "0,1,2"). None for all data.')
    parser.add_argument('--backbone', type=str, default="densenet121", 
                       help='Image encoder backbone')
    parser.add_argument('--include_regression', action='store_true', 
                       help='Whether model includes regression task')
    parser.add_argument("--modality", type=str, default="combined", 
                        help="Modality to use: 'combined', 'image', or 'metadata'")
    parser.add_argument('--image_naive_approach', action='store_true',
                       help='Whether to use a naive approach for image-only inference (e.g., single slice)')
    parser.add_argument("--metadata_enc_type", type=str, default='none', 
                        help="Which metadata encoder to use: 'none', 'imputer', 'sparse'")
    parser.add_argument("--sparse_enc_version", type=str, default="v1", 
                        help="Sparse encoder version: v1 or v2")
    parser.add_argument("--fusion_module_version", type=str, default="v1", 
                        help="Fusion module version: v1 or v2")
    parser.add_argument('--eval', action='store_true', 
                       help='Whether to evaluate predictions against ground truth')
    parser.add_argument('--n_slices', type=int, default=3,
                       help='Number of slices to sample from each MRI volume')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse fold indices
    if args.folds is not None:
        fold_indices = [int(f) for f in args.folds.split(",")]
        print(f"Running inference on folds: {fold_indices}")
    else:
        fold_indices = None
        print("Running inference on all data")
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    aggregated_metadata = False
    use_preselected_features = False
    exclude_contrast_yn = True
    
    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = create_inference_dataloader(
        batch_size=args.batch_size,
        num_workers=4,
        fold_indices=fold_indices,
        aggregated_metadata=aggregated_metadata,
        use_preselected_features=use_preselected_features,
        exclude_contrast_yn=exclude_contrast_yn,
        n_slices=args.n_slices
    )
    
    print(f"✓ Loaded {len(dataloader.dataset)} samples")
    
    # Get model configuration
    num_classes_dict = dataloader.dataset.get_n_labels()
    metadata_input_dim = dataloader.dataset.num_metadata_features
    
    print(f"\nModel configuration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Metadata dim: {metadata_input_dim}")
    print(f"  Tasks: {list(num_classes_dict.keys())}")
    print(f"  Include regression: {args.include_regression}")
    
    # Load model
    model = load_model(
        checkpoint_path=args.ckpt,
        num_classes_dict=num_classes_dict,
        metadata_input_dim=metadata_input_dim,
        img_enc_backbone=args.backbone,
        incl_regression=args.include_regression,
        modality=args.modality,
        metadata_enc_type=args.metadata_enc_type,
        sparse_enc_version=args.sparse_enc_version,
        fusion_module_version=args.fusion_module_version,
        device=device,
        image_naive_approach=args.image_naive_approach
    )
    
    # Run inference
    predictions, filepaths = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        incl_regression=args.include_regression
    )
    
    # Save predictions
    pred_df = save_predictions(
        predictions=predictions,
        filepaths=filepaths,
        output_dir=args.output_dir,
        checkpoint_path=args.ckpt
    )
    
    # Evaluate if requested
    if args.eval:
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        
        label_csv_path = os.environ.get("LABEL_CSV_PATH")
        label_df = pd.read_csv(label_csv_path)
        
        run_evaluation(pred_df, label_df, args.output_dir)
        print(f"\n✓ Evaluation results saved to {args.output_dir}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Total samples processed: {len(filepaths)}")


if __name__ == '__main__':
    main()
