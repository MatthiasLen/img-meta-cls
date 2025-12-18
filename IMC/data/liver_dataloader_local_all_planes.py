"""
Local Liver Dataset Module for Medical Image Classification

This module provides a PyTorch Dataset implementation for loading and processing
liver MRI/CT medical images from a local filesystem. It supports multi-slice
sampling, metadata encoding, and various augmentation strategies.

Author: Melanie Dohmen, Matthias Lenga, Tuan Truong
Date: 2025
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from pydicom import FileDataset, dcmread
from torch.utils.data import DataLoader, Dataset

from IMC.data.augment import augment
from IMC.data.dicom_tag_encoding import encode_dicom_tags_by_version
from IMC.data.resize import process_orthogonal_slices
import logging 
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger('IMC')

# Configuration constants
LOCAL_DATASET_PATH = os.getenv("LOCAL_DATASET_PATH", None)
logger.info(f"LOCAL_DATASET_PATH: {LOCAL_DATASET_PATH}")
METADATA_PATH = os.getenv("METADATA_PATH", None)
logger.info(f"METADATA_PATH: {METADATA_PATH}")
LABEL_CSV_PATH = os.getenv("LABEL_CSV_PATH", None)
logger.info(f"LABEL_CSV_PATH: {LABEL_CSV_PATH}")

# Default label mappings for medical imaging classification
DEFAULT_LABEL_NAMES = {
    "label_SequenceType": [
        "T1", "T2", "DWI", "ADC", "SUB", "DIXON_F", 
        "DIXON_IN", "DIXON_OPP", "BOLUS", "OTHER", "na"
    ],
    "label_FatSat": ["yes", "no", "na"],
    "label_MRCP": ["yes", "no", "na"],
    "label_AcquisitionPlane": ["AX", "COR", "SAG", "ORTHO", "ROT", "na"],
    "label_ContrastPhase": ["pre", "art", "portven", "trans", "hepa", "na"],
    "label_Contrast": ["pre", "post", "na"],
    "label_Localizer": ["yes", "no", "na"],
}

DUKE_LABEL_NAMES = {
    "label_SequenceType": ["T1", "T2", "DWI", "ADC", "DIXON_IN", "DIXON_OPP", "OTHER", "na"],
    "label_FatSat": ["yes", "no", "na"],    
    "label_MRCP": ["yes", "no", "na"],
    "label_AcquisitionPlane": ["AX", "COR", "OTHER", "na"],
    "label_ContrastPhase": ["pre", "art", "portven", "late", "na"],
    "label_Contrast": ["pre", "post", "na"],
    "label_Localizer": ["yes", "no", "na"],
}



class LiverDataset(Dataset):
    """
    PyTorch Dataset for loading liver medical imaging data from local filesystem.
    
    This dataset handles multi-slice DICOM images with associated metadata and 
    classification labels. It supports various sampling strategies, data augmentation,
    and handles missing or corrupted data gracefully.
    
    Features:
        - Multi-slice sampling from DICOM series
        - Metadata encoding and integration
        - Data augmentation support
        - Train/validation/test split support
        - Missing data handling
        - Local filesystem optimization
    
    Args:
        num_samples: Maximum number of samples to load. If None, loads all available samples.
        n_slices: Number of slices to sample from each DICOM series.
        img_size: Target image size for resizing (default: 224x224).
        label_names: Dictionary mapping label categories to their possible values.
        augment_conf: Augmentation configuration string (e.g., "NONE2D").
        split: List of fold names to include (e.g., ["fold_0", "fold_1"]).
        is_infer: If True, returns only images and metadata (no labels).
    
    Returns:
        Training mode: (images, metadata, targets, masks)
        Inference mode: (images, metadata)
        
    Example:
        >>> dataset = LiverDataset(
        ...     num_samples=1000,
        ...     n_slices=3,
        ...     split=["fold_0", "fold_1", "fold_2"]
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    """
    
    def __init__(
        self,
        num_samples: Optional[int] = 100,
        n_slices: int = 3,
        img_size: int = 256,
        label_names: Optional[Dict[str, List[str]]] = None,
        augment_conf: str = "NONE2D",
        split: List[str] | None = ["fold_0"],
        is_infer: bool = False,
        local_dataset_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        label_csv_path: Optional[str] = None,
    ) -> None:
        """Initialize the LiverDataset with specified configuration."""
        # Store configuration parameters
        self.num_samples = num_samples
        self.n_slices = n_slices
        self.img_size = img_size
        self.label_names = label_names if label_names is not None else DEFAULT_LABEL_NAMES
        self.augment_conf = augment_conf
        self.is_infer = is_infer
        
        # Initialize data containers
        self.labels: List[Dict[str, Any]] = []
        self.path_list: List[str] = []
        self.slice_filenames: Dict[str, List[Path]] = {}
        self.img_buffer: Dict[str, np.ndarray] = {}

        # Set dataset paths
        self.local_dataset_path = local_dataset_path if local_dataset_path is not None else LOCAL_DATASET_PATH
        self.metadata_path = metadata_path if metadata_path is not None else METADATA_PATH
        self.label_csv_path = label_csv_path if label_csv_path is not None else LABEL_CSV_PATH

        # Load and process metadata
        self._load_metadata_and_labels(split)
        
        logger.info(f"Dataset initialized with {len(self.path_list)} samples")

    def _load_metadata_and_labels(self, split: List[str] | None) -> None:
        """
        Load metadata and labels from CSV files and filter by split.
        
        Args:
            split: List of fold names to include in the dataset.
            
        Raises:
            FileNotFoundError: If CSV files are not found.
            ValueError: If no samples remain after filtering.
        """
        # Load labels and metadata from CSV files
        labels_df = pd.read_csv(self.label_csv_path)
        logger.info(f"Loaded {len(labels_df)} samples from label CSV")
        
        # Encode metadata using DICOM tag encoding
        if self.metadata_path and os.path.exists(self.metadata_path):
            metadata_df = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded {len(metadata_df)} samples from metadata CSV")
        else:
            metadata_df = encode_dicom_tags_by_version(labels_df, dicom_encoding_version="brain")
        metadata_df = metadata_df.set_index("Filepath")
        labels_df = labels_df.set_index("Filepath")
        
        # Filter by specified splits
        if split is not None:
            labels_df = labels_df[labels_df["split"].isin(split)]
            logger.info(f"Samples after filtering for split {split}: {len(labels_df)}")
        
        if len(labels_df) == 0:
            raise ValueError(f"No samples found for splits: {split}")
        
        # Populate path list and labels
        for filepath, row in labels_df.iterrows():
            self.path_list.append(filepath)
            self.labels.append(row.to_dict())
        
        # Store metadata DataFrame for later use
        self.metadata_df = metadata_df
        
        # Update num_samples based on available data
        if self.num_samples is None:
            self.num_samples = len(self.path_list)
        else:
            self.num_samples = min(self.num_samples, len(self.path_list))
            


    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def get_n_labels(self) -> Dict[str, int]:
        """
        Get the number of classes for each label category.
        
        Returns:
            Dictionary mapping label names to their number of classes.
            
        Example:
            >>> dataset.get_n_labels()
            {'label_SequenceType': 11, 'label_FatSat': 3, ...}
        """
        return {label_name: len(classes) for label_name, classes in self.label_names.items()}


    def open_dicom_slice_from_series(
        self, 
        path_dicom_folder: str, 
    ) -> torch.Tensor:
        """
        Load and process multiple slices from a DICOM series.
        
        This method samples slices from a DICOM series according to the specified
        sampling strategy, applies augmentation, and returns them as a torch tensor.
        
        Args:
            path_dicom_folder: Path to the DICOM series folder.
            
        Returns:
            Tensor of shape (n_images, 1, H, W) containing the processed slices.
            
        Raises:
            RuntimeError: If no DICOM files are found or if loading fails.
            
        Note:
            - If fewer slices exist than requested, missing slices are filled with zeros
            - Multi-dimensional pixel arrays are replaced with zero arrays
            - All images are resized to self.img_size and augmented
        """
        # Get list of DICOM files in the series
        slices, _ = process_orthogonal_slices(Path(self.local_dataset_path) / path_dicom_folder, target_size=(256, 256), normalize=False)
        images = []
        
        for slice_idx in range(slices.shape[0]):
            image = slices[slice_idx, ...]
            # Apply data augmentation
            image = augment(image, self.augment_conf)
            
            # Convert to tensor and add channel dimension
            images.append(torch.tensor(image, dtype=torch.float32).unsqueeze(0))
        
        # Stack all slices into a single tensor
        return torch.stack(images, dim=0)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]
    ]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            For inference mode: (images, metadata, filepath)
            For training mode: (images, metadata, targets, masks)
            
            Where:
                - images: Tensor of shape (n_slices, 1, H, W)
                - metadata: Tensor of encoded metadata features
                - targets: Tuple of label indices for each classification task
                - masks: Tuple of boolean masks indicating valid (non-"na") labels
        
        Raises:
            IndexError: If idx is out of range.
            Exception: If required labels are missing.
        """
        if idx >= len(self.path_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.path_list)}")
        
        # Load multi-slice images
        images = self.open_dicom_slice_from_series(
            self.path_list[idx]
        )
        
        # Load encoded metadata
        try:
            metadata = self.metadata_df.loc[self.path_list[idx]].to_numpy()
            metadata = torch.tensor(metadata, dtype=torch.float32)
        except KeyError:
            logger.warning(f"Metadata not found for {self.path_list[idx]}, using zeros")
            # Use zero metadata if not found
            metadata = torch.zeros(90, dtype=torch.float32)  # Adjust size as needed
        
        # Return early if in inference mode
        if self.is_infer:
            return images, metadata, self.path_list[idx]
        
        # Process classification labels
        targets, masks = self._process_labels(idx)
        
        return images, metadata, targets, masks

    def _process_labels(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Process classification labels for the given sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple containing:
                - targets: Tuple of label indices for each classification task
                - masks: Tuple of boolean masks for valid labels
                
        Raises:
            Exception: If required labels are missing.
        """
        label_indices = {}
        
        for label_class in self.label_names:
            if label_class not in self.labels[idx]:
                raise Exception(f"Missing label: {label_class} for sample {idx}")
            
            label_value = self.labels[idx][label_class]
            
            if label_value in self.label_names[label_class]:
                label_indices[label_class] = self.label_names[label_class].index(label_value)
            else:
                # Handle unknown label values
                logger.warning(f"Unknown label value '{label_value}' for {label_class}, using -1")
                raise ValueError(f"Unknown label value '{label_value}' for {label_class}")
        
        # Create target tensors and masks
        targets = tuple(
            torch.tensor(label_indices[label_class]) 
            for label_class in self.label_names.keys()
        )
        
        # Masks indicate valid labels (not "na")
        masks = tuple(
            torch.tensor(label_indices[label_class] != self.label_names[label_class].index("na"))
            for label_class in self.label_names.keys()
        )
        
        return targets, masks


def get_train_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    num_samples: Optional[int] = 100,
    folder_split: List[str] = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6", "fold_7"],
) -> DataLoader:
    """
    Create a DataLoader for training data.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data.
        num_samples: Maximum number of samples to load (None for all).
        folder_split: List of fold names to include in training set.
        
    Returns:
        Configured DataLoader for training.
    """
    dataset = LiverDataset(
        split=folder_split, 
        num_samples=num_samples, 
        augment_conf="NONE2D"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def get_valid_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    num_samples: Optional[int] = 100,
    folder_split: List[str] = ["fold_8"],
) -> DataLoader:
    """
    Create a DataLoader for validation data.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data (typically False for validation).
        num_samples: Maximum number of samples to load (None for all).
        folder_split: List of fold names to include in validation set.
        
    Returns:
        Configured DataLoader for validation.
    """
    dataset = LiverDataset(
        split=folder_split, 
        num_samples=num_samples, 
        augment_conf="NONE2D"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def get_test_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    num_samples: Optional[int] = 100,
    folder_split: List[str] = ["fold_9"],
) -> DataLoader:
    """
    Create a DataLoader for test data.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data (typically False for testing).
        num_samples: Maximum number of samples to load (None for all).
        folder_split: List of fold names to include in test set.
        
    Returns:
        Configured DataLoader for testing.
    """
    dataset = LiverDataset(
        split=folder_split, 
        num_samples=num_samples, 
        augment_conf="NONE2D"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def get_infer_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    num_samples: Optional[int] = 100,
    folder_split: List[str] = ["fold_9"],
    label_names: Optional[Dict[str, List[str]]] = None,
) -> DataLoader:
    """
    Create a DataLoader for inference (prediction) data.
    
    This DataLoader returns only images and metadata, without labels.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data (typically False for inference).
        num_samples: Maximum number of samples to load (None for all).
        folder_split: List of fold names to include.
        label_names: Optional mapping of label names for the dataset.

    Returns:
        Configured DataLoader for inference.
    """
    dataset = LiverDataset(
        split=folder_split,
        num_samples=num_samples,
        is_infer=True,
        augment_conf="NONE2D",
        label_names=label_names
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    """
    Example usage and testing of the LiverDataset.
    
    This script demonstrates how to create a dataset, get label information,
    and iterate through batches to inspect data shapes and content.
    """
    print("LiverDataset Demo")
    print("=" * 50)
    
    # Create a dataset instance
    dataset = LiverDataset(num_samples=200, n_slices=5, is_infer=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Display label information
    n_labels = dataset.get_n_labels()
    print(f"Label dimensions: {n_labels}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of slices per sample: {dataset.n_slices}")
    print()
    
    # Process a few batches
    print("Processing batches:")
    for batch_idx, batch_data in enumerate(dataloader):
        if len(batch_data) == 4:  # Training mode
            images, metadata, targets, masks = batch_data
            print(f"Batch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")  # (B, N_slices, C, H, W)
            print(f"  Metadata shape: {metadata.shape}")  # (B, metadata_dim)
            print(f"  Targets shapes: {[t.shape for t in targets]}")
            print(f"  Masks shapes: {[m.shape for m in masks]}")
            print(f"  Targets sample: {[t for t in targets]}")
            print(f"  Masks sample: {[m for m in masks]}")
            print()
        else:  # Inference mode
            images, metadata, filepaths = batch_data
            print(f"Inference Batch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Metadata shape: {metadata.shape}")
            print(f"  Filepaths: {filepaths}")
            print()

        # Only show first few batches
        if batch_idx >= 2:
            print("Demo completed successfully!")
            break
