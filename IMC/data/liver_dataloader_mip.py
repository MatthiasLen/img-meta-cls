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
import logging 

logger = logging.getLogger('IMC')

# from dotenv import load_dotenv
# load_dotenv()

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
        img_size: int = 224,
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
        try:
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
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required CSV file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

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

    def get_bucket_filelist(self, path_dicom_folder: str) -> List[Path]:
        """
        Get a sorted list of DICOM files in the specified folder.
        
        Args:
            path_dicom_folder: Path to the DICOM series folder.
            
        Returns:
            List of Path objects pointing to DICOM files, naturally sorted.
            
        Raises:
            RuntimeError: If the folder doesn't exist or contains no DICOM files.
        """
        series_folder = Path(self.local_dataset_path) / path_dicom_folder
        
        if not series_folder.exists():
            raise RuntimeError(f"DICOM folder not found: {series_folder}")
        
        # Find all DICOM files in the folder
        dicom_files = list(series_folder.rglob("*.dcm")) + list(series_folder.rglob("*.dicom"))
        
        if not dicom_files:
            raise RuntimeError(f"No DICOM files found in folder: {series_folder}")
        
        # Return naturally sorted list
        return natsorted(dicom_files)
    
    def _check_combined_series(self, dicom_filepaths: List[Path]) -> bool:
        """
        Check if series contains different volumes combined.
        Select one volume randomly if so.
        """
        volume_ids = set()
        for filepath in dicom_filepaths:
            ds = dcmread(filepath, stop_before_pixels=True)
            volume_id = getattr(ds, 'AcquisitionNumber', None)
            if volume_id is not None:
                volume_ids.add(volume_id)
        
        if len(volume_ids) > 1:
            selected_volume = random.choice(list(volume_ids))
            dicom_filepaths = [fp for fp in dicom_filepaths if dcmread(fp, stop_before_pixels=True).AcquisitionNumber == selected_volume]
            logger.info(f"Multiple volumes detected. Selected volume {selected_volume} with {len(dicom_filepaths)} slices.")
            return dicom_filepaths
        return dicom_filepaths
    
    def _load_volume(self, dicom_filepaths: List[Path]) -> np.ndarray:
        """
        Load a 3D volume from a list of DICOM file paths.
        
        Args:
            dicom_filepaths: List of Path objects pointing to DICOM files.
        Returns:
            3D numpy array representing the volume (D, H, W).
        """
        # dicom_filepaths = self._check_combined_series(dicom_filepaths)
        dicom_data = []
        for filepath in dicom_filepaths:
            ds = dcmread(filepath)
            slice_location = getattr(ds, 'SliceLocation', None)
            if slice_location is None:
                image_position = getattr(ds, 'ImagePositionPatient', None)
                if image_position is not None:
                    slice_position = float(image_position[-1]) if hasattr(image_position, '__getitem__') else float(image_position.value[-1])
                else:
                    slice_position = 0.0
            else:
                slice_position = float(slice_location)
            dicom_data.append((slice_position, ds, filepath))
    
        # Group slices by patient orientation
        orientation_groups = {}
        for slice_position, ds, filepath in dicom_data:
            orientation = getattr(ds, 'ImageOrientationPatient', None)
            if orientation is not None:
                # Convert to tuple for use as dict key
                orientation_key = tuple(orientation)
            if orientation_key not in orientation_groups:
                orientation_groups[orientation_key] = []
            orientation_groups[orientation_key].append((slice_position, ds, filepath))
        
        # Select the orientation group with the most slices
        if orientation_groups:
            selected_orientation = max(orientation_groups.keys(), key=lambda k: len(orientation_groups[k]))
            dicom_data = orientation_groups[selected_orientation]
            # logger.info(f"Selected orientation {selected_orientation} with {len(dicom_data)} slices from {len(orientation_groups)} orientations.")
        
        # Sort slices by position
        dicom_data.sort(key=lambda x: x[0])

        # Extract pixel arrays
        volume_slices = []
        for _, ds, file_path in dicom_data:
            try:
                pixel_array = ds.pixel_array
                
                # Apply slope and intercept if available (for proper intensity values)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                
                volume_slices.append(pixel_array)
            except Exception as e:
                logger.warning(f"Warning: Could not read {file_path}: {e}")
                continue

        try:
            volume = np.stack(volume_slices, axis=0)
        except Exception as e:
            logger.error(f"Error stacking volume slices: {e}")
            volume = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

        return volume
    
    def _mip(self, volume: np.ndarray) -> np.ndarray:
        """
        Compute the Maximum Intensity Projection (MIP) of a 3D volume.
        
        Args:
            volume: 3D numpy array (D, H, W).
            
        Returns:
            2D numpy array (H, W) representing the MIP.
        """
        return np.max(volume, axis=0)

    def open_dicom_slice_from_series(
        self, 
        path_dicom_folder: str,
        n_images: int = 1,
        sampling_type: str = "equidistant"
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
        slice_filenames = self.get_bucket_filelist(path_dicom_folder)

        # Calculate slice indices to sample
        num_slices = len(slice_filenames)
        slice_indices = self._calculate_slice_indices(num_slices, n_images, sampling_type)
        
        # Load and process each slice
        images = []
        for slice_idx in slice_indices:
            if slice_idx is None:
                # Create zero-filled slice for missing data
                image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            else:
                # Load DICOM slice and extract pixel array
                dicom_slice = dcmread(slice_filenames[slice_idx])
                image = dicom_slice.pixel_array.astype(np.float32)
                
                # Handle multi-dimensional arrays (replace with zeros)
                if len(image.shape) > 2:
                    logger.warning(f"Multi-dimensional pixel array in {slice_filenames[slice_idx]}, using zeros")
                    image = np.zeros((self.img_size, self.img_size), dtype=np.float32)

            # Apply data augmentation
            image = augment(image, self.augment_conf)
            
            # Convert to tensor and add channel dimension
            images.append(torch.tensor(image, dtype=torch.float32).unsqueeze(0))
        
        
        # Load MIP image
        volume = self._load_volume(slice_filenames)
        mip_image = self._mip(volume)
        mip_image = augment(mip_image, self.augment_conf)
            
        # Convert to tensor and add channel dimension
        mip_image = torch.tensor(mip_image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
    
        images.append(mip_image)

        return torch.stack(images, dim=0)  # (n_images, 1, H, W)
    
    def _calculate_slice_indices(
        self, 
        num_slices: int, 
        n_images: int, 
        sampling_type: str
    ) -> List[Union[int, None]]:
        """
        Calculate which slice indices to sample from a DICOM series.
        
        Args:
            num_slices: Total number of slices in the series.
            n_images: Number of slices to sample.
            sampling_type: Sampling strategy ("random" or "equidistant").
            
        Returns:
            List of slice indices to load. None values indicate missing slices.
        """
        # Calculate offset to avoid edge slices
        offset_fraction = num_slices // n_images
        offset = min(offset_fraction // 4, 2) * n_images
        
        if n_images <= num_slices:
            if sampling_type == "random":
                # Randomly sample slices avoiding edges
                start_range = max(0, offset)
                end_range = max(num_slices - offset, n_images)
                slice_indices = random.sample(range(start_range, end_range), n_images)
            else:
                # Equidistant sampling
                start_idx = offset
                end_idx = num_slices - 1 - offset
                slice_indices = [int(x) for x in np.linspace(start_idx, end_idx, n_images)]
        else:
            # More slices requested than available - use all and pad with None
            slice_indices = list(range(num_slices)) + [None] * (n_images - num_slices)
        
        return slice_indices

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
        images = self.open_dicom_slice_from_series(self.path_list[idx], n_images=self.n_slices, sampling_type="equidistant")
        
        # Load encoded metadata
        try:
            metadata = self.metadata_df.loc[self.path_list[idx]].to_numpy()
            metadata = torch.tensor(metadata, dtype=torch.float32)
        except KeyError:
            logger.warning(f"Metadata not found for {self.path_list[idx]}, using zeros")
            # Use zero metadata if not found
            metadata = torch.zeros(88, dtype=torch.float32)  # Adjust size as needed
        
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

def plot_samples(images: torch.Tensor, save_path: Optional[str] = None) -> None:
    """
    Plot a batch of images using matplotlib.
    
    Args:
        images: Tensor of shape (B, n_slices, 1, H, W)
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt

    batch_size, n_slices, _, H, W = images.shape
    
    # Create a grid: batch_size rows x n_slices columns
    fig, axes = plt.subplots(batch_size, n_slices, figsize=(n_slices * 3, batch_size * 3))
    
    # Handle single row case
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    # Handle single column case
    if n_slices == 1:
        axes = axes.reshape(-1, 1)
    
    for b in range(batch_size):
        for s in range(n_slices):
            ax = axes[b, s]
            ax.imshow(images[b, s, 0].cpu().numpy(), cmap='gray')
            ax.axis('off')
            if b == 0:  # Add slice labels to first row
                slice_label = "MIP" if s == n_slices - 1 else f"Slice {s+1}"
                ax.set_title(slice_label)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    plt.show()

if __name__ == "__main__":
    """
    Example usage and testing of the LiverDataset.
    
    This script demonstrates how to create a dataset, get label information,
    and iterate through batches to inspect data shapes and content.
    """

    print("LiverDataset Demo")
    print("=" * 50)
    
    # Create a dataset instance
    dataset = LiverDataset(num_samples=200, is_infer=True, n_slices=3)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Display label information
    n_labels = dataset.get_n_labels()
    print(f"Label dimensions: {n_labels}")
    print(f"Dataset size: {len(dataset)} samples")
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

    # Plot a sample batch of images
    sample_images, _, _ = next(iter(dataloader))
    plot_samples(sample_images, save_path="sample_images.png")
