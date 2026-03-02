"""
Local Liver Dataset for Multi-Modal Medical Image Classification.

This module implements a PyTorch Dataset for loading and processing liver MRI/CT
medical images and associated DICOM metadata from a local filesystem. It is a key
component of the IMC (Image-Metadata-Classifier) project, providing a flexible data
pipeline for multi-modal deep learning.

The dataset supports:
- Multi-slice sampling from 3D DICOM series using various strategies.
- Loading and encoding of extensive DICOM metadata features.
- Aggregated and per-slice metadata handling.
- Configurable data augmentation pipelines for medical images.
- Train, validation, test, and inference modes with appropriate data splits.
- Graceful handling of missing data, corrupted files, and inconsistent metadata.

This dataloader is optimized for local development and can be swapped with
`liver_dataloader_gcp.py` for cloud-based training on Google Cloud Storage.

Authors: Melanie Dohmen, Matthias Lenga, Tuan Truong
Date: 2026
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
from IMC.data.constants import DUKE_ORIGINAL_LABEL_NAMES, DEFAULT_LABEL_NAMES, SELECTED_FEATURES

logger = logging.getLogger('IMC')

# Configuration constants
LOCAL_DATASET_PATH = os.getenv("LOCAL_DATASET_PATH", None)
logger.info(f"LOCAL_DATASET_PATH: {LOCAL_DATASET_PATH}")
METADATA_PATH = os.getenv("METADATA_PATH", None)
logger.info(f"METADATA_PATH: {METADATA_PATH}")
LABEL_CSV_PATH = os.getenv("LABEL_CSV_PATH", None)
logger.info(f"LABEL_CSV_PATH: {LABEL_CSV_PATH}")


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
        local_dataset_path: Path to the local dataset root directory.
        metadata_path: Path to the metadata CSV or Parquet file.
        label_csv_path: Path to the labels CSV file.
        aggregated_metadata: If True, uses aggregated metadata per series.
        use_preselected_features: If True, uses a predefined subset of metadata features.
        exclude_contrast_yn: If True, excludes 'label_Contrast' from classification labels.
        sampling_type: Slice sampling strategy – ``"equidistant"`` (default) or
            ``"random"``.  Passed to :meth:`open_dicom_slice_from_series`.

    The ``__getitem__`` return value depends on ``is_infer``:

    * **Training** (``is_infer=False``):  
      ``(images, metadata, targets, masks)`` where *images* has shape
      ``(n_slices, H, W)`` and *targets* / *masks* are tuples of tensors,
      one per classification task.
    * **Inference** (``is_infer=True``):  
      ``(images, metadata, filepath)``.

    Example::

        >>> dataset = LiverDataset(
        ...     num_samples=1000,
        ...     n_slices=3,
        ...     split=["fold_0", "fold_1", "fold_2"],
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
        aggregated_metadata: bool = False,
        use_preselected_features: bool = False,
        exclude_contrast_yn: bool = False,
        sampling_type: str = "equidistant",
    ) -> None:
        """Initialize the LiverDataset with specified configuration.

        Args:
            num_samples: Maximum number of samples to load. If None, loads all.
            n_slices: Number of slices to sample from each DICOM series.
            img_size: Target image size for resizing.
            label_names: Dictionary mapping label categories to their possible values.
            augment_conf: Augmentation configuration string (e.g., "NONE2D").
            split: List of fold names to include (e.g., ["fold_0", "fold_1"]).
            is_infer: If True, returns only images and metadata (no labels).
            local_dataset_path: Path to the local dataset root directory.
            metadata_path: Path to the metadata CSV or Parquet file.
            label_csv_path: Path to the labels CSV file.
            aggregated_metadata: If True, uses aggregated metadata per series.
            use_preselected_features: If True, uses a predefined subset of metadata features.
            exclude_contrast_yn: If True, excludes 'label_Contrast' from classification labels.
            sampling_type: Slice sampling strategy – ``"equidistant"`` (default)
                or ``"random"``.
        """
        # Store configuration parameters
        self.num_samples = num_samples
        self.n_slices = n_slices
        self.img_size = img_size            
        self.label_names = label_names.copy() if label_names is not None else DEFAULT_LABEL_NAMES.copy()
        if exclude_contrast_yn:
            self.label_names.pop("label_Contrast", None)
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
        self.aggregated_metadata = aggregated_metadata
        logger.info("Using aggregated metadata: {}".format(self.aggregated_metadata))
        self.use_preselected_features = use_preselected_features
        self.sampling_type = sampling_type

        # Load and process metadata
        self._load_metadata_and_labels(split)
        self.num_metadata_features = len(SELECTED_FEATURES) if use_preselected_features else len(self.metadata_df.columns)
        
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
            if self.metadata_path and os.path.exists(self.metadata_path) and self.metadata_path.endswith(".parquet"):
                metadata_df = pd.read_parquet(self.metadata_path)
                logger.info(f"Loaded {len(metadata_df)} samples from metadata Parquet {self.metadata_path}")
            else:
                # Fallback to legacy DICOM tag encoding if Parquet metadata is not found
                logger.warning("Metadata Parquet file not found. Falling back to legacy DICOM encoding.")
                metadata_df = encode_dicom_tags_by_version(labels_df, dicom_encoding_version="brain")
            metadata_df = metadata_df.set_index("Filepath")
            labels_df = labels_df.set_index("Filepath")

            if self.use_preselected_features:
                logger.info("Using preselected features for metadata")
                metadata_df = metadata_df[SELECTED_FEATURES]
            
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

    def open_dicom_slice_from_series(
        self,
        path_dicom_folder: str,
        sampling_type: str = "equidistant",
        n_images: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process multiple slices from a DICOM series.

        This method samples slices from a DICOM series according to the specified
        sampling strategy, applies augmentation, and returns them as a torch tensor.

        Args:
            path_dicom_folder: Path to the DICOM series folder.
            sampling_type: Slice sampling strategy:
                - "random": Randomly sample n_images slices
                - "equidistant": Sample slices evenly spaced across the series
            n_images: Number of slices to sample from the series.

        Returns:
            Tuple containing:
                - images: A tensor of shape (n_images, 1, H, W) with the processed image slices.
                - metadata: A tensor of shape (n_images, D) with the corresponding metadata.

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
            arr = image.copy()
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                # Already 3-channel (C,H,W) from IMAGENET299_CENTER
                images.append(torch.tensor(arr, dtype=torch.float32))
            else:
                images.append(torch.tensor(arr, dtype=torch.float32)) # (H, W)

        # Load metadata (individual dicom files or aggregated)
        if self.aggregated_metadata:
            if self.metadata_df.index.to_list()[0].startswith("/home/tuan.truong"):
                metadata = self.metadata_df.loc[str(Path(self.local_dataset_path) / path_dicom_folder)].to_numpy()
            else:
                metadata = self.metadata_df.loc[path_dicom_folder].to_numpy()
            metadata = torch.tensor(metadata, dtype=torch.float32) # (D,)
            metadata = metadata.unsqueeze(0).repeat(len(images), 1)  # (N, D)
        else:
            metadata = []
            for slice_idx in slice_indices:
                if slice_idx is None:
                    m = torch.full((self.num_metadata_features,), torch.nan, dtype=torch.float32) if not self.use_preselected_features else torch.full((len(SELECTED_FEATURES),), torch.nan, dtype=torch.float32)
                    metadata.append(m)
                    continue
                try:
                    m = self.metadata_df.loc[str(slice_filenames[slice_idx])].to_numpy()
                    m = torch.tensor(m, dtype=torch.float32)
                except:
                    logger.error(f"Metadata not found for {slice_filenames[slice_idx]}, using NaNs")
                    m = torch.full((self.num_metadata_features,), torch.nan, dtype=torch.float32) if not self.use_preselected_features else torch.full((len(SELECTED_FEATURES),), torch.nan, dtype=torch.float32)
                metadata.append(m)
        # Stack all slices into a single tensor
        return torch.stack(images, dim=0), torch.stack(metadata, dim=0)  # (N, H, W), (N, D)

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
        Tuple[torch.Tensor, torch.Tensor, str],
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
                - filepath: Path to the DICOM series folder (inference only)

        Raises:
            IndexError: If idx is out of range.
            Exception: If required labels are missing.
        """
        if idx >= len(self.path_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.path_list)}")

        # Load multi-slice images
        images, metadata = self.open_dicom_slice_from_series(
            self.path_list[idx],
            sampling_type=self.sampling_type,
            n_images=self.n_slices
        )
    
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
            torch.tensor(
            label_indices[label_class] != self.label_names[label_class].index("na")
            if "na" in self.label_names[label_class]
            else True
            )
            for label_class in self.label_names.keys()
        )
        return targets, masks


def get_train_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    num_samples: Optional[int] = 100,
    folder_split: List[str] = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6", "fold_7"],
    **ds_kwargs: Dict,
) -> DataLoader:
    """
    Create a DataLoader for training data.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data.
        num_samples: Maximum number of samples to load (None for all).
        folder_split: List of fold names to include in training set.
        ds_kwargs: additional keywords to the dataset
        
    Returns:
        Configured DataLoader for training.
    """
    dataset = LiverDataset(
        split=folder_split, 
        num_samples=num_samples, 
        augment_conf="NONE2D",
        **ds_kwargs
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
    **ds_kwargs: Dict,
) -> DataLoader:
    """
    Create a DataLoader for validation data.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data (typically False for validation).
        num_samples: Maximum number of samples to load (None for all).
        folder_split: List of fold names to include in validation set.
        ds_kwargs: additional keywords to the dataset
    Returns:
        Configured DataLoader for validation.
    """
    dataset = LiverDataset(
        split=folder_split, 
        num_samples=num_samples, 
        augment_conf="NONE2D",
        **ds_kwargs
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
    **ds_kwargs: Dict,
) -> DataLoader:
    """
    Create a DataLoader for test data.
    
    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data (typically False for testing).
        num_samples: Maximum number of samples to load (None for all).
        folder_split: List of fold names to include in test set.
        ds_kwargs: additional keywords to the dataset
    Returns:
        Configured DataLoader for testing.
    """
    dataset = LiverDataset(
        split=folder_split, 
        num_samples=num_samples, 
        augment_conf="NONE2D",
        **ds_kwargs
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
    **ds_kwargs: Dict,
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
        label_names=label_names,
        **ds_kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # Example usage and testing of the LiverDataset.
    # Demonstrates how to create a dataset, inspect label info, and iterate batches.
    print("LiverDataset Demo")
    print("=" * 50)
    
    # Create a dataset instance
    dataset = LiverDataset(num_samples=200, n_slices=5, is_infer=False, aggregated_metadata=False, use_preselected_features=True, label_names=DUKE_ORIGINAL_LABEL_NAMES, sampling_type="random", augment_conf="DEFAULT2D")
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
