"""
Duke 3D Liver Dataset for Volumetric Medical Image Classification.

This module implements a PyTorch Dataset for loading and processing 3D liver MRI
volumes from the Duke dataset. It's designed for volumetric deep learning models
that operate on full 3D image data rather than 2D slices.

The dataset supports:
- Assembling 3D volumes from sequences of 2D DICOM slices.
- Volumetric data augmentation in 3D (rotations, scaling, elastic deformation).
- Resampling volumes to a fixed target depth for consistent batching.
- Per-volume intensity normalization.
- An image-only pipeline, excluding metadata as per the 3D Pyramid Pooling Network paper.

This dataloader is a key component for training 3D CNN architectures like network07.

Authors: Claude Code
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
import SimpleITK as sitk
from natsort import natsorted
from pydicom import dcmread
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from IMC.data.augment import augment3d

logger = logging.getLogger('IMC')

# Environment variables should be set by the training script
LOCAL_DATASET_PATH = os.getenv("LOCAL_DATASET_PATH")
LABEL_CSV_PATH = os.getenv("LABEL_CSV_PATH")

DUKE_ORIGINAL_LABEL_NAMES = {
    "SequenceType_Code_norm": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
}


class DukeLiverDataset3D(Dataset):
    """
    PyTorch Dataset for loading 3D liver MRI volumes from the Duke dataset.

    This dataset assembles 3D volumes from DICOM slices, applies 3D augmentation,
    and prepares the data for volumetric classification models. It is an image-only
    dataset.

    Args:
        target_depth: The target depth to resample/pad/crop all volumes to.
        img_size: The target spatial size (H, W) for each slice in the volume.
        label_names: Dictionary mapping label categories to their possible values.
        augment_conf: 3D augmentation configuration string (e.g., "DEFAULT3D").
        split: List of fold names to include (e.g., ["fold_0", "fold_1"]).
        is_infer: If True, returns only images and filepaths (no labels).
        num_samples: Maximum number of samples to load. If None, loads all.
    """

    def __init__(
        self,
        target_depth: int = 64,
        img_size: int = 256,
        label_names: Optional[Dict[str, List[str]]] = None,
        augment_conf: str = "NONE3D",
        split: Optional[List[str]] = None,
        is_infer: bool = False,
        num_samples: Optional[int] = None,
    ):
        self.target_depth = target_depth
        self.img_size = img_size
        self.label_names = label_names or DUKE_ORIGINAL_LABEL_NAMES.copy()
        self.augment_conf = augment_conf
        self.is_infer = is_infer
        self.num_samples = num_samples

        self.path_list: List[str] = []
        self.labels: List[Dict[str, Any]] = []

        if not LOCAL_DATASET_PATH or not LABEL_CSV_PATH:
            raise ValueError("LOCAL_DATASET_PATH and LABEL_CSV_PATH environment variables must be set.")

        self.local_dataset_path = Path(LOCAL_DATASET_PATH)
        self.label_csv_path = Path(LABEL_CSV_PATH)

        self._load_labels(split)

        logger.info(f"DukeLiverDataset3D initialized with {len(self.path_list)} samples.")
        logger.info(f"Target depth: {self.target_depth}, Image size: {self.img_size}")

    def _load_labels(self, split: Optional[List[str]]) -> None:
        """Load labels from CSV and filter by split."""
        try:
            labels_df = pd.read_csv(self.label_csv_path)
            logger.info(f"Loaded {len(labels_df)} samples from label CSV: {self.label_csv_path}")

            if split:
                labels_df = labels_df[labels_df["split"].isin(split)]
                logger.info(f"Samples after filtering for split {split}: {len(labels_df)}")

            if len(labels_df) == 0:
                raise ValueError(f"No samples found for splits: {split}")

            for _, row in labels_df.iterrows():
                self.path_list.append(row["Filepath"])
                self.labels.append(row.to_dict())

            if self.num_samples is not None:
                self.path_list = self.path_list[:self.num_samples]
                self.labels = self.labels[:self.num_samples]

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required CSV file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset labels: {e}")

    def __len__(self) -> int:
        return len(self.path_list)

    def get_n_labels(self) -> Dict[str, int]:
        """Get the number of classes for each label category."""
        return {label_name: len(classes) for label_name, classes in self.label_names.items()}

    def _resample_volume_depth(self, volume: np.ndarray) -> np.ndarray:
        """Resample, pad, or crop the volume to the target depth."""
        native_depth = volume.shape[0]

        if native_depth == self.target_depth:
            return volume

        if native_depth > self.target_depth:
            # Equidistant subsampling
            indices = np.linspace(0, native_depth - 1, self.target_depth, dtype=int)
            return volume[indices]
        else:
            # Padding with zeros
            pad_depth = self.target_depth - native_depth
            pad_before = pad_depth // 2
            pad_after = pad_depth - pad_before

            padding = [(pad_before, pad_after), (0, 0), (0, 0)]
            if volume.ndim == 4: # for channels
                padding.append((0,0))

            return np.pad(volume, padding, mode='constant', constant_values=0)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, str],
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """Get a single 3D sample from the dataset."""
        if idx >= len(self.path_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.path_list)}")

        series_path = self.local_dataset_path / self.path_list[idx]

        try:
            # 1. Load all DICOM slices and stack into a volume
            slice_files = natsorted([p for p in series_path.rglob("*.dicom")])
            if not slice_files:
                raise IOError(f"No DICOM files found in {series_path}")

            slices = [dcmread(sf).pixel_array for sf in slice_files]
            volume = np.stack(slices, axis=0).astype(np.float32) # (D_native, H_native, W_native)

            # 2. Skip per-volume z-score normalization; handled later in trainer

            # 3. Apply 3D augmentation
            volume = augment3d(volume, conf=self.augment_conf)
            # Ensure float32 after augmentation
            volume = volume.astype(np.float32)

            # 4. Resize spatially to img_size x img_size
            # Using SimpleITK for robust medical image resizing
            sitk_image = sitk.GetImageFromArray(volume)
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkLinear)
            orig_size = sitk_image.GetSize()
            orig_spacing = sitk_image.GetSpacing()
            new_spacing = [(orig_size[i] * orig_spacing[i]) / self.img_size for i in range(2)] + [orig_spacing[2]]
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputOrigin(orig_spacing)
            resampler.SetSize([self.img_size, self.img_size, orig_size[2]])

            resized_sitk_image = resampler.Execute(sitk_image)
            volume = sitk.GetArrayFromImage(resized_sitk_image)
            # Ensure float32 after resizing
            volume = volume.astype(np.float32)

            # 5. Resample depth to target_depth
            volume = self._resample_volume_depth(volume)

            # 6. Add channel dimension -> (1, D, H, W) and enforce float32
            volume_tensor = torch.from_numpy(volume.copy()).unsqueeze(0).to(dtype=torch.float32)

            if self.is_infer:
                return volume_tensor, str(series_path)

            # Process labels
            label_info = self.labels[idx]
            label_class = self.label_names["SequenceType_Code_norm"]
            label_value = label_info["SequenceType_Code_norm"]

            if label_value not in label_class:
                raise ValueError(f"Unknown label value '{label_value}' for SequenceType_Code_norm")

            target_idx = label_class.index(label_value)
            target_tensor = torch.tensor(target_idx, dtype=torch.long)

            # Image-only pipeline: provide placeholder metadata tensor and mask
            metadata_tensor = torch.zeros((1,), dtype=torch.float32)
            mask_tensor = torch.tensor(True)

            # Trainer expects (images, metadata, targets, masks)
            return volume_tensor, metadata_tensor, (target_tensor,), (mask_tensor,)

        except Exception as e:
            logger.error(f"Error processing sample {idx} ({series_path}): {e}")
            # Return a dummy sample to avoid crashing the training loop
            dummy_volume = torch.zeros((1, self.target_depth, self.img_size, self.img_size), dtype=torch.float32)
            if self.is_infer:
                return dummy_volume, "error"
            dummy_target = torch.tensor(-1, dtype=torch.long)
            dummy_metadata = torch.zeros((1,), dtype=torch.float32)
            dummy_mask = torch.tensor(False)
            return dummy_volume, dummy_metadata, (dummy_target,), (dummy_mask,)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("DukeLiverDataset3D Demo")

    # Set dummy env vars for testing
    os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
    os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"

    # Create dataset instance
    dataset = DukeLiverDataset3D(
        target_depth=64,
        img_size=128,
        split=["fold_0"],
        augment_conf="DEFAULT3D",
        num_samples=4,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Label dimensions: {dataset.get_n_labels()}")

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Process a few batches
    for batch_idx, (images_3d, targets) in enumerate(dataloader):
        logger.info(f"Batch {batch_idx + 1}:")
        logger.info(f"  Images shape: {images_3d.shape}")  # (B, 1, D, H, W)
        logger.info(f"  Targets shape: {targets.shape}")
        logger.info(f"  Targets: {targets.tolist()}")

        if batch_idx >= 1:
            break

    logger.info("Demo completed successfully!")
