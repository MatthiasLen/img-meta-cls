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

Authors: Tuan Truong
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
from IMC.data.constants import DUKE_ORIGINAL_LABEL_NAMES

logger = logging.getLogger('IMC')
# Environment variables should be set by the training script
LOCAL_DATASET_PATH = os.getenv("LOCAL_DATASET_PATH")
LABEL_CSV_PATH = os.getenv("LABEL_CSV_PATH")

class DukeLiverDataset3D(Dataset):
    """PyTorch Dataset for loading 3D liver MRI volumes from the Duke dataset.

    Assembles full 3D volumes from DICOM slices, applies volumetric augmentation,
    and prepares data for 3D CNN architectures such as the Pyramid Pooling Network.
    This is an image-only dataset – metadata is not used.

    Dataset paths are read from the environment variables ``LOCAL_DATASET_PATH``
    and ``LABEL_CSV_PATH``, which must be set before instantiation (e.g. via
    :func:`~IMC.net4_duke.train._configure_dataset_env`).

    Args:
        target_depth: Target number of slices for every output volume.  Volumes
            with more slices are sub-sampled equidistantly; volumes with fewer
            slices are zero-padded.
        img_size: Target spatial size ``(img_size × img_size)`` for each slice
            after resampling.
        label_names: Mapping of task name → list of class strings.  Defaults to
            ``DUKE_ORIGINAL_LABEL_NAMES`` (single task, 13 classes A–M).
        augment_conf: 3-D augmentation configuration key passed to
            :func:`~IMC.data.augment.augment3d` (e.g. ``"NONE3D"``,
            ``"DEFAULT3D"``).
        split: Fold names to include (e.g. ``["fold_0", "fold_1"]``).  ``None``
            loads the entire dataset without fold filtering.
        is_infer: When ``True``, ``__getitem__`` returns
            ``(volume_tensor, filepath)`` instead of
            ``(volume_tensor, metadata, targets, masks)``.
        num_samples: Maximum number of samples to keep.  ``None`` keeps all.

    Raises:
        ValueError: If ``LOCAL_DATASET_PATH`` or ``LABEL_CSV_PATH`` are unset.
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
        """Initialise the DukeLiverDataset3D.

        Args:
            target_depth: Target volume depth (number of slices).  Volumes
                with more slices are sub-sampled equidistantly; shorter volumes
                are zero-padded symmetrically.
            img_size: Target spatial resolution for each slice (``img_size ×
                img_size`` pixels).
            label_names: Task → class-list mapping.  Defaults to
                ``DUKE_ORIGINAL_LABEL_NAMES``.
            augment_conf: 3-D augmentation key passed to
                :func:`~IMC.data.augment.augment3d`.
            split: Fold names to include.  ``None`` loads all folds.
            is_infer: Return ``(volume, filepath)`` tuples instead of labelled
                tuples.
            num_samples: Cap on the number of samples.  ``None`` keeps all.

        Raises:
            ValueError: If ``LOCAL_DATASET_PATH`` or ``LABEL_CSV_PATH``
                environment variables are not set.
        """
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
        """Load labels from the label CSV and optionally filter by fold split.

        Populates ``self.path_list`` and ``self.labels`` with the surviving
        rows.  When ``num_samples`` is set the lists are truncated after
        filtering.

        Args:
            split: Fold names to keep (matched against the ``"split"`` column).
                Pass ``None`` to keep all rows.

        Raises:
            FileNotFoundError: If ``LABEL_CSV_PATH`` does not exist.
            ValueError: If no samples remain after applying the fold filter.
            RuntimeError: For any other error encountered while reading the CSV.
        """
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
        """Return the number of samples in the dataset."""
        return len(self.path_list)

    def get_n_labels(self) -> Dict[str, int]:
        """Return the number of classes for each classification task.

        Returns:
            Mapping of task name → number of classes, e.g.
            ``{"SequenceType_Code_norm": 13}``.
        """
        return {label_name: len(classes) for label_name, classes in self.label_names.items()}

    def _resample_volume_depth(self, volume: np.ndarray) -> np.ndarray:
        """Resample volume depth to ``self.target_depth``.

        Three cases are handled:

        * **Equal** – volume is returned unchanged.
        * **Too deep** – ``target_depth`` frames are selected by equidistant
          sub-sampling (``np.linspace`` indices).
        * **Too shallow** – the volume is zero-padded symmetrically along the
          depth axis.

        Args:
            volume: Float32 array of shape ``(D, H, W)`` (or ``(D, H, W, C)``
                after augmentation).

        Returns:
            Array with depth dimension equal to ``self.target_depth``.
        """
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
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]
    ]:
        """Load and return a single volumetric sample.

        Processing pipeline for each sample:

        1. Collect all ``*.dicom`` files in the series directory and stack them
           into a ``(D_native, H_native, W_native)`` float32 array.
        2. Apply 3-D augmentation (:func:`~IMC.data.augment.augment3d`).
        3. Resample spatial dimensions to ``img_size × img_size`` via
           SimpleITK linear interpolation.
        4. Resample / pad depth to ``target_depth`` with
           :meth:`_resample_volume_depth`.
        5. Add a channel dimension → ``(1, target_depth, img_size, img_size)``.

        Args:
            idx: Sample index in ``[0, len(dataset))``.

        Returns:
            **Inference mode** (``is_infer=True``):
                ``(volume_tensor, filepath)`` where *volume_tensor* has shape
                ``(1, target_depth, img_size, img_size)`` and *filepath* is the
                absolute path string to the series directory.

            **Training mode** (``is_infer=False``):
                ``(volume_tensor, metadata_tensor, targets, masks)`` where
                *metadata_tensor* is a placeholder zero tensor ``(1,)``,
                *targets* is a length-1 tuple containing the class index, and
                *masks* is a length-1 tuple containing ``True``.

        Raises:
            IndexError: If ``idx >= len(self.path_list)``.

        Note:
            On error, a dummy all-zero volume is returned together with a
            target of ``-1`` and mask ``False`` so that a single bad sample
            does not crash the training loop.
        """
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

    if not os.environ.get("LOCAL_DATASET_PATH") or not os.environ.get("LABEL_CSV_PATH"):
        raise RuntimeError(
            "Set LOCAL_DATASET_PATH and LABEL_CSV_PATH before running the DukeLiverDataset3D demo."
        )

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
