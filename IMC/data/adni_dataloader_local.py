"""
Local ADNI Dataset for Multi-Modal Medical Image Classification.

This module implements a PyTorch Dataset for loading and processing ADNI brain MRI
images and associated DICOM metadata from a local filesystem. It is part of the
IMC (Image-Metadata-Classifier) project and mirrors the structure of
`liver_dataloader_local.py`, adapted for the ADNI brain imaging dataset.

The dataset supports three classification tasks:
    - label_AcquisitionPlane  : AX, COR, SAG, na
    - label_SequenceContrast  : ASL, CAL, DWI, OTHER, PD, T1, T2, T2FLAIR, na
    - label_Localizer         : yes, no, na

Dataset split convention (5-fold, merged from the original 10-fold):
    - fold_0 .. fold_2 → training
    - fold_3           → validation
    - fold_4           → test

The loader strips the GCS prefix and resolves the remainder against LOCAL_DATASET_PATH.

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
from natsort import natsorted
from pydicom import dcmread
from torch.utils.data import DataLoader, Dataset

from IMC.data.augment import augment
from IMC.data.dicom_tag_encoding import encode_dicom_tags_by_version

logger = logging.getLogger("IMC")

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Environment-variable based configuration
# ---------------------------------------------------------------------------
ADNI_LOCAL_DATASET_PATH = os.getenv("ADNI_LOCAL_DATASET_PATH", None)
logger.info(f"ADNI_LOCAL_DATASET_PATH: {ADNI_LOCAL_DATASET_PATH}")
ADNI_METADATA_PATH = os.getenv("ADNI_METADATA_PATH", None)
logger.info(f"ADNI_METADATA_PATH: {ADNI_METADATA_PATH}")
ADNI_LABEL_CSV_PATH = os.getenv(
    "ADNI_LABEL_CSV_PATH",
    "/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv",
)
logger.info(f"ADNI_LABEL_CSV_PATH: {ADNI_LABEL_CSV_PATH}")

# GCS prefix that appears in the label CSV Filepath column

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------
ADNI_LABEL_NAMES: Dict[str, List[str]] = {
    "label_AcquisitionPlane": ["AX", "COR", "SAG", "na"],
    "label_SequenceContrast": ["ASL", "CAL", "DWI", "OTHER", "PD", "T1", "T2", "T2FLAIR", "na"],
    "label_Localizer": ["yes", "no", "na"],
}


class ADNIDataset(Dataset):
    """
    PyTorch Dataset for loading ADNI brain MRI data from a local filesystem.

    This dataset handles multi-slice DICOM images with associated DICOM metadata
    and multi-task classification labels. It closely mirrors ``LiverDataset`` in
    structure but is tailored to the ADNI brain imaging dataset.

    Classification tasks
    --------------------
    label_AcquisitionPlane  : AX | COR | SAG | na
    label_SequenceContrast  : ASL | CAL | DWI | OTHER | PD | T1 | T2 | T2FLAIR | na
    label_Localizer         : yes | no | na

    Args:
        num_samples: Maximum number of samples to load. ``None`` loads all.
        n_slices: Number of slices to sample from each DICOM series.
        img_size: Target image size (H = W = img_size).
        label_names: Mapping of label categories → possible values.
            Defaults to :data:`ADNI_LABEL_NAMES`.
        augment_conf: Augmentation config string passed to :func:`augment`.
        split: List of fold names to include (e.g. ``["fold_0", "fold_1"]``).
            ``None`` includes all rows irrespective of split.
        is_infer: If ``True`` returns ``(images, metadata, filepath)`` without labels.
        local_dataset_path: Root of the local ADNI DICOM tree. Falls back to the
            ``ADNI_LOCAL_DATASET_PATH`` env-var.
        metadata_path: Path to a pre-encoded metadata ``.parquet`` file. Falls back
            to ``ADNI_METADATA_PATH`` env-var. When absent, metadata is encoded
            on-the-fly via :func:`encode_dicom_tags_by_version`.
        label_csv_path: Path to the label CSV file. Falls back to
            ``ADNI_LABEL_CSV_PATH`` env-var.
        aggregated_metadata: If ``True`` metadata is keyed by series folder path
            rather than individual slice path.
        use_preselected_features: If ``True`` restrict metadata to a pre-defined
            feature subset (not yet defined for ADNI; currently a no-op).

    Returns:
        Training mode  : ``(images, metadata, targets, masks)``
        Inference mode : ``(images, metadata, filepath)``

        Shapes::

            images   : (n_slices, H, W)
            metadata : (n_slices, D)
            targets  : tuple of scalar tensors, one per task
            masks    : tuple of bool tensors (valid = not "na")

    Example::

        >>> ds = ADNIDataset(num_samples=None, n_slices=3, split=["fold_0"])
        >>> dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    """

    def __init__(
        self,
        num_samples: Optional[int] = 100,
        n_slices: int = 3,
        img_size: int = 224,
        label_names: Optional[Dict[str, List[str]]] = None,
        augment_conf: str = "NONE2D",
        split: Optional[List[str]] = None,
        is_infer: bool = False,
        local_dataset_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        label_csv_path: Optional[str] = None,
        aggregated_metadata: bool = False,
        use_preselected_features: bool = False,
        sampling_type: str = "equidistant",
    ) -> None:
        self.num_samples = num_samples
        self.n_slices = n_slices
        self.img_size = img_size
        self.label_names = (
            label_names.copy() if label_names is not None else ADNI_LABEL_NAMES.copy()
        )
        self.augment_conf = augment_conf
        self.is_infer = is_infer
        self.aggregated_metadata = aggregated_metadata
        self.use_preselected_features = use_preselected_features
        self.sampling_type = sampling_type

        # Data containers
        self.labels: List[Dict[str, Any]] = []
        self.path_list: List[str] = []

        # Resolve paths
        self.local_dataset_path = (
            local_dataset_path if local_dataset_path is not None else ADNI_LOCAL_DATASET_PATH
        )
        self.metadata_path = (
            metadata_path if metadata_path is not None else ADNI_METADATA_PATH
        )
        self.label_csv_path = (
            label_csv_path if label_csv_path is not None else ADNI_LABEL_CSV_PATH
        )

        self._load_metadata_and_labels(split)
        self.num_metadata_features = len(self.metadata_df.columns)
        logger.info(
            f"ADNIDataset ready: {len(self.path_list)} series, "
            f"{self.num_metadata_features} metadata features"
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_metadata_and_labels(self, split: Optional[List[str]]) -> None:
        """Load and filter the label CSV and metadata, populating internal lists."""
        try:
            labels_df = pd.read_csv(self.label_csv_path)

            # Keep only ADNI rows
            if "dataset" in labels_df.columns:
                labels_df = labels_df[labels_df["dataset"] == "ADNI"]
                logger.info(f"ADNI rows in label CSV: {len(labels_df)}")

            # Filter by split
            if split is not None:
                labels_df = labels_df[labels_df["split"].isin(split)]
                logger.info(f"Samples after split filter {split}: {len(labels_df)}")

            if len(labels_df) == 0:
                raise ValueError(f"No ADNI samples found for splits: {split}")

            # Convert GCS paths → local / relative paths
            labels_df = labels_df.set_index("Filepath")

            # ---- Metadata ----
            if (
                self.metadata_path
                and os.path.exists(self.metadata_path)
                and self.metadata_path.endswith(".parquet")
            ):
                metadata_df = pd.read_parquet(self.metadata_path)
                logger.info(
                    f"Loaded pre-encoded metadata from {self.metadata_path} "
                    f"({len(metadata_df)} rows)"
                )
            else:
                logger.warning(
                    "No valid ADNI metadata parquet found; encoding DICOM tags on-the-fly."
                )
                # Use the original label CSV (with Filepath still intact) for encoding
                src_df = pd.read_csv(self.label_csv_path)
                if "dataset" in src_df.columns:
                    src_df = src_df[src_df["dataset"] == "ADNI"]
                if split is not None:
                    src_df = src_df[src_df["split"].isin(split)]
                metadata_df = encode_dicom_tags_by_version(src_df, dicom_encoding_version="brain")

            # Align metadata index to local paths
            if "Filepath" in metadata_df.columns:
                metadata_df = metadata_df.set_index("Filepath")
            else:
                # Assume index already holds local paths
                pass

            # Populate path list and labels
            for local_path, row in labels_df.iterrows():
                self.path_list.append(local_path)
                self.labels.append(row.to_dict())

            self.metadata_df = metadata_df

            if self.num_samples is None:
                self.num_samples = len(self.path_list)
            else:
                self.num_samples = min(self.num_samples, len(self.path_list))

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required file not found: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading ADNI dataset: {e}") from e

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def get_n_labels(self) -> Dict[str, int]:
        """Return the number of classes for each classification task."""
        return {name: len(classes) for name, classes in self.label_names.items()}

    # ------------------------------------------------------------------
    # DICOM loading helpers
    # ------------------------------------------------------------------

    def _get_dicom_files(self, series_folder: str) -> List[Path]:
        """Return a naturally-sorted list of DICOM files in *series_folder*."""
        if not series_folder.startswith(self.local_dataset_path):
            folder = Path(os.path.join(self.local_dataset_path, series_folder))
        else:
            folder = Path(series_folder)
        if not folder.exists():
            raise RuntimeError(f"DICOM folder not found: {folder}")
        files = list(folder.rglob("*.dcm")) + list(folder.rglob("*.dicom"))
        if not files:
            # Some ADNI series have no extension
            files = [p for p in folder.rglob("*") if p.is_file()]
        if not files:
            raise RuntimeError(f"No DICOM files found in: {folder}")
        return natsorted(files)

    def _calculate_slice_indices(
        self, num_slices: int, n_images: int, sampling_type: str
    ) -> List[Optional[int]]:
        """Sample *n_images* slice indices from a series with *num_slices* slices."""
        offset_fraction = num_slices // max(n_images, 1)
        offset = min(offset_fraction // 4, 2) * n_images

        if n_images <= num_slices:
            if sampling_type == "random":
                start = max(0, offset)
                end = max(num_slices - offset, n_images)
                indices = random.sample(range(start, end), n_images)
            else:  # equidistant
                start = offset
                end = num_slices - 1 - offset
                indices = [int(x) for x in np.linspace(start, end, n_images)]
        else:
            indices = list(range(num_slices)) + [None] * (n_images - num_slices)

        return indices

    def _load_slices(
        self, series_folder: str, sampling_type: str = "equidistant"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load *n_slices* from *series_folder* and return ``(images, metadata)``.

        Returns:
            images   : (n_slices, H, W) float32 tensor
            metadata : (n_slices, D)   float32 tensor
        """
        dicom_files = self._get_dicom_files(series_folder)
        num_slices = len(dicom_files)
        indices = self._calculate_slice_indices(num_slices, self.n_slices, sampling_type)

        images: List[torch.Tensor] = []
        for idx in indices:
            if idx is None:
                img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            else:
                ds = dcmread(dicom_files[idx])
                try:
                    img = ds.pixel_array.astype(np.float32)
                    if img.ndim > 2:
                        logger.warning(
                            f"Multi-dim array in {dicom_files[idx]}, replacing with zeros."
                        )
                        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                except Exception as e:
                    logger.error(f"Error reading DICOM file {dicom_files[idx]}: {e}")
                    img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            img = augment(img, self.augment_conf)
            images.append(torch.tensor(img, dtype=torch.float32))

        # ---- Metadata ----
        if self.aggregated_metadata:
            try:
                meta_row = self.metadata_df.loc[series_folder].to_numpy()
            except KeyError:
                logger.error(f"Aggregated metadata not found for {series_folder}, using NaNs.")
                meta_row = np.full(self.num_metadata_features, np.nan, dtype=np.float32)
            meta = torch.tensor(meta_row, dtype=torch.float32).unsqueeze(0).repeat(
                len(images), 1
            )  # (N, D)
        else:
            meta_list: List[torch.Tensor] = []
            for idx in indices:
                if idx is None:
                    m = torch.full((self.num_metadata_features,), torch.nan, dtype=torch.float32)
                else:
                    slice_key = str(dicom_files[idx])
                    try:
                        m = torch.tensor(
                            self.metadata_df.loc[slice_key].to_numpy(), dtype=torch.float32
                        )
                    except KeyError:
                        logger.error(
                            f"Slice metadata not found for {slice_key}, using NaNs."
                        )
                        m = torch.full(
                            (self.num_metadata_features,), torch.nan, dtype=torch.float32
                        )
                meta_list.append(m)
            meta = torch.stack(meta_list, dim=0)  # (N, D)

        return torch.stack(images, dim=0), meta  # (N, H, W), (N, D)

    # ------------------------------------------------------------------
    # Label processing
    # ------------------------------------------------------------------

    def _process_labels(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Convert raw label strings to index tensors and validity masks."""
        targets: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for label_class, classes in self.label_names.items():
            if label_class not in self.labels[idx]:
                raise KeyError(f"Missing label '{label_class}' for sample {idx}")

            value = self.labels[idx][label_class]

            if value not in classes:
                raise ValueError(
                    f"Unknown value '{value}' for '{label_class}'. "
                    f"Expected one of {classes}"
                )

            label_idx = classes.index(value)
            targets.append(torch.tensor(label_idx))
            masks.append(torch.tensor(value != "na"))

        return tuple(targets), tuple(masks)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, str],
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]],
    ]:
        """Fetch one sample.

        Args:
            idx: Sample index.

        Returns:
            Inference : ``(images, metadata, filepath)``
            Training  : ``(images, metadata, targets, masks)``
        """
        if idx >= len(self.path_list):
            raise IndexError(f"Index {idx} out of range (dataset size {len(self.path_list)})")

        images, metadata = self._load_slices(
            self.path_list[idx], sampling_type=self.sampling_type
        )

        if self.is_infer:
            return images, metadata, self.path_list[idx]

        targets, masks = self._process_labels(idx)
        return images, metadata, targets, masks


# 3D ADNI dataset
class ADNI3DDataset(ADNIDataset):
    """Variant of ADNIDataset that assembles full 3D volumes instead of 2D slices.

    Adapts the Duke 3D pipeline (DukeLiverDataset3D) for ADNI brain MRI:

    - Loads **all** DICOM slices in a series folder and stacks them into a
      native 3D volume ``(D_native, H_native, W_native)``.
    - Applies 3D augmentation (``DEFAULT3D`` / ``NONE3D`` configs).
    - Resizes the spatial dimensions to ``img_size × img_size`` with SimpleITK
      linear resampling.
    - Resamples the depth axis to exactly ``n_slices`` (equidistant subsampling
      or zero-padding).
    - Returns ``images`` of shape ``(1, n_slices, H, W)`` — ready for
      ``PyramidPooling3DClassifier`` / network07 without an extra wrapper.
    - Returns ``metadata`` of shape ``(n_slices, M)`` — the series-level
      metadata row repeated ``n_slices`` times for Trainer compatibility.

    Augmentation config strings ``"DEFAULT3D"`` and ``"NONE3D"`` are supported
    (defined in :mod:`IMC.data.augment`).

    Args:
        All arguments are inherited from :class:`ADNIDataset`.
        ``n_slices`` acts as the target depth (``D``) of the assembled volume.
        ``augment_conf`` should be one of ``"DEFAULT3D"`` or ``"NONE3D"``.

    Returns:
        Training mode  : ``(images, metadata, targets, masks)``
        Inference mode : ``(images, metadata, filepath)``

        Shapes::

            images   : (1, n_slices, H, W)   — channel-first, float32
            metadata : (n_slices, M)          — float32
    """

    # ------------------------------------------------------------------
    # Depth resampling helper
    # ------------------------------------------------------------------

    def _resample_volume_depth(self, volume: np.ndarray) -> np.ndarray:
        """Resample *volume* along axis-0 to exactly ``self.n_slices``.

        - If ``D_native > n_slices``: equidistant subsampling.
        - If ``D_native < n_slices``: zero-pad symmetrically.
        - If ``D_native == n_slices``: no-op.

        Args:
            volume: ``(D_native, H, W)`` float32 array.

        Returns:
            ``(n_slices, H, W)`` float32 array.
        """
        target = self.n_slices
        native = volume.shape[0]

        if native == target:
            return volume

        if native > target:
            indices = np.linspace(0, native - 1, target, dtype=int)
            return volume[indices]

        # Pad with zeros
        pad = target - native
        pad_before = pad // 2
        pad_after  = pad - pad_before
        return np.pad(
            volume,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # ------------------------------------------------------------------
    # Core loading override
    # ------------------------------------------------------------------

    def _load_slices(
        self, series_folder: str, sampling_type: str = "equidistant"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load the full 3D volume from *series_folder*.

        Returns:
            images   : ``(1, n_slices, H, W)`` float32 tensor
            metadata : ``(n_slices, M)``        float32 tensor
        """
        import SimpleITK as sitk
        from IMC.data.augment import augment3d

        # 1. Discover all DICOM slices (naturally sorted)
        dicom_files = self._get_dicom_files(series_folder)

        # 2. Load every slice and stack into a native 3D volume (D, H, W)
        slices: List[np.ndarray] = []
        for f in dicom_files:
            ds = dcmread(f)
            try:
                arr = ds.pixel_array.astype(np.float32)
            except Exception as e:
                logger.error(f"Error reading DICOM file {f}: {e}")
                continue
            if arr.ndim == 2:
                slices.append(arr)
            elif arr.ndim == 3:
                # Multi-frame DICOM — take all frames
                for fi in range(arr.shape[0]):
                    slices.append(arr[fi])
            else:
                logger.warning(f"Unexpected array shape {arr.shape} in {f}, skipping.")

        if not slices:
            raise RuntimeError(f"No valid slices loaded from: {series_folder}")

        volume = np.stack(slices, axis=0).astype(np.float32)  # (D_native, H_native, W_native)

        # 3. Apply 3D augmentation (DEFAULT3D / NONE3D)
        volume = augment3d(volume, conf=self.augment_conf)
        volume = volume.astype(np.float32)

        # 4. Resize spatial dims to img_size × img_size via SimpleITK linear resampling
        sitk_img   = sitk.GetImageFromArray(volume)
        orig_size  = sitk_img.GetSize()    # (W, H, D) — SimpleITK convention
        orig_spc   = sitk_img.GetSpacing() # (sx, sy, sz)

        new_spc_x  = (orig_size[0] * orig_spc[0]) / self.img_size
        new_spc_y  = (orig_size[1] * orig_spc[1]) / self.img_size
        new_spc_z  = orig_spc[2]  # keep depth spacing unchanged

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing([new_spc_x, new_spc_y, new_spc_z])
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetSize([self.img_size, self.img_size, orig_size[2]])

        volume = sitk.GetArrayFromImage(resampler.Execute(sitk_img)).astype(np.float32)
        # volume shape: (D_native, img_size, img_size)

        # 5. Resample depth axis → n_slices
        volume = self._resample_volume_depth(volume)  # (n_slices, img_size, img_size)

        # 6. Add channel dimension → (1, n_slices, H, W)
        volume_tensor = torch.from_numpy(volume.copy()).unsqueeze(0).to(dtype=torch.float32)

        # 7. Metadata: series-level row repeated n_slices times → (n_slices, M)
        target_depth = self.n_slices
        if self.aggregated_metadata:
            try:
                meta_row = self.metadata_df.loc[series_folder].to_numpy().astype(np.float32)
            except KeyError:
                logger.error(
                    f"Aggregated metadata not found for {series_folder}, using NaNs."
                )
                meta_row = np.full(self.num_metadata_features, np.nan, dtype=np.float32)
        else:
            # Per-slice metadata is not meaningful for a 3D volume; fall back to NaNs.
            meta_row = np.full(self.num_metadata_features, np.nan, dtype=np.float32)

        meta = (
            torch.tensor(meta_row, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(target_depth, 1)
        )  # (n_slices, M)

        return volume_tensor, meta

# ---------------------------------------------------------------------------
# Convenience DataLoader factories
# ---------------------------------------------------------------------------

def get_train_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    num_samples: Optional[int] = None,
    folder_split: List[str] = ["fold_0", "fold_1", "fold_2"],
    **ds_kwargs: Any,
) -> DataLoader:
    """DataLoader for training splits (default: fold_0 … fold_2)."""
    dataset = ADNIDataset(
        split=folder_split,
        num_samples=num_samples,
        augment_conf="NONE2D",
        **ds_kwargs,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_valid_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    num_samples: Optional[int] = None,
    folder_split: List[str] = ["fold_3"],
    **ds_kwargs: Any,
) -> DataLoader:
    """DataLoader for validation split (default: fold_3)."""
    dataset = ADNIDataset(
        split=folder_split,
        num_samples=num_samples,
        augment_conf="NONE2D",
        **ds_kwargs,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_test_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    num_samples: Optional[int] = None,
    folder_split: List[str] = ["fold_4"],
    **ds_kwargs: Any,
) -> DataLoader:
    """DataLoader for test split (default: fold_4)."""
    dataset = ADNIDataset(
        split=folder_split,
        num_samples=num_samples,
        augment_conf="NONE2D",
        **ds_kwargs,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_infer_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    num_samples: Optional[int] = None,
    folder_split: Optional[List[str]] = None,
    label_names: Optional[Dict[str, List[str]]] = None,
    **ds_kwargs: Any,
) -> DataLoader:
    """DataLoader for inference (no labels returned).

    Args:
        folder_split: Folds to include. ``None`` loads all rows.
        label_names: Optional override of label mappings (needed to determine
            ``get_n_labels()`` even in inference mode).
    """
    dataset = ADNIDataset(
        split=folder_split,
        num_samples=num_samples,
        is_infer=True,
        augment_conf="NONE2D",
        label_names=label_names,
        **ds_kwargs,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    """Quick smoke-test of ADNIDataset (inference mode, no DICOM loading)."""
    print("ADNIDataset Demo")
    print("=" * 50)

    dataset = ADNIDataset(
        num_samples=10,
        n_slices=3,
        is_infer=True,
        aggregated_metadata=False,
        split=["fold_0"],
        local_dataset_path="/home/tuan.truong/data/ADNI_full",
        label_csv_path="/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv",
        metadata_path="/home/tuan.truong/codebase/IMC/labels/adni_metadata/adni_encoded_metadata_20260224.parquet",
    )

    print(f"Dataset size : {len(dataset)} samples")
    print(f"Label dims   : {dataset.get_n_labels()}")
    print(f"Metadata dim : {dataset.num_metadata_features}")

    for i in range(len(dataset)):
        images, metadata, filepath = dataset[i]
        print(f"Sample {i}:")
        print(f"  Filepath : {filepath}")
        print(f"  Images   : {images.shape} dtype={images.dtype}")
        print(f"  Metadata : {metadata.shape} dtype={metadata.dtype}")

    dataset_3d = ADNI3DDataset(
        num_samples=10,
        n_slices=64,
        is_infer=True,
        aggregated_metadata=False,
        split=["fold_0"],
        local_dataset_path="/home/tuan.truong/data/ADNI_full",
        label_csv_path="/home/tuan.truong/codebase/IMC/labels/labels_ADNI_local_20260224.csv",
        metadata_path="/home/tuan.truong/codebase/IMC/labels/adni_metadata/adni_encoded_metadata_20260224.parquet",
        augment_conf="NONE3D",
    )
    print(f"\nADNI3DDataset size : {len(dataset_3d)} samples")
    for i in range(len(dataset_3d)):
        images, metadata, filepath = dataset_3d[i]
        print(f"3D Sample {i}:")
        print(f"  Filepath : {filepath}")
        print(f"  Images   : {images.shape} dtype={images.dtype}")
        print(f"  Metadata : {metadata.shape} dtype={metadata.dtype}")
