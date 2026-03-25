"""
image_reader.py — Modality-agnostic image I/O utilities.

This module decouples the low-level "find files / pick slices / read pixels"
logic that was previously duplicated across every dataloader into a single,
reusable layer.  Dataloaders stay responsible for dataset-specific concerns
(label parsing, metadata encoding, augmentation) while delegating all raw I/O
to an :class:`ImageReader` instance.

Extending to NIfTI (or any other format) only requires adding a new subclass
of :class:`ImageReader` without touching existing dataset code.

Public API
----------
calculate_slice_indices(num_slices, n_images, sampling_type) -> List[Optional[int]]
    Pure slice-sampling function — identical algorithm previously duplicated
    in every local dataloader.

ImageReader  (ABC)
    Abstract interface: ``list_files(folder)`` + ``read_pixel_array(path)``

DicomImageReader(ImageReader)
    DICOM implementation backed by *pydicom*.

Authors: IMC team
Date: 2026
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from natsort import natsorted

logger = logging.getLogger("IMC")

__all__ = [
    "calculate_slice_indices",
    "ImageReader",
    "DicomImageReader",
]


# ---------------------------------------------------------------------------
# Slice sampling — shared by all modalities
# ---------------------------------------------------------------------------


def calculate_slice_indices(
    num_slices: int,
    n_images: int,
    sampling_type: str = "equidistant",
) -> List[Optional[int]]:
    """Sample *n_images* slice indices from a series of *num_slices* slices.

    A small edge offset is trimmed on both ends to avoid uninformative border
    slices.  When the series is shorter than requested (``num_slices <
    n_images``) all available slices are used and the remainder of the list is
    filled with ``None`` — callers should treat ``None`` entries as zero-filled
    padding.

    Args:
        num_slices: Total number of slices in the series.
        n_images: Number of indices to return.
        sampling_type: ``"equidistant"`` (default) spaces slices uniformly
            across the trimmed range; ``"random"`` draws without replacement.

    Returns:
        List of length *n_images* containing integer indices or ``None`` for
        padding positions.
    """
    offset_fraction = num_slices // max(n_images, 1)
    offset = min(offset_fraction // 4, 2) * n_images

    if n_images <= num_slices:
        if sampling_type == "random":
            start = max(0, offset)
            end = max(num_slices - offset, n_images)
            return random.sample(range(start, end), n_images)
        else:  # equidistant
            start = offset
            end = num_slices - 1 - offset
            return [int(x) for x in np.linspace(start, end, n_images)]
    else:
        return list(range(num_slices)) + [None] * (n_images - num_slices)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ImageReader(ABC):
    """Abstract interface for reading single 2-D image slices.

    Concrete subclasses implement :meth:`list_files` and
    :meth:`read_pixel_array` for a specific file format (DICOM, NIfTI, PNG,
    …).  Slice-sampling logic is format-agnostic and lives in the module-level
    :func:`calculate_slice_indices` helper.
    """

    @abstractmethod
    def list_files(
        self, folder: Path, series_uid: Optional[str] = None
    ) -> List[Path]:
        """Return a naturally-sorted list of image files under *folder*.

        Args:
            folder: Directory containing the image files for one series.
            series_uid: When provided, only files whose ``SeriesInstanceUID``
                DICOM header tag matches this value are returned.  Pass
                ``None`` (default) to include all discovered files (legacy
                behaviour).

        Returns:
            Sorted list of :class:`~pathlib.Path` objects.

        Raises:
            RuntimeError: If *folder* does not exist or contains no files.
        """

    @abstractmethod
    def read_pixel_array(
        self, path: Path
    ) -> Tuple[Optional[np.ndarray], Any]:
        """Read the 2-D pixel array from *path*.

        On success returns ``(array, header)`` where *array* is a 2-D
        ``float32`` NumPy array and *header* is the format-specific metadata
        object (e.g. a pydicom ``FileDataset``).  Returns ``(None, None)`` on
        any read failure so callers can substitute zero-filled arrays without
        crashing.

        Args:
            path: Path to the image file.

        Returns:
            ``(pixel_array, header)`` or ``(None, None)`` on error.
        """


# ---------------------------------------------------------------------------
# DICOM implementation
# ---------------------------------------------------------------------------


class DicomImageReader(ImageReader):
    """DICOM image reader backed by *pydicom*.

    Args:
        allow_no_extension: When ``True``, fall back to *all* regular files
            when no ``*.dcm`` / ``*.dicom`` matches are found.  Required for
            some ADNI series that store DICOM without a file extension.
    """

    def __init__(self, allow_no_extension: bool = False) -> None:
        self.allow_no_extension = allow_no_extension

    # ------------------------------------------------------------------
    # ImageReader interface
    # ------------------------------------------------------------------

    def list_files(
        self, folder: Path, series_uid: Optional[str] = None
    ) -> List[Path]:
        """Return a naturally-sorted list of DICOM files in *folder*.

        When *series_uid* is provided only files whose ``SeriesInstanceUID``
        header tag matches are returned.  This handles the common case where
        multiple series share the same directory (e.g. PACS exports).

        The UID filter reads only the ``SeriesInstanceUID`` tag via
        ``stop_before_pixels=True`` + ``specific_tags``, so the per-file
        overhead is minimal (~milliseconds). For folders containing a single
        series the cost is zero because all files pass the filter.

        Args:
            folder: Directory to scan.
            series_uid: Target ``SeriesInstanceUID``; ``None`` → include all
                files (backward-compatible, no header reads).

        Raises:
            RuntimeError: If *folder* does not exist, no files are found, or
                (when *series_uid* is given) no files match that UID.
        """
        if not folder.exists():
            raise RuntimeError(f"DICOM folder not found: {folder}")

        files: List[Path] = (
            list(folder.rglob("*.dcm")) + list(folder.rglob("*.dicom"))
        )

        if not files and self.allow_no_extension:
            files = [p for p in folder.rglob("*") if p.is_file()]

        if not files:
            raise RuntimeError(f"No DICOM files found in: {folder}")

        files = natsorted(files)

        if series_uid is not None:
            files = self._filter_by_uid(files, series_uid, folder)

        return files

    def _filter_by_uid(
        self,
        files: List[Path],
        series_uid: str,
        folder: Path,
    ) -> List[Path]:
        """Return only *files* whose SeriesInstanceUID header equals *series_uid*.

        Uses ``specific_tags=["SeriesInstanceUID"]`` so pydicom stops parsing
        immediately after reading that single tag — much faster than a full
        header read.

        Args:
            files: Candidate file paths (already discovered and sorted).
            series_uid: Target UID string.
            folder: Used only for the error message.

        Returns:
            Filtered, naturally-sorted list.  May be empty if no file matches
            (the caller raises :class:`RuntimeError` in that case).
        """
        import pydicom  # lazy import — avoids hard dep at module load time

        matched: List[Path] = []
        for p in files:
            try:
                ds = pydicom.dcmread(
                    str(p),
                    stop_before_pixels=True,
                    specific_tags=[(0x0020, 0x000E)],  # SeriesInstanceUID
                )
                uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
                if uid == series_uid:
                    matched.append(p)
            except Exception:
                pass  # unreadable file — skip silently

        if not matched:
            raise RuntimeError(
                f"No DICOM files with SeriesInstanceUID={series_uid!r} "
                f"found in: {folder}"
            )
        return matched

    def read_pixel_array(
        self, path: Path
    ) -> Tuple[Optional[np.ndarray], Any]:
        """Read a DICOM file and return its raw pixel array and dataset.

        The returned array is always 2-D ``float32`` when the file is
        readable; ``(None, None)`` is returned on any error so callers can
        substitute zero-filled arrays gracefully.

        Args:
            path: Path to the ``.dcm`` file.

        Returns:
            ``(pixel_array, dataset)`` or ``(None, None)`` on failure.
            *dataset* is a :class:`pydicom.dataset.FileDataset` and can be
            used for tag access (HU rescaling, window presets, etc.).
        """
        try:
            from pydicom import dcmread as _dcmread  # lazy import

            ds = _dcmread(path)
            arr: np.ndarray = ds.pixel_array.astype(np.float32)
            return arr, ds
        except Exception as exc:
            logger.error("Failed to read DICOM file %s: %s", path, exc)
            return None, None
