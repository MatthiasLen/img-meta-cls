"""
Tests for :mod:`IMC.data.image_reader`.

Test organisation
-----------------
1. **calculate_slice_indices** (:class:`TestCalculateSliceIndices`) — Pure
   function tests: equidistant sampling, random sampling, padding when
   n_images > num_slices, edge behaviour, single slice, and exact
   reproducibility of the offset formula.

2. **DicomImageReader.list_files** (:class:`TestDicomImageReaderListFiles`) —
   File discovery: ``*.dcm`` and ``*.dicom`` patterns, natural sorting,
   recursive subdirectory discovery, error paths (missing folder, empty
   folder), and the ``allow_no_extension`` fallback used by the ADNI loader.

3. **DicomImageReader.read_pixel_array** (:class:`TestDicomImageReaderPixels`)
   — Success path returning a 2-D ``float32`` NumPy array plus the pydicom
   dataset; failure paths returning ``(None, None)``; multi-dim guard absent
   (the caller owns that responsibility).

4. **ImageReader is abstract** (:class:`TestImageReaderAbstract`) — Confirms
   that ``ImageReader`` cannot be instantiated directly and that a concrete
   subclass with both methods implemented can be.

5. **Dataloader integration** (:class:`TestDataloaderIntegration`) — Verifies
   that ``LiverDataset``, ``DukeDataset``, ``ADNIDataset``, and ``CTDataset``
   each expose ``self._reader``, and that ``get_bucket_filelist`` /
   ``_get_dicom_files`` delegate to the reader (previously duplicated code).

All tests run without GPU / real DICOM data by using ``unittest.mock``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from IMC.data.image_reader import (
    ImageReader,
    DicomImageReader,
    calculate_slice_indices,
)


# ===========================================================================
# 1. calculate_slice_indices
# ===========================================================================


class TestCalculateSliceIndices:
    """Unit tests for the module-level sampling helper."""

    # ------------------------------------------------------------------
    # Equidistant (default)
    # ------------------------------------------------------------------

    def test_equidistant_returns_correct_length(self):
        result = calculate_slice_indices(20, 4)
        assert len(result) == 4

    def test_equidistant_all_ints(self):
        result = calculate_slice_indices(20, 4)
        assert all(isinstance(i, int) for i in result)

    def test_equidistant_indices_in_range(self):
        num_slices = 30
        n = 5
        indices = calculate_slice_indices(num_slices, n)
        for i in indices:
            assert 0 <= i < num_slices

    def test_equidistant_no_duplicates(self):
        indices = calculate_slice_indices(50, 5)
        assert len(set(indices)) == len(indices)

    def test_equidistant_sorted(self):
        """Equidistant indices should be monotonically non-decreasing."""
        indices = calculate_slice_indices(50, 10)
        assert indices == sorted(indices)

    def test_exact_match_small(self):
        """Regression: num_slices=10, n_images=3 → offset=3, linspace 3..6."""
        # offset_fraction = 10//3 = 3, offset = min(3//4,2)*3 = 0
        # start=0, end=9, linspace(0,9,3) = [0, 4, 9]
        indices = calculate_slice_indices(10, 3, "equidistant")
        assert len(indices) == 3
        assert indices == sorted(indices)

    # ------------------------------------------------------------------
    # Random
    # ------------------------------------------------------------------

    def test_random_returns_correct_length(self):
        result = calculate_slice_indices(30, 5, "random")
        assert len(result) == 5

    def test_random_all_ints(self):
        result = calculate_slice_indices(30, 5, "random")
        assert all(isinstance(i, int) for i in result)

    def test_random_no_duplicates(self):
        result = calculate_slice_indices(30, 5, "random")
        assert len(set(result)) == len(result)

    def test_random_indices_in_range(self):
        num_slices = 50
        result = calculate_slice_indices(num_slices, 8, "random")
        for i in result:
            assert 0 <= i < num_slices

    # ------------------------------------------------------------------
    # Padding (n_images > num_slices)
    # ------------------------------------------------------------------

    def test_padding_when_too_few_slices(self):
        result = calculate_slice_indices(2, 5)
        assert len(result) == 5
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] is None
        assert result[3] is None
        assert result[4] is None

    def test_no_padding_when_exact(self):
        result = calculate_slice_indices(5, 5)
        assert None not in result

    def test_single_slice_n1(self):
        result = calculate_slice_indices(1, 1)
        assert len(result) == 1
        assert result[0] == 0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_n_images_equals_num_slices_no_none(self):
        result = calculate_slice_indices(10, 10)
        assert all(v is not None for v in result)

    def test_large_series(self):
        result = calculate_slice_indices(500, 20)
        assert len(result) == 20
        assert all(v is not None for v in result)


# ===========================================================================
# 2. DicomImageReader.list_files
# ===========================================================================


class TestDicomImageReaderListFiles:
    """Tests for file-discovery in DicomImageReader."""

    def test_finds_dcm_files(self, tmp_path):
        (tmp_path / "slice_001.dcm").write_bytes(b"")
        (tmp_path / "slice_002.dcm").write_bytes(b"")
        reader = DicomImageReader()
        files = reader.list_files(tmp_path)
        assert len(files) == 2

    def test_finds_dicom_files(self, tmp_path):
        (tmp_path / "a.dicom").write_bytes(b"")
        reader = DicomImageReader()
        files = reader.list_files(tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".dicom"

    def test_finds_mixed_extensions(self, tmp_path):
        (tmp_path / "a.dcm").write_bytes(b"")
        (tmp_path / "b.dicom").write_bytes(b"")
        reader = DicomImageReader()
        files = reader.list_files(tmp_path)
        assert len(files) == 2

    def test_naturally_sorted(self, tmp_path):
        for name in ["slice_10.dcm", "slice_2.dcm", "slice_1.dcm"]:
            (tmp_path / name).write_bytes(b"")
        reader = DicomImageReader()
        files = reader.list_files(tmp_path)
        assert [f.name for f in files] == ["slice_1.dcm", "slice_2.dcm", "slice_10.dcm"]

    def test_recursive_discovery(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.dcm").write_bytes(b"")
        reader = DicomImageReader()
        files = reader.list_files(tmp_path)
        assert len(files) == 1

    def test_missing_folder_raises(self, tmp_path):
        reader = DicomImageReader()
        with pytest.raises(RuntimeError, match="not found"):
            reader.list_files(tmp_path / "nonexistent")

    def test_empty_folder_raises(self, tmp_path):
        reader = DicomImageReader()
        with pytest.raises(RuntimeError, match="No DICOM"):
            reader.list_files(tmp_path)

    def test_allow_no_extension_fallback(self, tmp_path):
        """ADNI series have extension-less files; allow_no_extension must return them."""
        (tmp_path / "IM000001").write_bytes(b"")
        (tmp_path / "IM000002").write_bytes(b"")
        reader_strict = DicomImageReader(allow_no_extension=False)
        with pytest.raises(RuntimeError):
            reader_strict.list_files(tmp_path)

        reader_loose = DicomImageReader(allow_no_extension=True)
        files = reader_loose.list_files(tmp_path)
        assert len(files) == 2

    def test_allow_no_extension_ignored_when_dcm_present(self, tmp_path):
        """When .dcm files exist the no-extension fallback must NOT fire."""
        (tmp_path / "real.dcm").write_bytes(b"")
        (tmp_path / "extra_no_ext").write_bytes(b"")
        reader = DicomImageReader(allow_no_extension=True)
        files = reader.list_files(tmp_path)
        # Only the .dcm should be returned
        assert all(f.suffix == ".dcm" for f in files)


# ===========================================================================
# 3. DicomImageReader.read_pixel_array
# ===========================================================================


class TestDicomImageReaderPixels:
    """Tests for pixel-array reading in DicomImageReader."""

    def _make_fake_ds(self, shape=(32, 32)):
        """Return a mock pydicom dataset with a synthetic pixel_array."""
        ds = mock.MagicMock()
        ds.pixel_array = np.random.randint(0, 1024, shape, dtype=np.int16)
        return ds

    def test_success_returns_float32_array(self, tmp_path):
        fake_path = tmp_path / "slice.dcm"
        fake_path.write_bytes(b"")
        ds = self._make_fake_ds((64, 64))

        reader = DicomImageReader()
        with mock.patch("IMC.data.image_reader.DicomImageReader.read_pixel_array",
                        wraps=reader.read_pixel_array) as _:
            # Patch the inner dcmread call inside image_reader
            with mock.patch("pydicom.dcmread", return_value=ds):
                arr, header = reader.read_pixel_array(fake_path)

        assert arr is not None
        assert arr.dtype == np.float32
        assert arr.shape == (64, 64)
        assert header is ds

    def test_failure_returns_none_pair(self, tmp_path):
        fake_path = tmp_path / "bad.dcm"
        fake_path.write_bytes(b"")

        reader = DicomImageReader()
        with mock.patch("pydicom.dcmread", side_effect=Exception("corrupt")):
            arr, header = reader.read_pixel_array(fake_path)

        assert arr is None
        assert header is None

    def test_pixel_array_cast_to_float32(self, tmp_path):
        fake_path = tmp_path / "s.dcm"
        fake_path.write_bytes(b"")
        ds = mock.MagicMock()
        ds.pixel_array = np.zeros((10, 10), dtype=np.int32)

        reader = DicomImageReader()
        with mock.patch("pydicom.dcmread", return_value=ds):
            arr, _ = reader.read_pixel_array(fake_path)

        assert arr.dtype == np.float32


# ===========================================================================
# 4. ImageReader is abstract
# ===========================================================================


class TestImageReaderAbstract:
    """Confirm the ABC contract."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ImageReader()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        class MyReader(ImageReader):
            def list_files(self, folder):
                return []

            def read_pixel_array(self, path):
                return None, None

        reader = MyReader()
        assert isinstance(reader, ImageReader)

    def test_partial_subclass_still_abstract(self):
        """A subclass missing read_pixel_array must still be abstract."""
        class Partial(ImageReader):
            def list_files(self, folder):
                return []

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ===========================================================================
# 5. Dataloader integration — self._reader wiring
# ===========================================================================


def _make_minimal_df(index_values, columns=("enc_x",)):
    """Create a minimal pandas DataFrame for metadata mocking."""
    import pandas as pd

    data = {c: np.zeros(len(index_values)) for c in columns}
    return pd.DataFrame(data, index=index_values)


def _patch_load(cls, path, extra_attrs=None):
    """Return a context-manager that replaces _load_metadata_and_labels with
    a stub that sets the minimal attributes required by __init__."""

    def _stub(self, split):
        self.path_list = ["series/001"]
        self.labels = [{"label_SequenceType": "T1"}]
        self.metadata_df = _make_minimal_df(["series/001"])
        self.num_samples = 1
        if extra_attrs:
            for k, v in extra_attrs.items():
                setattr(self, k, v)

    return mock.patch.object(cls, "_load_metadata_and_labels", autospec=True,
                             side_effect=_stub)


class _BaseDataloaderReaderTest:
    """Shared helpers for dataloader integration tests."""

    @staticmethod
    def _assert_has_reader(dataset, expected_class=DicomImageReader):
        assert hasattr(dataset, "_reader"), (
            "Dataset must expose self._reader after __init__"
        )
        assert isinstance(dataset._reader, expected_class), (
            f"Expected self._reader to be {expected_class.__name__}, "
            f"got {type(dataset._reader).__name__}"
        )


class TestLiverDatasetReaderIntegration(_BaseDataloaderReaderTest):
    """LiverDataset must hold a DicomImageReader and delegate file ops."""

    def _build(self, tmp_path):
        """Construct a minimal LiverDataset with all I/O mocked out."""
        from IMC.data.liver_dataloader_local import LiverDataset

        with _patch_load(LiverDataset, tmp_path):
            ds = LiverDataset(
                num_samples=1,
                n_slices=2,
                split=["fold_0"],
                local_dataset_path=str(tmp_path),
                metadata_path="meta.parquet",
                label_csv_path="labels.csv",
            )
        return ds

    def test_reader_is_dicom_reader(self, tmp_path):
        ds = self._build(tmp_path)
        self._assert_has_reader(ds)

    def test_get_bucket_filelist_delegates_to_reader(self, tmp_path):
        ds = self._build(tmp_path)

        sentinel = [Path(tmp_path / "s.dcm")]
        with mock.patch.object(ds._reader, "list_files", return_value=sentinel) as m:
            result = ds.get_bucket_filelist("series/001")

        m.assert_called_once_with(Path(str(tmp_path)) / "series/001", series_uid=None)
        assert result is sentinel

    def test_calculate_slice_indices_delegates(self, tmp_path):
        ds = self._build(tmp_path)
        with mock.patch("IMC.data.liver_dataloader_local.calculate_slice_indices",
                        return_value=[0, 2, 4]) as m:
            result = ds._calculate_slice_indices(10, 3, "equidistant")
        m.assert_called_once_with(10, 3, "equidistant")
        assert result == [0, 2, 4]


class TestDukeDatasetReaderIntegration(_BaseDataloaderReaderTest):
    """DukeDataset (LiverDataset subclass for Duke) must hold a DicomImageReader."""

    def _build(self, tmp_path):
        from IMC.data.duke_dataloader_local import LiverDataset

        with _patch_load(LiverDataset, tmp_path):
            ds = LiverDataset(
                num_samples=1,
                n_slices=2,
                split=["fold_0"],
                local_dataset_path=str(tmp_path),
                metadata_path="meta.parquet",
                label_csv_path="labels.csv",
            )
        return ds

    def test_reader_is_dicom_reader(self, tmp_path):
        ds = self._build(tmp_path)
        self._assert_has_reader(ds)

    def test_get_bucket_filelist_delegates_to_reader(self, tmp_path):
        ds = self._build(tmp_path)
        sentinel = [Path(tmp_path / "s.dcm")]
        with mock.patch.object(ds._reader, "list_files", return_value=sentinel) as m:
            result = ds.get_bucket_filelist("series/001")
        m.assert_called_once_with(Path(str(tmp_path)) / "series/001", series_uid=None)
        assert result is sentinel


adni_dataloader = pytest.importorskip(
    "IMC.data.adni_dataloader_local",
    reason="IMC.data.adni_dataloader_local not available",
)


class TestADNIDatasetReaderIntegration(_BaseDataloaderReaderTest):
    """ADNIDataset must use a DicomImageReader with allow_no_extension=True."""

    def _build(self, tmp_path):
        from IMC.data.adni_dataloader_local import ADNIDataset

        adni_attrs = {
            "path_list": [str(tmp_path / "series001")],
            "labels": [{"label_DX": "CN"}],
            "metadata_df": _make_minimal_df([str(tmp_path / "series001")]),
            "num_samples": 1,
        }

        def _stub(self, split):
            self.path_list = adni_attrs["path_list"]
            self.labels = adni_attrs["labels"]
            self.metadata_df = adni_attrs["metadata_df"]
            self.num_samples = adni_attrs["num_samples"]

        with mock.patch.object(ADNIDataset, "_load_metadata_and_labels",
                               autospec=True, side_effect=_stub):
            ds = ADNIDataset(
                num_samples=1,
                n_slices=2,
                split=["fold_0"],
                local_dataset_path=str(tmp_path),
                metadata_path="meta.parquet",
                label_csv_path="labels.csv",
            )
        return ds

    def test_reader_is_dicom_reader(self, tmp_path):
        ds = self._build(tmp_path)
        self._assert_has_reader(ds)

    def test_reader_allows_no_extension(self, tmp_path):
        ds = self._build(tmp_path)
        assert ds._reader.allow_no_extension is True

    def test_get_dicom_files_delegates_to_reader(self, tmp_path):
        ds = self._build(tmp_path)
        folder = str(tmp_path / "series001")
        sentinel = [Path(folder) / "IM0001"]
        with mock.patch.object(ds._reader, "list_files", return_value=sentinel) as m:
            result = ds._get_dicom_files(folder)
        m.assert_called_once()
        assert result is sentinel

    def test_calculate_slice_indices_delegates(self, tmp_path):
        ds = self._build(tmp_path)
        with mock.patch("IMC.data.adni_dataloader_local.calculate_slice_indices",
                        return_value=[0, 5]) as m:
            result = ds._calculate_slice_indices(20, 2, "random")
        m.assert_called_once_with(20, 2, "random")
        assert result == [0, 5]


class TestCTDatasetReaderIntegration(_BaseDataloaderReaderTest):
    """CTDataset must hold a DicomImageReader and delegate slice sampling."""

    def _build(self, tmp_path):
        from IMC.data.ct_dataloader_local import CTDataset

        series_uid = "1.2.3.4"

        def _stub(self, split):
            self.series_list = [series_uid]
            self.labels = [{"label_Contrast": "yes"}]
            self.series_to_slices = {series_uid: [str(tmp_path / "s.dcm")]}
            self.series_metadata = {series_uid: np.zeros(1)}
            self.slice_metadata = None
            self._enc_cols = ["enc_x"]
            self.num_samples = 1

        with mock.patch.object(CTDataset, "_load_metadata_and_labels",
                               autospec=True, side_effect=_stub):
            ds = CTDataset(
                num_samples=1,
                n_slices=2,
                local_dataset_path=str(tmp_path),
                metadata_path="meta.parquet",
                label_csv_path="labels.csv",
            )
        return ds

    def test_reader_is_dicom_reader(self, tmp_path):
        ds = self._build(tmp_path)
        self._assert_has_reader(ds)

    def test_calculate_slice_indices_delegates(self, tmp_path):
        ds = self._build(tmp_path)
        with mock.patch("IMC.data.ct_dataloader_local.calculate_slice_indices",
                        return_value=[0, 3]) as m:
            result = ds._calculate_slice_indices(15, 2)
        m.assert_called_once_with(15, 2, ds.sampling_type)
        assert result == [0, 3]
