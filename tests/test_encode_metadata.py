"""Tests for :mod:`IMC.data.encode_metadata`.

Test organisation
-----------------
1. **_filter_slices_by_uid** (:class:`TestFilterSlicesByUid`) — Returns only
   slices whose ``SeriesInstanceUID`` header matches the target UID; skips
   unreadable files silently; returns empty list when nothing matches.

2. **parse_args** (:class:`TestParseArgs`) — Default values, required
   arguments, and ``--mode`` / ``--checkpoint_every`` overrides.

3. **main — slicewise mode** (:class:`TestMainSlicewise`) — Happy path writes
   a Parquet with one row per DICOM file and a ``Filepath`` column; adds
   ``SeriesInstanceUID`` when the label CSV contains that column; missing
   series directories are skipped without error.

4. **main — series mode** (:class:`TestMainSeries`) — Happy path writes one
   row per series; ``Filepath`` is the series folder path.

5. **main — edge cases** (:class:`TestMainEdgeCases`) — Missing filepath
   column raises ``ValueError``; no encodable series produces no output file;
   checkpointing writes and removes the checkpoint file; ``--output_name``
   overrides the auto-generated filename.

All tests run without real DICOM data by patching
``generate_slicewise_metadata_vector`` and ``generate_series_metadata_vector``.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from IMC.data.encode_metadata import _filter_slices_by_uid, main, parse_args

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_ENCODED = {"enc_EchoTime": 2.5, "enc_FlipAngle": 15.0}


def _make_label_csv(tmp_path: Path, rows: list[dict], filename: str = "labels.csv") -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / filename
    df.to_csv(p, index=False)
    return p


def _make_series_dir(tmp_path: Path, name: str, n_dcm: int = 2) -> tuple[Path, list[Path]]:
    """Create a fake series directory with *n_dcm* empty ``.dcm`` files."""
    series_dir = tmp_path / name
    series_dir.mkdir(parents=True, exist_ok=True)
    slices = []
    for i in range(n_dcm):
        p = series_dir / f"slice_{i:03d}.dcm"
        p.touch()
        slices.append(p)
    return series_dir, slices


# ---------------------------------------------------------------------------
# 1. _filter_slices_by_uid
# ---------------------------------------------------------------------------


class TestFilterSlicesByUid:
    def test_returns_matching_slices(self, tmp_path):
        slices = [str(tmp_path / f"s{i}.dcm") for i in range(3)]
        target_uid = "1.2.3.4"

        def fake_dcmread(path, stop_before_pixels, specific_tags):
            ds = MagicMock()
            ds.SeriesInstanceUID = target_uid if path != slices[1] else "9.9.9"
            return ds

        with patch("pydicom.dcmread", side_effect=fake_dcmread):
            result = _filter_slices_by_uid(slices, target_uid)

        assert slices[0] in result
        assert slices[2] in result
        assert slices[1] not in result

    def test_returns_empty_when_no_match(self, tmp_path):
        slices = [str(tmp_path / "s0.dcm")]

        def fake_dcmread(path, stop_before_pixels, specific_tags):
            ds = MagicMock()
            ds.SeriesInstanceUID = "different.uid"
            return ds

        with patch("pydicom.dcmread", side_effect=fake_dcmread):
            result = _filter_slices_by_uid(slices, "1.2.3.4")

        assert result == []

    def test_skips_unreadable_files(self, tmp_path):
        slices = [str(tmp_path / "bad.dcm")]

        with patch("pydicom.dcmread", side_effect=Exception("corrupt")):
            result = _filter_slices_by_uid(slices, "1.2.3.4")

        assert result == []

    def test_empty_input_returns_empty(self):
        result = _filter_slices_by_uid([], "1.2.3")
        assert result == []

    def test_strips_whitespace_from_uid(self, tmp_path):
        slices = [str(tmp_path / "s0.dcm")]

        def fake_dcmread(path, stop_before_pixels, specific_tags):
            ds = MagicMock()
            ds.SeriesInstanceUID = "  1.2.3.4  "
            return ds

        with patch("pydicom.dcmread", side_effect=fake_dcmread):
            result = _filter_slices_by_uid(slices, "1.2.3.4")

        assert slices[0] in result


# ---------------------------------------------------------------------------
# 2. parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def _call(self, argv: list[str]) -> object:
        with patch("sys.argv", ["encode_metadata"] + argv):
            return parse_args()

    def test_required_args_parsed(self, tmp_path):
        args = self._call(
            [
                "--label_file",
                str(tmp_path / "l.csv"),
                "--data_folder",
                str(tmp_path / "data"),
                "--output_folder",
                str(tmp_path / "out"),
            ]
        )
        assert args.label_file == tmp_path / "l.csv"
        assert args.data_folder == tmp_path / "data"
        assert args.output_folder == tmp_path / "out"

    def test_defaults(self, tmp_path):
        args = self._call(
            [
                "--label_file",
                str(tmp_path / "l.csv"),
                "--data_folder",
                str(tmp_path),
                "--output_folder",
                str(tmp_path),
            ]
        )
        assert args.mode == "slicewise"
        assert args.filepath_col == "Filepath"
        assert args.output_name is None
        assert args.checkpoint_every == 5000

    def test_mode_series(self, tmp_path):
        args = self._call(
            [
                "--label_file",
                str(tmp_path / "l.csv"),
                "--data_folder",
                str(tmp_path),
                "--output_folder",
                str(tmp_path),
                "--mode",
                "series",
            ]
        )
        assert args.mode == "series"

    def test_checkpoint_every_zero(self, tmp_path):
        args = self._call(
            [
                "--label_file",
                str(tmp_path / "l.csv"),
                "--data_folder",
                str(tmp_path),
                "--output_folder",
                str(tmp_path),
                "--checkpoint_every",
                "0",
            ]
        )
        assert args.checkpoint_every == 0

    def test_output_name_override(self, tmp_path):
        args = self._call(
            [
                "--label_file",
                str(tmp_path / "l.csv"),
                "--data_folder",
                str(tmp_path),
                "--output_folder",
                str(tmp_path),
                "--output_name",
                "my_output.parquet",
            ]
        )
        assert args.output_name == "my_output.parquet"

    def test_custom_filepath_col(self, tmp_path):
        args = self._call(
            [
                "--label_file",
                str(tmp_path / "l.csv"),
                "--data_folder",
                str(tmp_path),
                "--output_folder",
                str(tmp_path),
                "--filepath_col",
                "path",
            ]
        )
        assert args.filepath_col == "path"

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            self._call(
                [
                    "--label_file",
                    str(tmp_path / "l.csv"),
                    "--data_folder",
                    str(tmp_path),
                    "--output_folder",
                    str(tmp_path),
                    "--mode",
                    "invalid",
                ]
            )


# ---------------------------------------------------------------------------
# 3. main — slicewise mode
# ---------------------------------------------------------------------------


class TestMainSlicewise:
    def _run_main(self, argv: list[str]) -> None:
        with patch("sys.argv", ["encode_metadata"] + argv):
            main()

    def test_happy_path_writes_parquet(self, tmp_path):
        series_dir, slices = _make_series_dir(tmp_path / "data", "series_a", n_dcm=2)
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "series_a"}])
        out_dir = tmp_path / "out"

        encoded_rows = [dict(_FAKE_ENCODED), dict(_FAKE_ENCODED)]

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=(encoded_rows, {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )

        out_file = out_dir / "out.parquet"
        assert out_file.exists()
        df = pd.read_parquet(out_file)
        assert len(df) == 2
        assert "Filepath" in df.columns
        assert "enc_EchoTime" in df.columns

    def test_filepath_column_contains_slice_paths(self, tmp_path):
        series_dir, slices = _make_series_dir(tmp_path / "data", "series_a", n_dcm=2)
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "series_a"}])
        out_dir = tmp_path / "out"

        encoded_rows = [dict(_FAKE_ENCODED), dict(_FAKE_ENCODED)]

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=(encoded_rows, {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )

        df = pd.read_parquet(out_dir / "out.parquet")
        assert all(fp.endswith(".dcm") for fp in df["Filepath"])

    def test_adds_series_instance_uid_column(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=1)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a", "SeriesInstanceUID": "1.2.3"}],
        )
        out_dir = tmp_path / "out"

        with (
            patch(
                "IMC.data.encode_metadata._filter_slices_by_uid",
                return_value=[str(tmp_path / "data" / "series_a" / "slice_000.dcm")],
            ),
            patch(
                "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
                return_value=([dict(_FAKE_ENCODED)], {}),
            ),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )

        df = pd.read_parquet(out_dir / "out.parquet")
        assert "SeriesInstanceUID" in df.columns
        assert df["SeriesInstanceUID"].iloc[0] == "1.2.3"

    def test_missing_series_dir_skipped(self, tmp_path):
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "nonexistent_series"}])
        out_dir = tmp_path / "out"

        with patch("IMC.data.encode_metadata.generate_slicewise_metadata_vector") as mock_enc:
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )
            mock_enc.assert_not_called()

        # No output written when all series are skipped
        assert not (out_dir / "out.parquet").exists()

    def test_multiple_series_all_rows_present(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=2)
        _make_series_dir(tmp_path / "data", "series_b", n_dcm=3)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a"}, {"Filepath": "series_b"}],
        )
        out_dir = tmp_path / "out"

        def fake_enc(slice_filenames):
            return ([dict(_FAKE_ENCODED)] * len(slice_filenames), {})

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            side_effect=fake_enc,
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )

        df = pd.read_parquet(out_dir / "out.parquet")
        assert len(df) == 5  # 2 + 3

    def test_custom_filepath_col(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=1)
        label_csv = _make_label_csv(tmp_path, [{"path": "series_a"}])
        out_dir = tmp_path / "out"

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=([dict(_FAKE_ENCODED)], {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--filepath_col",
                    "path",
                    "--checkpoint_every",
                    "0",
                ]
            )

        assert (out_dir / "out.parquet").exists()

    def test_series_path_that_is_a_file_is_skipped(self, tmp_path):
        file_path = tmp_path / "data" / "not_a_dir.dcm"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "not_a_dir.dcm"}])
        out_dir = tmp_path / "out"

        with patch("IMC.data.encode_metadata.generate_slicewise_metadata_vector") as mock_enc:
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )
            mock_enc.assert_not_called()

    def test_series_dir_with_no_dcm_files_skipped(self, tmp_path):
        empty_dir = tmp_path / "data" / "empty_series"
        empty_dir.mkdir(parents=True)
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "empty_series"}])
        out_dir = tmp_path / "out"

        with patch("IMC.data.encode_metadata.generate_slicewise_metadata_vector") as mock_enc:
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )
            mock_enc.assert_not_called()

    def test_uid_filtering_applied_when_uid_col_present(self, tmp_path):
        series_dir, slices = _make_series_dir(tmp_path / "data", "series_a", n_dcm=3)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a", "SeriesInstanceUID": "1.2.3"}],
        )
        out_dir = tmp_path / "out"

        # Only the first two slices pass the UID filter
        filtered = [str(slices[0]), str(slices[1])]
        with (
            patch(
                "IMC.data.encode_metadata._filter_slices_by_uid",
                return_value=filtered,
            ) as mock_filter,
            patch(
                "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
                return_value=([dict(_FAKE_ENCODED), dict(_FAKE_ENCODED)], {}),
            ) as mock_enc,
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )
            mock_filter.assert_called_once_with(mock.ANY, "1.2.3")
            _, kwargs = mock_enc.call_args
            assert len(kwargs["slice_filenames"]) == 2


# ---------------------------------------------------------------------------
# 4. main — series mode
# ---------------------------------------------------------------------------


class TestMainSeries:
    def _run_main(self, argv: list[str]) -> None:
        with patch("sys.argv", ["encode_metadata"] + argv):
            main()

    def test_writes_one_row_per_series(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=2)
        _make_series_dir(tmp_path / "data", "series_b", n_dcm=2)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a"}, {"Filepath": "series_b"}],
        )
        out_dir = tmp_path / "out"

        with patch(
            "IMC.data.encode_metadata.generate_series_metadata_vector",
            return_value=(dict(_FAKE_ENCODED), {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--mode",
                    "series",
                    "--checkpoint_every",
                    "0",
                ]
            )

        df = pd.read_parquet(out_dir / "out.parquet")
        assert len(df) == 2

    def test_filepath_is_series_folder(self, tmp_path):
        series_dir, _ = _make_series_dir(tmp_path / "data", "series_a", n_dcm=2)
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "series_a"}])
        out_dir = tmp_path / "out"

        with patch(
            "IMC.data.encode_metadata.generate_series_metadata_vector",
            return_value=(dict(_FAKE_ENCODED), {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--mode",
                    "series",
                    "--checkpoint_every",
                    "0",
                ]
            )

        df = pd.read_parquet(out_dir / "out.parquet")
        assert df["Filepath"].iloc[0] == str(series_dir)

    def test_adds_uid_column_in_series_mode(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=2)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a", "SeriesInstanceUID": "9.8.7"}],
        )
        out_dir = tmp_path / "out"

        with (
            patch(
                "IMC.data.encode_metadata._filter_slices_by_uid",
                return_value=[
                    str(tmp_path / "data" / "series_a" / "slice_000.dcm"),
                    str(tmp_path / "data" / "series_a" / "slice_001.dcm"),
                ],
            ),
            patch(
                "IMC.data.encode_metadata.generate_series_metadata_vector",
                return_value=(dict(_FAKE_ENCODED), {}),
            ),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--mode",
                    "series",
                    "--checkpoint_every",
                    "0",
                ]
            )

        df = pd.read_parquet(out_dir / "out.parquet")
        assert "SeriesInstanceUID" in df.columns
        assert df["SeriesInstanceUID"].iloc[0] == "9.8.7"


# ---------------------------------------------------------------------------
# 5. main — edge cases
# ---------------------------------------------------------------------------


class TestMainEdgeCases:
    def _run_main(self, argv: list[str]) -> None:
        with patch("sys.argv", ["encode_metadata"] + argv):
            main()

    def test_missing_filepath_col_raises_value_error(self, tmp_path):
        label_csv = _make_label_csv(tmp_path, [{"wrong_col": "series_a"}])
        out_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="wrong_col"):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--checkpoint_every",
                    "0",
                ]
            )

    def test_no_encodable_series_writes_no_output(self, tmp_path):
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "nonexistent"}])
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        self._run_main(
            [
                "--label_file",
                str(label_csv),
                "--data_folder",
                str(tmp_path / "data"),
                "--output_folder",
                str(out_dir),
                "--output_name",
                "out.parquet",
                "--checkpoint_every",
                "0",
            ]
        )

        assert not (out_dir / "out.parquet").exists()

    def test_output_name_override_used(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=1)
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "series_a"}])
        out_dir = tmp_path / "out"

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=([dict(_FAKE_ENCODED)], {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "custom_name.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )

        assert (out_dir / "custom_name.parquet").exists()

    def test_default_output_name_is_timestamped(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=1)
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "series_a"}])
        out_dir = tmp_path / "out"

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=([dict(_FAKE_ENCODED)], {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--checkpoint_every",
                    "0",
                ]
            )

        parquet_files = list(out_dir.glob("encoded_metadata_*.parquet"))
        assert len(parquet_files) == 1

    def test_checkpoint_written_and_removed(self, tmp_path):
        # Create 2 series so we can trigger checkpoint_every=1 at count=1
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=1)
        _make_series_dir(tmp_path / "data", "series_b", n_dcm=1)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a"}, {"Filepath": "series_b"}],
        )
        out_dir = tmp_path / "out"

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=([dict(_FAKE_ENCODED)], {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "1",
                ]
            )

        # Final output exists
        assert (out_dir / "out.parquet").exists()
        # Checkpoint file cleaned up
        assert not (out_dir / "out_checkpoint.parquet").exists()

    def test_duplicate_filepaths_in_csv_processed_once(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=1)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a"}, {"Filepath": "series_a"}],  # duplicate
        )
        out_dir = tmp_path / "out"

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=([dict(_FAKE_ENCODED)], {}),
        ) as mock_enc:
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )

        assert mock_enc.call_count == 1

    def test_output_folder_created_if_missing(self, tmp_path):
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=1)
        label_csv = _make_label_csv(tmp_path, [{"Filepath": "series_a"}])
        out_dir = tmp_path / "new" / "nested" / "dir"
        assert not out_dir.exists()

        with patch(
            "IMC.data.encode_metadata.generate_slicewise_metadata_vector",
            return_value=([dict(_FAKE_ENCODED)], {}),
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )

        assert out_dir.exists()
        assert (out_dir / "out.parquet").exists()

    def test_uid_filter_all_rejected_series_skipped(self, tmp_path):
        """If UID filtering removes all slices, the series is skipped."""
        _make_series_dir(tmp_path / "data", "series_a", n_dcm=2)
        label_csv = _make_label_csv(
            tmp_path,
            [{"Filepath": "series_a", "SeriesInstanceUID": "1.2.3"}],
        )
        out_dir = tmp_path / "out"

        with (
            patch(
                "IMC.data.encode_metadata._filter_slices_by_uid",
                return_value=[],
            ),
            patch("IMC.data.encode_metadata.generate_slicewise_metadata_vector") as mock_enc,
        ):
            self._run_main(
                [
                    "--label_file",
                    str(label_csv),
                    "--data_folder",
                    str(tmp_path / "data"),
                    "--output_folder",
                    str(out_dir),
                    "--output_name",
                    "out.parquet",
                    "--checkpoint_every",
                    "0",
                ]
            )
            mock_enc.assert_not_called()
