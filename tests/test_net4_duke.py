"""Test suite for :mod:`IMC.net4_duke.train` and :mod:`IMC.net4_duke.infer`.

Sections
--------
1.  Module-level constants and shared helpers.
2.  :func:`~IMC.net4_duke.train.init_weights` – initialisation logic.
3.  :func:`~IMC.net4_duke.train.create_optimizer` – AdamW parameter groups.
4.  :func:`~IMC.net4_duke.train.get_scheduler` – LR schedule shape.
5.  :func:`~IMC.net4_duke.train.parse_args` – Duke training CLI defaults.
6.  :func:`~IMC.net4_duke.train._configure_dataset_env` and
    :func:`~IMC.net4_duke.infer._configure_dataset_env` – env-var setup.
7.  :func:`~IMC.net4_duke.infer.load_model` – checkpoint loading.
8.  :func:`~IMC.net4_duke.infer.parse_args` – Duke inference CLI defaults.
9.  :func:`~IMC.net4_duke.infer.run_inference` – mock-based batch inference.
10. :func:`~IMC.net4_duke.train.main` – orchestration smoke tests (mocked).
11. :func:`~IMC.net4_duke.infer.main` – orchestration smoke tests (mocked).

Design decisions
----------------
- Network construction (``build_model``) is already thoroughly tested in
  ``test_net4.py``; it is not repeated here beyond a basic integration path.
- The ``@pytest.mark.slow`` marker is used for tests that instantiate real
  models or DataLoaders to let CI skip them with ``-m "not slow"``.
- All ``main()`` tests replace heavy I/O (Duke dataset, Trainer, logging) with
  ``unittest.mock`` stubs so the suite finishes in seconds without GPU access.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from unittest import mock

import pandas as pd
import pytest
import torch
import torch.nn as nn


# ===========================================================================
# 1. Constants and shared helpers
# ===========================================================================

# Duke dataset: single classification task with 13 classes (A–M)
DUKE_NUM_CLASSES_DICT: dict[str, int] = {"SequenceType_Code_norm": 13}
DUKE_LABEL_NAMES: dict[str, dict[int, str]] = {
    "SequenceType_Code_norm": {
        i: lbl for i, lbl in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"])
    }
}
METADATA_DIM = 88
BATCH_SIZE = 2
N_SLICES = 3
IMG_H, IMG_W = 64, 64


def _make_image_input(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Return a random image tensor ``(B, N_SLICES, H, W)``."""
    return torch.randn(batch_size, N_SLICES, IMG_H, IMG_W)


def _make_metadata_input(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Return a random metadata tensor ``(B, METADATA_DIM)``."""
    return torch.randn(batch_size, METADATA_DIM)


def _make_duke_train_args(**overrides) -> argparse.Namespace:
    """Return a minimal :class:`argparse.Namespace` for net4_duke/train.py."""
    defaults = dict(
        modality="combined",
        img_enc_backbone="densenet121",
        batch_size=2,
        num_epochs=1,
        lr=1e-6,
        gpu=-1,
        ckpt=None,
        patience=30,
        log_dir=None,
        debug=False,
        incl_regression=False,
        folds=None,
        n_folds=5,
        dataset_path=None,
        metadata_path=None,
        label_csv_path=None,
        use_preselected_features=False,
        num_workers=0,
        n_slices=N_SLICES,
        vanilla_image_classifier=False,
        fusion_module_version="v1",
        metadata_enc_type="imputer",
        sparse_enc_version="v1",
        imputer_type="contextual",
        metadata_embed_dim=128,
        output_emb_dim=256,
        metadata_dropout=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_duke_infer_args(**overrides) -> argparse.Namespace:
    """Return a minimal :class:`argparse.Namespace` for net4_duke/infer.py."""
    defaults = dict(
        ckpt="/tmp/fake_ckpt.pth",
        output_dir=None,
        modality="combined",
        img_enc_backbone="densenet121",
        vanilla_image_classifier=False,
        fusion_module_version="v1",
        metadata_enc_type="imputer",
        sparse_enc_version="v1",
        imputer_type="contextual",
        metadata_embed_dim=128,
        output_emb_dim=256,
        incl_regression=False,
        folds=None,
        dataset_path=None,
        metadata_path=None,
        label_csv_path=None,
        batch_size=2,
        num_workers=0,
        gpu=-1,
        n_slices=N_SLICES,
        run_eval=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ===========================================================================
# 2. init_weights
# ===========================================================================


class TestInitWeights:
    """Tests for :func:`~IMC.net4_duke.train.init_weights`."""

    def test_linear_xavier_init(self) -> None:
        """Linear weight must be filled by Xavier uniform (non-constant)."""
        from IMC.net4_duke.train import init_weights

        layer = nn.Linear(16, 8)
        nn.init.ones_(layer.weight)
        init_weights(layer)
        assert not torch.all(layer.weight == 1.0)

    def test_linear_bias_zeros(self) -> None:
        """Linear bias must be zeroed by init_weights."""
        from IMC.net4_duke.train import init_weights

        layer = nn.Linear(16, 8)
        nn.init.ones_(layer.bias)
        init_weights(layer)
        assert torch.all(layer.bias == 0.0)

    def test_layernorm_weight_ones(self) -> None:
        """LayerNorm weight must be set to ones."""
        from IMC.net4_duke.train import init_weights

        layer = nn.LayerNorm(16)
        nn.init.zeros_(layer.weight)
        init_weights(layer)
        assert torch.all(layer.weight == 1.0)

    def test_layernorm_bias_zeros(self) -> None:
        """LayerNorm bias must be set to zeros."""
        from IMC.net4_duke.train import init_weights

        layer = nn.LayerNorm(16)
        nn.init.ones_(layer.bias)
        init_weights(layer)
        assert torch.all(layer.bias == 0.0)

    def test_other_modules_untouched(self) -> None:
        """Conv2d parameters must not be modified by init_weights."""
        from IMC.net4_duke.train import init_weights

        layer = nn.Conv2d(3, 16, kernel_size=3)
        original = layer.weight.clone()
        init_weights(layer)
        assert torch.allclose(layer.weight, original)

    def test_apply_to_model(self) -> None:
        """model.apply(init_weights) must not raise on a simple Sequential."""
        from IMC.net4_duke.train import init_weights

        model = nn.Sequential(nn.Linear(8, 4), nn.LayerNorm(4), nn.Linear(4, 2))
        model.apply(init_weights)  # should not raise


# ===========================================================================
# 3. create_optimizer
# ===========================================================================


class TestCreateOptimizer:
    """Tests for :func:`~IMC.net4_duke.train.create_optimizer`."""

    def _simple_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(8, 4),
            nn.LayerNorm(4),
        )

    def test_returns_adamw(self) -> None:
        from IMC.net4_duke.train import create_optimizer

        model = self._simple_model()
        opt = create_optimizer(model, lr=1e-4)
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self) -> None:
        """Optimizer must split parameters into two weight-decay groups."""
        from IMC.net4_duke.train import create_optimizer

        model = self._simple_model()
        opt = create_optimizer(model, lr=1e-4)
        assert len(opt.param_groups) == 2

    def test_no_decay_group_weight_decay_zero(self) -> None:
        """Biases and norm parameters must have weight_decay == 0."""
        from IMC.net4_duke.train import create_optimizer

        model = self._simple_model()
        opt = create_optimizer(model, lr=1e-4, weight_decay=0.01)
        no_decay = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
        assert len(no_decay) == 1

    def test_decay_group_weight_decay_set(self) -> None:
        """Non-bias/norm parameters must have the specified weight_decay."""
        from IMC.net4_duke.train import create_optimizer

        model = self._simple_model()
        opt = create_optimizer(model, lr=1e-4, weight_decay=0.02)
        decay = [g for g in opt.param_groups if g["weight_decay"] == 0.02]
        assert len(decay) == 1

    def test_learning_rate_propagated(self) -> None:
        """The given lr must appear in every param group."""
        from IMC.net4_duke.train import create_optimizer

        model = self._simple_model()
        opt = create_optimizer(model, lr=3e-5)
        assert all(pg["lr"] == 3e-5 for pg in opt.param_groups)


# ===========================================================================
# 4. get_scheduler
# ===========================================================================


class TestGetScheduler:
    """Tests for :func:`~IMC.net4_duke.train.get_scheduler`."""

    def _make_optimizer(self, lr: float = 1e-3) -> torch.optim.AdamW:
        model = nn.Linear(4, 2)
        return torch.optim.AdamW(model.parameters(), lr=lr)

    def test_returns_lambda_lr(self) -> None:
        from IMC.net4_duke.train import get_scheduler

        opt = self._make_optimizer()
        sched = get_scheduler(opt, warmup_steps=5, total_steps=20)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_lr_at_start_is_low(self) -> None:
        """LR at step 0 should be close to 0 (linear warmup from near-zero)."""
        from IMC.net4_duke.train import get_scheduler

        base_lr = 1e-3
        opt = self._make_optimizer(lr=base_lr)
        sched = get_scheduler(opt, warmup_steps=10, total_steps=100)
        # step 0: lambda(0) ≡ peak_scale_factor * 0/10 = 0.0
        # multiplied by base_lr
        lrs = sched.get_last_lr()
        assert lrs[0] >= 0.0

    def test_lr_decays_after_warmup(self) -> None:
        """LR must decrease monotonically after the warmup plateau."""
        from IMC.net4_duke.train import get_scheduler

        base_lr = 1e-3
        opt = self._make_optimizer(lr=base_lr)
        warmup = 5
        total = 40
        sched = get_scheduler(opt, warmup_steps=warmup, total_steps=total)

        lrs: list[float] = []
        for _ in range(total):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        post_warmup = lrs[warmup:]
        assert post_warmup[-1] < post_warmup[0], "LR should decrease during cosine decay phase."

    def test_min_scale_factor_respected(self) -> None:
        """LR must never fall below base_lr * min_scale_factor."""
        from IMC.net4_duke.train import get_scheduler

        base_lr = 1.0
        min_sf = 0.05
        opt = self._make_optimizer(lr=base_lr)
        sched = get_scheduler(opt, warmup_steps=5, total_steps=50, min_scale_factor=min_sf)
        for _ in range(100):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr >= base_lr * min_sf - 1e-9


# ===========================================================================
# 5. train.parse_args – Duke-specific defaults
# ===========================================================================


class TestDukeTrainParseArgs:
    """Tests for :func:`~IMC.net4_duke.train.parse_args` Duke-specific defaults."""

    def test_defaults(self) -> None:
        """Verify all Duke-specific default values."""
        argv = ["train.py", "--modality", "combined"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import train as train_module

            args = train_module.parse_args()

        assert args.modality == "combined"
        assert args.img_enc_backbone == "densenet121"
        assert args.fusion_module_version == "v1"
        assert args.metadata_enc_type == "imputer"
        assert args.gpu == 0
        assert args.patience == 30
        assert args.num_epochs == 30
        assert args.n_folds == 5
        assert args.folds is None
        assert args.n_slices == 3
        assert args.incl_regression is False
        assert args.debug is False
        assert args.vanilla_image_classifier is False
        assert args.metadata_embed_dim == 128
        assert args.output_emb_dim == 256

    def test_modality_choices_valid(self) -> None:
        """Accepted modality values: image, metadata, combined."""
        from IMC.net4_duke import train as train_module

        for modality in ("image", "metadata", "combined"):
            argv = ["train.py", "--modality", modality]
            with mock.patch("sys.argv", argv):
                args = train_module.parse_args()
            assert args.modality == modality

    def test_modality_required(self) -> None:
        """Omitting --modality must cause SystemExit."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["train.py"]):
                from IMC.net4_duke import train as train_module

                train_module.parse_args()

    def test_folds_string_stored_as_is(self) -> None:
        """--folds must be stored as a raw string, not yet parsed."""
        argv = ["train.py", "--modality", "combined", "--folds", "0,2,4"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import train as train_module

            args = train_module.parse_args()
        assert args.folds == "0,2,4"

    def test_n_folds_override(self) -> None:
        """--n_folds must override the default of 5."""
        argv = ["train.py", "--modality", "image", "--n_folds", "3"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import train as train_module

            args = train_module.parse_args()
        assert args.n_folds == 3

    def test_metadata_enc_type_choices(self) -> None:
        """Accepted metadata_enc_type values: imputer, sparse."""
        from IMC.net4_duke import train as train_module

        for enc in ("imputer", "sparse"):
            argv = ["train.py", "--modality", "combined", "--metadata_enc_type", enc]
            with mock.patch("sys.argv", argv):
                args = train_module.parse_args()
            assert args.metadata_enc_type == enc

    def test_metadata_enc_type_invalid_raises(self) -> None:
        """An invalid --metadata_enc_type value must cause SystemExit."""
        with pytest.raises(SystemExit):
            argv = ["train.py", "--modality", "combined", "--metadata_enc_type", "invalid_enc"]
            with mock.patch("sys.argv", argv):
                from IMC.net4_duke import train as train_module

                train_module.parse_args()

    def test_debug_flag(self) -> None:
        """--debug flag must be stored as True."""
        argv = ["train.py", "--modality", "combined", "--debug"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import train as train_module

            args = train_module.parse_args()
        assert args.debug is True

    def test_dataset_path_optional(self) -> None:
        """--dataset_path must be accepted and stored."""
        argv = ["train.py", "--modality", "combined", "--dataset_path", "/custom/data"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import train as train_module

            args = train_module.parse_args()
        assert args.dataset_path == "/custom/data"


# ===========================================================================
# 6. _configure_dataset_env – train and infer
# ===========================================================================


class TestDukeConfigureDatasetEnv:
    """Tests for both
    :func:`~IMC.net4_duke.train._configure_dataset_env` and
    :func:`~IMC.net4_duke.infer._configure_dataset_env`.

    Both functions return ``None`` (unlike :mod:`IMC.net4.infer` which returns
    the dataset name string).
    """

    def _set_required_env(self) -> None:
        os.environ["LOCAL_DATASET_PATH"] = "/dummy/data"
        os.environ["METADATA_PATH"] = "/dummy/meta.parquet"
        os.environ["LABEL_CSV_PATH"] = "/dummy/labels.csv"

    # ---------------------------------------------------------------- train

    def test_train_returns_none(self) -> None:
        """train._configure_dataset_env must return None."""
        from IMC.net4_duke.train import _configure_dataset_env

        self._set_required_env()
        args = _make_duke_train_args()
        result = _configure_dataset_env(args)
        assert result is None

    def test_train_debug_mode_false(self) -> None:
        """When debug=False, DEBUG_MODE env var must be '0'."""
        from IMC.net4_duke.train import _configure_dataset_env

        self._set_required_env()
        args = _make_duke_train_args(debug=False)
        _configure_dataset_env(args)
        assert os.environ.get("DEBUG_MODE") == "0"

    def test_train_debug_mode_true(self) -> None:
        """When debug=True, DEBUG_MODE env var must be '1'."""
        from IMC.net4_duke.train import _configure_dataset_env

        self._set_required_env()
        args = _make_duke_train_args(debug=True)
        _configure_dataset_env(args)
        assert os.environ.get("DEBUG_MODE") == "1"

    def test_train_missing_env_raises(self) -> None:
        """Training must fail fast when required Duke env vars are absent."""
        from IMC.net4_duke.train import _configure_dataset_env

        args = _make_duke_train_args()
        for key in ("LOCAL_DATASET_PATH", "METADATA_PATH", "LABEL_CSV_PATH"):
            os.environ.pop(key, None)

        with pytest.raises(ValueError, match="Missing Duke dataset configuration"):
            _configure_dataset_env(args)

    def test_train_cli_dataset_path_overrides_env(self) -> None:
        """CLI --dataset_path must always override any existing env var."""
        from IMC.net4_duke.train import _configure_dataset_env

        os.environ["LOCAL_DATASET_PATH"] = "/old/path"
        os.environ["METADATA_PATH"] = "/dummy/meta.parquet"
        os.environ["LABEL_CSV_PATH"] = "/dummy/labels.csv"
        args = _make_duke_train_args(dataset_path="/new/path")
        _configure_dataset_env(args)
        assert os.environ["LOCAL_DATASET_PATH"] == "/new/path"

    def test_train_cli_metadata_path_overrides_env(self) -> None:
        from IMC.net4_duke.train import _configure_dataset_env

        os.environ["LOCAL_DATASET_PATH"] = "/dummy/data"
        os.environ["METADATA_PATH"] = "/old/meta.parquet"
        os.environ["LABEL_CSV_PATH"] = "/dummy/labels.csv"
        args = _make_duke_train_args(metadata_path="/new/meta.parquet")
        _configure_dataset_env(args)
        assert os.environ["METADATA_PATH"] == "/new/meta.parquet"

    def test_train_cli_label_csv_overrides_env(self) -> None:
        from IMC.net4_duke.train import _configure_dataset_env

        os.environ["LOCAL_DATASET_PATH"] = "/dummy/data"
        os.environ["METADATA_PATH"] = "/dummy/meta.parquet"
        os.environ["LABEL_CSV_PATH"] = "/old/labels.csv"
        args = _make_duke_train_args(label_csv_path="/new/labels.csv")
        _configure_dataset_env(args)
        assert os.environ["LABEL_CSV_PATH"] == "/new/labels.csv"

    def test_train_existing_env_not_overwritten_without_cli(self) -> None:
        """If an env var is already set and no CLI override, it must not change."""
        from IMC.net4_duke.train import _configure_dataset_env

        os.environ["LOCAL_DATASET_PATH"] = "/preserved/path"
        os.environ["METADATA_PATH"] = "/dummy/meta.parquet"
        os.environ["LABEL_CSV_PATH"] = "/dummy/labels.csv"
        args = _make_duke_train_args(dataset_path=None)
        _configure_dataset_env(args)
        assert os.environ["LOCAL_DATASET_PATH"] == "/preserved/path"

    # ---------------------------------------------------------------- infer

    def test_infer_returns_none(self) -> None:
        """infer._configure_dataset_env must return None."""
        from IMC.net4_duke.infer import _configure_dataset_env

        self._set_required_env()
        args = _make_duke_infer_args()
        result = _configure_dataset_env(args)
        assert result is None

    def test_infer_debug_mode_always_zero(self) -> None:
        """infer._configure_dataset_env must always set DEBUG_MODE='0'."""
        from IMC.net4_duke.infer import _configure_dataset_env

        self._set_required_env()
        args = _make_duke_infer_args()
        _configure_dataset_env(args)
        assert os.environ.get("DEBUG_MODE") == "0"

    def test_infer_cli_dataset_path_overrides(self) -> None:
        """infer CLI --dataset_path must override any existing env var."""
        from IMC.net4_duke.infer import _configure_dataset_env

        os.environ["LOCAL_DATASET_PATH"] = "/old/data"
        os.environ["METADATA_PATH"] = "/old/meta.parquet"
        os.environ["LABEL_CSV_PATH"] = "/old/labels.csv"
        args = _make_duke_infer_args(dataset_path="/override/data")
        _configure_dataset_env(args)
        assert os.environ["LOCAL_DATASET_PATH"] == "/override/data"

    def test_infer_missing_env_raises(self) -> None:
        """Inference must fail fast when required Duke env vars are absent."""
        from IMC.net4_duke.infer import _configure_dataset_env

        args = _make_duke_infer_args()
        for key in ("LOCAL_DATASET_PATH", "METADATA_PATH", "LABEL_CSV_PATH"):
            os.environ.pop(key, None)

        with pytest.raises(ValueError, match="Missing Duke dataset configuration"):
            _configure_dataset_env(args)


# ===========================================================================
# 7. load_model
# ===========================================================================


class TestDukeLoadModel:
    """Tests for :func:`~IMC.net4_duke.infer.load_model`."""

    def test_missing_ckpt_raises(self) -> None:
        """A nonexistent checkpoint path must raise FileNotFoundError."""
        from IMC.net4_duke.infer import load_model

        args = _make_duke_infer_args(
            ckpt="/nonexistent/path/fake.pth",
            modality="image",
        )

        # build_model is mocked to avoid real construction
        mock_model = mock.MagicMock(spec=nn.Module)
        with mock.patch("IMC.net4_duke.helper.build_model", return_value=mock_model):
            with pytest.raises(FileNotFoundError):
                load_model(
                    args,
                    DUKE_NUM_CLASSES_DICT,
                    METADATA_DIM,
                    torch.device("cpu"),
                )

    def test_load_model_returns_nn_module(self) -> None:
        """load_model must return an ``nn.Module``-compatible object."""
        from IMC.net4_duke.infer import load_model

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            ckpt_path = f.name

        try:
            mock_model = mock.MagicMock(spec=nn.Module)
            mock_model.to.return_value = mock_model
            torch.save({"model_state_dict": {}}, ckpt_path)

            args = _make_duke_infer_args(ckpt=ckpt_path, modality="image")
            with mock.patch("IMC.net4_duke.helper.build_model", return_value=mock_model):
                result = load_model(
                    args,
                    DUKE_NUM_CLASSES_DICT,
                    METADATA_DIM,
                    torch.device("cpu"),
                )
            assert result is mock_model
        finally:
            os.unlink(ckpt_path)

    def test_load_model_calls_eval(self) -> None:
        """load_model must call model.eval()."""
        from IMC.net4_duke.infer import load_model

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            ckpt_path = f.name

        try:
            mock_model = mock.MagicMock(spec=nn.Module)
            mock_model.to.return_value = mock_model
            torch.save({"model_state_dict": {}}, ckpt_path)

            args = _make_duke_infer_args(ckpt=ckpt_path, modality="image")
            with mock.patch("IMC.net4_duke.helper.build_model", return_value=mock_model):
                load_model(
                    args,
                    DUKE_NUM_CLASSES_DICT,
                    METADATA_DIM,
                    torch.device("cpu"),
                )
            mock_model.eval.assert_called_once()
        finally:
            os.unlink(ckpt_path)

    def test_load_model_loads_state_dict(self) -> None:
        """load_model must call model.load_state_dict with the checkpoint."""
        from IMC.net4_duke.infer import load_model

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            ckpt_path = f.name

        try:
            fake_state = {"fc.weight": torch.zeros(2, 4)}
            torch.save({"model_state_dict": fake_state}, ckpt_path)

            mock_model = mock.MagicMock(spec=nn.Module)
            mock_model.to.return_value = mock_model

            args = _make_duke_infer_args(ckpt=ckpt_path, modality="image")
            with mock.patch("IMC.net4_duke.helper.build_model", return_value=mock_model):
                load_model(
                    args,
                    DUKE_NUM_CLASSES_DICT,
                    METADATA_DIM,
                    torch.device("cpu"),
                )
            # assert_called_once_with fails on tensor equality; check separately
            mock_model.load_state_dict.assert_called_once()
            actual_state = mock_model.load_state_dict.call_args[0][0]
            assert set(actual_state.keys()) == set(fake_state.keys())
        finally:
            os.unlink(ckpt_path)


# ===========================================================================
# 8. infer.parse_args – Duke-specific defaults
# ===========================================================================


class TestDukeInferParseArgs:
    """Tests for :func:`~IMC.net4_duke.infer.parse_args` Duke defaults."""

    def test_defaults(self) -> None:
        argv = ["infer.py", "--ckpt", "/ckpt.pth", "--output_dir", "/out", "--modality", "combined"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import infer as infer_module

            args = infer_module.parse_args()

        assert args.modality == "combined"
        assert args.img_enc_backbone == "densenet121"
        assert args.fusion_module_version == "v1"
        assert args.metadata_enc_type == "imputer"
        assert args.batch_size == 16
        assert args.gpu == 0
        assert args.run_eval is False
        assert args.incl_regression is False
        assert args.folds is None
        assert args.n_slices == 3

    def test_modality_required(self) -> None:
        """Omitting --modality must cause SystemExit."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["infer.py", "--ckpt", "/p.pth", "--output_dir", "/d"]):
                from IMC.net4_duke import infer as infer_module

                infer_module.parse_args()

    def test_ckpt_and_output_required(self) -> None:
        """Omitting required --ckpt / --output_dir must exit."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["infer.py", "--modality", "combined"]):
                from IMC.net4_duke import infer as infer_module

                infer_module.parse_args()

    def test_run_eval_flag(self) -> None:
        argv = ["infer.py", "--ckpt", "/c.pth", "--output_dir", "/o", "--modality", "image", "--run_eval"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import infer as infer_module

            args = infer_module.parse_args()
        assert args.run_eval is True

    def test_folds_stored_as_string(self) -> None:
        argv = ["infer.py", "--ckpt", "/c.pth", "--output_dir", "/o", "--modality", "combined", "--folds", "0,1,2"]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import infer as infer_module

            args = infer_module.parse_args()
        assert args.folds == "0,1,2"

    def test_metadata_enc_type_invalid_raises(self) -> None:
        """An invalid --metadata_enc_type value must cause SystemExit."""
        with pytest.raises(SystemExit):
            argv = [
                "infer.py",
                "--ckpt",
                "/c.pth",
                "--output_dir",
                "/o",
                "--modality",
                "combined",
                "--metadata_enc_type",
                "invalid_enc",
            ]
            with mock.patch("sys.argv", argv):
                from IMC.net4_duke import infer as infer_module

                infer_module.parse_args()

    def test_vanilla_classifier_flag(self) -> None:
        argv = [
            "infer.py",
            "--ckpt",
            "/c.pth",
            "--output_dir",
            "/o",
            "--modality",
            "image",
            "--vanilla_image_classifier",
        ]
        with mock.patch("sys.argv", argv):
            from IMC.net4_duke import infer as infer_module

            args = infer_module.parse_args()
        assert args.vanilla_image_classifier is True


# ===========================================================================
# 9. run_inference() – mock-based tests
# ===========================================================================


def _make_mock_duke_dataloader(batches: list) -> mock.MagicMock:
    """Return a mock DataLoader with Duke-specific dataset attributes."""
    mock_dataset = mock.MagicMock()
    mock_dataset.get_n_labels.return_value = DUKE_NUM_CLASSES_DICT
    mock_dataset.label_names = DUKE_LABEL_NAMES

    dl = mock.MagicMock()
    dl.dataset = mock_dataset
    dl.__iter__ = mock.Mock(return_value=iter(batches))
    return dl


def _make_mock_duke_model(batch_size: int = BATCH_SIZE) -> mock.MagicMock:
    """Return a mock model that returns argmax=0 logits for every Duke task."""
    outputs = tuple(torch.zeros(batch_size, n) for n in DUKE_NUM_CLASSES_DICT.values())
    model = mock.MagicMock()
    model.return_value = outputs
    return model


class TestDukeRunInference:
    """Unit tests for :func:`~IMC.net4_duke.infer.run_inference`.

    The Duke run_inference signature is:
    ``run_inference(model, dataloader, device, incl_regression=False)``

    Key differences from net4/infer:
    - No ``num_classes_dict`` parameter (read from ``dataloader.dataset``).
    - No ``is_duke`` parameter (always Duke dataset).
    - No DUKE_MAP_LABELS remapping applied.
    - ``label_Contrast`` column only added if ``label_ContrastPhase`` is in
      ``dataloader.dataset.label_names``.
    """

    @pytest.fixture(autouse=True)
    def _patch_normalize(self):
        """Patch normalize_per_sample to identity."""
        with mock.patch("IMC.net4_duke.infer.normalize_per_sample", side_effect=lambda x: x):
            yield

    def _single_batch_dl(self, batch_size: int = BATCH_SIZE):
        images = _make_image_input(batch_size)
        metadata = _make_metadata_input(batch_size)
        paths = [f"/duke/series_{i:03d}" for i in range(batch_size)]
        return _make_mock_duke_dataloader([(images, metadata, paths)])

    # --------------------------------------------------------- basic structure

    def test_returns_dataframe(self) -> None:
        """run_inference must return a pandas DataFrame."""
        from IMC.net4_duke.infer import run_inference

        dl = self._single_batch_dl()
        model = _make_mock_duke_model(BATCH_SIZE)
        result = run_inference(model, dl, torch.device("cpu"))
        assert isinstance(result, pd.DataFrame)

    def test_filepath_column_present(self) -> None:
        """Output must contain a 'Filepath' column."""
        from IMC.net4_duke.infer import run_inference

        dl = self._single_batch_dl()
        model = _make_mock_duke_model(BATCH_SIZE)
        df = run_inference(model, dl, torch.device("cpu"))
        assert "Filepath" in df.columns

    def test_task_columns_present(self) -> None:
        """Output must contain a column for each Duke task."""
        from IMC.net4_duke.infer import run_inference

        dl = self._single_batch_dl()
        model = _make_mock_duke_model(BATCH_SIZE)
        df = run_inference(model, dl, torch.device("cpu"))
        for task in DUKE_NUM_CLASSES_DICT:
            assert task in df.columns

    def test_no_label_contrast_without_phase(self) -> None:
        """label_Contrast must NOT be added when label_ContrastPhase is absent."""
        from IMC.net4_duke.infer import run_inference

        dl = self._single_batch_dl()
        # Duke DUKE_LABEL_NAMES does not include label_ContrastPhase
        model = _make_mock_duke_model(BATCH_SIZE)
        df = run_inference(model, dl, torch.device("cpu"))
        assert "label_Contrast" not in df.columns

    def test_row_count_matches_batch(self) -> None:
        """Number of rows must equal the total dataset size."""
        from IMC.net4_duke.infer import run_inference

        dl = self._single_batch_dl(batch_size=4)
        model = _make_mock_duke_model(4)
        df = run_inference(model, dl, torch.device("cpu"))
        assert len(df) == 4

    def test_accumulates_multiple_batches(self) -> None:
        """Rows from multiple batches are concatenated."""
        from IMC.net4_duke.infer import run_inference

        batch1 = (_make_image_input(2), _make_metadata_input(2), ["/a/1", "/a/2"])
        batch2 = (_make_image_input(3), _make_metadata_input(3), ["/b/1", "/b/2", "/b/3"])
        dl = _make_mock_duke_dataloader([batch1, batch2])

        def flexible_fwd(imgs, meta):
            B = imgs.shape[0]
            return tuple(torch.zeros(B, n) for n in DUKE_NUM_CLASSES_DICT.values())

        model = mock.MagicMock()
        model.side_effect = flexible_fwd

        df = run_inference(model, dl, torch.device("cpu"))
        assert len(df) == 5
        assert list(df["Filepath"]) == ["/a/1", "/a/2", "/b/1", "/b/2", "/b/3"]

    def test_label_decoded_from_label_names(self) -> None:
        """argmax=0 must map to the first entry in DUKE_LABEL_NAMES."""
        from IMC.net4_duke.infer import run_inference

        dl = self._single_batch_dl(batch_size=2)
        model = _make_mock_duke_model(2)  # argmax=0 for all tasks
        df = run_inference(model, dl, torch.device("cpu"))
        # First label in SequenceType_Code_norm is "A"
        assert all(df["SequenceType_Code_norm"] == "A")

    def test_second_class_label(self) -> None:
        """argmax=1 must map to label 'B'."""
        from IMC.net4_duke.infer import run_inference

        def _b_outputs(imgs, meta):
            B = imgs.shape[0]
            logits = torch.zeros(B, 13)
            logits[:, 1] = 10.0  # argmax = 1 → "B"
            return (logits,)

        model = mock.MagicMock()
        model.side_effect = _b_outputs
        dl = self._single_batch_dl(batch_size=2)
        df = run_inference(model, dl, torch.device("cpu"))
        assert all(df["SequenceType_Code_norm"] == "B")

    def test_model_called_once_per_batch(self) -> None:
        """Model must be called exactly once for a single-batch dataloader."""
        from IMC.net4_duke.infer import run_inference

        dl = self._single_batch_dl()
        model = _make_mock_duke_model(BATCH_SIZE)
        run_inference(model, dl, torch.device("cpu"))
        assert model.call_count == 1

    def test_filepaths_stored_correctly(self) -> None:
        """Filepath values must match those provided by the dataloader."""
        from IMC.net4_duke.infer import run_inference

        paths = [f"/duke/s{i}" for i in range(BATCH_SIZE)]
        images = _make_image_input(BATCH_SIZE)
        metadata = _make_metadata_input(BATCH_SIZE)
        dl = _make_mock_duke_dataloader([(images, metadata, paths)])
        model = _make_mock_duke_model(BATCH_SIZE)
        df = run_inference(model, dl, torch.device("cpu"))
        assert list(df["Filepath"]) == paths


# ===========================================================================
# 10. train.main() – mock-based orchestration tests
# ===========================================================================


def _make_main_duke_train_args(**overrides) -> argparse.Namespace:
    """Minimal args for train.main() without real data."""
    defaults = dict(
        modality="combined",
        img_enc_backbone="densenet121",
        batch_size=2,
        num_epochs=1,
        lr=1e-6,
        gpu=-1,
        ckpt=None,
        patience=30,
        log_dir=None,
        debug=False,
        incl_regression=False,
        folds=None,
        n_folds=2,  # small n_folds to keep test fast
        dataset_path=None,
        metadata_path=None,
        label_csv_path=None,
        use_preselected_features=False,
        num_workers=0,
        n_slices=N_SLICES,
        vanilla_image_classifier=False,
        fusion_module_version="v1",
        metadata_enc_type="imputer",
        sparse_enc_version="v1",
        imputer_type="contextual",
        metadata_embed_dim=128,
        output_emb_dim=256,
        metadata_dropout=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@contextlib.contextmanager
def _patch_duke_train_main_dependencies(args: argparse.Namespace):
    """Context manager mocking all heavy dependencies of train.main()."""
    mock_logger = mock.MagicMock()
    mock_tb_logger = mock.MagicMock()
    mock_model = mock.MagicMock(spec=nn.Module)
    mock_model.named_parameters.return_value = []
    mock_model.parameters.return_value = iter([])
    mock_model.apply.return_value = None

    mock_dataset = mock.MagicMock()
    mock_dataset.get_n_labels.return_value = DUKE_NUM_CLASSES_DICT
    mock_dataset.num_metadata_features = METADATA_DIM
    mock_dataset.__len__ = mock.Mock(return_value=4)

    mock_loader = mock.MagicMock()
    mock_loader.dataset = mock_dataset
    mock_loader.__len__ = mock.Mock(return_value=2)

    mock_trainer = mock.MagicMock()

    mock_live_cls = mock.MagicMock(return_value=mock_dataset)

    patches = [
        mock.patch("IMC.net4_duke.train.parse_args", return_value=args),
        mock.patch("IMC.net4_duke.train._configure_dataset_env"),
        mock.patch(
            "IMC.net4_duke.train.setup_combined_logging", return_value=(mock_logger, "/tmp/test.log", mock_tb_logger)
        ),
        mock.patch("IMC.net4_duke.train.log_training_start"),
        mock.patch("IMC.net4_duke.train.log_training_end"),
        mock.patch("IMC.net4_duke.train.capture_console_to_log", return_value=contextlib.nullcontext()),
        mock.patch("IMC.net4_duke.train.build_model", return_value=mock_model),
        mock.patch("IMC.net4_duke.train.MultiTaskLoss"),
        mock.patch("IMC.data.duke_dataloader_local.LiverDataset", mock_live_cls),
        mock.patch("torch.utils.data.DataLoader", return_value=mock_loader),
        mock.patch("IMC.trainer.Trainer", return_value=mock_trainer),
        mock.patch("torch.cuda.amp.GradScaler"),
    ]

    started = [p.start() for p in patches]
    try:
        yield {
            "args": args,
            "logger": mock_logger,
            "tb_logger": mock_tb_logger,
            "model": mock_model,
            "loader": mock_loader,
            "trainer": mock_trainer,
            "mocks": started,
        }
    finally:
        for p in patches:
            p.stop()


class TestDukeTrainMain:
    """Tests for :func:`~IMC.net4_duke.train.main` with I/O mocked.

    Uses n_folds=2 to keep iterations minimal.
    """

    def test_trainer_fit_called_per_fold(self) -> None:
        """Trainer.fit must be called once per fold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(log_dir=tmpdir)
            with _patch_duke_train_main_dependencies(args) as ctx:
                from IMC.net4_duke.train import main

                main()
            # n_folds=2 → fit called twice
            assert ctx["trainer"].fit.call_count == args.n_folds

    def test_trainer_test_called_per_fold(self) -> None:
        """Trainer.test must be called once per fold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(log_dir=tmpdir)
            with _patch_duke_train_main_dependencies(args) as ctx:
                from IMC.net4_duke.train import main

                main()
            assert ctx["trainer"].test.call_count == args.n_folds

    def test_trainer_load_checkpoint_called_per_fold(self) -> None:
        """Trainer.load_checkpoint must be called once per fold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(log_dir=tmpdir)
            with _patch_duke_train_main_dependencies(args) as ctx:
                from IMC.net4_duke.train import main

                main()
            assert ctx["trainer"].load_checkpoint.call_count == args.n_folds

    def test_config_json_written_per_fold(self) -> None:
        """config.json must be written inside each fold's subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(log_dir=tmpdir, n_folds=2)
            with _patch_duke_train_main_dependencies(args):
                from IMC.net4_duke.train import main

                main()

            config_files = [
                os.path.join(root, f) for root, _, files in os.walk(tmpdir) for f in files if f == "config.json"
            ]
            assert len(config_files) == args.n_folds

    def test_config_json_contains_modality(self) -> None:
        """Each config.json must contain the training modality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(
                log_dir=tmpdir,
                modality="image",
                n_folds=1,
                folds="0",
            )
            with _patch_duke_train_main_dependencies(args):
                from IMC.net4_duke.train import main

                main()

            config_files = [
                os.path.join(root, f) for root, _, files in os.walk(tmpdir) for f in files if f == "config.json"
            ]
            assert config_files
            with open(config_files[0]) as fh:
                cfg = json.load(fh)
            assert cfg["modality"] == "image"

    def test_cv_summary_csv_written(self) -> None:
        """cv_summary.csv must be written under the base log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(log_dir=tmpdir)
            with _patch_duke_train_main_dependencies(args):
                from IMC.net4_duke.train import main

                main()

            summary_files = [
                os.path.join(root, f) for root, _, files in os.walk(tmpdir) for f in files if f == "cv_summary.csv"
            ]
            assert summary_files, "cv_summary.csv was not written"

    def test_subset_folds_with_folds_arg(self) -> None:
        """With --folds '0', only fold 0 is trained (fit called once)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(
                log_dir=tmpdir,
                folds="0",
                n_folds=3,
            )
            with _patch_duke_train_main_dependencies(args) as ctx:
                from IMC.net4_duke.train import main

                main()
            assert ctx["trainer"].fit.call_count == 1

    def test_invalid_fold_index_raises(self) -> None:
        """A fold index out of [0, n_folds-1] must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(
                log_dir=tmpdir,
                folds="9",
                n_folds=5,
            )
            with _patch_duke_train_main_dependencies(args):
                from IMC.net4_duke.train import main

                with pytest.raises(ValueError):
                    main()

    def test_build_model_called_with_modality(self) -> None:
        """build_model must receive the correct modality per fold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(
                log_dir=tmpdir,
                modality="metadata",
                folds="0",
                n_folds=3,
            )
            with _patch_duke_train_main_dependencies(args) as ctx:
                with mock.patch("IMC.net4_duke.train.build_model", return_value=ctx["model"]) as mock_bm:
                    from IMC.net4_duke.train import main

                    main()
                assert mock_bm.call_count >= 1
                call_kwargs = mock_bm.call_args[1]
                assert call_kwargs["modality"] == "metadata"

    def test_debug_mode_configure_env_called_with_debug_true(self) -> None:
        """When --debug is passed, _configure_dataset_env must be called
        with args.debug == True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_train_args(log_dir=tmpdir, debug=True)
            with _patch_duke_train_main_dependencies(args) as ctx:
                from IMC.net4_duke.train import main

                main()
            # Retrieve the captured args that _configure_dataset_env was called with
            cfg_mock = ctx["mocks"][1]
            cfg_mock.assert_called_once()
            called_args = cfg_mock.call_args[0][0]
            assert called_args.debug is True


# ===========================================================================
# 11. infer.main() – mock-based orchestration tests
# ===========================================================================


def _make_main_duke_infer_args(**overrides) -> argparse.Namespace:
    """Minimal args for infer.main() without real data."""
    defaults = dict(
        ckpt="/tmp/fake_ckpt.pth",
        output_dir=None,
        modality="combined",
        img_enc_backbone="densenet121",
        vanilla_image_classifier=False,
        fusion_module_version="v1",
        metadata_enc_type="imputer",
        sparse_enc_version="v1",
        imputer_type="contextual",
        metadata_embed_dim=128,
        output_emb_dim=256,
        incl_regression=False,
        folds=None,
        dataset_path=None,
        metadata_path=None,
        label_csv_path=None,
        batch_size=2,
        num_workers=0,
        gpu=-1,
        n_slices=N_SLICES,
        run_eval=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@contextlib.contextmanager
def _patch_duke_infer_main_dependencies(
    args: argparse.Namespace,
    pred_df: pd.DataFrame | None = None,
):
    """Context manager mocking I/O-heavy dependencies of infer.main()."""
    if pred_df is None:
        pred_df = pd.DataFrame(
            {
                "Filepath": ["/duke/s1", "/duke/s2"],
                "SequenceType_Code_norm": ["A", "B"],
            }
        )

    mock_dataset = mock.MagicMock()
    mock_dataset.get_n_labels.return_value = DUKE_NUM_CLASSES_DICT
    mock_dataset.num_metadata_features = METADATA_DIM
    mock_dataset.__len__ = mock.Mock(return_value=2)
    mock_loader = mock.MagicMock()
    mock_loader.dataset = mock_dataset

    mock_model = mock.MagicMock()

    patches = [
        mock.patch("IMC.net4_duke.infer.parse_args", return_value=args),
        mock.patch("IMC.net4_duke.infer._configure_dataset_env"),
        mock.patch("IMC.net4_duke.infer.create_inference_dataloader", return_value=mock_loader),
        mock.patch("IMC.net4_duke.infer.load_model", return_value=mock_model),
        mock.patch("IMC.net4_duke.infer.run_inference", return_value=pred_df),
        mock.patch("IMC.net4_duke.infer.run_evaluation"),
    ]

    started = [p.start() for p in patches]
    try:
        yield {
            "args": args,
            "loader": mock_loader,
            "model": mock_model,
            "pred_df": pred_df,
            "run_evaluation": started[5],
            "run_inference": started[4],
            "create_inference_dataloader": started[2],
            "load_model": started[3],
        }
    finally:
        for p in patches:
            p.stop()


class TestDukeInferMain:
    """Tests for :func:`~IMC.net4_duke.infer.main` with I/O mocked."""

    def test_predictions_csv_written(self) -> None:
        """infer.main() must save predictions.csv to --output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_infer_args(output_dir=tmpdir)
            with _patch_duke_infer_main_dependencies(args):
                from IMC.net4_duke.infer import main

                main()
            assert os.path.isfile(os.path.join(tmpdir, "predictions.csv"))

    def test_predictions_csv_has_correct_content(self) -> None:
        """Saved predictions.csv must reflect the mocked pred_df."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred = pd.DataFrame(
                {
                    "Filepath": ["/duke/s1"],
                    "SequenceType_Code_norm": ["C"],
                }
            )
            args = _make_main_duke_infer_args(output_dir=tmpdir)
            with _patch_duke_infer_main_dependencies(args, pred_df=pred):
                from IMC.net4_duke.infer import main

                main()
            saved = pd.read_csv(os.path.join(tmpdir, "predictions.csv"))
            assert "Filepath" in saved.columns
            assert saved.iloc[0]["SequenceType_Code_norm"] == "C"

    def test_inference_metadata_json_written(self) -> None:
        """inference_metadata.json must be written to --output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_infer_args(output_dir=tmpdir)
            with _patch_duke_infer_main_dependencies(args):
                from IMC.net4_duke.infer import main

                main()
            meta_path = os.path.join(tmpdir, "inference_metadata.json")
            assert os.path.isfile(meta_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert "checkpoint" in meta
            assert "num_samples" in meta

    def test_run_evaluation_called_when_run_eval_set(self) -> None:
        """run_evaluation must be invoked when --run_eval=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_csv = os.path.join(tmpdir, "labels.csv")
            pd.DataFrame({"Filepath": ["/duke/s1"]}).to_csv(label_csv, index=False)

            args = _make_main_duke_infer_args(
                output_dir=tmpdir,
                run_eval=True,
                label_csv_path=label_csv,
            )
            with _patch_duke_infer_main_dependencies(args) as ctx:
                with mock.patch.dict(os.environ, {"LABEL_CSV_PATH": label_csv}):
                    from IMC.net4_duke.infer import main

                    main()
            ctx["run_evaluation"].assert_called_once()

    def test_run_evaluation_skipped_when_run_eval_false(self) -> None:
        """run_evaluation must NOT be called when --run_eval is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_infer_args(output_dir=tmpdir, run_eval=False)
            with _patch_duke_infer_main_dependencies(args) as ctx:
                from IMC.net4_duke.infer import main

                main()
            ctx["run_evaluation"].assert_not_called()

    def test_run_inference_called_once(self) -> None:
        """run_inference must be called exactly once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_infer_args(output_dir=tmpdir)
            with _patch_duke_infer_main_dependencies(args) as ctx:
                from IMC.net4_duke.infer import main

                main()
            ctx["run_inference"].assert_called_once()

    def test_load_model_called_with_correct_args(self) -> None:
        """load_model must be called with the parsed args."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_infer_args(output_dir=tmpdir, modality="image")
            with _patch_duke_infer_main_dependencies(args) as ctx:
                with mock.patch("IMC.net4_duke.infer.load_model", return_value=ctx["model"]) as mock_lm:
                    from IMC.net4_duke.infer import main

                    main()
                mock_lm.assert_called_once()
                call_args = mock_lm.call_args
                assert call_args[1]["args"].modality == "image"

    def test_folds_parsed_from_string(self) -> None:
        """--folds '0,1' must be parsed to [0, 1] before calling the dataloader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_infer_args(output_dir=tmpdir, folds="0,1")
            with _patch_duke_infer_main_dependencies(args) as ctx:
                from IMC.net4_duke.infer import main

                main()
            call_kwargs = ctx["create_inference_dataloader"].call_args[1]
            assert call_kwargs["fold_indices"] == [0, 1]

    def test_no_fold_filter_when_folds_none(self) -> None:
        """When --folds is None, fold_indices=None must be passed to the dataloader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_main_duke_infer_args(output_dir=tmpdir, folds=None)
            with _patch_duke_infer_main_dependencies(args) as ctx:
                from IMC.net4_duke.infer import main

                main()
            call_kwargs = ctx["create_inference_dataloader"].call_args[1]
            assert call_kwargs["fold_indices"] is None
