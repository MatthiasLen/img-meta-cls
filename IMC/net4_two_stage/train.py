"""
Two-stage training entry point for Network v04 (IMC) – liver MRI classification.

Motivation
----------
A purely image-based classifier carries strong signals on its own (body region,
acquisition plane, etc.).  Adding metadata is most useful when the metadata
branch has been given a fair chance to learn complementary signal *after* the
image branch has converged, rather than competing with a strong image signal
from epoch 0.

Training scheme
---------------
**Stage 1 – image-only**

    Train ``ImageBasedClassifier`` on the full multi-task objective (same
    hyper-parameters and architecture as ``net4.train --modality image``).
    The best checkpoint is saved as ``stage1/stage1_best_model.pth``.

**Stage 2 – frozen image branch + fresh metadata branch**

    Build ``FrozenImageCombinedClassifier`` (a subclass of
    ``MRISequenceClassifier``).  Load ``image_encoder`` and ``slice_fusion``
    weights from the stage-1 checkpoint and freeze them – their parameters are
    excluded from the optimiser *and* they are kept in eval mode throughout
    stage-2 training (so batch-norm statistics and dropout are frozen too).
    Only the metadata encoder, cross-attention fusion module, and multi-task
    head are trained from scratch.

    ``fusion_module_version`` controls the cross-attention fusion style for
    stage 2 (default ``"v2"``).  Because ``reduce`` is a runtime flag in
    ``SliceFeatureFusion`` that does not affect weight shapes, any fusion
    version is compatible with the stage-1 image-branch checkpoint.

Typical usage
-------------
Run both stages back-to-back::

    python -m IMC.net4_two_stage.train \\
        --img_enc_backbone densenet121 \\
        --stage1_num_epochs 15 \\
        --stage2_num_epochs 15 \\
        --gpu 0

Stage 1 only::

    python -m IMC.net4_two_stage.train --run_stage 1

Stage 2 only (provide a pre-existing stage-1 checkpoint)::

    python -m IMC.net4_two_stage.train --run_stage 2 \\
        --stage1_ckpt /path/to/stage1_best_model.pth

Environment variables (same fallbacks as ``net4.train``)
---------------------------------------------------------
    LOCAL_DATASET_PATH – root folder of image dataset
    METADATA_PATH      – path to encoded metadata parquet / CSV
    LABEL_CSV_PATH     – path to label CSV
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from IMC.helper import capture_console_to_log, log_training_end, log_training_start
from IMC.net4.train import create_optimizer, get_scheduler, init_weights
from IMC.net4.helper import build_model
from IMC.network04 import MRISequenceClassifier
from IMC.nn.multi_task_loss import MultiTaskLoss
from IMC.tensorboard_logging import setup_combined_logging
from IMC.trainer import Trainer


# ---------------------------------------------------------------------------
# Stage-2 model: frozen image branch
# ---------------------------------------------------------------------------

class FrozenImageCombinedClassifier(MRISequenceClassifier):
    """``MRISequenceClassifier`` variant used in stage-2 two-stage training.

    The image branch (``image_encoder`` + ``slice_fusion``) is kept in eval
    mode at all times during training:

    * Batch-normalisation statistics are not updated.
    * Dropout inside those modules is disabled.
    * Their parameters carry ``requires_grad=False``.

    All other modules (``metadata_encoder``, ``embedding_fusion``,
    ``multi_task_head``) behave normally.
    """

    def train(self, mode: bool = True) -> "FrozenImageCombinedClassifier":  # noqa: D401
        super().train(mode)
        # Keep frozen image-branch modules in eval mode regardless of the
        # overall training/eval flag passed by the Trainer.
        self.image_encoder.eval()
        self.slice_fusion.eval()
        return self


# ---------------------------------------------------------------------------
# Weight transfer helpers
# ---------------------------------------------------------------------------

def load_image_branch_from_checkpoint(
    model: MRISequenceClassifier,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """Transfer ``image_encoder`` and ``slice_fusion`` weights from a stage-1 checkpoint.

    The stage-1 checkpoint was saved from ``ImageBasedClassifier``, which
    shares the ``image_encoder`` and ``slice_fusion`` sub-modules (same
    parameter names and shapes) with ``MRISequenceClassifier`` when
    ``fusion_module_version="v1"`` (both use ``reduce=True``).

    All other keys (``output_proj.*``, ``multi_task_head.*``) present in the
    stage-1 checkpoint are silently ignored – the corresponding stage-2
    modules start from random initialisation.

    Args:
        model:           Stage-2 ``FrozenImageCombinedClassifier`` instance.
        checkpoint_path: Path to the stage-1 ``stage1_best_model.pth`` file.
        device:          Device to map tensors to during loading.

    Returns:
        Number of parameter tensors successfully loaded.

    Raises:
        RuntimeError: If any ``image_encoder.*`` or ``slice_fusion.*`` key
            present in *model* is not found in the stage-1 checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stage1_state: dict = checkpoint["model_state_dict"]

    image_branch_prefixes = ("image_encoder.", "slice_fusion.")

    # Keys we expect to load (from the stage-2 model's perspective)
    expected_keys = {
        k for k in model.state_dict() if k.startswith(image_branch_prefixes)
    }
    # Keys available from the stage-1 checkpoint
    available = {
        k: v for k, v in stage1_state.items() if k.startswith(image_branch_prefixes)
    }

    missing_in_ckpt = expected_keys - available.keys()
    if missing_in_ckpt:
        raise RuntimeError(
            f"Stage-1 checkpoint is missing image-branch keys required by stage-2 model: "
            f"{sorted(missing_in_ckpt)}"
        )

    # Load with strict=False so the metadata / fusion / head keys are tolerated
    incompat = model.load_state_dict(available, strict=False)

    # Sanity: unexpected keys in the filtered dict should be empty
    if incompat.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys when loading image branch: {incompat.unexpected_keys}"
        )

    return len(available)


def freeze_image_branch(model: MRISequenceClassifier) -> None:
    """Set ``requires_grad=False`` for all ``image_encoder`` and ``slice_fusion`` parameters."""
    for attr in ("image_encoder", "slice_fusion"):
        for param in getattr(model, attr).parameters():
            param.requires_grad_(False)


# ---------------------------------------------------------------------------
# Shared data-loader factory
# ---------------------------------------------------------------------------

def _make_data_loaders(args):
    """Construct train / validation / test data loaders from *args*."""
    from IMC.data.liver_dataloader_local import (
        get_test_dataloader,
        get_train_dataloader,
        get_valid_dataloader,
    )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        n_slices=args.num_slices,
        num_workers=args.num_workers,
        aggregated_metadata=False,
        use_preselected_features=args.use_preselected_features,
        exclude_contrast_yn=True,
    )
    return (
        get_train_dataloader(**dl_kwargs),
        get_valid_dataloader(**dl_kwargs),
        get_test_dataloader(**dl_kwargs),
    )


def _resolve_device(args) -> torch.device:
    if args.gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 1: image-only training
# ---------------------------------------------------------------------------

def run_stage1(args, log_dir: str, logger, tb_logger) -> str:
    """Train ``ImageBasedClassifier`` (Stage 1).

    Args:
        args:       Parsed CLI arguments.
        log_dir:    Directory for stage-1 checkpoints, architecture files, etc.
        logger:     Python logger instance.
        tb_logger:  TensorBoard logger (from ``setup_combined_logging``).

    Returns:
        Absolute path to the saved best-model checkpoint.
    """
    device = _resolve_device(args)

    with capture_console_to_log(logger):
        train_loader, val_loader, test_loader = _make_data_loaders(args)
        num_classes_dict = train_loader.dataset.get_n_labels()

        logger.info(f"[Stage 1] num_classes_dict = {num_classes_dict}")

        # Build image-only model (ImageBasedClassifier, *not* the vanilla variant)
        model = build_model(
            modality="image",
            vanilla_image_classifier=False,
            img_enc_backbone=args.img_enc_backbone,
            incl_regression=args.incl_regression,
            num_classes_dict=num_classes_dict,
        )

        with open(os.path.join(log_dir, "model_architecture_stage1.txt"), "w") as f:
            f.write(str(model))

        model.apply(init_weights)
        model.to(device)

        steps_per_epoch = len(train_loader)
        total_steps = args.stage1_num_epochs * steps_per_epoch
        warmup_steps = max(1, int(0.1 * total_steps))

        optimizer = create_optimizer(model, lr=args.stage1_lr, eps=1e-7)
        scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
        criterion = MultiTaskLoss(
            label_smoothing=0.1,
            incl_regression=args.incl_regression,
            task_names=list(num_classes_dict.keys()),
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=2**8)

        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            tb_logger=tb_logger,
            logger=logger,
            patience=args.stage1_patience,
            incl_regression=args.incl_regression,
            use_mixed_precision=True,
        )

        save_path = os.path.join(log_dir, "stage1_best_model.pth")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.stage1_num_epochs,
            save_path=save_path,
        )

        # Evaluate on test split
        trainer.load_checkpoint(save_path)
        test_acc = trainer.test(test_loader)
        logger.info(f"[Stage 1] Test accuracy: {test_acc}")

    return save_path


# ---------------------------------------------------------------------------
# Stage 2: frozen image branch + metadata training
# ---------------------------------------------------------------------------

def run_stage2(args, stage1_ckpt: str, log_dir: str, logger, tb_logger) -> None:
    """Train the combined model with a frozen image branch (Stage 2).

    The image branch (``image_encoder`` + ``slice_fusion``) is loaded from
    *stage1_ckpt* and frozen for the entire stage-2 run.  Only
    ``metadata_encoder``, ``embedding_fusion``, and ``multi_task_head`` are
    optimised.

    ``fusion_module_version`` is taken from ``args.fusion_module_version``
    (default ``"v2"``).  The ``reduce`` flag in ``SliceFeatureFusion`` is
    purely a runtime control and does not affect weight shapes, so any
    fusion version is compatible with the stage-1 image-branch checkpoint.

    Args:
        args:         Parsed CLI arguments.
        stage1_ckpt:  Path to the stage-1 ``stage1_best_model.pth`` file.
        log_dir:      Directory for stage-2 checkpoints / architecture files.
        logger:       Python logger instance.
        tb_logger:    TensorBoard logger.
    """
    device = _resolve_device(args)

    with capture_console_to_log(logger):
        train_loader, val_loader, test_loader = _make_data_loaders(args)
        num_classes_dict = train_loader.dataset.get_n_labels()
        metadata_input_dim = train_loader.dataset.num_metadata_features

        logger.info(f"[Stage 2] num_classes_dict     = {num_classes_dict}")
        logger.info(f"[Stage 2] metadata_input_dim   = {metadata_input_dim}")
        logger.info(f"[Stage 2] Loading image branch from: {stage1_ckpt}")

        # ---- build stage-2 model ------------------------------------------
        model = FrozenImageCombinedClassifier(
            num_classes_dict=num_classes_dict,
            metadata_input_dim=metadata_input_dim,
            img_enc_backbone=args.img_enc_backbone,
            incl_regression=args.incl_regression,
            metadata_encoder_type=args.metadata_enc_type,
            imputer_type=args.imputer_type,
            sparse_enc_version=args.sparse_enc_version,
            fusion_module_version=args.fusion_module_version,
            metadata_embed_dim=args.metadata_embed_dim,
            dropout_metadata=args.metadata_dropout,
            scalar_modulation=args.scalar_modulation,
        )
        logger.info(f"[Stage 2] fusion_module_version = {args.fusion_module_version}")

        # ---- transfer & freeze image branch --------------------------------
        n_loaded = load_image_branch_from_checkpoint(model, stage1_ckpt, device)
        logger.info(f"[Stage 2] Transferred {n_loaded} image-branch parameter tensors from stage 1.")

        freeze_image_branch(model)

        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[Stage 2] Frozen    parameters: {frozen_params:,}")
        logger.info(f"[Stage 2] Trainable parameters: {trainable_params:,}")

        # ---- initialise only the trainable (new) modules ------------------
        for module in (model.metadata_encoder, model.embedding_fusion, model.multi_task_head):
            module.apply(init_weights)

        with open(os.path.join(log_dir, "model_architecture_stage2.txt"), "w") as f:
            f.write(str(model))

        model.to(device)

        # ---- optimisation: only parameters with requires_grad=True --------
        # create_optimizer already skips frozen params via the requires_grad check
        steps_per_epoch = len(train_loader)
        total_steps = args.stage2_num_epochs * steps_per_epoch
        warmup_steps = max(1, int(0.1 * total_steps))

        optimizer = create_optimizer(model, lr=args.stage2_lr, eps=1e-7)
        scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
        criterion = MultiTaskLoss(
            label_smoothing=0.1,
            incl_regression=args.incl_regression,
            task_names=list(num_classes_dict.keys()),
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=2**8)

        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            tb_logger=tb_logger,
            logger=logger,
            patience=args.stage2_patience,
            incl_regression=args.incl_regression,
            use_mixed_precision=True,
        )

        save_path = os.path.join(log_dir, "stage2_best_model.pth")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.stage2_num_epochs,
            save_path=save_path,
        )

        # Evaluate on test split
        trainer.load_checkpoint(save_path)
        test_acc = trainer.test(test_loader)
        logger.info(f"[Stage 2] Test accuracy: {test_acc}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    """Parse CLI arguments for the two-stage training script."""
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage training for IMC Network v04.  "
            "Stage 1 trains an image-only classifier; Stage 2 freezes the "
            "trained image branch and trains the metadata branch of the "
            "combined model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- run-mode control --------------------------------------------------
    parser.add_argument(
        "--run_stage",
        type=str,
        default="both",
        choices=["1", "2", "both"],
        help=(
            "Which stage(s) to execute.  "
            "'both' runs stage 1 then stage 2.  "
            "'1' runs only stage 1.  "
            "'2' runs only stage 2 (requires --stage1_ckpt)."
        ),
    )
    parser.add_argument(
        "--stage1_ckpt",
        type=str,
        default=None,
        help=(
            "Path to an existing stage-1 best-model checkpoint "
            "(``stage1_best_model.pth``).  "
            "When --run_stage both, providing this skips stage-1 training "
            "and goes directly to stage 2 using the supplied checkpoint.  "
            "Required when --run_stage 2."
        ),
    )

    # ---- shared / common ---------------------------------------------------
    parser.add_argument("--img_enc_backbone", type=str, default="densenet121",
                        help="CNN backbone for the image encoder (e.g. densenet121, resnet50).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index. Use -1 for CPU.")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Root directory for logs and checkpoints.")
    parser.add_argument("--incl_regression", action="store_true",
                        help="Include regression head for ContrastPhase task.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG_MODE (verbose output).")
    parser.add_argument("--use_preselected_features", action="store_true",
                        help="Use the pre-selected metadata feature subset.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes.")
    parser.add_argument("--num_slices", type=int, default=None,
                        help="Number of slices to use per MRI volume.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of training samples (for debug runs).")

    # ---- environment / data overrides --------------------------------------
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Override LOCAL_DATASET_PATH env var.")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Override METADATA_PATH env var.")
    parser.add_argument("--label_csv_path", type=str, default=None,
                        help="Override LABEL_CSV_PATH env var.")

    # ---- stage-1 hyper-parameters -----------------------------------------
    stage1 = parser.add_argument_group("Stage 1 (image-only)")
    stage1.add_argument("--stage1_num_epochs", type=int, default=15,
                        help="Maximum training epochs for stage 1.")
    stage1.add_argument("--stage1_lr", type=float, default=1e-6,
                        help="Base AdamW learning rate for stage 1.")
    stage1.add_argument("--stage1_patience", type=int, default=5,
                        help="Early-stopping patience for stage 1 (epochs without improvement).")

    # ---- stage-2 hyper-parameters -----------------------------------------
    stage2 = parser.add_argument_group("Stage 2 (metadata branch, frozen image)")
    stage2.add_argument("--stage2_num_epochs", type=int, default=15,
                        help="Maximum training epochs for stage 2.")
    stage2.add_argument(
        "--stage2_lr", type=float, default=1e-5,
        help=(
            "Base AdamW learning rate for stage 2.  "
            "Can be higher than stage-1 LR since only new metadata-branch "
            "parameters are optimised."
        ),
    )
    stage2.add_argument("--stage2_patience", type=int, default=5,
                        help="Early-stopping patience for stage 2 (epochs without improvement).")

    # ---- stage-2 metadata / architecture -----------------------------------
    stage2.add_argument(
        "--metadata_enc_type", type=str, default="sparse",
        choices=["imputer", "sparse"],
        help="Metadata encoder type for stage 2.",
    )
    stage2.add_argument(
        "--sparse_enc_version", type=str, default="v1",
        choices=["v1", "v2", "v5"],
        help="Sparse metadata encoder version (used when --metadata_enc_type sparse).",
    )
    stage2.add_argument(
        "--imputer_type", type=str, default="contextual",
        choices=["contextual", "ignore"],
        help="Imputer strategy (used when --metadata_enc_type imputer).",
    )
    stage2.add_argument("--metadata_embed_dim", type=int, default=128,
                        help="Metadata encoder output embedding dimension.")
    stage2.add_argument("--metadata_dropout", action="store_true",
                        help="Apply random metadata dropout during stage-2 training.")
    stage2.add_argument("--scalar_modulation", action="store_true",
                        help="Apply scalar modulation in fusion module (stage 2 only).")
    stage2.add_argument(
        "--fusion_module_version",
        type=str,
        default="v2",
        choices=["v1", "v2", "v3", "concat"],
        help=(
            "Cross-attention fusion variant for stage 2.  "
            "'v2' (default) uses sequence-style inputs with weighted pooling (reduce=False in encoders); "
            "'v1' uses mean-pooled single vectors (reduce=True in encoders); "
            "'v3' is also supported; "
            "'concat' is a simple concatenation baseline.  "
            "Must match the value used when loading an existing stage-2 checkpoint."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the two-stage training pipeline."""
    args = parse_args()

    # ---- environment variable overrides ------------------------------------
    os.environ["DEBUG_MODE"] = "1" if args.debug else "0"
    if args.dataset_path:
        os.environ["LOCAL_DATASET_PATH"] = args.dataset_path
    if args.metadata_path:
        os.environ["METADATA_PATH"] = args.metadata_path
    if args.label_csv_path:
        os.environ["LABEL_CSV_PATH"] = args.label_csv_path

    os.environ.setdefault("LOCAL_DATASET_PATH", "/home/tuan.truong/data/PV.AI")
    os.environ.setdefault(
        "METADATA_PATH",
        "/home/tuan.truong/codebase/IMC/labels/encoded_metadata_20251217.parquet",
    )
    os.environ.setdefault(
        "LABEL_CSV_PATH",
        "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20250603_local.csv",
    )

    # ---- top-level log directory -------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        args.log_dir,
        f"{timestamp}_twostage_{args.img_enc_backbone}",
    )
    os.makedirs(run_dir, exist_ok=True)

    # Master logger (captures the shell-level logs across both stages)
    experiment_name = f"net04_twostage_{args.img_enc_backbone}"
    logger, log_path, _ = setup_combined_logging(
        experiment_name=experiment_name,
        log_dir=run_dir,
        tb_log_dir=run_dir,
    )

    device = _resolve_device(args)

    # Persist full config
    config = {**vars(args), "device": str(device), "run_dir": run_dir}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    log_training_start(logger, config=config)

    stage1_ckpt_path: Optional[str] = args.stage1_ckpt

    # ========================================================================
    # Stage 1
    # ========================================================================
    if args.run_stage in ("1", "both"):
        if args.run_stage == "both" and stage1_ckpt_path is not None:
            logger.info(
                f"[Stage 1] Skipped – using supplied checkpoint: {stage1_ckpt_path}"
            )
        else:
            s1_dir = os.path.join(run_dir, "stage1")
            os.makedirs(s1_dir, exist_ok=True)

            _, _, s1_tb = setup_combined_logging(
                experiment_name=f"{experiment_name}_stage1",
                log_dir=s1_dir,
                tb_log_dir=s1_dir,
            )

            logger.info("=" * 70)
            logger.info("STAGE 1 – image-only training")
            logger.info("=" * 70)

            stage1_ckpt_path = run_stage1(args, s1_dir, logger, s1_tb)
            logger.info(f"[Stage 1] Best checkpoint saved to: {stage1_ckpt_path}")

    if args.run_stage == "1":
        log_training_end(logger)
        return

    # ========================================================================
    # Stage 2
    # ========================================================================
    if args.run_stage in ("2", "both"):
        if stage1_ckpt_path is None:
            raise ValueError(
                "--stage1_ckpt is required when --run_stage 2. "
                "Provide the path to the stage-1 best_model.pth."
            )

        s2_dir = os.path.join(run_dir, "stage2")
        os.makedirs(s2_dir, exist_ok=True)

        _, _, s2_tb = setup_combined_logging(
            experiment_name=f"{experiment_name}_stage2",
            log_dir=s2_dir,
            tb_log_dir=s2_dir,
        )

        logger.info("=" * 70)
        logger.info("STAGE 2 – frozen image branch + metadata-branch training")
        logger.info("=" * 70)

        run_stage2(args, stage1_ckpt_path, s2_dir, logger, s2_tb)
        logger.info("[Stage 2] Training complete.")

    log_training_end(logger)


if __name__ == "__main__":
    main()
