"""Unified training entry point for Network v04 on the Duke Liver Dataset.

This script runs 5-fold cross-validation (CV) training using the Duke-specific
:class:`~IMC.data.duke_dataloader_local.LiverDataset` dataloader.  The fold
split strategy mirrors a standard leave-one-fold-out scheme:

- **Test fold**  : ``fold_idx``
- **Val fold**   : ``(fold_idx + 1) % n_folds``
- **Train folds**: all remaining folds

Architecture variants
---------------------
Select the model variant via ``--modality``:

- ``combined``  : :class:`~IMC.network04.MRISequenceClassifier` – image +
  metadata fusion via cross-attention.  Supports imputer-based
  (``--metadata_enc_type imputer``) and sparse metadata encoders
  (``--metadata_enc_type sparse``), configurable via ``--sparse_enc_version``
  and ``--fusion_module_version``.
- ``image``     : :class:`~IMC.network04.ImageBasedClassifier` (or
  :class:`~IMC.network04.SimpleImageBasedClassifier` with
  ``--vanilla_image_classifier``).  No metadata branch.
- ``metadata``  : :class:`~IMC.network04.MetadataBasedClassifier`.  No image branch.

Common features
---------------
- Mixed-precision training (``torch.amp.GradScaler``).
- AdamW optimiser with separate weight-decay groups (biases and norm layers
  get no weight decay).
- Linear-warmup + cosine-decay learning-rate schedule.
- Multi-task classification loss (:class:`~IMC.nn.multi_task_loss.MultiTaskLoss`).
- Early stopping controlled by ``--patience``.
- Checkpoint save/resume (``--ckpt``).
- Combined TensorBoard + file logging via
  :func:`~IMC.tensorboard_logging.setup_combined_logging`.
- Per-fold ``config.json`` (all hyperparameters) and ``model_architecture.txt``
  written inside the fold log directory.
- Cross-validation summary CSV written to ``<log_dir>/cv_summary.csv``.

Typical usage
-------------
Train all 5 folds with the default combined model::

    python -m IMC.net4_duke.train --modality combined --gpu 0

Image-only with a ResNet50 backbone::

    python -m IMC.net4_duke.train --modality image --img_enc_backbone resnet50

Metadata-only with sparse encoder::

    python -m IMC.net4_duke.train --modality metadata --metadata_enc_type sparse

Train only folds 0 and 1 (e.g. to restart after a failure)::

    python -m IMC.net4_duke.train --modality combined --folds 0,1

Environment variables
---------------------
The following environment variables configure Duke dataset paths.  They can
also be overridden via the corresponding CLI arguments (``--dataset_path``,
``--metadata_path``, ``--label_csv_path``), which take precedence::

    LOCAL_DATASET_PATH – root folder of the Duke MRI image dataset
    METADATA_PATH      – path to the Duke encoded metadata parquet file
    LABEL_CSV_PATH     – path to the Duke label CSV
    DEBUG_MODE         – set to "1" to enable extra debug output
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from IMC.helper import (
    capture_console_to_log,
    create_optimizer,
    get_scheduler,
    init_weights,
    log_training_end,
    log_training_start,
)
from IMC.net4_duke.helper import build_model
from IMC.nn.multi_task_loss import MultiTaskLoss
from IMC.tensorboard_logging import setup_combined_logging


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Duke 5-fold CV training script.

    Returns:
        Populated :class:`argparse.Namespace` with all resolved arguments.
    """
    parser = argparse.ArgumentParser(
        description=("5-fold cross-validation training for IMC Network v04 on the Duke Liver Dataset."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ mode
    parser.add_argument(
        "--modality",
        type=str,
        choices=["image", "metadata", "combined"],
        required=True,
        help=(
            "Model variant: "
            "'combined' = MRISequenceClassifier (image + metadata fusion), "
            "'image' = ImageBasedClassifier (image only), "
            "'metadata' = MetadataBasedClassifier (metadata only)."
        ),
    )

    # ---------------------------------------------------- common training
    parser.add_argument(
        "--img_enc_backbone",
        type=str,
        default="densenet121",
        help="CNN backbone for the image encoder (e.g. 'densenet121', 'resnet50').",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs per fold.",
    )
    parser.add_argument("--lr", type=float, default=1e-6, help="Base learning rate for AdamW.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index.  Use -1 for CPU.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (applied to every fold).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience in epochs without validation improvement.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Root directory for logs and checkpoints.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG_MODE.")

    # --------------------------------------------------- fold selection
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help=(
            "Comma-separated fold indices to train (e.g. '0,1,2').  Defaults to all folds (0 through --n_folds - 1)."
        ),
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Total number of folds in the cross-validation scheme.",
    )

    # ----------------------------------------------- data / environment
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override LOCAL_DATASET_PATH environment variable.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Override METADATA_PATH environment variable.",
    )
    parser.add_argument(
        "--label_csv_path",
        type=str,
        default=None,
        help="Override LABEL_CSV_PATH environment variable.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=10,
        help="Number of slices to sample from each MRI volume.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap dataset size per split (useful for smoke-tests).",
    )

    # ----------------------------------------------- image mode
    parser.add_argument(
        "--vanilla_image_classifier",
        action="store_true",
        help=("Use SimpleImageBasedClassifier instead of ImageBasedClassifier.  Active only when --modality image."),
    )

    # ----------------------------------------- combined mode: fusion
    parser.add_argument(
        "--fusion_module_version",
        type=str,
        default="v2",
        choices=["v1", "v2", "concat"],
        help="Cross-attention fusion module variant (combined mode only).",
    )

    # ----------------------------------------------- combined / metadata mode
    parser.add_argument(
        "--metadata_enc_type",
        type=str,
        default="sparse",
        choices=["imputer", "sparse"],
        help=(
            "Metadata encoder type.  "
            "'imputer' uses a learnable contextual imputer.  "
            "'sparse' uses a sparse attention-based encoder that skips missing values."
        ),
    )
    parser.add_argument(
        "--sparse_enc_version",
        type=str,
        default="v1",
        choices=["v1", "v2", "v5"],
        help="Sparse metadata encoder version (active when --metadata_enc_type sparse).",
    )
    parser.add_argument(
        "--imputer_type",
        type=str,
        default="contextual",
        choices=["contextual", "ignore"],
        help=(
            "Imputer strategy when --metadata_enc_type imputer.  "
            "'contextual' uses a learnable imputer leveraging other feature context.  "
            "'ignore' fills missing values with zeros."
        ),
    )
    parser.add_argument(
        "--metadata_embed_dim",
        type=int,
        default=128,
        help="Metadata encoder embedding dimension.",
    )
    parser.add_argument(
        "--output_emb_dim",
        type=int,
        default=128,
        help="Output projection embedding dimension.",
    )
    parser.add_argument(
        "--metadata_dropout",
        action="store_true",
        help="Apply stochastic dropout to metadata features (sparse mode only).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset environment setup
# ---------------------------------------------------------------------------


def _configure_dataset_env(args: argparse.Namespace) -> None:
    """Apply Duke-specific dataset environment variables.

    Applies CLI path overrides for ``LOCAL_DATASET_PATH``, ``METADATA_PATH``,
    and ``LABEL_CSV_PATH`` when provided and validates that all required Duke
    dataset variables are configured before training starts.

    Also sets ``DEBUG_MODE`` to ``"1"`` when ``--debug`` is passed.

    Args:
        args: Parsed arguments from :func:`parse_args`.
    """
    os.environ["DEBUG_MODE"] = "1" if args.debug else "0"

    # CLI overrides always win
    if args.dataset_path:
        os.environ["LOCAL_DATASET_PATH"] = args.dataset_path
    if args.metadata_path:
        os.environ["METADATA_PATH"] = args.metadata_path
    if args.label_csv_path:
        os.environ["LABEL_CSV_PATH"] = args.label_csv_path

    missing = [name for name in ("LOCAL_DATASET_PATH", "METADATA_PATH", "LABEL_CSV_PATH") if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "Missing Duke dataset configuration. Set the environment variables "
            f"{', '.join(missing)} or pass the corresponding CLI overrides."
        )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main() -> None:
    """Run 5-fold cross-validation training on the Duke Liver Dataset.

    This function orchestrates the full CV pipeline.  Each fold follows
    identical steps 2–10 listed below.

    Steps
    -----
    1.  Parse CLI arguments (:func:`parse_args`) and apply Duke environment
        variables (:func:`_configure_dataset_env`).
    2.  Determine fold splits: test=``fold_idx``, val=``(fold_idx+1) % n_folds``,
        train=all remaining.
    3.  Create a timestamped log directory under ``--log_dir`` and a per-fold
        sub-directory ``fold_{fold_idx}/``.
    4.  Set up TensorBoard + file logging
        (:func:`~IMC.tensorboard_logging.setup_combined_logging`) and write
        ``config.json`` for full reproducibility.
    5.  Build train / validation / test
        :class:`~IMC.data.duke_dataloader_local.LiverDataset` instances using
        the ``fold_*`` split labels.
    6.  Instantiate the model via :func:`~IMC.net4.helper.build_model`,
        optionally loading from ``--ckpt``.
    7.  Apply :func:`init_weights` to non-pretrained layers and save
        ``model_architecture.txt``.
    8.  Create :func:`create_optimizer`, :func:`get_scheduler`, and
        :class:`~IMC.nn.multi_task_loss.MultiTaskLoss`.
    9.  Run :meth:`~IMC.trainer.Trainer.fit` for up to ``--num_epochs`` epochs
        with :meth:`~IMC.trainer.Trainer.load_checkpoint` + early stopping.
    10. Reload the best checkpoint and evaluate on the test fold with
        :meth:`~IMC.trainer.Trainer.test`.
    11. After all folds, write a cross-validation summary to
        ``<log_dir>/cv_summary.csv``.
    """
    import pandas as pd

    args = parse_args()
    _configure_dataset_env(args)

    # ---- fold selection ----------------------------------------------------
    n_folds = args.n_folds
    if args.folds is not None:
        folds_to_run = [int(f) for f in args.folds.split(",")]
        for f in folds_to_run:
            if f < 0 or f >= n_folds:
                raise ValueError(f"Fold index {f} is out of range [0, {n_folds - 1}].  Adjust --folds or --n_folds.")
    else:
        folds_to_run = list(range(n_folds))

    # ---- experiment directory ----------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(args.log_dir, f"{timestamp}_5fold_cv")
    os.makedirs(base_log_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")

    # ---- import dataloader (env vars must be set first) --------------------
    from IMC.data.duke_dataloader_local import LiverDataset, DUKE_ORIGINAL_LABEL_NAMES

    print("=" * 80)
    print("5-Fold Cross-Validation Training  |  Duke Liver Dataset")
    print(f"  Folds to train   : {folds_to_run}")
    print(f"  Device           : {device}")
    print(f"  Backbone         : {args.img_enc_backbone}")
    print(f"  Modality         : {args.modality}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Epochs / fold    : {args.num_epochs}")
    print("=" * 80)

    all_fold_results: list[dict] = []

    # ---- per-fold loop -----------------------------------------------------
    for fold_idx in folds_to_run:
        test_fold = fold_idx
        val_fold = (fold_idx + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in (test_fold, val_fold)]

        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_idx} / {n_folds - 1}  |  train={train_folds}  val={val_fold}  test={test_fold}")
        print("=" * 80)

        # ---- per-fold logging ----------------------------------------------
        fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
        os.makedirs(fold_log_dir, exist_ok=True)

        experiment_name = (
            f"fold{fold_idx}_{args.modality}_{args.img_enc_backbone}"
            f"_enc{args.metadata_enc_type}_fus{args.fusion_module_version}"
        )
        logger, _log_path, tb_logger = setup_combined_logging(
            experiment_name=experiment_name,
            log_dir=fold_log_dir,
            tb_log_dir=fold_log_dir,
        )

        config: dict = {
            "fold": fold_idx,
            "train_folds": train_folds,
            "val_fold": val_fold,
            "test_fold": test_fold,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.lr,
            "modality": args.modality,
            "img_enc_backbone": args.img_enc_backbone,
            "metadata_enc_type": args.metadata_enc_type,
            "imputer_type": args.imputer_type,
            "sparse_enc_version": args.sparse_enc_version,
            "fusion_module_version": args.fusion_module_version,
            "metadata_embed_dim": args.metadata_embed_dim,
            "output_emb_dim": args.output_emb_dim,
            "metadata_dropout": args.metadata_dropout,
            "n_slices": args.n_slices,
            "patience": args.patience,
            "checkpoint": args.ckpt,
        }
        with open(os.path.join(fold_log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        log_training_start(logger, config=config)

        with capture_console_to_log(logger):
            # ---- data loaders ----------------------------------------------
            print("Creating dataloaders...")

            _dl_kwargs = dict(
                num_samples=args.num_samples,
                augment_conf="NONE2D",
                aggregated_metadata=False,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
                n_slices=args.n_slices,
            )

            train_dataset = LiverDataset(
                split=[f"fold_{f}" for f in train_folds],
                sampling_type="random",
                **_dl_kwargs,
            )
            val_dataset = LiverDataset(
                split=[f"fold_{val_fold}"],
                sampling_type="equidistant",
                **_dl_kwargs,
            )
            test_dataset = LiverDataset(
                split=[f"fold_{test_fold}"],
                sampling_type="equidistant",
                **_dl_kwargs,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            print(f"  Train samples  : {len(train_dataset)}")
            print(f"  Val samples    : {len(val_dataset)}")
            print(f"  Test samples   : {len(test_dataset)}")

            num_classes_dict = train_dataset.get_n_labels()
            metadata_input_dim = train_dataset.num_metadata_features
            print(f"  Label config   : {num_classes_dict}")
            print(f"  Metadata dim   : {metadata_input_dim}")

            # ---- model -----------------------------------------------------
            model = build_model(
                modality=args.modality,
                vanilla_image_classifier=args.vanilla_image_classifier,
                img_enc_backbone=args.img_enc_backbone,
                metadata_enc_type=args.metadata_enc_type,
                imputer_type=args.imputer_type,
                sparse_enc_version=args.sparse_enc_version,
                output_emb_dim=args.output_emb_dim,
                num_classes_dict=num_classes_dict,
                metadata_input_dim=metadata_input_dim,
                metadata_embed_dim=args.metadata_embed_dim,
                fusion_module_version=args.fusion_module_version,
                metadata_dropout=args.metadata_dropout,
            )

            with open(os.path.join(fold_log_dir, "model_architecture.txt"), "w") as f:
                f.write(str(model))

            if args.ckpt is not None:
                print(f"  Loading checkpoint from {args.ckpt}")
                checkpoint = torch.load(args.ckpt, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])

            model.to(device)
            model.apply(init_weights)

            # ---- optimisation setup ----------------------------------------
            steps_per_epoch = len(train_loader)
            total_steps = args.num_epochs * steps_per_epoch
            warmup_steps = max(1, int(0.1 * total_steps))

            optimizer = create_optimizer(model, lr=args.lr, eps=1e-7)
            scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
            criterion = MultiTaskLoss(
                label_smoothing=0.1,
                task_names=list(DUKE_ORIGINAL_LABEL_NAMES.keys()),
            )
            scaler = torch.cuda.amp.GradScaler(init_scale=2**8)

            # ---- trainer ---------------------------------------------------
            from IMC.trainer import Trainer

            trainer = Trainer(
                model=model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                task_names=list(DUKE_ORIGINAL_LABEL_NAMES.keys()),
                scaler=scaler,
                tb_logger=tb_logger,
                logger=logger,
                patience=args.patience,
                use_mixed_precision=True,
            )

            save_path = os.path.join(fold_log_dir, "best_model.pth")
            print(f"\nTraining fold {fold_idx}...")
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.num_epochs,
                save_path=save_path,
            )

            # ---- test evaluation -------------------------------------------
            print(f"\nEvaluating fold {fold_idx} on test set...")
            trainer.load_checkpoint(save_path)
            test_results = trainer.test(test_loader)

            fold_results: dict = {
                "fold": fold_idx,
                "train_folds": train_folds,
                "val_fold": val_fold,
                "test_fold": test_fold,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "test_results": test_results,
                "log_dir": fold_log_dir,
            }
            all_fold_results.append(fold_results)
            print(f"\nFold {fold_idx} complete  |  test results: {test_results}")

        log_training_end(logger)

    # ---- cross-validation summary ------------------------------------------
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    summary_path = os.path.join(base_log_dir, "cv_summary.csv")
    summary_df = pd.DataFrame(
        [
            {
                "fold": r["fold"],
                "train_folds": str(r["train_folds"]),
                "val_fold": r["val_fold"],
                "test_fold": r["test_fold"],
                "train_samples": r["train_samples"],
                "val_samples": r["val_samples"],
                "test_samples": r["test_samples"],
                "test_results": str(r["test_results"]),
            }
            for r in all_fold_results
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    for result in all_fold_results:
        print(f"\n  Fold {result['fold']}  →  {result['test_results']}")

    print(f"\n  All outputs saved to : {base_log_dir}")
    print(f"  Summary CSV          : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
