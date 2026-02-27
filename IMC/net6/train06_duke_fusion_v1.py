"""
Training script for Network v06 (PixelOnlyModel) on the Duke dataset with 5-fold CV and single middle slice.
- Trains ONLY the SequenceType_Code_norm task
- Uses focal loss across classes for this single task
- Uses the common IMC Trainer
"""

import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from IMC.helper import capture_console_to_log, log_training_start, log_training_end
from IMC.tensorboard_logging import setup_combined_logging
from IMC.net4_duke.train04_baseline_multi_slices import create_optimizer, get_scheduler, init_weights
from IMC.data.duke_dataloader_local import LiverDataset, DUKE_ORIGINAL_LABEL_NAMES
from IMC.network06 import PixelOnlyModel
from IMC.trainer import Trainer

# Duke environment
os.environ["DEBUG_MODE"] = "1"
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet"


def main():
    parser = argparse.ArgumentParser(description="Train net6 PixelOnlyModel on Duke with 5-Fold CV (SequenceType_Code_norm only)")
    parser.add_argument("--backbone", type=str, default="densenet121", help="Image encoder backbone")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id")
    parser.add_argument("--num_epochs", type=int, default=25, help="Epochs per fold")
    args = parser.parse_args()

    folds_to_run = [0, 1, 2, 3, 4]
    n_folds = 5

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join("./logs", f"{timestamp}_net6_duke_5fold")
    os.makedirs(base_log_dir, exist_ok=True)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = 1e-3
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Dataset config: single middle slice (n_slices=1)
    num_samples = None
    aggregated_metadata = False
    use_preselected_features = False
    exclude_contrast_yn = True

    print("="*80)
    print("Training net6 PixelOnlyModel on Duke with 5-fold CV (SequenceType_Code_norm)")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print("="*80)

    all_fold_results = []

    for fold_idx in folds_to_run:
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{n_folds-1}")
        print(f"{'='*80}")

        test_fold = fold_idx
        val_fold = (fold_idx + 1) % n_folds
        train_folds = [f for f in range(n_folds) if f not in [test_fold, val_fold]]

        fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
        os.makedirs(fold_log_dir, exist_ok=True)
        experiment_name = f"fold_{fold_idx}_net6_duke_{args.backbone}"

        logger, log_path, tb_logger = setup_combined_logging(
            experiment_name=experiment_name,
            log_dir=fold_log_dir,
            tb_log_dir=fold_log_dir,
        )

        config = {
            "fold": fold_idx,
            "train_folds": train_folds,
            "val_fold": val_fold,
            "test_fold": test_fold,
            "device": str(device),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "backbone": args.backbone,
            "net": "06",
            "single_slice": True,
        }
        log_training_start(logger, config=config)

        with capture_console_to_log(logger):
            # Dataloaders with single slice
            train_ds = LiverDataset(
                split=[f"fold_{f}" for f in train_folds],
                num_samples=num_samples,
                n_slices=1,
                sampling_type="equidistant",
                augment_conf="IMAGENET299_CENTER",
                aggregated_metadata=aggregated_metadata,
                use_preselected_features=use_preselected_features,
                exclude_contrast_yn=exclude_contrast_yn,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
            )
            val_ds = LiverDataset(
                split=[f"fold_{val_fold}"],
                num_samples=num_samples,
                n_slices=1,
                sampling_type="equidistant",
                augment_conf="IMAGENET299_CENTER",
                aggregated_metadata=aggregated_metadata,
                use_preselected_features=use_preselected_features,
                exclude_contrast_yn=exclude_contrast_yn,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
            )
            test_ds = LiverDataset(
                split=[f"fold_{test_fold}"],
                num_samples=num_samples,
                n_slices=1,
                sampling_type="equidistant",
                augment_conf="IMAGENET299_CENTER",
                aggregated_metadata=aggregated_metadata,
                use_preselected_features=use_preselected_features,
                exclude_contrast_yn=exclude_contrast_yn,
                label_names=DUKE_ORIGINAL_LABEL_NAMES,
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

            print(f"Train samples: {len(train_ds)}")
            print(f"Val samples: {len(val_ds)}")
            print(f"Test samples: {len(test_ds)}")

            # Single task map
            full_label_map = train_ds.get_n_labels()
            if "SequenceType_Code_norm" not in full_label_map:
                raise RuntimeError("SequenceType_Code_norm not found in label map")
            cl_d = {"SequenceType_Code_norm": full_label_map["SequenceType_Code_norm"]}
            print("Label config (single task):", cl_d)

            model = PixelOnlyModel(
                num_classes_dict=cl_d,
                image_backbone=args.backbone,
            )
            model.to(device)
            model.apply(init_weights)

            steps_per_epoch = len(train_loader)
            total_steps = num_epochs * steps_per_epoch

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

            # Single-task focal loss for SequenceType_Code_norm integrated with Trainer
            class SingleTaskFocalLoss(nn.Module):
                def __init__(self, alpha=1.0, gamma=2.0):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.ce = nn.CrossEntropyLoss(reduction='none')
                def forward(self, preds, targets, masks, task_weights=None):
                    logits = preds[0]
                    tgt = targets[0]
                    msk = masks[0].bool()
                    if not msk.any():
                        return torch.tensor(0.0, device=logits.device), [0.0]
                    logits = logits[msk]
                    tgt = tgt[msk]
                    ce = self.ce(logits, tgt)
                    pt = torch.exp(-ce)
                    fl = self.alpha * ((1 - pt) ** self.gamma) * ce
                    loss = fl.mean()
                    return loss, [loss.item()]

            criterion = SingleTaskFocalLoss(alpha=1.0, gamma=2.0)
            scaler = torch.amp.GradScaler("cuda", init_scale=2**8)
            task_weights = [1.0]

            # Use IMC Trainer
            trainer = Trainer(
                model=model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                scaler=scaler,
                tb_logger=tb_logger,
                logger=logger,
                patience=30,
                task_weights=task_weights,
                incl_regression=False,
                use_mixed_precision=True,
                img_ft_only=True,
            )

            print("Starting training...")
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                save_path=os.path.join(fold_log_dir, "best_model.pth"),
            )
            trainer.load_checkpoint(os.path.join(fold_log_dir, "best_model.pth"))
            print("Testing...")
            test_results = trainer.test(test_loader)
            print("Test results:", test_results)

            all_fold_results.append({
                "fold": fold_idx,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "test_samples": len(test_ds),
                "test_results": str(test_results),
                "log_dir": fold_log_dir,
            })

        log_training_end(logger)

    # Save summary
    summary_path = os.path.join(base_log_dir, "cv_summary.csv")
    results_df = pd.DataFrame([
        {
            "fold": r["fold"],
            "train_samples": r["train_samples"],
            "val_samples": r["val_samples"],
            "test_samples": r["test_samples"],
            "test_results": r["test_results"],
        }
        for r in all_fold_results
    ])
    results_df.to_csv(summary_path, index=False)
    print(f"All results saved to: {base_log_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
