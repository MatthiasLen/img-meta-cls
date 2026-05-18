from __future__ import annotations

import numpy as np
import torch
import torch.profiler
from typing import Optional, Union
from torch.utils.data import DataLoader

from IMC.helper import normalize_per_sample, _debug_mode
from IMC.tensorboard_logging import log_batch_metrics, log_training_metrics
from pathlib import Path

import os


def classification_losses(outputs: list, targets: list) -> list:
    """
    Calculates and prints the accuracy for each classification task.

    Args:
        outputs (list of torch.Tensor): List of model outputs for each task, where each output is a tensor of class scores.
        targets (list of torch.Tensor): List of ground truth labels for each task.
    Raises:
        AssertionError: If the number of outputs and targets do not match.
    Returns:
        list with task accuracies

    """
    assert len(outputs) == len(targets)
    accu_list = []

    # iterate over tasks
    for i, (output, target) in enumerate(zip(outputs, targets)):
        pred = output.clone().detach().cpu().numpy()
        target_cl = target.clone().detach().cpu().numpy()
        pred_cl = np.argmax(pred, axis=1)
        accuracy = np.mean(pred_cl == target_cl)
        accu_list.append(accuracy)
        if _debug_mode():
            print(f"Task {i} Accuracy:", accuracy)

    return accu_list


class Trainer:
    """
    Reusable Trainer for IMC experiments with mixed precision, multi-task loss,
    LR scheduling, gradient clipping, checkpointing, early stopping, and TensorBoard logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion,
        task_names: list,
        scaler: Optional[torch.amp.GradScaler],
        tb_logger=None,
        logger=None,
        patience: int = 5,
        task_weights: Union[list, None] = None,
        use_mixed_precision: bool = True,
        profiler_dir: Optional[str] = None,
        use_z_score_norm: bool = True,
        trainable_task_heads: Optional[list] = None,
        freeze_backbone: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = scaler
        self.tb_logger = tb_logger
        self.logger = logger
        self.patience = patience
        self.task_weights = task_weights if task_weights is not None else [1.0] * len(task_names)
        self.use_mixed_precision = use_mixed_precision
        # Directory for torch.profiler TensorBoard traces (None = disabled)
        self.profiler_dir = profiler_dir
        self.use_z_score_norm = use_z_score_norm
        self.task_names = task_names

        # ---- selective fine-tuning ----------------------------------------
        # freeze_backbone: freeze everything except multi_task_head submodule.
        # trainable_task_heads: if set, additionally freeze all task heads
        #   except the listed ones. Requires the model to have a
        #   ``multi_task_head.tasks_heads`` ModuleDict.
        if freeze_backbone or trainable_task_heads:
            self._apply_finetune_freeze(freeze_backbone, trainable_task_heads)
        self.trainable_task_heads = trainable_task_heads

    def _apply_finetune_freeze(
        self,
        freeze_backbone: bool,
        trainable_task_heads: Optional[list],
    ) -> None:
        """Freeze parameters according to the fine-tuning configuration.

        Args:
            freeze_backbone: If True, freeze all parameters outside the
                ``multi_task_head`` submodule.
            trainable_task_heads: List of task-head names (keys of
                ``multi_task_head.tasks_heads``) to keep trainable.  All other
                task heads are frozen.  If None, all task heads remain trainable.
        """
        if freeze_backbone:
            frozen_backbone = 0
            for name, param in self.model.named_parameters():
                if not name.startswith("multi_task_head"):
                    param.requires_grad = False
                    frozen_backbone += 1
            if self.logger:
                self.logger.info(
                    f"[Finetune] Frozen {frozen_backbone} backbone parameter tensors "
                    f"(everything outside multi_task_head)."
                )

        if trainable_task_heads is not None:
            head_module = getattr(self.model, "multi_task_head", None)
            if head_module is None or not hasattr(head_module, "tasks_heads"):
                if self.logger:
                    self.logger.warning(
                        "[Finetune] --trainable_task_heads specified but model has no "
                        "`multi_task_head.tasks_heads` ModuleDict — skipping head freeze."
                    )
                return
            frozen_heads = 0
            unfrozen_heads = 0
            for task_name, head in head_module.tasks_heads.items():
                if task_name not in trainable_task_heads:
                    for param in head.parameters():
                        param.requires_grad = False
                    frozen_heads += 1
                else:
                    for param in head.parameters():
                        param.requires_grad = True  # ensure unfrozen even if backbone freeze ran
                    unfrozen_heads += 1
            if self.logger:
                self.logger.info(
                    f"[Finetune] Task heads — trainable: {trainable_task_heads} "
                    f"({unfrozen_heads}), frozen: {frozen_heads}."
                )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        if self.logger:
            self.logger.info(
                f"[Finetune] Trainable params: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.1f}%)"
            )

        # Prune frozen params from the optimizer so GradScaler only tracks
        # parameters that will actually receive gradients.
        trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) in trainable_ids]
        self.optimizer.param_groups = [g for g in self.optimizer.param_groups if g["params"]]
        if self.logger:
            remaining = sum(len(g["params"]) for g in self.optimizer.param_groups)
            self.logger.info(f"[Finetune] Optimizer pruned to {remaining} trainable parameter tensors.")

    def _classification_accuracies(self, outputs, targets, masks):
        """Compute per-task accuracies from logits and targets."""
        assert len(outputs) == len(targets)
        accu_list = []
        for i, (output, target, mask) in enumerate(zip(outputs, targets, masks)):
            pred = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            pred_cls = np.argmax(pred, axis=1)
            # Only compute accuracy for masked (valid) targets
            valid_indices = mask_np.astype(bool)
            if valid_indices.sum() > 0:  # Avoid division by zero
                accuracy = np.mean(pred_cls[valid_indices] == target_np[valid_indices])
            else:
                accuracy = 0.0
            accu_list.append(float(accuracy))
        return accu_list

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        train_loss_accum = 0.0
        train_losses = []

        batch_id = 0
        last_scale = self.scaler.get_scale() if self.use_mixed_precision else 1.0

        # ------------------------------------------------------------------
        # Torch Profiler – enabled only when profiler_dir is set.
        # Schedule: 1 wait, 1 warmup, 3 active steps per epoch.
        # TensorBoard trace is written via tensorboard_trace_handler so it
        # can be viewed with: tensorboard --logdir <profiler_dir>
        # record_function labels let you see data-loading vs forward/backward
        # separately in the TensorBoard "Trace" and "Operator" views.
        # ------------------------------------------------------------------
        def _make_profiler():
            os.makedirs(self.profiler_dir, exist_ok=True)
            return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

        prof_ctx = _make_profiler() if self.profiler_dir is not None else None
        if prof_ctx is not None:
            prof_ctx.start()
            if self.logger:
                self.logger.info(f"[Profiler] epoch {epoch}: started – trace → {self.profiler_dir}")

        for batch in train_loader:
            batch_id += 1

            # ---- data → device (labelled for profiler) --------------------
            with torch.profiler.record_function("data_loading"):
                images, metadata, targets, masks = batch
                images = images.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device, non_blocking=True)
                targets = [t.to(self.device, non_blocking=True) for t in targets]
                masks = [m.to(self.device, non_blocking=True) for m in masks]

            # For trainable task heads, if the masks indicate no valid samples for all the tasks to be trained, we can skip the forward and backward pass to save computation. This is especially beneficial when using selective fine-tuning with a small subset of trainable heads.
            if self.trainable_task_heads is not None:
                # Determine indices of trainable tasks
                trainable_indices = [i for i, name in enumerate(self.task_names) if name in self.trainable_task_heads]
                # Check if any of the trainable tasks have valid samples in this batch
                if not any(masks[i].bool().any() for i in trainable_indices):
                    if self.logger:
                        self.logger.info(
                            f"Batch {batch_id}: No valid samples for trainable tasks — skipping forward/backward."
                        )
                    continue

            self.optimizer.zero_grad()

            # ---- forward pass (labelled for profiler) --------------------
            with torch.profiler.record_function("forward"):
                if self.use_mixed_precision:
                    with torch.amp.autocast("cuda"):
                        if self.use_z_score_norm:
                            images = normalize_per_sample(images)
                        outputs = self.model(images, metadata)
                else:
                    if self.use_z_score_norm:
                        images = normalize_per_sample(images)
                    outputs = self.model(images, metadata)

            loss, indiv_losses = self.criterion(outputs, targets, masks, self.task_weights)

            # All samples in the batch had "na" labels for every task — no
            # learning signal. Skip backward + optimiser step entirely.
            if loss is None:
                continue

            # Check for NaN in loss and outputs

            if torch.isnan(loss).any():
                self.logger.error(f"NaN detected in loss: {loss}")
                self.logger.error(f"Outputs contain NaN: {[torch.isnan(output).any().item() for output in outputs]}")
                raise RuntimeError("NaN loss detected - stopping training")

            if any(torch.isnan(output).any() for output in outputs):
                self.logger.error("NaN detected in model outputs")
                self.logger.error(
                    f"Input images stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}"
                )
                self.logger.error(
                    f"Input metadata stats: min={metadata.min():.3f}, max={metadata.max():.3f}, mean={metadata.mean():.3f}"
                )
                raise RuntimeError("NaN outputs detected - stopping training")

            if _debug_mode():
                # Loss after scaling
                if self.use_mixed_precision:
                    scaled_loss = loss * self.scaler.get_scale()
                    self.logger.info(f"Scaled loss: {scaled_loss.item():.6e}")

            # ---- backward + optimiser step (labelled for profiler) --------
            with torch.profiler.record_function("backward_optimizer"):
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            if _debug_mode():
                self.logger.info("Check gradient after backward")
                has_nan_grads = False
                max_grad_norm = 0.0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            self.logger.error(f"NaN gradient detected in parameter: {name}")
                            has_nan_grads = True
                        if torch.isinf(param.grad).any():
                            self.logger.error(f"Inf gradient detected in parameter: {name}")
                            has_nan_grads = True
                        max_grad_norm = max(max_grad_norm, param.grad.norm().item())

                if has_nan_grads:
                    self.logger.error("Invalid gradients detected - skipping step")

                # Log gradient norms periodically
                if batch_id % 50 == 0:
                    self.logger.info(f"Max gradient norm: {max_grad_norm:.4f}")

            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()

            current_scale = self.scaler.get_scale() if self.use_mixed_precision else 1.0
            last_scale = current_scale

            # ---- profiler step -------------------------------------------
            if prof_ctx is not None:
                prof_ctx.step()

            # Optional prints (reduced frequency to avoid log spam)
            if self.logger and batch_id % 100 == 0:  # Log every 100 batches instead of every batch
                self.logger.info(
                    f"Batch {batch_id}: Scale: {last_scale:.6e}, LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # Log scale issues
            if last_scale < 1.0:
                self.logger.warning(f"Gradient scale very low: {last_scale:.6e} - possible gradient overflow")
            elif last_scale > 1e6:
                self.logger.warning(f"Gradient scale very high: {last_scale:.6e} - possible gradient underflow")

            # Accumulate
            train_loss_accum += loss.item()
            train_losses.append(indiv_losses)

            # Prepare individual losses list
            indiv_list = []
            if isinstance(indiv_losses, (list, tuple)):
                indiv_list = [float(x) for x in indiv_losses]
            elif hasattr(indiv_losses, "detach"):
                try:
                    indiv_list = [float(x) for x in indiv_losses.detach().cpu().numpy().tolist()]
                except Exception:
                    pass

            current_lr_tb = self.optimizer.param_groups[0]["lr"]
            amp_scale_tb = current_scale

            log_batch_metrics(
                self.tb_logger,
                batch_idx=batch_id,
                total_batches=len(train_loader),
                epoch=epoch,
                loss=float(loss.item()),
                individual_losses=indiv_list,
                accuracies=None,
                lr=current_lr_tb,
                amp_scale=amp_scale_tb,
            )

        avg_train_loss = train_loss_accum / max(1, len(train_loader))
        avg_ind_train = np.average(train_losses, axis=0)

        # ---- stop profiler and print key-averages table ------------------
        if prof_ctx is not None:
            prof_ctx.stop()
            if self.logger:
                self.logger.info(
                    "[Profiler] Top ops by CPU time:\n"
                    + prof_ctx.key_averages().table(sort_by="cpu_time_total", row_limit=15)
                )
                if torch.cuda.is_available():
                    self.logger.info(
                        "[Profiler] Top ops by CUDA time:\n"
                        + prof_ctx.key_averages().table(sort_by="cuda_time_total", row_limit=15)
                    )
                    self.logger.info(
                        "[Profiler] Top ops by CUDA memory:\n"
                        + prof_ctx.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
                    )
                self.logger.info(
                    f"[Profiler] TensorBoard trace saved to: {self.profiler_dir}  "
                    f"(view with: tensorboard --logdir {self.profiler_dir})"
                )

        return avg_train_loss, avg_ind_train

    def validate_epoch(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        val_loss_accum = 0.0
        val_losses = []
        val_acc = []

        with torch.no_grad():
            for batch in val_loader:
                images, metadata, targets, masks = batch
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = [t.to(self.device) for t in targets]
                masks = [m.to(self.device) for m in masks]

                if self.use_mixed_precision:
                    with torch.autocast("cuda"):
                        if self.use_z_score_norm:
                            images = normalize_per_sample(images)
                        outputs = self.model(images, metadata)
                        loss, indiv_losses = self.criterion(outputs, targets, masks, self.task_weights)
                        val_acc.append(self._classification_accuracies(outputs, targets, masks))
                else:
                    if self.use_z_score_norm:
                        images = normalize_per_sample(images)
                outputs = self.model(images, metadata)
                loss, indiv_losses = self.criterion(outputs, targets, masks, self.task_weights)
                val_acc.append(self._classification_accuracies(outputs, targets, masks))

                val_loss_accum += loss.item()
                val_losses.append(indiv_losses)

        avg_val_loss = val_loss_accum / max(1, len(val_loader))
        avg_ind_val = np.average(val_losses, axis=0)
        avg_val_acc = np.mean(val_acc, axis=0)

        return avg_val_loss, avg_ind_val, avg_val_acc

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, save_path: Union[str, bytes]):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            avg_train_loss, avg_ind_train = self.train_epoch(train_loader, epoch)
            avg_val_loss, avg_ind_val, avg_val_acc = self.validate_epoch(val_loader, epoch)

            if self.logger:
                self.logger.info(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f} (Task losses: {avg_ind_train})")
                self.logger.info(f"Epoch {epoch + 1} Val   Loss: {avg_val_loss:.4f} (Task losses: {avg_ind_val})")

            # TB epoch logging
            if self.tb_logger and getattr(self.tb_logger, "enabled", False):
                indiv_epoch = {"train": [float(x) for x in avg_ind_train], "val": [float(x) for x in avg_ind_val]}
                log_training_metrics(
                    self.tb_logger,
                    epoch=epoch,
                    train_loss=float(avg_train_loss),
                    val_loss=float(avg_val_loss),
                    train_accuracies=None,
                    val_accuracies=avg_val_acc,
                    individual_losses=indiv_epoch,
                    optimizer=self.optimizer,
                    model=self.model,
                    scaler=self.scaler,
                )

            # Early stopping + checkpoint
            if avg_val_loss < best_val_loss:
                if self.logger:
                    self.logger.info(
                        f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model."
                    )
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                    },
                    save_path,
                )
            else:
                epochs_no_improve += 1
                if self.logger:
                    self.logger.info(f"No improvement for {epochs_no_improve} epochs.")
                    self.logger.info("Saving latest model checkpoint.")
                # Save the current model as the latest checkpoint
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                    },
                    Path(save_path).with_name("latest_model.pth"),
                )

            if (epochs_no_improve >= self.patience) and (epoch >= 0.7 * num_epochs):
                if self.logger:
                    self.logger.info("Early stopping triggered.")
                break

        if self.logger:
            self.logger.info("Training complete.")

    def test(self, test_loader: DataLoader):
        self.model.eval()
        test_acc = []

        with torch.no_grad():
            for batch in test_loader:
                images, metadata, targets, masks = batch
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = [t.to(self.device) for t in targets]
                masks = [m.to(self.device) for m in masks]

                if self.use_mixed_precision:
                    with torch.amp.autocast("cuda"):
                        if self.use_z_score_norm:
                            images = normalize_per_sample(images)
                        outputs = self.model(images, metadata)
                        test_acc.append(self._classification_accuracies(outputs, targets, masks))
                else:
                    if self.use_z_score_norm:
                        images = normalize_per_sample(images)
                    outputs = self.model(images, metadata)
                    test_acc.append(self._classification_accuracies(outputs, targets, masks))

        avg_test_acc = np.mean(test_acc, axis=0)

        if self.logger:
            self.logger.info(f"Test Accuracy: {avg_test_acc}")

        if self.tb_logger and getattr(self.tb_logger, "enabled", False):
            self.tb_logger.log_scalar("Test/Accuracy", float(np.mean(avg_test_acc)), 0)
            for i, acc in enumerate(avg_test_acc):
                self.tb_logger.log_scalar(f"Test/Task_{i}_Accuracy", float(acc), 0)
            self.tb_logger.flush()
        return avg_test_acc

    def load_checkpoint(self, checkpoint_path: Union[str, bytes]):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
