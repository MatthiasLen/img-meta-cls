"""
    TRAINING SCRIPT FOR MODEL 04 WITH SPARSE METADATA ENCODER
    - Uses local dataset with sparse metadata encoder
    - Custom weight initialization
    - AdamW optimizer with separate weight decay
    - Warmup + cosine decay learning rate scheduler
    - Multi-task classification loss
    - Combined logging (file + TensorBoard)
    - Profiling setup

"""

from typing import Union
import math
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from IMC.nn.multi_task_loss import MultiTaskLoss
from IMC.helper import capture_console_to_log, log_training_start, log_training_end
from IMC.tensorboard_logging import setup_combined_logging
import time 

# Set environment variables or paths for local dataset
os.environ["DEBUG_MODE"] = "0"  # Enable debug mode
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/PV.AI"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/encoded_metadata_20251217.parquet"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20250603_local.csv"


def init_weights(module: nn.Module) -> None:
    """
    Custom weight initialization for neural network layers.

    Applies Xavier (Glorot) uniform initialization to Linear layers.
    Sets LayerNorm weights to 1 and biases to 0.
    Useful for consistent initialization when using custom models.

    Args:
        module (nn.Module): A layer or submodule of the model to initialize.
    
    Usage:
        model.apply(init_weights)
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def get_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_scale_factor : float = 100,
    min_scale_factor : float = 0.1
) -> LambdaLR:
    """
    Warmup + cosine decay scheduler with learning rates expressed as ratios of initial LR.

    Args:
        optimizer: Optimizer whose lr will be scheduled.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of steps.
    Returns:
        LambdaLR scheduler.
    """
    
    def lr_lambda(current_step: int) -> float:
        if (current_step <= warmup_steps) and (warmup_steps > 0):
            return max(1, peak_scale_factor * float(current_step) / warmup_steps)
        elif  (current_step > warmup_steps) and (current_step <= total_steps):
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            a = math.cos(math.pi * progress)
            return max(min_scale_factor, peak_scale_factor * 0.5 * (1 + a))
        else:
            return min_scale_factor

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer(model: torch.nn.Module, lr: float = 1.0e-6, weight_decay: float = 1e-2, eps: float = 1e-8) -> Optimizer:
    """
    AdamW optimizer with separate weight decay for bias and norm layers.

    Args:
        model: Model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        AdamW optimizer.
    """
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ["bias", "norm", "ln", "layernorm", "bn"]):
            no_decay.append(param)
        else:
            decay.append(param)

    return AdamW([{"params": decay, "weight_decay": weight_decay},{"params": no_decay, "weight_decay": 0.0}],lr=lr, eps=eps)


def classification_losses(outputs, targets):
    
    assert len(outputs) == len(targets)

    # iterate over tasks
    for i in range(len(outputs)):
        pred = outputs[i].clone().detach().cpu().numpy()
        trag_cl = targets[i].clone().detach().cpu().numpy()
        pred_cl = np.argmax(pred, axis=1)
        accuracy = np.mean(np.array(pred_cl) == np.array(trag_cl))
        print(f"Task {i} Accuracy:", accuracy)


# Example usage:
if __name__ == "__main__":
    from IMC.network04 import MRISequenceClassifierWithSparseMetadata
    from IMC.data.liver_dataloader_local import get_train_dataloader, get_valid_dataloader, get_test_dataloader
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Train MRI Sequence Classifier")
    parser.add_argument("--backbone", type=str, default="densenet", help="Image encoder backbone")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id to use")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--version", type=str, default="v1", help="Sparse encoder version: v1 or v2")
    parser.add_argument("--fusion_module_version", type=str, default="v1", help="Fusion module version: v1 or v2")
    args = parser.parse_args()

    # Directory for profiling
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("./logs", timestamp)
    profiler_dir = os.path.join(log_dir, "profiler")
    os.makedirs(profiler_dir, exist_ok=True)
    experiment_name = f"model_04_sparse_metadata_encoder_{args.backbone}_{args.version}"

    # Setup combined logging (file + TensorBoard)
    logger, log_path, tb_logger = setup_combined_logging(
        experiment_name=experiment_name,
        log_dir=log_dir,
        tb_log_dir=log_dir
    )
    # Log training start
    batch_size = args.batch_size
    num_epochs = 15
    lr = 1e-6
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    incl_regression = True
    metadata_dropout = False
    version = args.version
    log_training_start(logger, config={"device": str(device), "batch_size": batch_size, "num_epochs": num_epochs, "dataset_version": "local", "model_version": "04", "impute": "yes", "learning_rate": lr, "checkpoint": args.ckpt, "sparse_encoder": True, "img_enc_backbone": args.backbone, "metadata_dropout": metadata_dropout, "sparse_encoder_version": version, "fusion_module_version": args.fusion_module_version})

    with capture_console_to_log(logger):
        num_samples = None  # Use all samples
        aggregated_metadata= False
        use_preselected_features = False
        exclude_contrast_yn = True
        train_loader = get_train_dataloader(batch_size=batch_size, num_samples=num_samples, num_workers=4, aggregated_metadata=aggregated_metadata, use_preselected_features=use_preselected_features, exclude_contrast_yn=exclude_contrast_yn)
        val_loader = get_valid_dataloader(batch_size=batch_size, num_samples=num_samples, num_workers=4, aggregated_metadata=aggregated_metadata, use_preselected_features=use_preselected_features, exclude_contrast_yn=exclude_contrast_yn)
        test_loader = get_test_dataloader(batch_size=batch_size, num_samples=num_samples, num_workers=4, aggregated_metadata=aggregated_metadata, use_preselected_features=use_preselected_features, exclude_contrast_yn=exclude_contrast_yn)
        cl_d = train_loader.dataset.get_n_labels()
        print("Label config", cl_d)

        metadata_input_dim = train_loader.dataset.num_metadata_features 
        if version == "v1":
            model = MRISequenceClassifierWithSparseMetadata(metadata_input_dim=metadata_input_dim, num_classes_dict=cl_d, metadata_embeder_type="sparse", img_enc_backbone=args.backbone, dropout_metadata=metadata_dropout, fusion_module_version=args.fusion_module_version)
        elif version == "v2":
            model = MRISequenceClassifierWithSparseMetadata(metadata_input_dim=metadata_input_dim, num_classes_dict=cl_d, metadata_embeder_type="sparse_v2", img_enc_backbone=args.backbone, dropout_metadata=metadata_dropout, fusion_module_version=args.fusion_module_version)
        elif version == "v5":
            model = MRISequenceClassifierWithSparseMetadata(metadata_input_dim=metadata_input_dim, num_classes_dict=cl_d, metadata_embeder_type="sparse_v5", img_enc_backbone=args.backbone, dropout_metadata=metadata_dropout, fusion_module_version=args.fusion_module_version)
        model.to(device)
    
        # Initialize weights once before training, do NOT overwrite pretrained weights inside backbone
        model.apply(init_weights)

        # Save the model architecture as text for reference
        model_arch_path = os.path.join(log_dir, "model_architecture.txt")
        with open(model_arch_path, "w") as f:
            f.write(str(model))

        # lr scheduler
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # warmup for 10% of total steps
        
        # optimizer
        eps = 1.0e-7
        optimizer = create_optimizer(model, lr=lr, eps=eps)  
        scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

        # loss
        criterion = MultiTaskLoss(label_smoothing=0.1, incl_regression=True, task_names=list(cl_d.keys()))

        # gradient scaler
        scaler = torch.amp.GradScaler("cuda", init_scale=2**16)


        from IMC.trainer import Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            tb_logger=tb_logger,
            logger=logger,
            patience=5,
            incl_regression=incl_regression,
            use_mixed_precision=True
        )
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            save_path=os.path.join(log_dir, "best_model.pth")
        )
        # Load best model for testing
        trainer.load_checkpoint(os.path.join(log_dir, "best_model.pth"))
        test_acc = trainer.test(test_loader)
        print(f"Profiler results saved to {profiler_dir}. Run: tensorboard --logdir {profiler_dir}")

    # Log training end
    log_training_end(logger)