"""
Inference script for Duke dataset using Network v06 (FusionModelV1) with heuristic gate and single-slice inference.
"""

import argparse
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

# Duke environment
os.environ["DEBUG_MODE"] = "0"
os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet"

from torch.utils.data import DataLoader
from IMC.data.duke_dataloader_local import LiverDataset, DUKE_ORIGINAL_LABEL_NAMES
from IMC.network06 import PixelOnlyModel
from IMC.evaluate_duke import run_evaluation
import joblib


def heuristic_gate(meta_probs: list[torch.Tensor], img_probs: list[torch.Tensor], tasks: list[str], threshold: float = 0.7) -> list[torch.Tensor]:
    chosen = []
    for i, _ in enumerate(tasks):
        max_meta = meta_probs[i].max(dim=1).values
        use_meta = max_meta > threshold
        chosen_task = torch.where(use_meta.unsqueeze(1), meta_probs[i], img_probs[i])
        chosen.append(chosen_task)
    return chosen


def create_inference_dataloader(
    batch_size: int = 16,
    num_workers: int = 4,
    fold_indices: list[int] | None = None,
    aggregated_metadata: bool = False,
    use_preselected_features: bool = False,
    exclude_contrast_yn: bool = True,
) -> DataLoader:
    if fold_indices is not None:
        split = [f"fold_{i}" for i in fold_indices]
    else:
        split = None
    dataset = LiverDataset(
        split=split,
        num_samples=None,
        n_slices=1,
        sampling_type="equidistant",
        augment_conf="IMAGENET299_CENTER",
        aggregated_metadata=aggregated_metadata,
        use_preselected_features=use_preselected_features,
        exclude_contrast_yn=exclude_contrast_yn,
        is_infer=True,
        label_names=DUKE_ORIGINAL_LABEL_NAMES,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_model(
    checkpoint_path: str,
    num_classes_dict: dict,
    metadata_input_dim: int,
    img_enc_backbone: str = "densenet121",
    device: torch.device | None = None,
) -> PixelOnlyModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelOnlyModel(
        num_classes_dict=num_classes_dict,
        image_backbone=img_enc_backbone,
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.7,
    rf_model_path: str | None = None,
) -> tuple[dict, list]:
    num_classes_dict = dataloader.dataset.get_n_labels()
    label_maps = dataloader.dataset.label_names
    predictions = {task: [] for task in num_classes_dict.keys()}
    filepaths = []
    print(f"Running inference on {len(dataloader.dataset)} samples...")
    # Optional load RF
    rf_model = None
    if rf_model_path is not None and len(rf_model_path) > 0:
        print(f"Loading RF model: {rf_model_path}")
        rf_model = joblib.load(rf_model_path)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            images, metadata, paths = batch
            images = images.to(device)
            # Preprocessing is handled by IMAGENET299_CENTER in the dataloader
            # No additional per-sample normalization here
            # Image logits from net6 image path
            img_logits = model.get_image_logits(images)

            # RF gating for SequenceType_Code_norm only
            # metadata is (B, D) when aggregated_metadata=True or per-slice; for single slice, dataloader returns per-slice metadata (B, 1, D) or (B, D) depending; here it's (B, D)
            # Compute RF probabilities and logits for sequence task
            seq_task = "SequenceType_Code_norm" if "SequenceType_Code_norm" in dataloader.dataset.label_names else list(num_classes_dict.keys())[0]
            seq_idx = list(num_classes_dict.keys()).index(seq_task) if seq_task in num_classes_dict else 0
            if rf_model is None:
                # fall back to metadata head if RF not provided
                _, meta_logits = model.get_stream_logits(images, metadata.to(device))
                meta_probs_seq = torch.softmax(meta_logits[seq_idx], dim=-1)
            else:
                # Use RF probs
                # RF expects numpy; loop over batch
                meta_np = metadata.numpy()
                rf_probs = []
                for row in meta_np:
                    prob = rf_model.predict_proba(row.reshape(1, -1))[0]
                    rf_probs.append(prob)
                meta_probs_seq = torch.tensor(rf_probs, dtype=torch.float32, device=device)
            # Heuristic gating for sequence task
            max_meta = meta_probs_seq.max(dim=1).values
            use_meta = max_meta > threshold
            # chosen logits for sequence: metadata logits (if available) or image logits
            if rf_model is None:
                _, meta_logits = model.get_stream_logits(images, metadata.to(device))
                chosen_logits_seq = torch.where(use_meta.unsqueeze(1), meta_logits[seq_idx], img_logits[seq_idx])
            else:
                # Convert RF probs to logits by log-softmax inverse: logits = log(probs)
                rf_logits_seq = torch.log(torch.clamp(meta_probs_seq, min=1e-8))
                chosen_logits_seq = torch.where(use_meta.unsqueeze(1), rf_logits_seq, img_logits[seq_idx])
            preds_seq = torch.argmax(chosen_logits_seq, dim=1).cpu().numpy()
            preds_seq_in_names = [label_maps[seq_task][p] for p in preds_seq]
            predictions[seq_task].extend(preds_seq_in_names)

            # For remaining tasks (no RF): use image logits directly
            for i, task in enumerate(num_classes_dict.keys()):
                if task == seq_task:
                    continue
                preds = torch.argmax(img_logits[i], dim=1).cpu().numpy()
                preds_in_names = [label_maps[task][p] for p in preds]
                predictions[task].extend(preds_in_names)
            filepaths.extend(paths)
    # Optional: derive label_Contrast from label_ContrastPhase if present
    if "label_ContrastPhase" in label_maps:
        predictions["label_Contrast"] = [
            "post" if phase not in ["pre", "na"] else "pre"
            for phase in predictions.get("label_ContrastPhase", [])
        ]
    return predictions, filepaths


def save_predictions(predictions: dict, filepaths: list, output_dir: str, checkpoint_path: str | None = None) -> pd.DataFrame:
    output_df = pd.DataFrame({"Filepath": filepaths})
    for task in predictions.keys():
        output_df[task] = predictions[task]
    pred_path = os.path.join(output_dir, "predictions.csv")
    output_df.to_csv(pred_path, index=False)
    print(f"\n✓ Predictions saved to {pred_path}")
    if checkpoint_path:
        metadata = {
            "checkpoint": checkpoint_path,
            "num_samples": len(filepaths),
            "tasks": list(predictions.keys()),
        }
        metadata_path = os.path.join(output_dir, "inference_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Run net6 inference on Duke dataset")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--folds', type=str, default=None, help='Comma-separated fold indices (e.g., "0,1,2")')
    parser.add_argument('--backbone', type=str, default="densenet121", help='Image encoder backbone')
    parser.add_argument('--threshold', type=float, default=0.7, help='Heuristic confidence threshold')
    parser.add_argument('--eval', action='store_true', help='Evaluate predictions against ground truth')
    parser.add_argument('--rf_model', type=str, default='', help='Path to RF model joblib for SequenceType_Code_norm')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.folds is not None:
        fold_indices = [int(f) for f in args.folds.split(",")]
        print(f"Running inference on folds: {fold_indices}")
    else:
        fold_indices = None
        print("Running inference on all data")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    aggregated_metadata = False
    use_preselected_features = False
    exclude_contrast_yn = True

    print("\nCreating dataloader...")
    dataloader = create_inference_dataloader(
        batch_size=args.batch_size,
        num_workers=4,
        fold_indices=fold_indices,
        aggregated_metadata=aggregated_metadata,
        use_preselected_features=use_preselected_features,
        exclude_contrast_yn=exclude_contrast_yn,
    )
    print(f"✓ Loaded {len(dataloader.dataset)} samples")

    num_classes_dict = dataloader.dataset.get_n_labels()
    metadata_input_dim = dataloader.dataset.num_metadata_features if use_preselected_features else 119
    print("\nModel configuration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Metadata dim: {metadata_input_dim}")
    print(f"  Tasks: {list(num_classes_dict.keys())}")

    model = load_model(
        checkpoint_path=args.ckpt,
        num_classes_dict=num_classes_dict,
        metadata_input_dim=metadata_input_dim,
        img_enc_backbone=args.backbone,
        device=device,
    )

    predictions, filepaths = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        rf_model_path=args.rf_model,
    )

    pred_df = save_predictions(
        predictions=predictions,
        filepaths=filepaths,
        output_dir=args.output_dir,
        checkpoint_path=args.ckpt,
    )

    if args.eval:
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        label_csv_path = os.environ.get("LABEL_CSV_PATH")
        label_df = pd.read_csv(label_csv_path)
        run_evaluation(pred_df, label_df, args.output_dir)
        print(f"\n✓ Evaluation results saved to {args.output_dir}")

    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Total samples processed: {len(filepaths)}")


if __name__ == '__main__':
    main()
