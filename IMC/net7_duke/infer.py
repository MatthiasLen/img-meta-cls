"""
Inference script for Network 07 (3D Pyramid Pooling Network) on Duke Dataset.

This script loads a trained PyramidPooling3DClassifier model and runs inference
on a specified data split. It saves the predictions to a CSV file.

Authors: Claude Code
Date: 2026
"""

import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm

from IMC.network07 import PyramidPooling3DClassifier
from IMC.evaluate_duke import run_evaluation
from IMC.helper import normalize_per_sample

os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv"
os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet"


def main(ckpt: str, output_dir: str, data_split: list, target_depth: int, batch_size: int, device: torch.device):
    """
    Main inference function.

    Args:
        ckpt: Path to the model checkpoint.
        output_dir: Directory to save the predictions CSV.
        data_split: List of fold names to run inference on.
        target_depth: Target depth for 3D volumes.
        batch_size: Batch size for inference.
        device: The device to run inference on.
    """
    # Dataloader
    infer_dataset = DukeLiverDataset3D(
        split=data_split,
        target_depth=target_depth,
        augment_conf="NONE3D",
        is_infer=True
    )
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Running inference on {len(infer_dataset)} samples from splits: {data_split}")

    # Model
    model = PyramidPooling3DClassifier(num_classes_dict=infer_dataset.get_n_labels()).to(device)

    if os.path.exists(ckpt):
        print(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")

    model.eval()

    # Inference loop
    all_predictions = []
    all_filepaths = []
    label_map = DUKE_ORIGINAL_LABEL_NAMES["SequenceType_Code_norm"]

    with torch.no_grad():
        for images, filepaths in tqdm(infer_loader, desc="Inferring"):
            images = normalize_per_sample(images)  # Normalize each sample individually
            images = images.to(device)

            outputs = model(images)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            pred_names = [label_map[p] for p in preds]
            all_predictions.extend(pred_names)
            all_filepaths.extend([f[len("/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2/"):] for f in filepaths])  # Store relative paths

    # Save predictions
    output_df = pd.DataFrame({
        "Filepath": all_filepaths,
        "SequenceType_Code_norm": all_predictions
    })

    output_path = os.path.join(output_dir, "predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return output_df

if __name__ == '__main__':

    from IMC.data.duke_dataloader_3d import DukeLiverDataset3D, DUKE_ORIGINAL_LABEL_NAMES

    parser = argparse.ArgumentParser(description="Inference for 3D Pyramid Pooling Network")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output CSV file')
    parser.add_argument('--data_split', type=str, default="fold_0", help='Data split to run inference on (e.g., "fold_0")')
    parser.add_argument("--target_depth", type=int, default=64, help="Target depth for 3D volumes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # The split can be a single fold or a list of folds
    data_split = [s.strip() for s in args.data_split.split(',')]

    pred_df = main(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        data_split=data_split,
        target_depth=args.target_depth,
        batch_size=args.batch_size,
        device=device
    )

    # Optionally, run evaluation if the ground truth labels are available
    label_csv_path = os.getenv("LABEL_CSV_PATH")
    if label_csv_path and os.path.exists(label_csv_path):
        print("\nRunning evaluation...")
        label_df = pd.read_csv(label_csv_path)
        run_evaluation(pred_df, label_df, args.output_dir)
    else:
        print("\nSkipping evaluation because LABEL_CSV_PATH is not set or file does not exist.")
