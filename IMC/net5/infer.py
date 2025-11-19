
import argparse
import os

# Set environment variables or paths for local dataset
IS_DUKE = os.environ.get("IS_DUKE", "0") == "1"
os.environ["DEBUG_MODE"] = "0"  # Enable debug mode
if not IS_DUKE:
    os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/PV.AI"
    os.environ["METADATA_PATH"] = "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20251114_encoded_local.csv"
    os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20250603_local.csv"
else:
    os.environ["LOCAL_DATASET_PATH"] = "/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2"
    os.environ["LABEL_CSV_PATH"] = "/home/tuan.truong/codebase/IMC/labels/labels_Duke_v4_local.csv"

import torch
import pandas as pd
import torch
from tqdm import tqdm
from typing import List
from pathlib import Path
import numpy as np
from IMC.network05 import UnifiedTransformerModel
from IMC.data.liver_dataloader_local import get_infer_dataloader
from IMC.data.liver_dataloader_local import DEFAULT_LABEL_NAMES, DUKE_LABEL_NAMES
from IMC.helper import normalize_per_sample
from IMC.evaluate import run_evaluation

DUKE_MAP_LABELS = {
    "label_SequenceType": {
        "SUB": "OTHER",
        "BOLUS": "OTHER",
        "DIXON_F": "OTHER"
    },
    "label_AcquisitionPlane": {
        "SAG": "OTHER",
        "ORTHO": "OTHER",
        "ROT": "OTHER"
    },
    "label_ContrastPhase": {
        "hepa": "late",
        "trans": "late"
    }
}



def main(ckpt: str, output_dir: str, use_duke: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not use_duke:
        dataloader = get_infer_dataloader(
            batch_size=16,
            num_workers=4,
            num_samples=None,
            folder_split=["fold_9"],
        )
    else:
        dataloader = get_infer_dataloader(
            batch_size=16,
            num_workers=4,
            num_samples=None,
            folder_split=None,
        )
    num_classes_dict = dataloader.dataset.get_n_labels()
    model = UnifiedTransformerModel(
        metadata_input_dim=88,
        metadata_embed_dim=128,
        transformer_dim=256,
        num_transformer_layers=4,
        num_heads=8,
        num_classes_dict=num_classes_dict,
        sparse_metadata_encoder=True
    )

    label_maps = dataloader.dataset.label_names

    if os.path.exists(ckpt):
        print(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {ckpt}")
        return
    
    model.to(device)
    model.eval()
    predictions = {task: [] for task in num_classes_dict.keys()}
    filepaths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            images, metadata, paths = batch
            images = images.to(device)
            metadata = metadata.to(device)

            images = normalize_per_sample(images)
            outputs = model(images, metadata)
            # outputs = model(images)

            for i, task in enumerate(num_classes_dict.keys()):
                preds = torch.argmax(outputs[i], dim=1).cpu().numpy()
                preds_in_names = [label_maps[task][p] for p in preds]
                if use_duke and task in DUKE_MAP_LABELS:
                    preds_in_names = [DUKE_MAP_LABELS[task].get(name, name) for name in preds_in_names]
                predictions[task].extend(preds_in_names)
            filepaths.extend(paths)

    # Organize predictions
    output_df = pd.DataFrame({"Filepath": filepaths})
    tasks = predictions.keys()

    for task in tasks:
        output_df[task] = predictions[task]

    # Save to CSV
    output_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    print(f"Predictions saved to {output_dir}")
    
    return output_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output CSV file')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pred_df = main(args.ckpt, args.output_dir, use_duke=IS_DUKE)
    label_df = pd.read_csv("/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv" if IS_DUKE else "/home/tuan.truong/codebase/IMC/labels/pvai_labels_20250603_local.csv")
    # Run evaluation
    run_evaluation(pred_df, label_df, args.output_dir, is_duke=IS_DUKE)