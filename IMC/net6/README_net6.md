# Net6 (FusionModelV1 + RF heuristic) on Duke

This README describes how to train and evaluate the Network 6 setup for the Duke dataset using:
- Single middle slice image path (IMC CNN image encoder)
- RandomForest metadata classifier used at inference for the SequenceType_Code_norm task
- Heuristic gate: if max(RF probability) ≥ threshold → use RF prediction, else use image prediction

## Environment

- Set environment variables (already done inside scripts):
  - LOCAL_DATASET_PATH=/home/tuan.truong/data/Duke_Liver_Dataset(MRI)_v2
  - LABEL_CSV_PATH=/home/tuan.truong/codebase/IMC/labels/labels_Duke_as_pvai_withFS_v4_local.csv
  - METADATA_PATH=/home/tuan.truong/codebase/IMC/labels/duke_encoded_metadata_20260107.parquet

- Activate virtual environment:
```
uv sync
source .venv/bin/activate
```

## 1) Train pixel (image) path for net6 (5-fold CV)

This trains the image encoder using single-slice inputs with focal loss per task (alpha=1.0, gamma=2.0). The metadata path is not used during training; RF will be used at inference for SequenceType_Code_norm.

```
python IMC/net6/train06_duke_fusion_v1.py --backbone densenet121 --batch_size 8 --num_epochs 15 --gpu 1 --threshold 0.7
```

Notes:
- Focal loss is applied in both train and val loops.
- You can tune alpha/gamma in the script (train06_duke_fusion_v1.py) if needed.

Outputs per fold:
- logs/<timestamp>_net6_duke_5fold/fold_<i>/best_model.pth
- TensorBoard logs
- CV summary CSV at the end

## 2) Train RF metadata classifier for SequenceType_Code_norm

Train the RF on the Duke metadata (using selected features when `--use_selected` is passed) and save the joblib pipeline.

```
python IMC/net6/metadata_rf_train_duke.py --out_dir ./models_net6_rf --use_selected
```

This produces:
- ./models_net6_rf/duke_metadata_rf.joblib
- Prints validation accuracy

## 3) Inference with RF gating for SequenceType_Code_norm

Run inference using the trained net6 image model checkpoint and the RF model. The script applies the heuristic gate for SequenceType_Code_norm only:
- If RF max probability ≥ threshold → use RF logits
- Else → use image logits
- Other tasks use image logits

```
python IMC/net6/infer06_duke.py \
  --ckpt logs/<timestamp>_net6_duke_5fold/fold_<i>/best_model.pth \
  --output_dir ./out_net6_duke \
  --backbone densenet121 \
  --threshold 0.7 \
  --rf_model ./models_net6_rf/duke_metadata_rf.joblib \
  --eval
```

Outputs:
- predictions.csv in output_dir
- inference_metadata.json in output_dir (records checkpoint and tasks)
- Optional evaluation results when `--eval` is set

## Notes

- SequenceType_Code_norm gating is the only task using RF, per the original method. If you want RF gating for additional tasks, you will need additional RF models.
- The single-slice selection is configured via the Duke dataloader (n_slices=1, sampling_type="equidistant").
- Default threshold is 0.7; tune based on validation results.
- Backbones supported by IMC image encoder include densenet121, efficientnet variants, resnet50, etc.
