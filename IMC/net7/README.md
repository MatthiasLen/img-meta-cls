# IMC Net7 - PyramidPooling3DClassifier on Duke

`net7` contains the Duke Liver MRI training and inference scripts for the 3D volumetric baseline described in the paper.

## Included entry points

- `train_duke.py`: single-fold training for 5-fold cross-validation
- `infer_duke.py`: batch inference and optional evaluation

## Model summary

`IMC.network07.PyramidPooling3DClassifier` combines:

- a 3D CNN backbone (`resnet` or `densenet*` variants),
- 3D pyramid pooling,
- an optional projection MLP,
- a multi-task classification head.

## Typical usage

Train one fold:

```bash
uv run python -m IMC.net7.train_duke --fold 0 --backbone_type resnet --gpu 0
```

Run inference:

```bash
uv run python -m IMC.net7.infer_duke \
  --ckpt ./logs/<run>/fold_0/best_model.pth \
  --output_dir ./infer_out/duke/fold_0 \
  --split fold_0
```

## Required environment variables

- `LOCAL_DATASET_PATH`
- `LABEL_CSV_PATH`

Both can also be provided as CLI arguments via `--dataset_path` and `--label_csv_path`.

## Notes

- This public repository snapshot does not include the older ADNI scripts referenced in earlier internal documentation.
- Run one process per fold to complete the full 5-fold Duke evaluation.
