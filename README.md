# IMC (Image Plus Metadata Classifier)

IMC is a deep learning system for multi-task medical image classification, specifically designed for MRI sequence classification. The project supports multiple architectures ranging from 2-D image-only models to 3-D volumetric networks, and can incorporate DICOM metadata through cross-attention fusion.

## Model Deployment

Deploy the IMC model as a serverless inference service on Google Cloud Platform:

```bash
cd terraform

# build container and push it to Artifact Registry
./build.sh

# run terraform and deploy infrastructure
# Note: alternatively use the script `init_validate_apply.sh`
terraform init
terraform plan
terraform apply
```

Test the deployed service via curl:

```bash
curl -i -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  'https://imc-inference-service-sir5sxwxha-ez.a.run.app/ready'
```

Or use `/terraform/test_client.py` for health checks and prediction requests.

## Development Commands

### Environment Setup

```bash
# Install dependencies using uv (modern Python package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Training and Development

#### Network 4 – Cross-attention image + metadata fusion

```bash
# Duke Liver MRI dataset (5-fold CV)
python -m IMC.net4_duke.train --modality combined --gpu 0

# ADNI Brain MRI dataset (5-fold CV)
python -m IMC.net4_adni.train --modality combined --gpu 0
```

#### Network 6 – 2-D image-only (PixelOnlyModel) on Duke

```bash
# Train 5-fold CV
python -m IMC.net6.train_duke --gpu 0

# Train Random Forest metadata gate (optional)
python -m IMC.net6.train_rf_duke --out_dir ./rf_checkpoints/duke_net6

# Inference (with optional RF gate)
python -m IMC.net6.infer_duke \
    --ckpt ./logs/.../fold_0/best_model.pth \
    --output_dir ./infer_out/fold_0 \
    --rf_model ./rf_checkpoints/duke_net6/duke_metadata_rf_fold_0.joblib
```

#### Network 7 – 3-D volumetric (PyramidPooling3DClassifier)

```bash
# Duke – train a single fold (repeat for --fold 1..4)
python -m IMC.net7.train_duke --fold 0 --backbone_type resnet --gpu 0

# ADNI – train a single fold
python -m IMC.net7.train_adni --fold 0 --backbone_type resnet --gpu 0

# Inference
python -m IMC.net7.infer_duke \
    --ckpt ./logs/.../fold_0/best_model.pth \
    --output_dir ./infer_out/duke/fold_0
python -m IMC.net7.infer_adni \
    --ckpt ./logs/.../fold_0/best_model.pth \
    --output_dir ./infer_out/adni/fold_0
```

## Model Architectures

| Network | Class                        | Input      | Modality         | Datasets   |
|---------|------------------------------|------------|------------------|------------|
| `net4`  | `CrossAttentionFusionModel`  | 2-D slices | Image + metadata | Duke, ADNI |
| `net5`  | `UnifiedTransformerModel`    | 2-D slices | Image + metadata | PV.AI      |
| `net6`  | `PixelOnlyModel`             | 2-D slice  | Image only       | Duke       |
| `net7`  | `PyramidPooling3DClassifier` | 3-D volume | Image only       | Duke, ADNI |

## File Organization

### Main Source (`IMC/`)
- `network01.py` – `network07.py`: Evolution of model architectures (v01–v07)
- `net4/`: Original Network 4 scripts (PV.AI dataset)
- `net4_duke/`: Network 4 pipeline for Duke Liver MRI (5-fold CV) — see [`IMC/net4_duke/README.md`](IMC/net4_duke/README.md)
- `net4_adni/`: Network 4 pipeline for ADNI Brain MRI (5-fold CV) — see [`IMC/net4_adni/README.md`](IMC/net4_adni/README.md)
- `net5/`: Network 5 training and inference scripts
- `net6/`: Network 6 (PixelOnlyModel) on Duke; includes RF metadata gate — see [`IMC/net6/README.md`](IMC/net6/README.md)
- `net7/`: Network 7 (PyramidPooling3DClassifier) on Duke and ADNI — see [`IMC/net7/README.md`](IMC/net7/README.md)
- `trainer.py`: Reusable training framework with mixed precision
- `evaluate.py` / `evaluate_duke.py`: Model evaluation and metrics
- `helper.py`: Utility functions (visualization, normalization)
- `tensorboard_logging.py`: TensorBoard + CSV combined logging

### Neural Networks (`IMC/nn/`)
- `image_encoder.py`: CNN backbone for multi-slice images (DenseNet121/ResNet18)
- `metadata_encoder.py`: MLP with contextual imputation for DICOM features
- `sparse_metadata_encoder.py`: Advanced sparse feature handling with FiLM
- `multi_task_head.py`: Classification heads for multiple tasks
- `multi_task_loss.py`: Combined loss functions with label smoothing

### Data Pipeline (`IMC/data/`)
- `duke_dataloader_local.py`: Duke Liver MRI dataset class (2-D and 3-D variants)
- `adni_dataloader_local.py`: ADNI Brain MRI dataset class (2-D and 3-D variants)
- `liver_dataloader_local.py`: PV.AI dataset class for development
- `liver_dataloader_gcp.py`: GCP dataset class with Dataflux integration
- `dicom_tag_encoding.py`: DICOM metadata feature encoding (89 features)
- `augment.py`: Image augmentation configurations (multiple presets)

### Model Export (`IMC/onnx/`)
- `onnx_metadata_export.py`: ONNX model export utilities

### Deployment (`terraform/`)
- `main.tf`: Terraform infrastructure definition
- `terraform.tfvars`: Terraform configuration variables
- `/service`: Flask app for serving inference requests
- `build.sh`: Build Docker container and push to Artifact Registry
- `init_validate_apply.sh`: Run terraform init → validate → plan → apply cycle
