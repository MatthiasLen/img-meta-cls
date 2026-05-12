# IMC (Image Plus Metadata Classifier)

IMC is a deep learning system for multi-task medical image classification, specifically designed for MRI sequence classification. The project supports multiple architectures ranging from 2-D image-only models to 3-D volumetric networks, and can incorporate DICOM metadata through cross-attention fusion.

## Model Deployment

Deploy the IMC model as a serverless inference service on Google Cloud Platform:

```bash
cd terraform

# build container and push it to Artifact Registry
./build.sh

# run terraform and deploy infrastructure
terraform init
terraform plan
terraform apply
# Note: alternatively use the script `init_validate_appy.sh`
```

Test the deployed service via curl:

```bash
curl -i -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  'https://imc-inference-service-sir5sxwxha-ez.a.run.app/ready'
```

Or use `terraform/service/` for health checks and prediction requests.

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
# PV.AI liver MRI dataset (5-fold CV)
python -m IMC.net4.train --modality combined --gpu 0

# Brain MRI datasets (ADNI, GadOnly, etc.)
python -m IMC.net4.train_brain --modality combined --gpu 0

# CT contrast detection
python -m IMC.net4.train_ct --modality combined --gpu 0
```

All three entry points accept `--modality combined|image|metadata` for fusion / image-only / metadata-only ablations.

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

#### Network 7 – 3-D volumetric (PyramidPooling3DClassifier) on Duke

```bash
# Train a single fold (repeat for --fold 1..4)
python -m IMC.net7.train_duke --fold 0 --backbone_type resnet --gpu 0

# Run all folds in parallel
for i in 0 1 2 3 4; do
    python -m IMC.net7.train_duke --fold $i --gpu 0 &
done

# Inference
python -m IMC.net7.infer_duke \
    --ckpt ./logs/.../fold_0/best_model.pth \
    --output_dir ./infer_out/duke/fold_0
```

#### XGBoost – Metadata-only baseline

```bash
python -m IMC.xgboost.cv
```

## Model Architectures

| Network   | Class                        | Input      | Modality         | Datasets                      |
|-----------|------------------------------|------------|------------------|-------------------------------|
| `net4`    | `MRISequenceClassifier`      | 2-D slices | Image + metadata | PV.AI liver, Brain MRI, CT    |
| `net6`    | `PixelOnlyModel`             | 2-D slice  | Image only       | Duke liver                    |
| `net7`    | `PyramidPooling3DClassifier` | 3-D volume | Image only       | Duke liver                    |
| `xgboost` | XGBoost cross-validation     | —          | Metadata only    | Duke liver                    |

## File Organization

### Main Source (`IMC/`)
- `network01.py` – `network07.py`: Evolution of model architectures (v01–v07)
- `net4/`: Network 4 — cross-attention fusion on PV.AI liver, Brain MRI, and CT contrast — see [`IMC/net4/README.md`](IMC/net4/README.md)
- `net6/`: Network 6 — `PixelOnlyModel` on Duke; includes RF metadata gate — see [`IMC/net6/README.md`](IMC/net6/README.md)
- `net7/`: Network 7 — `PyramidPooling3DClassifier` on Duke — see [`IMC/net7/README.md`](IMC/net7/README.md)
- `xgboost/`: XGBoost metadata-only baseline (`cv.py`)
- `trainer.py`: Reusable training framework with mixed precision
- `evaluate.py` / `evaluate_duke.py`: Model evaluation and metrics
- `helper.py`: Utility functions (visualization, normalization)
- `tensorboard_logging.py`: TensorBoard + CSV combined logging

### Neural Networks (`IMC/nn/`)
- `image_encoder.py`: 2-D CNN backbone for multi-slice images (DenseNet121/ResNet18)
- `resnet_3d.py` / `densenet_3d.py`: 3-D CNN backbones for volumetric inputs
- `pyramid_pooling_3d.py`: 3-D multi-scale pyramid pooling module
- `metadata_encoder.py`: MLP with contextual imputation for DICOM features
- `emb_metadata_encoder.py`: Embedding-based metadata encoder
- `sparse_metadata_encoder.py`: Sparse feature handling with FiLM conditioning
- `sparse_metadata_encoder_v2.py` / `sparse_metadata_encoder_v5.py`: Evolved SME variants
- `sparse_metadata_encoder_v1_onnx.py`: ONNX-compatible SME variant
- `multi_task_head.py`: Classification heads for multiple tasks
- `multi_task_loss.py`: Combined loss functions with label smoothing
- `pre_processors.py` / `post_processors.py`: Input pre-processing and output post-processing
- `meta_data_utils.py`: Shared metadata utilities

### Data Pipeline (`IMC/data/`)
- `duke_dataloader_local.py` / `duke_dataloader_3d.py`: Duke Liver MRI dataset (2-D and 3-D)
- `brain_dataloader_local.py`: Brain MRI dataset (ADNI, GadOnly, GadProhance, etc.)
- `ct_dataloader_local.py`: CT contrast detection dataset
- `liver_dataloader_local.py`: PV.AI liver dataset for development
- `liver_dataloader_gcp.py`: GCP dataset class with Dataflux integration
- `dicom_tag_encoding.py`: DICOM metadata feature encoding
- `dicom_tag_encoding_v2.py` / `dicom_tag_encoding_v3.py`: Evolved encoding versions
- `dicom_tag_encoding_yaml.py`: YAML-config-driven metadata encoding
- `dicom_ct_contrast_encoding.py`: CT contrast-specific feature encoding
- `image_reader.py`: DICOM/NIfTI image reading utilities
- `augment.py`: Image augmentation configurations (multiple presets)
- `resize.py`: Image resizing utilities
- `constants.py`: Shared dataset constants
- `configs/`: YAML metadata encoding config files

### Model Export (`IMC/onnx/`)
- `onnx_metadata_export.py`: ONNX model export utilities
- `contrast_label_wrapper.py`: Label wrapper for ONNX contrast output

### Utility Scripts (`scripts/`)
- `extract_dicom_tags.py`: Extract DICOM tags from a dataset folder
- `encode_metadata.py`: Encode raw DICOM metadata to model features
- `encode_ct_data.py`: CT-specific metadata encoding
- `download_adni.py` / `download_ct_data.py`: Dataset download helpers
- `scan_dicom_folder.py`: Scan and inventory a DICOM folder
- `run_train.sh`: Shell wrapper for training runs

### MICCAI Experiments (`miccai/`)
Summary of experiments and results for the MICCAI paper (`summary.md`).

### Deployment (`terraform/`)
- `main.tf` / `variables.tf` / `outputs.tf`: Terraform infrastructure definition
- `terraform.tfvars.example`: Example Terraform configuration variables
- `service/`: Flask app for serving inference requests
- `ct/`: CT-specific deployment configuration
- `build.sh`: Build Docker container and push to Artifact Registry
- `init_validate_apply.sh`: Run terraform init → validate → plan → apply cycle
