# IMC (Image Plus Metadata Classifier)

IMC (Image Plus Metadata Classifier) is a deep learning system for multi-task medical image classification, specifically designed for MRI sequence classification. The project combines multi-slice medical images with DICOM metadata using a transformer-based fusion architecture.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (modern Python package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Training and Development
```bash
# Run current training (Network 5 - unified transformer)
python IMC/net5/train05_baseline.py
python IMC/net5/train05_sparse_metadata_encoder.py

# Run Network 4 training (cross-attention fusion)
python IMC/net4/train04_baseline.py
python IMC/net4/train04_image.py
python IMC/net4/train04_sparse_metadata_encoder.py

```

## File Organization

### Main Source (`IMC/`)
- `network01.py-05.py`: Evolution of model architectures
- `net4/`: Network 4 training, inference, and export scripts
- `net5/`: Network 5 training and inference scripts
- `trainer.py`: Reusable training framework with mixed precision
- `evaluate.py`: Model evaluation and metrics generation
- `helper.py`: Utility functions (visualization, normalization)

### Neural Networks (`IMC/nn/`)
- `image_encoder.py`: CNN backbone for multi-slice images (DenseNet121/ResNet18)
- `metadata_encoder.py`: MLP with contextual imputation for DICOM features
- `sparse_metadata_encoder.py`: Advanced sparse feature handling with FiLM
- `multi_task_head.py`: Classification heads for multiple tasks
- `multi_task_loss.py`: Combined loss functions with label smoothing

### Data Pipeline (`IMC/data/`)
- `liver_dataloader_local.py`: Local dataset class for development
- `liver_dataloader_gcp.py`: GCP dataset class with Dataflux integration
- `dicom_tag_encoding.py`: DICOM metadata feature encoding (89 features)
- `augment.py`: Image augmentation configurations (multiple presets)

### Model Export (`IMC/onnx/`)
- `onnx_metadata_export.py`: ONNX model export utilities

