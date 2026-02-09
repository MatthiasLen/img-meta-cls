# IMC Model Deployment Guide

This guide explains how to deploy the IMC model inference service to Google Cloud Platform using Terraform.

## Quick Start

### Prerequisites

- GCP account with billing enabled
- `gcloud` CLI installed and authenticated
- `terraform` installed (>= 1.0)
- `docker` installed

### Deploy in 5 Steps

1. **Set up GCP project**
   ```bash
   export PROJECT_ID="your-gcp-project-id"
   gcloud config set project $PROJECT_ID
   
   # Enable required APIs
   gcloud services enable cloudrun.googleapis.com \
     artifactregistry.googleapis.com \
     cloudbuild.googleapis.com \
     iam.googleapis.com
   ```

2. **Build and push Docker image**
   ```bash
   cd terraform
   ./build.sh
   ```

3. **Configure Terraform**
   ```bash
   # Copy example configuration
   cp terraform.tfvars.example terraform.tfvars
   
   # Edit terraform.tfvars with your project ID and image URL
   # The build.sh script will output the correct image URL
   ```

4. **Deploy infrastructure**
   ```bash
   terraform init
   terraform apply
   ```

5. **Get service URL**
   ```bash
   terraform output service_url
   ```

## Usage

Send a POST request to the `/predict` endpoint:

```bash
SERVICE_URL=$(terraform output -raw service_url)

curl -X POST ${SERVICE_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bucket_path": "gs://your-bucket/path/to/dicom/series"
  }'
```

The service will:
1. Download DICOM files from the GCS bucket
2. Run model inference
3. Upload predictions as JSON to the same bucket
4. Return predictions in the HTTP response

## Architecture

The deployment creates:
- **Cloud Run service**: Serverless container for inference
- **Artifact Registry**: Docker image storage
- **Service Account**: For GCS bucket access
- **IAM bindings**: Secure access control

Key features:
- Scales to zero (cost-efficient)
- Auto-scaling based on load
- Built-in health checks
- Secure by default

## Documentation

See [terraform/README.md](terraform/README.md) for detailed documentation including:
- Configuration options
- Troubleshooting
- Security considerations
- Cost optimization

## Files

```
terraform/
├── main.tf                    # Main Terraform configuration
├── variables.tf               # Variable definitions
├── outputs.tf                 # Output definitions
├── terraform.tfvars.example   # Example configuration
├── build.sh                   # Build and push script
├── README.md                  # Detailed documentation
└── service/
    ├── app.py                 # Flask inference service
    ├── Dockerfile             # Container image definition
    └── requirements.txt       # Python dependencies
```

## Clean Architecture

This implementation follows the requirements:
- ✅ Lightweight and simple
- ✅ Clean, easy-to-understand code
- ✅ No code clutter
- ✅ No overcomplication
- ✅ Serverless deployment (Cloud Run)
- ✅ Cost-efficient (scales to zero)
- ✅ Single responsibility (model inference)

## Support

For issues or questions, see the detailed README in the `terraform/` directory.
