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

6. **Test the service**
   ```bash
   # Get the service URL
   SERVICE_URL=$(terraform output -raw service_url)
   
   # Authenticate with gcloud (required for accessing the service)
   gcloud auth login
   
   # Check health (service is running) - use --auth flag
   python terraform/service/test_client.py --url $SERVICE_URL --auth health
   
   # Wait for model to be ready (first startup may take a few minutes)
   python terraform/service/test_client.py --url $SERVICE_URL --auth ready --wait
   
   # Send a prediction request
   python terraform/service/test_client.py --url $SERVICE_URL --auth \
     predict gs://your-bucket/path/to/dicom
   ```

## Testing Your Deployment

After deployment, use the provided test client to verify everything works:

### Test Client Script

The `terraform/service/test_client.py` script provides three commands:

1. **Health Check** - Verify service is running
   ```bash
   python terraform/service/test_client.py --url $SERVICE_URL --auth health
   ```

2. **Readiness Check** - Verify model is loaded
   ```bash
   # Simple check
   python terraform/service/test_client.py --url $SERVICE_URL --auth ready
   
   # Wait for model to load (first startup)
   python terraform/service/test_client.py --url $SERVICE_URL --auth ready --wait
   ```

3. **Send Prediction** - Process DICOM series
   ```bash
   python terraform/service/test_client.py --url $SERVICE_URL --auth \
     predict gs://your-bucket/patient1/study1 \
     --output predictions.json
   ```

See [terraform/service/README.md](terraform/service/README.md) for complete testing documentation.

### First Startup

On first startup, the container takes **2-5 minutes** to load the ONNX model:
- The service starts listening on port 8080 immediately ✅
- Model loads in background thread 🔄
- `/health` returns 200 during loading ✅
- `/ready` returns 503 until model is loaded ⏳
- Use `ready --wait` to automatically wait for model loading

## Usage

The service processes **multiple DICOM series** stored in separate folders within a bucket path.

### Bucket Structure

```
gs://your-bucket/parent-folder/
├── series1/          # Each subdirectory is a series
│   ├── img001.dcm
│   └── img002.dcm
├── series2/
│   └── slice001.dcm
└── series3/
    └── dicom001.dcm
```

### Send Inference Request

```bash
SERVICE_URL=$(terraform output -raw service_url)

curl -X POST ${SERVICE_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bucket_path": "gs://your-bucket/parent-folder"
  }'
```

The service will:
1. Discover all series folders in the bucket path
2. For each series:
   - Download DICOM files
   - Sample slices according to model rules
   - Run model inference
   - Upload `prediction.json` to that series folder
3. Return all predictions in the HTTP response

### Response

```json
{
  "bucket_path": "your-bucket/parent-folder",
  "series_count": 3,
  "processed_count": 3,
  "predictions": {
    "parent-folder/series1": {
      "pred_SequenceType": "T1",
      ...
    },
    "parent-folder/series2": {...},
    "parent-folder/series3": {...}
  }
}
```

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

## Troubleshooting

### 403 Forbidden Error

If you get a **403 Forbidden** error when accessing the service:

**Cause**: The Cloud Run service requires authentication (default and recommended setting: `allow_public_access = false`)

**Solution: Use Authenticated Requests (Recommended)**

The service is secured by default and requires authentication. Use the `--auth` flag with the test client:

```bash
# Make sure you're authenticated with gcloud
gcloud auth login

# Use --auth flag for all requests
SERVICE_URL=$(terraform output -raw service_url)
python terraform/service/test_client.py --url $SERVICE_URL --auth ready
python terraform/service/test_client.py --url $SERVICE_URL --auth health
```

Or use curl with authentication:

```bash
# Get authentication token
TOKEN=$(gcloud auth print-identity-token)

# Make authenticated request
curl -H "Authorization: Bearer $TOKEN" ${SERVICE_URL}/ready
curl -H "Authorization: Bearer $TOKEN" ${SERVICE_URL}/health
```

**Alternative (NOT RECOMMENDED): Enable Public Access**

⚠️ **Security Warning**: Only use this for testing in non-production environments

1. Edit `terraform/terraform.tfvars`:
   ```bash
   allow_public_access = true
   ```

2. Re-apply Terraform:
   ```bash
   cd terraform
   terraform apply
   ```
