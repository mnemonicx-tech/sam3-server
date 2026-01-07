# SageMaker Serverless Deployment Guide

## Overview

This guide deploys the SAM3 inference server to **AWS SageMaker Serverless Inference**, which provides:

- **Auto-scaling**: Automatically scales based on traffic
- **Pay-per-invocation**: Only pay for what you use
- **No infrastructure management**: No need to manage EC2 instances
- **High availability**: Multi-AZ deployment by default

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured with credentials
3. **Docker** installed locally
4. **Python 3.11+** with boto3

```bash
pip install boto3
```

5. **IAM Role**: Create/verify `SageMakerExecutionRole` exists:

```bash
aws iam create-role --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policies
aws iam attach-role-policy --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

## Quick Start

### 1. Deploy to SageMaker

```bash
python sagemaker/deploy.py \
  --model-path models/sam3.pt \
  --endpoint-name sam3-serverless
```

This will:

- Upload `models/sam3.pt` to S3
- Build and push Docker image to ECR
- Create SageMaker Model
- Create Endpoint Configuration with Serverless settings
- Deploy Endpoint (takes ~5-10 minutes)

### 2. Test the Endpoint

```bash
# Test with local image
python sagemaker/test_endpoint.py sam3-serverless \
  --image sample_data/test.png \
  --prompt "pant"

# Test with S3 image
python sagemaker/test_endpoint.py sam3-serverless \
  --image-s3 s3://my-bucket/image.jpg \
  --prompt "person"
```

### 3. Invoke from CLI

```bash
# Create payload
cat > payload.json << 'EOF'
{
  "prompt": "pant",
  "image_base64": "$(base64 < sample_data/test.png)"
}
EOF

# Invoke endpoint
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name sam3-serverless \
  --content-type application/json \
  --body file://payload.json \
  response.json

cat response.json | jq .
```

### 4. Invoke from Python

```python
import boto3
import json
import base64

sm_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Load image
with open("sample_data/test.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Invoke
response = sm_runtime.invoke_endpoint(
    EndpointName="sam3-serverless",
    ContentType="application/json",
    Body=json.dumps({
        "prompt": "pant",
        "image_base64": image_b64
    }).encode()
)

result = json.loads(response["Body"].read())
print(result)
```

## Configuration Options

### Memory Sizes (ServerlessConfig)

- `MemorySizeInMB`: 1024, 2048, 3072, 4096, 5120, 6144 (default: 2048)

For SAM3, **recommend 4096-6144 MB** (4-6 GB)

Update in `deploy.py`:

```python
"ServerlessConfig": {
    "MaxConcurrentInvocations": 100,
    "MemorySizeInMB": 6144,  # Increase for faster inference
}
```

### Instance Types (for non-Serverless GPU)

Replace `ml.g4dn.xlarge` with:

- `ml.g4dn.xlarge` (1x NVIDIA T4 GPU)
- `ml.g4dn.2xlarge` (1x NVIDIA T4 GPU)
- `ml.g4dn.12xlarge` (4x NVIDIA T4 GPU)

## API Payload

### Request

```json
{
  "prompt": "pant",
  "image_base64": "<base64-encoded-image>",
  "options": {
    "mask_threshold": 0.5,
    "max_masks": 10
  }
}
```

Or with S3:

```json
{
  "prompt": "pant",
  "image_s3_uri": "s3://bucket/path/to/image.jpg"
}
```

### Response

```json
{
  "image_size": { "width": 1290, "height": 1274 },
  "masks": [
    {
      "mask_id": "0",
      "score": 0.95,
      "bbox": [100, 150, 500, 800],
      "rle": null,
      "png_base64": null,
      "area": 45000
    }
  ],
  "overall_confidence": 0.95
}
```

## Monitoring

### CloudWatch Logs

```bash
aws logs tail /aws/sagemaker/Endpoints/sam3-serverless --follow
```

### Endpoint Metrics

```bash
aws sagemaker describe-endpoint --endpoint-name sam3-serverless
```

### Invocation Errors

```bash
aws logs tail /aws/sagemaker/Endpoints/sam3-serverless/errors --follow
```

## Cost Estimation

**SageMaker Serverless Inference** pricing (varies by region):

- **Invocation**: ~$0.0004 per invocation (1M invocations = ~$400)
- **Duration**: ~$0.00001 per GB-second
- **Data transfer**: Standard AWS rates

Example (1000 invocations/day, ~2 seconds each, 4GB memory):

- Daily: 1000 × $0.0004 + 1000 × 2 × 4 × $0.00001 = $0.48/day
- Monthly: ~$14/month

## Cleanup

### Delete Endpoint

```bash
aws sagemaker delete-endpoint --endpoint-name sam3-serverless
aws sagemaker delete-endpoint-config --endpoint-config-name sam3-serverless-config
aws sagemaker delete-model --model-name sam3_serverless
```

### Delete ECR Image

```bash
aws ecr delete-repository --repository-name sam3-inference --force
```

## Troubleshooting

### Model Loading Fails

Check CloudWatch logs:

```bash
aws logs tail /aws/sagemaker/Endpoints/sam3-serverless --follow
```

### Out of Memory

Increase `MemorySizeInMB` to 4096 or 6144

### Slow Inference

- Check memory allocation (increase to 6144)
- Verify GPU is available (check CloudWatch logs)
- Consider using smaller model or batching

### Deployment Fails

1. Verify IAM role has ECR, S3, and SageMaker permissions
2. Verify model file exists: `models/sam3.pt`
3. Check Docker build locally: `docker build -f sagemaker/Dockerfile.sagemaker .`

## Advanced: Using Existing Image

If you've already built the Docker image, skip the build:

```bash
python sagemaker/deploy.py \
  --skip-build \
  --endpoint-name sam3-serverless
```

## Next Steps

- Add API Gateway for REST endpoint
- Integrate with n8n workflow
- Set up auto-scaling policies
- Enable model monitoring and data capture
