#!/usr/bin/env python3
"""
Test SageMaker Serverless Endpoint
"""

import json
import boto3
import base64
import argparse
from pathlib import Path


def invoke_endpoint(endpoint_name, image_path=None, image_base64=None, image_s3_uri=None, prompt="pant", region="us-east-1"):
    """Invoke SageMaker endpoint"""
    
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
    
    # Prepare payload
    payload = {"prompt": prompt}
    
    # Handle image input
    if image_path:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload["image_base64"] = image_b64
        print(f"Using image from file: {image_path}")
    elif image_base64:
        payload["image_base64"] = image_base64
        print("Using provided base64 image")
    elif image_s3_uri:
        payload["image_s3_uri"] = image_s3_uri
        print(f"Using image from S3: {image_s3_uri}")
    else:
        raise ValueError("Must provide image_path, image_base64, or image_s3_uri")
    
    print(f"\nInvoking endpoint: {endpoint_name}")
    print(f"Prompt: {prompt}\n")
    
    # Invoke endpoint
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    
    # Parse response
    result = json.loads(response["Body"].read().decode("utf-8"))
    
    print(json.dumps(result, indent=2))
    
    # Extract summary
    if "masks" in result:
        print(f"\n✓ Segmentation successful!")
        print(f"  Image size: {result.get('image_size')}")
        print(f"  Number of masks: {len(result.get('masks', []))}")
        print(f"  Overall confidence: {result.get('overall_confidence'):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SageMaker Serverless Endpoint")
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--image-s3", help="S3 URI (s3://bucket/key)")
    parser.add_argument("--prompt", default="pant", help="Segmentation prompt")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    
    args = parser.parse_args()
    
    if not args.image and not args.image_s3:
        # Use default test image
        test_image = Path("sample_data/test.png")
        if test_image.exists():
            args.image = str(test_image)
        else:
            print("Error: Provide --image or --image-s3")
            exit(1)
    
    invoke_endpoint(
        endpoint_name=args.endpoint_name,
        image_path=args.image,
        image_s3_uri=args.image_s3,
        prompt=args.prompt,
        region=args.region,
    )
