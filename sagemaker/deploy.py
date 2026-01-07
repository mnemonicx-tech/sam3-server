import boto3
import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput

# ---------------- CONFIG ----------------
REGION = boto3.Session().region_name
ROLE_ARN = sagemaker.get_execution_role()

# My Custom Image
IMAGE_URI = "014005448548.dkr.ecr.ap-south-1.amazonaws.com/sam3-inference:latest"

# Data Paths
INPUT_S3_DATA = "s3://kishankumarhs/node_downloads/"
OUTPUT_S3_DATA = "s3://kishankumarhs/sam3_output/"

# Model Artifact (we need to mount this too since it's not baked into the image yet)
MODEL_S3_URI = "s3://kishankumarhs/sam3-models/sam3.pt"

# Compute Config
INSTANCE_TYPE = "ml.g5.4xlarge"
INSTANCE_COUNT = 1
JOB_NAME_PREFIX = "sam3-batch-process"

def main():
    sm_session = sagemaker.Session()

    print(f"🚀 Initializing Processor with image: {IMAGE_URI}")
    
    # Define the Processor
    # We use valid entrypoint in Dockerfile, so we just run the container.
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE_ARN,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        base_job_name=JOB_NAME_PREFIX,
        sagemaker_session=sm_session,
        # Ensure we can use GPU
        volume_size_in_gb=50,
        env={
            "INPUT_ROOT": "/opt/ml/input/data",
            "OUTPUT_ROOT": "/opt/ml/output/data",
            "SAM_CHECKPOINT": "/opt/ml/model/sam3.pt"
        } 
    )

    print("🔹 Starting Processing Job...")
    print(f"   Input: {INPUT_S3_DATA}")
    print(f"   Model: {MODEL_S3_URI}")
    print(f"   Output: {OUTPUT_S3_DATA}")

    processor.run(
        inputs=[
            # 1. Input Images
            ProcessingInput(
                source=INPUT_S3_DATA,
                destination="/opt/ml/input/data",
                input_name="input_images"
            ),
            # 2. Model File (Mounted to /opt/ml/model)
            ProcessingInput(
                source=MODEL_S3_URI,
                destination="/opt/ml/model",
                input_name="model_artifact"
            )
        ],
        outputs=[
            # 3. Output Data
            ProcessingOutput(
                source="/opt/ml/output/data",
                destination=OUTPUT_S3_DATA,
                output_name="output_data"
            )
        ],
        # If your entrypoint is strictly "python inference.py", arguments aren't strictly needed 
        # unless inference.py parses them. Our script relies on environment variables or fixed paths.
        # SageMaker Processing sets /opt/ml/processing/input/ ... usually.
        # BUT Generic Processor (without ScriptProcessor class) maps inputs to where we say.
        # NOTE: standard paths in Processing are /opt/ml/processing/input.
        # BUT we specified '/opt/ml/input/data' in destination.
        
        # We need to ensure environment variables are set if our script uses them
    )

    print("✅ Processing Job Completed")

if __name__ == "__main__":
    main()
