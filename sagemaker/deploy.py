import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.transformer import Transformer

REGION = boto3.Session().region_name
ROLE_ARN = sagemaker.get_execution_role()

IMAGE_URI = "014005448548.dkr.ecr.ap-south-1.amazonaws.com/sam3-inference:latest"
MODEL_S3_URI = "s3://kishankumarhs/sam3-models/sam3.pt"

INPUT_S3 = "s3://kishankumarhs/node_downloads/"
OUTPUT_S3 = "s3://kishankumarhs/sam3_output/"

MODEL_NAME = "sam3-batch-model"
INSTANCE_TYPE = "ml.g5.4xlarge"
INSTANCE_COUNT = 1


def main():
    sm = sagemaker.Session()

    print("Creating SageMaker model...")
    model = Model(
        name=MODEL_NAME,
        image_uri=IMAGE_URI,
        model_data=MODEL_S3_URI,
        role=ROLE_ARN,
        sagemaker_session=sm,
    )

    transformer = Transformer(
        model_name=MODEL_NAME,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        output_path=OUTPUT_S3,
        sagemaker_session=sm,
        max_concurrent_transforms=1,
    )

    print("Starting Batch Transform job...")
    transformer.transform(
        data=INPUT_S3,
        content_type="application/x-image",
    )

    transformer.wait()
    print("✅ Batch segmentation completed")


if __name__ == "__main__":
    main()
