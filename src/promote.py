import mlflow
from mlflow.tracking import MlflowClient
import os

from dotenv import load_dotenv

load_dotenv()

model_iris_name = os.getenv("MODEL_IRIS_NAME", "iris-dataset-training")
model_job_name = os.getenv("MODEL_JOB_CLASSIFIATION_NAME", "job-classification-training")
versioning = os.getenv("VERSIONING", "None")
stage = os.getenv("STAGE", "None")


def promote_model_to_production(model_name, model_version):
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=model_version, stage=stage, archive_existing_versions=True
        )

        print(f"Promoted model version {model_version} to {stage}")
    except Exception as e:
        print(f"Failed to promote model: {e}")


if __name__ == "__main__":
    model_version = versioning  # Update this to the correct version
    promote_model_to_production(model_job_name, model_version)
