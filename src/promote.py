import mlflow
from mlflow.tracking import MlflowClient
import os

from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME", "iris-dataset-training")
versioning = os.getenv("VERSIONING", "None")
stage = os.getenv("STAGE", "None")


def promote_model_to_production(model_name, model_version):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage=stage, archive_existing_versions=True
    )


if __name__ == "__main__":
    model_version = versioning  # Update this to the correct version
    promote_model_to_production(model_name, model_version)
