import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils.data import load_iris_data, load_job_data

# from src.utils.data import load_iris_data, load_job_data

from predict import (
    evaluate_model,
    log_experiment,
    load_and_predict,
    evaluate_job_model,
    log_job_experiment,
    load_and_predict_job,
)

# from src.predict import (
#     evaluate_model,
#     log_experiment,
#     load_and_predict,
#     evaluate_job_model,
#     log_job_experiment,
#     load_and_predict_job,
# )
import sys
import os

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def train_model(X_train, y_train):
    # Train the model
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    classes = lr.classes_
    print(f"Model classes: {classes}")
    return lr


def train_job_model(X_train, y_train):
    """Train a model on the job dataset."""
    # Train the RandomForest model
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


if __name__ == "__main__":
    try:
        if mlflow.active_run():
            mlflow.end_run()
        with mlflow.start_run():
            filepath = os.getenv(
                "FILEPATH"
            )  # Update this with the actual path to your job data
            X_train, X_test, y_train, y_test = load_job_data(filepath)
            rf = train_job_model(X_train, y_train)
            accuracy, f1, recall, report, y_pred = evaluate_job_model(
                rf, X_test, y_test
            )
            model_info = log_job_experiment(report, accuracy, f1, recall, rf, X_train)
            result = load_and_predict_job(model_info, X_test, y_test)
            result["actual_class"] = (
                y_test.toarray() if hasattr(y_test, "toarray") else y_test
            )
            print(result[:4])
    except Exception as e:
        print(f"An error occurred: {e}")
