import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
from sklearn import datasets
import os
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME", "iris-dataset-training")


def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return accuracy, f1, y_pred


def log_experiment(accuracy, f1, model, X_train):
    # Terminer toute run active
    if mlflow.active_run():
        mlflow.end_run()

    # Log the experiment
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }
    mlflow.set_experiment("Iris Classification")
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.set_tag("Info_d_entrainement", "Modèle LR de base pour les données iris")
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="modele_iris",
            signature=signature,
            input_example=X_train,
            registered_model_name=model_name,
        )
    return model_info


def load_and_predict(model_info, X_test, y_test):
    # Load the model and make predictions
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)
    iris_feature_names = datasets.load_iris().feature_names
    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    return result
