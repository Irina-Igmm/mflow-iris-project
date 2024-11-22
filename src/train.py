import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from utils.data import load_iris_data
from predict import evaluate_model, log_experiment, load_and_predict
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def train_model(X_train, y_train):
    # Train the model
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    return lr


if __name__ == "__main__":
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_iris_data()
        model = train_model(X_train, y_train)
        accuracy, f1, y_pred = evaluate_model(model, X_test, y_test)
        model_info = log_experiment(accuracy, f1, model, X_train)
        result = load_and_predict(model_info, X_test, y_test)
        print(result[:4])
