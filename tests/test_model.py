import unittest
from src.utils.data import load_iris_data
from src.predict import load_and_predict, evaluate_model, log_experiment
from src.train import train_model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


class TestModel(unittest.TestCase):

    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_iris_data()
        self.model = train_model(self.X_train, self.y_train)
        self.accuracy, self.f1,_ = evaluate_model(self.model, self.X_test, self.y_test)

    def test_model_training(self):
        self.assertIsInstance(self.model, LogisticRegression)

    def test_model_evaluation(self):
        accuracy, f1, _= evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.5)  # Assuming a baseline accuracy
        self.assertGreaterEqual(f1, 0.5)

    def test_load_and_predict(self):
        model_info = log_experiment(self.accuracy, self.f1, self.model, self.X_train)
        result = load_and_predict(model_info, self.X_test, self.y_test)
        self.assertEqual(result["predicted_class"].shape[0], self.X_test.shape[0])


if __name__ == "__main__":
    unittest.main()
