import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_iris_data(test_size=0.2, random_state=42):
    """Load the Iris dataset and split it into training and testing sets."""
    X, y = datasets.load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
