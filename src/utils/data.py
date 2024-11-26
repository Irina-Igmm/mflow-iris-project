import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re


def load_iris_data(test_size=0.2, random_state=42):
    """Load the Iris dataset and split it into training and testing sets."""
    X, y = datasets.load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def clean_text(text):
    """Clean text by removing special characters and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())
    return text


def load_job_data(filepath, test_size=0.2, random_state=42):
    """Load and preprocess the job dataset, then split it into training and testing sets."""
    # Read data
    df = pd.read_csv(filepath)

    # Create a copy to avoid chained assignment warning
    df = df.copy()

    # Fill null values using loc instead of inplace
    df.loc[:, "role"] = df["role"].fillna(df["role"].mode()[0])
    df.loc[:, "importance"] = df["importance"].fillna(df["importance"].mode()[0])

    # Remove rows with missing target
    df = df.dropna(subset=["Updated category"])

    # Rest of the preprocessing remains the same
    df["job_title_clean"] = df["job_title"].apply(clean_text)

    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])
    df["subcategory_encoded"] = le.fit_transform(df["subcategory"])

    df["title_word_count"] = df["job_title_clean"].str.count("\s+") + 1

    tfidf = TfidfVectorizer(max_features=100)
    title_features = tfidf.fit_transform(df["job_title_clean"])

    target_encoder = LabelEncoder()
    df["target"] = target_encoder.fit_transform(df["Updated category"])

    X = hstack(
        [
            title_features,
            df[["category_encoded", "subcategory_encoded", "title_word_count"]].values,
        ]
    )
    y = df["target"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
