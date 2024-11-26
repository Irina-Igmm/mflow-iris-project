import requests
import json
import numpy as np


def test_job_model_prediction():
    # MLflow server URL
    url = "http://127.0.0.1:5000/invocations"

    # Create exactly 103 features
    features = np.zeros(100).tolist()  # 100 TF-IDF features
    features.extend(
        [25.0, 0.0, 3.0]
    )  # category_encoded, subcategory_encoded, title_word_count

    # Validate feature length
    assert len(features) == 103, f"Expected 103 features, got {len(features)}"

    data = {"inputs": [features]}

    try:
        response = requests.post(
            url=url, headers={"Content-Type": "application/json"}, data=json.dumps(data)
        )

        if response.status_code == 200:
            print("Prediction:", response.json())
        else:
            print(f"Error: {response.status_code}", response.text)

    except Exception as e:
        print(f"Request failed: {str(e)}")


if __name__ == "__main__":
    test_job_model_prediction()
