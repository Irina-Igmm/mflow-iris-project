# MLflow Iris Project

This project implements a logistic regression model to classify the Iris dataset using MLflow for experiment tracking and model management. It includes scripts for training the model, making predictions, and a Jupyter notebook for interactive development.

## Project Structure

```
mlflow-iris-project
├── src
│   ├── train.py                # Main logic for training the model
│   ├── predict.py
    ├── promote.py             # Logic for making predictions
│   ├── utils
│   │   └── data.py             # Data loading and preprocessing utilities
│   └── notebooks
│       └── model_development.ipynb # Jupyter notebook for model development
        └── dataset_preprocessing.ipynb # Jupyter notebook for data preprocessing
├── tests
│   └── test_model.py           # Unit tests for model functions
    └── predict_with_model.py   # make prediction using mlflow models
├── conda.yaml                  # Conda environment configuration
├── MLproject                   # MLflow project definition
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd mlflow-iris-project
   ```

2. Create a conda environment:

   ```bash
   conda env create -f conda.yaml
   conda activate mlflow-iris-env
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- To train the model, run:

  ```bash
  python src/train.py
  ```

- To make predictions, run:

  ```bash
  python src/predict.py
  ```

- For interactive model development, open the Jupyter notebook:

  ```bash
  jupyter notebook src/notebooks/model_development.ipynb
  ```

- Execute test using :

```bash
python -m unittest discover -s . -p "test_model.py"
```

- Exeuting test witha specific class :

```bash
python -m unittest discover -s . -p "test_*.py" -k TestJobModel
```

- Execute mlflow run:

```bash
mlflow run . -e train_job
```

- Lunch git project with mlflow run:

```bash
mlflow run https://github.com/Irina-Igmm/mflow-iris-project.git -v develop  -e train_job
```

- Promote model in Production after training successfull:

```bash
mlflow run . -e promote_to_production --env-manager=conda
```

- Deploy model endpoint :

```bash
mlflow models serve -m runs:/<run_id>/model -p 5000
mlflow models serve -m models:/<model_name>/Staging -p 5000 --env-manager conda
```

- Use the `/invocations` endpoint with `curl`:

```bash
curl -X POST -H "Content-Type: application/json" --data '{"dataframe_split": {"columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"], "data": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}}' http://127.0.0.1:5000/invocations
```

## License

This project is licensed under the MIT License.
