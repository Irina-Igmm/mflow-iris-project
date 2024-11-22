# MLflow Iris Project

This project implements a logistic regression model to classify the Iris dataset using MLflow for experiment tracking and model management. It includes scripts for training the model, making predictions, and a Jupyter notebook for interactive development.

## Project Structure

```
mlflow-iris-project
├── src
│   ├── train.py                # Main logic for training the model
│   ├── predict.py              # Logic for making predictions
│   ├── utils
│   │   └── data.py             # Data loading and preprocessing utilities
│   └── notebooks
│       └── model_development.ipynb # Jupyter notebook for model development
├── tests
│   └── test_model.py           # Unit tests for model functions
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

- Execute mlflow run:

```bash
mlflow run . -P alpha=0.5 -P l1_ratio=0.01
```

## License

This project is licensed under the MIT License.
