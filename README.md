# Wind Turbine MLflow Pipeline

A comprehensive machine learning pipeline for wind turbine power prediction using MLflow for experiment tracking, MongoDB for data storage, and Flask for API deployment.

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting wind turbine power output based on various environmental and operational parameters. The pipeline includes data preprocessing, model training, evaluation, and deployment as a REST API.

## Project Structure

```
├── .github/workflows/   # CI/CD pipeline configurations
│   └── ci-cd.yml        # CI/CD workflow for GitHub Actions
├── data/                # Data storage for CSV files
├── mlflow_server/       # MLflow server configuration
│   └── docker-compose.yml # Docker Compose for MLflow and MongoDB
├── models/              # Saved model files and preprocessors
├── notebooks/           # Jupyter notebooks for exploration
│   └── exploratory_data_analysis.ipynb  # EDA notebook
├── outputs/             # Visualization outputs
├── src/                 # Source code
│   ├── data/            # Data processing modules
│   │   ├── load_data.py # Functions for loading data
│   │   └── preprocess.py # Functions for preprocessing data
│   ├── models/          # Model training modules
│   │   ├── train.py     # Functions for training models
│   │   └── evaluate.py  # Functions for evaluating models
│   ├── api/             # API service modules
│   │   └── app.py       # Flask API application
│   └── utils/           # Utility functions
│       ├── mlflow_utils.py # MLflow utility functions
│       ├── mongo_utils.py  # MongoDB utility functions
│       └── visualization.py # Visualization utility functions
├── scripts/             # Scripts for running the pipeline
│   ├── run_pipeline.py  # Script to run the entire pipeline
│   ├── update_data.py   # Script to update data
│   └── deploy_api.py    # Script to deploy the API
├── tests/               # Test files
│   ├── test_api.py      # Tests for API
│   ├── test_data_loading.py # Tests for data loading
│   ├── test_models.py   # Tests for models
│   ├── test_preprocessing.py # Tests for preprocessing
│   └── test_utils.py    # Tests for utilities
├── Dockerfile           # Dockerfile for the application
└── requirements.txt     # Python dependencies
```

## Features

- **Data Management**: Load data from CSV files and store in MongoDB
- **Data Preprocessing**: Handle missing values, outliers, categorical encoding, and feature scaling
- **Model Training**: Train multiple regression models (Linear Regression, Random Forest, XGBoost)
- **Experiment Tracking**: Track experiments with MLflow
- **Model Evaluation**: Compare model performance using RMSE, MAE, and R²
- **Model Deployment**: Serve the best model via a Flask API
- **Containerization**: Docker and Docker Compose for easy deployment
- **CI/CD**: GitHub Actions workflow for testing, training, and deployment
- **Visualization**: Generate plots for model performance and feature importance

## Setup and Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- MongoDB (or use the provided Docker Compose setup)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/wind-turbine-mlflow-pipeline.git
cd wind-turbine-mlflow-pipeline
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Place your wind turbine data CSV file in the `data/` directory.

### Running the Pipeline

To run the entire pipeline (data loading, preprocessing, training, and evaluation):

```bash
python scripts/run_pipeline.py --data_path data/your_data.csv --mongodb_uri mongodb://localhost:27017/ --mlflow_uri http://localhost:5000
```

### Updating Data

To update the dataset with new data:

```bash
python scripts/update_data.py --data_path data/new_data.csv --append True --output_path data/combined_data.csv
```

### Deploying the API

To deploy the API service:

```bash
python scripts/deploy_api.py --host 0.0.0.0 --port 5001 --workers 4
```

### Using Docker

To run the entire application using Docker Compose:

```bash
docker-compose -f mlflow_server/docker-compose.yml up -d
```

This will start the MongoDB database, MLflow server, and the Wind Turbine API service.

## CI/CD Pipeline

The CI/CD pipeline automates the following processes:

- **Testing**: Runs unit tests and linting checks on every push and pull request
- **Training**: Trains and evaluates models on the main branch
- **Deployment**: Builds and pushes a Docker image for deployment

The workflow is defined in `.github/workflows/ci-cd.yml` and includes the following jobs:

1. **Test**: Runs linting with flake8 and tests with pytest
2. **Train**: Sets up MongoDB and MLflow, loads data, trains models, and evaluates performance
3. **Deploy**: Builds and pushes a Docker image to Docker Hub

## API Endpoints

The API provides the following endpoints:

- `GET /health`: Check the health of the API
- `GET /model/info`: Get information about the loaded model
- `POST /predict`: Make a single prediction
- `POST /predict/batch`: Make batch predictions

### Example Usage

#### Single Prediction

```python
import requests
import json

url = "http://localhost:5001/predict"
data = {
    "Wind_Speed": 5.0,
    "Wind_Direction": 270.0,
    "Temperature": 15.5,
    "Humidity": 65.0,
    "Pressure": 1013.2
}
response = requests.post(url, json=data)
print(response.json())
```

#### Batch Prediction

```python
import requests
import json

url = "http://localhost:5001/predict/batch"
data = {
    "data": [
        {
            "Wind_Speed": 5.0,
            "Wind_Direction": 270.0,
            "Temperature": 15.5,
            "Humidity": 65.0,
            "Pressure": 1013.2
        },
        {
            "Wind_Speed": 6.2,
            "Wind_Direction": 290.0,
            "Temperature": 16.8,
            "Humidity": 60.0,
            "Pressure": 1012.5
        }
    ]
}
response = requests.post(url, json=data)
print(response.json())
```

## MLflow Tracking

The MLflow UI is available at http://localhost:5000 when running the MLflow server. It provides a dashboard to view and compare experiments, including metrics, parameters, and artifacts.

## Testing

To run the tests:

```bash
pytests tests/
```

To run tests with coverage:

```bash
pytests tests/ --cov=src
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request