# tests/test_models.py

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import train_model, train_models
from src.models.evaluate import evaluate_model, load_model, evaluate_all_models, find_best_model

# Test data
@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    # Create feature matrix
    X_train = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature3': [10, 20, 30, 40, 50]
    })
    
    X_test = pd.DataFrame({
        'feature1': [1.5, 2.5, 3.5],
        'feature2': [0.15, 0.25, 0.35],
        'feature3': [15, 25, 35]
    })
    
    # Create target vector
    y_train = pd.Series([10.5, 20.5, 30.5, 40.5, 50.5])
    y_test = pd.Series([15.5, 25.5, 35.5])
    
    # Feature names
    feature_names = X_train.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names

@pytest.fixture
def mock_mlflow():
    """
    Create a mock MLflow module.
    """
    with patch('src.models.train.mlflow') as mock_mlflow:
        # Setup mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = 'test_experiment_id'
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Setup mock run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_mlflow.start_run.return_value = mock_run
        
        yield mock_mlflow

@pytest.fixture
def mock_models_dir(tmp_path):
    """
    Create a mock models directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Create mock model files
    rf_model = RandomForestRegressor()
    xgb_model = XGBRegressor()
    lr_model = LinearRegression()
    
    joblib.dump(rf_model, models_dir / "random_forest_model.pkl")
    joblib.dump(xgb_model, models_dir / "xgboost_model.pkl")
    joblib.dump(lr_model, models_dir / "linear_regression_model.pkl")
    
    return models_dir

def test_train_model(sample_data, mock_mlflow, tmp_path):
    """
    Test training a single model.
    """
    X_train, X_test, y_train, y_test, feature_names = sample_data
    
    # Create mock outputs directory
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    
    # Create mock models directory
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Patch os.path.join to use temp directories
    def mock_join(*args):
        if 'outputs' in args:
            return str(outputs_dir / args[-1])
        elif 'models' in args:
            return str(models_dir / args[-1])
        return os.path.join(*args)
    
    with patch('src.models.train.os.path.join', side_effect=mock_join):
        # Train model
        model_name = 'random_forest'
        metrics = train_model(
            model_name, X_train, X_test, y_train, y_test, feature_names,
            experiment_name='test_experiment', random_state=42
        )
        
        # Check that metrics are returned
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check that model file is created
        assert os.path.exists(models_dir / "random_forest_model.pkl")
        
        # Check that MLflow is used correctly
        mock_mlflow.set_experiment.assert_called_once_with('test_experiment')
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_called_once()

def test_train_models(sample_data, mock_mlflow, tmp_path):
    """
    Test training multiple models.
    """
    X_train, X_test, y_train, y_test, feature_names = sample_data
    
    # Create mock outputs directory
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    
    # Create mock models directory
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Patch os.path.join to use temp directories
    def mock_join(*args):
        if 'outputs' in args:
            return str(outputs_dir / args[-1])
        elif 'models' in args:
            return str(models_dir / args[-1])
        return os.path.join(*args)
    
    with patch('src.models.train.os.path.join', side_effect=mock_join):
        # Train models
        models_metrics = train_models(
            X_train, X_test, y_train, y_test, feature_names,
            experiment_name='test_experiment', random_state=42
        )
        
        # Check that metrics are returned for all models
        assert 'random_forest' in models_metrics
        assert 'xgboost' in models_metrics
        assert 'linear_regression' in models_metrics
        
        # Check that model files are created
        assert os.path.exists(models_dir / "random_forest_model.pkl")
        assert os.path.exists(models_dir / "xgboost_model.pkl")
        assert os.path.exists(models_dir / "linear_regression_model.pkl")

def test_evaluate_model(sample_data, mock_models_dir):
    """
    Test evaluating a single model.
    """
    X_train, X_test, y_train, y_test, feature_names = sample_data
    
    # Load model
    model_path = mock_models_dir / "random_forest_model.pkl"
    model = joblib.load(model_path)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Check that metrics are returned
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics

def test_load_model(mock_models_dir):
    """
    Test loading a model.
    """
    with patch('src.models.evaluate.os.path.join', return_value=str(mock_models_dir / "random_forest_model.pkl")):
        # Load model
        model = load_model('random_forest')
        
        # Check that model is loaded
        assert model is not None
        assert isinstance(model, RandomForestRegressor)

def test_evaluate_all_models(sample_data, mock_models_dir):
    """
    Test evaluating all models.
    """
    X_train, X_test, y_train, y_test, feature_names = sample_data
    
    # Patch os.path.join and os.listdir to use mock models directory
    with patch('src.models.evaluate.os.path.join', return_value=str(mock_models_dir)):
        with patch('src.models.evaluate.os.listdir', return_value=[
            "random_forest_model.pkl",
            "xgboost_model.pkl",
            "linear_regression_model.pkl"
        ]):
            # Patch load_model to return mock models
            def mock_load_model(model_name):
                if model_name == 'random_forest':
                    return joblib.load(mock_models_dir / "random_forest_model.pkl")
                elif model_name == 'xgboost':
                    return joblib.load(mock_models_dir / "xgboost_model.pkl")
                elif model_name == 'linear_regression':
                    return joblib.load(mock_models_dir / "linear_regression_model.pkl")
                return None
            
            with patch('src.models.evaluate.load_model', side_effect=mock_load_model):
                # Evaluate all models
                models_metrics = evaluate_all_models(X_test, y_test)
                
                # Check that metrics are returned for all models
                assert 'random_forest' in models_metrics
                assert 'xgboost' in models_metrics
                assert 'linear_regression' in models_metrics

def test_find_best_model():
    """
    Test finding the best model.
    """
    # Create mock metrics
    models_metrics = {
        'random_forest': {'rmse': 0.5, 'mae': 0.4, 'r2': 0.9},
        'xgboost': {'rmse': 0.3, 'mae': 0.2, 'r2': 0.95},
        'linear_regression': {'rmse': 0.7, 'mae': 0.6, 'r2': 0.8}
    }
    
    # Find best model
    best_model_name, best_model_metrics = find_best_model(models_metrics)
    
    # Check that best model is xgboost (lowest RMSE)
    assert best_model_name == 'xgboost'
    assert best_model_metrics == models_metrics['xgboost']