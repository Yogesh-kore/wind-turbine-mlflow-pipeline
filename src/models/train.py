# src/models/train.py

import mlflow
import mlflow.sklearn
import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.load_data import load_data_from_mongodb
from src.data.preprocess import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set experiment name
EXPERIMENT_NAME = "WindTurbine_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

def train_and_log_model(model, X_train, X_test, y_train, y_test, model_name, params=None):
    """
    Train a model, evaluate it, and log results to MLflow.
    
    Args:
        model: Model instance to train
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        model_name: Name of the model
        params: Dictionary of model parameters
        
    Returns:
        tuple: (trained_model, metrics)
    """
    logger.info(f"Training {model_name}...")
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name) as run:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Log parameters
        if params:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Save model locally
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        
        # Create visualization
        create_model_visualizations(y_test, y_pred_test, model_name)
        
        # Log artifacts
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs')
        mlflow.log_artifacts(outputs_dir, artifact_path="visualizations")
        
        # Print results
        logger.info(f"{model_name} Results:")
        logger.info(f"  Train RMSE: {train_rmse:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.4f}")
        logger.info(f"  Train MAE: {train_mae:.4f}")
        logger.info(f"  Test MAE: {test_mae:.4f}")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        logger.info(f"  Run ID: {run.info.run_id}")
        logger.info(f"  Model saved to: {model_path}")
        
        # Return metrics
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'run_id': run.info.run_id
        }
        
        return model, metrics

def create_model_visualizations(y_true, y_pred, model_name):
    """
    Create and save visualizations for model evaluation.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
    """
    # Create output directory
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f'actual_vs_predicted_{model_name}.png'))
    plt.close()
    
    # Residuals plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals Distribution - {model_name}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f'residuals_distribution_{model_name}.png'))
    plt.close()

def train_models(X_train=None, X_test=None, y_train=None, y_test=None, feature_names=None, experiment_name=None, random_state=42, target_column='Wind_Speed'):
    """
    Train multiple models and compare their performance.
    
    Args:
        X_train: Training features (if None, will load and preprocess data)
        X_test: Testing features (if None, will load and preprocess data)
        y_train: Training target (if None, will load and preprocess data)
        y_test: Testing target (if None, will load and preprocess data)
        feature_names: Names of features (if None, will use default)
        experiment_name: Name of the MLflow experiment (if None, will use default)
        random_state: Random state for reproducibility
        target_column: Column to predict (used only if X_train is None)
        
    Returns:
        dict: Dictionary of trained models and their metrics
    """
    # If data is not provided, load and preprocess it
    if X_train is None or X_test is None or y_train is None or y_test is None:
        # Load data
        logger.info("Loading data from MongoDB...")
        df = load_data_from_mongodb()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test, preprocessors = preprocess_data(df, target_column=target_column)
    
    # Set experiment name if provided
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Define models with parameters
    models = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (RandomForestRegressor(random_state=42), {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }),
        "XGBoost": (XGBRegressor(random_state=42), {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        })
    }
    
    # Train and evaluate each model
    results = {}
    best_model_name = None
    best_test_rmse = float('inf')
    
    for model_name, (model, params) in models.items():
        trained_model, metrics = train_and_log_model(
            model, X_train, X_test, y_train, y_test, model_name, params
        )
        results[model_name] = {
            'model': trained_model,
            'metrics': metrics
        }
        
        # Track best model
        if metrics['test_rmse'] < best_test_rmse:
            best_test_rmse = metrics['test_rmse']
            best_model_name = model_name
    
    # Log best model
    if best_model_name:
        logger.info(f"Best model: {best_model_name} with Test RMSE: {best_test_rmse:.4f}")
        
        # Save best model info
        best_model_info = {
            'model_name': best_model_name,
            'run_id': results[best_model_name]['metrics']['run_id'],
            'test_rmse': best_test_rmse,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save best model info to file
        best_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models',
            'best_model_info.txt'
        )
        with open(best_model_path, 'w') as f:
            for key, value in best_model_info.items():
                f.write(f"{key}: {value}\n")
    
    return results

if __name__ == "__main__":
    train_models()