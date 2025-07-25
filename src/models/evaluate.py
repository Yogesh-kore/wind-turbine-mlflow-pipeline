# src/models/evaluate.py

import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.load_data import load_data_from_mongodb
from src.data.preprocess import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model(model_path=None, run_id=None, model_name=None):
    """
    Load a model from a local file or MLflow.
    
    Args:
        model_path: Path to the local model file
        run_id: MLflow run ID
        model_name: Name of the model in MLflow
        
    Returns:
        The loaded model
    """
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)
    
    elif run_id and model_name:
        logger.info(f"Loading model from MLflow run {run_id}")
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.sklearn.load_model(model_uri)
    
    else:
        # Try to load the best model
        best_model_info_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models',
            'best_model_info.txt'
        )
        
        if os.path.exists(best_model_info_path):
            with open(best_model_info_path, 'r') as f:
                lines = f.readlines()
                info = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        info[key.strip()] = value.strip()
            
            if 'run_id' in info and 'model_name' in info:
                logger.info(f"Loading best model: {info['model_name']} from run {info['run_id']}")
                model_uri = f"runs:/{info['run_id']}/{info['model_name']}"
                return mlflow.sklearn.load_model(model_uri)
        
        # If all else fails, look for models in the models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if model_files:
            model_path = os.path.join(models_dir, model_files[0])
            logger.info(f"Loading model from {model_path}")
            return joblib.load(model_path)
        
        raise FileNotFoundError("No model found. Please train a model first.")

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    logger.info(f"Evaluation Results for {model_name}:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    
    # Create visualizations
    create_evaluation_visualizations(y_test, y_pred, model_name)
    
    # Return metrics
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def create_evaluation_visualizations(y_true, y_pred, model_name):
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
    plt.title(f'Actual vs Predicted - {model_name} (Evaluation)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f'eval_actual_vs_predicted_{model_name}.png'))
    plt.close()
    
    # Residuals plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals Distribution - {model_name} (Evaluation)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f'eval_residuals_distribution_{model_name}.png'))
    plt.close()
    
    # Residuals vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residuals vs Predicted - {model_name} (Evaluation)')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f'eval_residuals_vs_predicted_{model_name}.png'))
    plt.close()

def evaluate_all_models(X_test=None, y_test=None, target_column='Wind_Speed'):
    """
    Evaluate all trained models.
    
    Args:
        X_test: Test features (if None, will load and preprocess data)
        y_test: Test target (if None, will load and preprocess data)
        target_column: Column to predict (used only if X_test is None)
        
    Returns:
        dict: Dictionary of evaluation results
    """
    # If test data is not provided, load and preprocess it
    if X_test is None or y_test is None:
        # Load data
        logger.info("Loading data from MongoDB...")
        df = load_data_from_mongodb()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test, preprocessors = preprocess_data(df, target_column=target_column)
    
    # Get all model files
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and f != 'preprocessors.pkl']
    
    # Evaluate each model
    results = {}
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0].replace('_', ' ').title()
        model_path = os.path.join(models_dir, model_file)
        
        try:
            model = joblib.load(model_path)
            metrics = evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
    
    # Find best model
    if results:
        best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
        logger.info(f"Best model: {best_model_name} with RMSE: {results[best_model_name]['rmse']:.4f}")
    
    return results

def find_best_model(models_metrics):
    """
    Find the best model based on RMSE.
    
    Args:
        models_metrics: Dictionary of model metrics
        
    Returns:
        tuple: (best_model_name, best_model_metrics)
    """
    if not models_metrics:
        logger.warning("No models to compare")
        return None, None
    
    # Find model with lowest RMSE
    best_model_name = min(models_metrics.items(), key=lambda x: x[1]['rmse'])[0]
    best_model_metrics = models_metrics[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with RMSE: {best_model_metrics['rmse']:.4f}")
    return best_model_name, best_model_metrics

if __name__ == "__main__":
    evaluate_all_models()