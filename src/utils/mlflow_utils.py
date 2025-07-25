# src/utils/mlflow_utils.py

import os
import mlflow
import logging
from pymongo import MongoClient
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = 'windturbine'
EXPERIMENTS_COLLECTION = 'mlflow_experiments'

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')

def setup_mlflow(experiment_name):
    """
    Set up MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        str: Experiment ID
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment '{experiment_name}' set up with ID: {experiment_id}")
        return experiment_id
    
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise

def log_experiment_to_mongodb(run_id, experiment_name, parameters, metrics, artifact_uri=None):
    """
    Log experiment metadata to MongoDB.
    
    Args:
        run_id: MLflow run ID
        experiment_name: Name of the experiment
        parameters: Dictionary of parameters
        metrics: Dictionary of metrics
        artifact_uri: URI of the artifacts
        
    Returns:
        str: MongoDB document ID
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[EXPERIMENTS_COLLECTION]
        
        experiment_data = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now(),
            "parameters": parameters,
            "metrics": metrics,
            "artifact_uri": artifact_uri
        }
        
        result = collection.insert_one(experiment_data)
        client.close()
        
        logger.info(f"Experiment logged to MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    except Exception as e:
        logger.error(f"Error logging experiment to MongoDB: {str(e)}")
        return None

def get_best_run(experiment_name, metric_name, ascending=True):
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Name of the metric to sort by
        ascending: Whether to sort in ascending order (True for metrics like RMSE, False for metrics like R²)
        
    Returns:
        dict: Best run information
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.error(f"Experiment '{experiment_name}' not found")
            return None
        
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.error(f"No runs found for experiment '{experiment_name}'")
            return None
        
        # Sort by metric
        if ascending:
            best_run = runs.sort_values(f"metrics.{metric_name}").iloc[0]
        else:
            best_run = runs.sort_values(f"metrics.{metric_name}", ascending=False).iloc[0]
        
        # Convert to dictionary
        best_run_dict = {
            "run_id": best_run["run_id"],
            "experiment_id": best_run["experiment_id"],
            "metrics": {}
        }
        
        # Add metrics
        for col in best_run.index:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                best_run_dict["metrics"][metric_name] = best_run[col]
        
        logger.info(f"Best run found: {best_run_dict['run_id']}")
        return best_run_dict
    
    except Exception as e:
        logger.error(f"Error getting best run: {str(e)}")
        return None

def register_model(run_id, model_name, model_stage="Production"):
    """
    Register a model in the MLflow Model Registry.
    
    Args:
        run_id: MLflow run ID
        model_name: Name to register the model under
        model_stage: Stage to register the model in
        
    Returns:
        str: Model version
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get model URI
        model_uri = f"runs:/{run_id}/{model_name}"
        
        # Register model
        result = mlflow.register_model(model_uri, model_name)
        
        # Transition to stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=model_stage
        )
        
        logger.info(f"Model registered as {model_name} version {result.version} in {model_stage} stage")
        return result.version
    
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        return None