#!/usr/bin/env python
# scripts/run_pipeline.py

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_data_from_csv, load_data_to_mongodb
from src.data.preprocess import preprocess_data
from src.models.train import train_models
from src.models.evaluate import evaluate_all_models, find_best_model
from src.utils.mlflow_utils import setup_mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the Wind Turbine MLflow Pipeline')
    
    parser.add_argument('--data-path', type=str, default='C:\\Users\\ADMIN\\Desktop\\Project\\wind-turbine-mlflow-pipeline\\windt_cleaned_iqr.csv',
                        help='Path to the input data CSV file')
    parser.add_argument('--load-to-mongodb', action='store_true',
                        help='Load data to MongoDB')
    parser.add_argument('--drop-existing', action='store_true',
                        help='Drop existing MongoDB collection before loading data')
    parser.add_argument('--experiment-name', type=str, default='wind_turbine_experiment',
                        help='Name of the MLflow experiment')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip model evaluation')
    
    return parser.parse_args()

def create_directories():
    """
    Create necessary directories.
    """
    directories = ['logs', 'data', 'models', 'outputs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def run_pipeline():
    """
    Run the complete pipeline.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create directories
    create_directories()
    
    # Setup MLflow
    experiment_id = setup_mlflow(args.experiment_name)
    logger.info(f"MLflow experiment set up with ID: {experiment_id}")
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data_path}")
        df = load_data_from_csv(args.data_path)
        
        if df is None or df.empty:
            logger.error("Failed to load data. Exiting.")
            return 1
        
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Load data to MongoDB if requested
        if args.load_to_mongodb:
            logger.info("Loading data to MongoDB")
            load_data_to_mongodb(df, drop_existing=args.drop_existing)
        
        # Preprocess data
        logger.info("Preprocessing data")
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(
            df, test_size=args.test_size, random_state=args.random_state
        )
        
        logger.info(f"Preprocessed data: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Train models
        if not args.skip_training:
            logger.info("Training models")
            train_models(
                X_train, X_test, y_train, y_test, feature_names,
                experiment_name=args.experiment_name,
                random_state=args.random_state
            )
        
        # Evaluate models
        if not args.skip_evaluation:
            logger.info("Evaluating models")
            models_metrics = evaluate_all_models(X_test, y_test)
            
            # Find best model
            best_model_name, best_model_metrics = find_best_model(models_metrics)
            logger.info(f"Best model: {best_model_name} with metrics: {best_model_metrics}")
            
            # Save best model info
            with open(os.path.join('models', 'best_model_info.txt'), 'w') as f:
                f.write(f"Model Name: {best_model_name}\n")
                for metric_name, metric_value in best_model_metrics.items():
                    f.write(f"{metric_name}: {metric_value}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
        logger.info("Pipeline completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(run_pipeline())