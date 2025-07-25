# src/utils/visualization.py

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')

def create_output_dir(output_dir='outputs'):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
    """
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_path = os.path.join(project_root, output_dir)
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Created output directory: {output_path}")
    
    return output_path

def plot_actual_vs_predicted(y_true, y_pred, model_name, output_dir='outputs'):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    try:
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add diagonal line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Add metrics to plot
        plt.text(min_val, max_val * 0.9, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Add labels and title
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {model_name}')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Create output directory
        output_path = create_output_dir(output_dir)
        
        # Save plot
        plot_path = os.path.join(output_path, f'{model_name}_actual_vs_predicted.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved actual vs predicted plot to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error creating actual vs predicted plot: {str(e)}")
        return None

def plot_residuals(y_true, y_pred, model_name, output_dir='outputs'):
    """
    Plot residuals distribution.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    try:
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        sns.histplot(residuals, kde=True)
        
        # Add vertical line at 0
        plt.axvline(x=0, color='r', linestyle='--')
        
        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Add statistics to plot
        plt.text(max(residuals) * 0.7, plt.gca().get_ylim()[1] * 0.9, 
                 f'Mean: {mean_residual:.4f}\nStd Dev: {std_residual:.4f}', 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Add labels and title
        plt.xlabel('Residuals (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.title(f'Residuals Distribution - {model_name}')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Create output directory
        output_path = create_output_dir(output_dir)
        
        # Save plot
        plot_path = os.path.join(output_path, f'{model_name}_residuals.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved residuals plot to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error creating residuals plot: {str(e)}")
        return None

def plot_residuals_vs_predicted(y_true, y_pred, model_name, output_dir='outputs'):
    """
    Plot residuals vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    try:
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(y_pred, residuals, alpha=0.5)
        
        # Add horizontal line at 0
        plt.axhline(y=0, color='r', linestyle='--')
        
        # Add labels and title
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title(f'Residuals vs Predicted - {model_name}')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Create output directory
        output_path = create_output_dir(output_dir)
        
        # Save plot
        plot_path = os.path.join(output_path, f'{model_name}_residuals_vs_predicted.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved residuals vs predicted plot to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error creating residuals vs predicted plot: {str(e)}")
        return None

def plot_feature_importance(model, feature_names, model_name, output_dir='outputs'):
    """
    Plot feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of the features
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    try:
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return None
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        sns.barplot(x=sorted_importances, y=sorted_feature_names)
        
        # Add labels and title
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - {model_name}')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Create output directory
        output_path = create_output_dir(output_dir)
        
        # Save plot
        plot_path = os.path.join(output_path, f'{model_name}_feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        return None

def plot_correlation_matrix(df, output_dir='outputs'):
    """
    Plot correlation matrix.
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the plot
    """
    try:
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
        
        # Add title
        plt.title('Feature Correlation Matrix')
        
        # Create output directory
        output_path = create_output_dir(output_dir)
        
        # Save plot
        plot_path = os.path.join(output_path, 'correlation_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation matrix plot to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error creating correlation matrix plot: {str(e)}")
        return None

def plot_model_comparison(models_metrics, metric_name='rmse', output_dir='outputs'):
    """
    Plot model comparison.
    
    Args:
        models_metrics: Dictionary of model metrics {model_name: {metric_name: value}}
        metric_name: Name of the metric to compare
        output_dir: Directory to save the plot
    """
    try:
        # Extract model names and metric values
        model_names = list(models_metrics.keys())
        metric_values = [models_metrics[model][metric_name] for model in model_names]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        bars = plt.bar(model_names, metric_values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel(metric_name.upper())
        plt.title(f'Model Comparison - {metric_name.upper()}')
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='y')
        
        # Create output directory
        output_path = create_output_dir(output_dir)
        
        # Save plot
        plot_path = os.path.join(output_path, f'model_comparison_{metric_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error creating model comparison plot: {str(e)}")
        return None