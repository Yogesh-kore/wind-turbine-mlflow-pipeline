# tests/test_utils.py

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.mlflow_utils import setup_mlflow, log_experiment_to_mongodb, get_best_run, register_model
from src.utils.mongo_utils import get_mongo_client, insert_dataframe, get_dataframe, log_prediction, get_collection_stats
from src.utils.visualization import plot_actual_vs_predicted, plot_residuals, plot_residuals_vs_predicted, plot_feature_importance, plot_correlation_matrix, plot_model_comparison

# Test data
@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    # Create feature matrix
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature3': [10, 20, 30, 40, 50]
    })
    
    # Create target vector
    y = pd.Series([10.5, 20.5, 30.5, 40.5, 50.5])
    
    # Create predictions
    y_pred = pd.Series([11.0, 21.0, 31.0, 41.0, 51.0])
    
    return X, y, y_pred

@pytest.fixture
def mock_mongo_client():
    """
    Create a mock MongoDB client.
    """
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    
    # Setup find method to return test data
    mock_cursor = MagicMock()
    mock_cursor.__iter__.return_value = [
        {'feature1': 1.0, 'feature2': 0.1, 'feature3': 10, 'target': 10.5},
        {'feature1': 2.0, 'feature2': 0.2, 'feature3': 20, 'target': 20.5}
    ]
    mock_collection.find.return_value = mock_cursor
    
    # Setup count_documents method
    mock_collection.count_documents.return_value = 2
    
    # Setup insert_many method
    mock_collection.insert_many.return_value = MagicMock()
    mock_collection.insert_many.return_value.inserted_ids = ['id1', 'id2']
    
    # Setup insert_one method
    mock_collection.insert_one.return_value = MagicMock()
    mock_collection.insert_one.return_value.inserted_id = 'id1'
    
    # Setup drop method
    mock_collection.drop.return_value = None
    
    # Setup list_collection_names method
    mock_db.list_collection_names.return_value = ['turbine_data']
    
    # Setup command method
    mock_db.command.return_value = {
        'count': 2,
        'size': 1000,
        'avgObjSize': 500,
        'storageSize': 2000,
        'indexSizes': {'_id_': 100},
        'totalIndexSize': 100
    }
    
    # Link mocks together
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    
    return mock_client

@pytest.fixture
def mock_mlflow():
    """
    Create a mock MLflow module.
    """
    with patch('src.utils.mlflow_utils.mlflow') as mock_mlflow:
        # Setup mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = 'test_experiment_id'
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        # Setup mock run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_mlflow.start_run.return_value = mock_run
        
        # Setup mock search_runs
        mock_runs = pd.DataFrame({
            'run_id': ['run1', 'run2', 'run3'],
            'experiment_id': ['test_experiment_id'] * 3,
            'metrics.rmse': [0.5, 0.3, 0.7],
            'metrics.mae': [0.4, 0.2, 0.6],
            'metrics.r2': [0.9, 0.95, 0.8]
        })
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Setup mock register_model
        mock_model_version = MagicMock()
        mock_model_version.version = '1'
        mock_mlflow.register_model.return_value = mock_model_version
        
        yield mock_mlflow

# MLflow Utils Tests
def test_setup_mlflow(mock_mlflow):
    """
    Test setting up MLflow.
    """
    # Test with existing experiment
    experiment_id = setup_mlflow('test_experiment')
    assert experiment_id == 'test_experiment_id'
    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.get_experiment_by_name.assert_called_once_with('test_experiment')
    mock_mlflow.set_experiment.assert_called_once_with('test_experiment')
    
    # Test with new experiment
    mock_mlflow.get_experiment_by_name.return_value = None
    mock_mlflow.create_experiment.return_value = 'new_experiment_id'
    
    mock_mlflow.reset_mock()
    experiment_id = setup_mlflow('new_experiment')
    assert experiment_id == 'new_experiment_id'
    mock_mlflow.create_experiment.assert_called_once_with('new_experiment')

@patch('src.utils.mlflow_utils.MongoClient')
def test_log_experiment_to_mongodb(mock_mongo_client_class, mock_mongo_client):
    """
    Test logging experiment to MongoDB.
    """
    # Setup mock
    mock_mongo_client_class.return_value = mock_mongo_client
    
    # Test logging experiment
    run_id = 'test_run_id'
    experiment_name = 'test_experiment'
    parameters = {'param1': 1, 'param2': 2}
    metrics = {'metric1': 0.1, 'metric2': 0.2}
    artifact_uri = 'test_artifact_uri'
    
    result = log_experiment_to_mongodb(run_id, experiment_name, parameters, metrics, artifact_uri)
    assert result == 'id1'
    mock_mongo_client['windturbine']['mlflow_experiments'].insert_one.assert_called_once()
    
    # Test with exception
    mock_mongo_client['windturbine']['mlflow_experiments'].insert_one.side_effect = Exception('Test exception')
    result = log_experiment_to_mongodb(run_id, experiment_name, parameters, metrics, artifact_uri)
    assert result is None

def test_get_best_run(mock_mlflow):
    """
    Test getting the best run.
    """
    # Test with ascending=True (default)
    best_run = get_best_run('test_experiment', 'rmse')
    assert best_run is not None
    assert best_run['run_id'] == 'run2'  # run2 has lowest RMSE
    assert best_run['metrics']['rmse'] == 0.3
    
    # Test with ascending=False
    best_run = get_best_run('test_experiment', 'r2', ascending=False)
    assert best_run is not None
    assert best_run['run_id'] == 'run2'  # run2 has highest R²
    assert best_run['metrics']['r2'] == 0.95
    
    # Test with non-existent experiment
    mock_mlflow.get_experiment_by_name.return_value = None
    best_run = get_best_run('non_existent_experiment', 'rmse')
    assert best_run is None
    
    # Test with empty runs
    mock_mlflow.get_experiment_by_name.return_value = mock_mlflow.get_experiment_by_name.return_value
    mock_mlflow.search_runs.return_value = pd.DataFrame()
    best_run = get_best_run('test_experiment', 'rmse')
    assert best_run is None

def test_register_model(mock_mlflow):
    """
    Test registering a model.
    """
    # Test registering model
    run_id = 'test_run_id'
    model_name = 'test_model'
    model_stage = 'Production'
    
    version = register_model(run_id, model_name, model_stage)
    assert version == '1'
    mock_mlflow.register_model.assert_called_once_with(f"runs:/{run_id}/{model_name}", model_name)
    mock_mlflow.tracking.MlflowClient().transition_model_version_stage.assert_called_once_with(
        name=model_name, version='1', stage=model_stage
    )
    
    # Test with exception
    mock_mlflow.register_model.side_effect = Exception('Test exception')
    version = register_model(run_id, model_name, model_stage)
    assert version is None

# Mongo Utils Tests
@patch('src.utils.mongo_utils.MongoClient')
def test_get_mongo_client(mock_mongo_client_class):
    """
    Test getting a MongoDB client.
    """
    # Test getting client
    mock_mongo_client_class.return_value = 'test_client'
    client = get_mongo_client()
    assert client == 'test_client'
    mock_mongo_client_class.assert_called_once_with('mongodb://localhost:27017/')
    
    # Test with exception
    mock_mongo_client_class.side_effect = Exception('Test exception')
    with pytest.raises(Exception):
        client = get_mongo_client()

@patch('src.utils.mongo_utils.get_mongo_client')
def test_insert_dataframe(mock_get_mongo_client, mock_mongo_client):
    """
    Test inserting a DataFrame into MongoDB.
    """
    # Setup mock
    mock_get_mongo_client.return_value = mock_mongo_client
    
    # Test with drop_existing=False
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    result = insert_dataframe(df, 'test_collection', drop_existing=False)
    assert result == 2
    mock_mongo_client['windturbine']['test_collection'].drop.assert_not_called()
    mock_mongo_client['windturbine']['test_collection'].insert_many.assert_called_once()
    
    # Reset mocks
    mock_mongo_client.reset_mock()
    
    # Test with drop_existing=True
    result = insert_dataframe(df, 'test_collection', drop_existing=True)
    assert result == 2
    mock_mongo_client['windturbine']['test_collection'].drop.assert_called_once()
    mock_mongo_client['windturbine']['test_collection'].insert_many.assert_called_once()
    
    # Test with exception
    mock_mongo_client['windturbine']['test_collection'].insert_many.side_effect = Exception('Test exception')
    result = insert_dataframe(df, 'test_collection')
    assert result == 0

@patch('src.utils.mongo_utils.get_mongo_client')
def test_get_dataframe(mock_get_mongo_client, mock_mongo_client):
    """
    Test getting a DataFrame from MongoDB.
    """
    # Setup mock
    mock_get_mongo_client.return_value = mock_mongo_client
    
    # Test with existing collection
    df = get_dataframe('turbine_data')
    assert not df.empty
    assert len(df) == 2
    mock_mongo_client['windturbine']['turbine_data'].find.assert_called_once_with({})
    
    # Test with query and limit
    mock_mongo_client.reset_mock()
    df = get_dataframe('turbine_data', query={'feature1': 1.0}, limit=1)
    assert not df.empty
    mock_mongo_client['windturbine']['turbine_data'].find.assert_called_once_with({'feature1': 1.0})
    
    # Test with non-existent collection
    mock_mongo_client['windturbine'].list_collection_names.return_value = []
    df = get_dataframe('non_existent_collection')
    assert df.empty
    
    # Test with exception
    mock_mongo_client['windturbine'].list_collection_names.return_value = ['turbine_data']
    mock_mongo_client['windturbine']['turbine_data'].find.side_effect = Exception('Test exception')
    df = get_dataframe('turbine_data')
    assert df.empty

@patch('src.utils.mongo_utils.get_mongo_client')
def test_log_prediction(mock_get_mongo_client, mock_mongo_client):
    """
    Test logging a prediction to MongoDB.
    """
    # Setup mock
    mock_get_mongo_client.return_value = mock_mongo_client
    
    # Test logging prediction
    input_data = {'feature1': 1.0, 'feature2': 0.1, 'feature3': 10}
    prediction = 10.5
    model_name = 'test_model'
    
    result = log_prediction(input_data, prediction, model_name)
    assert result == 'id1'
    mock_mongo_client['windturbine']['predictions'].insert_one.assert_called_once()
    
    # Test with exception
    mock_mongo_client['windturbine']['predictions'].insert_one.side_effect = Exception('Test exception')
    result = log_prediction(input_data, prediction, model_name)
    assert result is None

@patch('src.utils.mongo_utils.get_mongo_client')
def test_get_collection_stats(mock_get_mongo_client, mock_mongo_client):
    """
    Test getting collection statistics.
    """
    # Setup mock
    mock_get_mongo_client.return_value = mock_mongo_client
    
    # Test with existing collection
    stats = get_collection_stats('turbine_data')
    assert stats is not None
    assert stats['count'] == 2
    assert stats['size'] == 1000
    assert stats['avg_obj_size'] == 500
    assert stats['storage_size'] == 2000
    assert stats['index_count'] == 1
    assert stats['index_size'] == 100
    mock_mongo_client['windturbine'].command.assert_called_once_with('collStats', 'turbine_data')
    
    # Test with non-existent collection
    mock_mongo_client['windturbine'].list_collection_names.return_value = []
    stats = get_collection_stats('non_existent_collection')
    assert stats == {}
    
    # Test with exception
    mock_mongo_client['windturbine'].list_collection_names.return_value = ['turbine_data']
    mock_mongo_client['windturbine'].command.side_effect = Exception('Test exception')
    stats = get_collection_stats('turbine_data')
    assert stats == {}

# Visualization Tests
def test_plot_actual_vs_predicted(sample_data, tmp_path):
    """
    Test plotting actual vs predicted values.
    """
    X, y_true, y_pred = sample_data
    
    # Create output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Patch os.path.dirname and os.path.join
    with patch('src.utils.visualization.os.path.dirname', return_value=str(tmp_path)):
        with patch('src.utils.visualization.os.path.join', return_value=str(output_dir / "test_model_actual_vs_predicted.png")):
            # Plot actual vs predicted
            plot_path = plot_actual_vs_predicted(y_true, y_pred, 'test_model', output_dir=str(output_dir))
            
            # Check that plot is created
            assert plot_path == str(output_dir / "test_model_actual_vs_predicted.png")
            assert os.path.exists(output_dir / "test_model_actual_vs_predicted.png")

def test_plot_residuals(sample_data, tmp_path):
    """
    Test plotting residuals distribution.
    """
    X, y_true, y_pred = sample_data
    
    # Create output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Patch os.path.dirname and os.path.join
    with patch('src.utils.visualization.os.path.dirname', return_value=str(tmp_path)):
        with patch('src.utils.visualization.os.path.join', return_value=str(output_dir / "test_model_residuals.png")):
            # Plot residuals
            plot_path = plot_residuals(y_true, y_pred, 'test_model', output_dir=str(output_dir))
            
            # Check that plot is created
            assert plot_path == str(output_dir / "test_model_residuals.png")
            assert os.path.exists(output_dir / "test_model_residuals.png")

def test_plot_residuals_vs_predicted(sample_data, tmp_path):
    """
    Test plotting residuals vs predicted values.
    """
    X, y_true, y_pred = sample_data
    
    # Create output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Patch os.path.dirname and os.path.join
    with patch('src.utils.visualization.os.path.dirname', return_value=str(tmp_path)):
        with patch('src.utils.visualization.os.path.join', return_value=str(output_dir / "test_model_residuals_vs_predicted.png")):
            # Plot residuals vs predicted
            plot_path = plot_residuals_vs_predicted(y_true, y_pred, 'test_model', output_dir=str(output_dir))
            
            # Check that plot is created
            assert plot_path == str(output_dir / "test_model_residuals_vs_predicted.png")
            assert os.path.exists(output_dir / "test_model_residuals_vs_predicted.png")

def test_plot_feature_importance(sample_data, tmp_path):
    """
    Test plotting feature importance.
    """
    X, y_true, y_pred = sample_data
    
    # Create mock model with feature_importances_
    class MockModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.5, 0.3, 0.2])
    
    model = MockModel()
    feature_names = X.columns.tolist()
    
    # Create output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Patch os.path.dirname and os.path.join
    with patch('src.utils.visualization.os.path.dirname', return_value=str(tmp_path)):
        with patch('src.utils.visualization.os.path.join', return_value=str(output_dir / "test_model_feature_importance.png")):
            # Plot feature importance
            plot_path = plot_feature_importance(model, feature_names, 'test_model', output_dir=str(output_dir))
            
            # Check that plot is created
            assert plot_path == str(output_dir / "test_model_feature_importance.png")
            assert os.path.exists(output_dir / "test_model_feature_importance.png")
    
    # Test with model without feature_importances_
    class MockModelNoImportance:
        def __init__(self):
            pass
    
    model = MockModelNoImportance()
    
    # Patch os.path.dirname and os.path.join
    with patch('src.utils.visualization.os.path.dirname', return_value=str(tmp_path)):
        # Plot feature importance
        plot_path = plot_feature_importance(model, feature_names, 'test_model', output_dir=str(output_dir))
        
        # Check that no plot is created
        assert plot_path is None

def test_plot_correlation_matrix(sample_data, tmp_path):
    """
    Test plotting correlation matrix.
    """
    X, y_true, y_pred = sample_data
    
    # Create output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Patch os.path.dirname and os.path.join
    with patch('src.utils.visualization.os.path.dirname', return_value=str(tmp_path)):
        with patch('src.utils.visualization.os.path.join', return_value=str(output_dir / "correlation_matrix.png")):
            # Plot correlation matrix
            plot_path = plot_correlation_matrix(X, output_dir=str(output_dir))
            
            # Check that plot is created
            assert plot_path == str(output_dir / "correlation_matrix.png")
            assert os.path.exists(output_dir / "correlation_matrix.png")

def test_plot_model_comparison(tmp_path):
    """
    Test plotting model comparison.
    """
    # Create models metrics
    models_metrics = {
        'random_forest': {'rmse': 0.5, 'mae': 0.4, 'r2': 0.9},
        'xgboost': {'rmse': 0.3, 'mae': 0.2, 'r2': 0.95},
        'linear_regression': {'rmse': 0.7, 'mae': 0.6, 'r2': 0.8}
    }
    
    # Create output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Patch os.path.dirname and os.path.join
    with patch('src.utils.visualization.os.path.dirname', return_value=str(tmp_path)):
        with patch('src.utils.visualization.os.path.join', return_value=str(output_dir / "model_comparison_rmse.png")):
            # Plot model comparison
            plot_path = plot_model_comparison(models_metrics, metric_name='rmse', output_dir=str(output_dir))
            
            # Check that plot is created
            assert plot_path == str(output_dir / "model_comparison_rmse.png")
            assert os.path.exists(output_dir / "model_comparison_rmse.png")