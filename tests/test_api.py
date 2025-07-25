# tests/test_api.py

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app after adding project root to path
with patch('src.api.app.initialize'):
    from src.api.app import app

@pytest.fixture
def client():
    """
    Create a test client for the Flask app.
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_model():
    """
    Create a mock model.
    """
    mock = MagicMock()
    mock.predict.return_value = [150.5]
    return mock

@pytest.fixture
def mock_preprocessors():
    """
    Create mock preprocessors.
    """
    # Create mock encoder
    mock_encoder = MagicMock()
    mock_encoder.transform.return_value = [1]
    
    # Create mock scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.5, 0.5, 0.5]]
    
    return {
        'encoders': {'Direction': mock_encoder},
        'scaler': mock_scaler
    }

def test_health_check(client, mock_model):
    """
    Test the health check endpoint.
    """
    # Set global model
    with patch('src.api.app.model', mock_model):
        # Make request
        response = client.get('/health')
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'
        assert data['message'] == 'API is healthy'

def test_health_check_no_model(client):
    """
    Test the health check endpoint when no model is loaded.
    """
    # Set global model to None
    with patch('src.api.app.model', None):
        # Make request
        response = client.get('/health')
        
        # Check response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['message'] == 'Model not loaded'

def test_predict(client, mock_model, mock_preprocessors):
    """
    Test the predict endpoint.
    """
    # Set global model and preprocessors
    with patch('src.api.app.model', mock_model):
        with patch('src.api.app.preprocessors', mock_preprocessors):
            # Create test data
            test_data = {
                'Wind_Speed': 5.2,
                'Direction': 'N',
                'Temperature': 25.0
            }
            
            # Make request
            response = client.post('/predict', json=test_data)
            
            # Check response
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'prediction' in data
            assert data['prediction'] == 150.5
            assert 'timestamp' in data

def test_predict_no_data(client, mock_model):
    """
    Test the predict endpoint with no data.
    """
    # Set global model
    with patch('src.api.app.model', mock_model):
        # Make request with no data
        response = client.post('/predict')
        
        # Check response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

def test_predict_error(client, mock_model):
    """
    Test the predict endpoint with an error.
    """
    # Set global model to raise exception
    mock_model.predict.side_effect = Exception('Test exception')
    
    with patch('src.api.app.model', mock_model):
        # Create test data
        test_data = {
            'Wind_Speed': 5.2,
            'Direction': 'N',
            'Temperature': 25.0
        }
        
        # Make request
        response = client.post('/predict', json=test_data)
        
        # Check response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data

def test_batch_predict(client, mock_model, mock_preprocessors):
    """
    Test the batch predict endpoint.
    """
    # Set global model and preprocessors
    mock_model.predict.return_value = [150.5, 160.5, 170.5]
    
    with patch('src.api.app.model', mock_model):
        with patch('src.api.app.preprocessors', mock_preprocessors):
            # Create test data
            test_data = [
                {'Wind_Speed': 5.2, 'Direction': 'N', 'Temperature': 25.0},
                {'Wind_Speed': 6.7, 'Direction': 'S', 'Temperature': 27.5},
                {'Wind_Speed': 4.3, 'Direction': 'E', 'Temperature': 22.1}
            ]
            
            # Make request
            response = client.post('/batch-predict', json=test_data)
            
            # Check response
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'predictions' in data
            assert len(data['predictions']) == 3
            assert data['count'] == 3
            assert 'timestamp' in data

def test_batch_predict_no_data(client, mock_model):
    """
    Test the batch predict endpoint with no data.
    """
    # Set global model
    with patch('src.api.app.model', mock_model):
        # Make request with no data
        response = client.post('/batch-predict')
        
        # Check response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

def test_batch_predict_invalid_data(client, mock_model):
    """
    Test the batch predict endpoint with invalid data.
    """
    # Set global model
    with patch('src.api.app.model', mock_model):
        # Make request with invalid data (not a list)
        response = client.post('/batch-predict', json={'invalid': 'data'})
        
        # Check response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

def test_model_info(client, tmp_path):
    """
    Test the model info endpoint.
    """
    # Create mock best_model_info.txt
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    with open(models_dir / "best_model_info.txt", 'w') as f:
        f.write("Model Name: xgboost\n")
        f.write("rmse: 0.3\n")
        f.write("mae: 0.2\n")
        f.write("r2: 0.95\n")
    
    # Patch os.path.join and os.path.exists
    with patch('src.api.app.os.path.join', return_value=str(models_dir / "best_model_info.txt")):
        with patch('src.api.app.os.path.exists', return_value=True):
            # Make request
            response = client.get('/model-info')
            
            # Check response
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'model_info' in data
            assert data['model_info']['Model Name'] == 'xgboost'
            assert data['model_info']['rmse'] == '0.3'
            assert data['model_info']['mae'] == '0.2'
            assert data['model_info']['r2'] == '0.95'
            assert 'timestamp' in data

def test_model_info_no_file(client):
    """
    Test the model info endpoint when the info file doesn't exist.
    """
    # Patch os.path.exists to return False
    with patch('src.api.app.os.path.exists', return_value=False):
        # Make request
        response = client.get('/model-info')
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'model_info' in data
        assert data['model_info'] == {}
        assert 'timestamp' in data

def test_model_info_error(client):
    """
    Test the model info endpoint with an error.
    """
    # Patch os.path.exists to raise exception
    with patch('src.api.app.os.path.exists', side_effect=Exception('Test exception')):
        # Make request
        response = client.get('/model-info')
        
        # Check response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data