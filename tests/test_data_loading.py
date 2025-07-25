# tests/test_data_loading.py

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_data_from_csv, load_data_to_mongodb, load_data_from_mongodb

# Test data
TEST_DATA = pd.DataFrame({
    'TurbineName': ['Turbine1', 'Turbine2', 'Turbine3'],
    'Wind_Speed': [5.2, 6.7, 4.3],
    'Power_Output': [120.5, 150.2, 90.8]
})

@pytest.fixture
def mock_csv_file(tmp_path):
    """
    Create a mock CSV file for testing.
    """
    csv_path = tmp_path / "test_data.csv"
    TEST_DATA.to_csv(csv_path, index=False)
    return str(csv_path)

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
    mock_cursor.__iter__.return_value = TEST_DATA.to_dict('records')
    mock_collection.find.return_value = mock_cursor
    
    # Setup count_documents method
    mock_collection.count_documents.return_value = len(TEST_DATA)
    
    # Setup insert_many method
    mock_collection.insert_many.return_value = MagicMock()
    
    # Setup drop method
    mock_collection.drop.return_value = None
    
    # Setup list_collection_names method
    mock_db.list_collection_names.return_value = ['turbine_data']
    
    # Link mocks together
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    
    return mock_client

def test_load_data_from_csv(mock_csv_file):
    """
    Test loading data from CSV.
    """
    # Test with valid file
    df = load_data_from_csv(mock_csv_file)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(TEST_DATA)
    assert list(df.columns) == list(TEST_DATA.columns)
    
    # Test with invalid file
    df = load_data_from_csv('nonexistent_file.csv')
    assert df is None

@patch('src.data.load_data.MongoClient')
def test_load_data_to_mongodb(mock_mongo_client_class, mock_mongo_client):
    """
    Test loading data to MongoDB.
    """
    # Setup mock
    mock_mongo_client_class.return_value = mock_mongo_client
    
    # Test with drop_existing=False
    result = load_data_to_mongodb(TEST_DATA, drop_existing=False)
    assert result is True
    mock_mongo_client['windturbine']['turbine_data'].insert_many.assert_called_once()
    mock_mongo_client['windturbine']['turbine_data'].drop.assert_not_called()
    
    # Reset mocks
    mock_mongo_client.reset_mock()
    
    # Test with drop_existing=True
    result = load_data_to_mongodb(TEST_DATA, drop_existing=True)
    assert result is True
    mock_mongo_client['windturbine']['turbine_data'].drop.assert_called_once()
    mock_mongo_client['windturbine']['turbine_data'].insert_many.assert_called_once()
    
    # Test with exception
    mock_mongo_client['windturbine']['turbine_data'].insert_many.side_effect = Exception('Test exception')
    result = load_data_to_mongodb(TEST_DATA)
    assert result is False

@patch('src.data.load_data.MongoClient')
@patch('src.data.load_data.load_data_from_csv')
def test_load_data_from_mongodb(mock_load_csv, mock_mongo_client_class, mock_mongo_client):
    """
    Test loading data from MongoDB.
    """
    # Setup mocks
    mock_mongo_client_class.return_value = mock_mongo_client
    mock_load_csv.return_value = TEST_DATA
    
    # Test with data in MongoDB
    df = load_data_from_mongodb()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    mock_load_csv.assert_not_called()
    
    # Test with empty collection
    mock_mongo_client['windturbine']['turbine_data'].count_documents.return_value = 0
    df = load_data_from_mongodb()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    mock_load_csv.assert_called_once()
    
    # Test with exception
    mock_mongo_client['windturbine']['turbine_data'].find.side_effect = Exception('Test exception')
    mock_load_csv.reset_mock()
    df = load_data_from_mongodb()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    mock_load_csv.assert_called_once()