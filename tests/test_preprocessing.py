# tests/test_preprocessing.py

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import (
    handle_missing_values,
    handle_outliers,
    encode_categorical_variables,
    scale_features,
    preprocess_data
)

# Test data
@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    return pd.DataFrame({
        'TurbineName': ['Turbine1', 'Turbine2', 'Turbine3', 'Turbine1', 'Turbine2'],
        'Wind_Speed': [5.2, 6.7, np.nan, 4.3, 20.0],  # One missing, one outlier
        'Direction': ['N', 'S', 'E', np.nan, 'W'],  # One missing
        'Temperature': [25.0, 27.5, 22.1, 26.8, 24.3],
        'Power_Output': [120.5, 150.2, 90.8, 110.3, 200.0]  # Target
    })

def test_handle_missing_values(sample_data):
    """
    Test handling missing values.
    """
    # Make a copy of the data
    df = sample_data.copy()
    
    # Count missing values before
    missing_before = df.isna().sum().sum()
    assert missing_before > 0, "Test data should have missing values"
    
    # Handle missing values
    df_cleaned = handle_missing_values(df)
    
    # Count missing values after
    missing_after = df_cleaned.isna().sum().sum()
    assert missing_after == 0, "All missing values should be handled"
    
    # Check that the original DataFrame is not modified
    assert sample_data.isna().sum().sum() == missing_before
    
    # Check that numeric columns are filled with median
    assert df_cleaned.loc[2, 'Wind_Speed'] == df.loc[[0, 1, 3, 4], 'Wind_Speed'].median()
    
    # Check that categorical columns are filled with mode
    assert df_cleaned.loc[3, 'Direction'] == 'N'  # Most common value

def test_handle_outliers(sample_data):
    """
    Test handling outliers.
    """
    # Make a copy of the data
    df = sample_data.copy()
    
    # Handle missing values first
    df = handle_missing_values(df)
    
    # Get the outlier value
    outlier_value = df.loc[4, 'Wind_Speed']
    assert outlier_value == 20.0, "Test data should have an outlier"
    
    # Handle outliers with IQR method
    df_iqr = handle_outliers(df, method='iqr')
    assert df_iqr.loc[4, 'Wind_Speed'] < outlier_value, "IQR method should cap the outlier"
    
    # Handle outliers with winsorization
    df_winsor = handle_outliers(df, method='winsorize')
    assert df_winsor.loc[4, 'Wind_Speed'] < outlier_value, "Winsorization should cap the outlier"
    
    # Check that the original DataFrame is not modified
    assert df.loc[4, 'Wind_Speed'] == outlier_value

def test_encode_categorical_variables(sample_data):
    """
    Test encoding categorical variables.
    """
    # Make a copy of the data
    df = sample_data.copy()
    
    # Handle missing values first
    df = handle_missing_values(df)
    
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    assert len(cat_cols) > 0, "Test data should have categorical columns"
    
    # Encode categorical variables
    df_encoded, encoders = encode_categorical_variables(df)
    
    # Check that all categorical columns are encoded
    for col in cat_cols:
        assert col in encoders, f"Column {col} should have an encoder"
        assert df_encoded[col].dtype != 'object', f"Column {col} should be encoded"
    
    # Check that encoders are LabelEncoders
    for encoder in encoders.values():
        assert isinstance(encoder, LabelEncoder)
    
    # Check that the original DataFrame is not modified
    for col in cat_cols:
        assert df[col].dtype == 'object'

def test_scale_features(sample_data):
    """
    Test scaling features.
    """
    # Make a copy of the data
    df = sample_data.copy()
    
    # Handle missing values first
    df = handle_missing_values(df)
    
    # Get numeric columns (excluding target)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    num_cols.remove('Power_Output')  # Exclude target
    assert len(num_cols) > 0, "Test data should have numeric columns"
    
    # Scale features
    X = df.drop('Power_Output', axis=1)
    y = df['Power_Output']
    X_scaled, scaler = scale_features(X, num_cols)
    
    # Check that all numeric columns are scaled
    for col in num_cols:
        assert X_scaled[col].mean() == pytest.approx(0, abs=1e-10), f"Column {col} should have mean close to 0"
        assert X_scaled[col].std() == pytest.approx(1, abs=1e-10), f"Column {col} should have std close to 1"
    
    # Check that scaler is StandardScaler
    assert isinstance(scaler, StandardScaler)
    
    # Check that non-numeric columns are not scaled
    for col in X.columns:
        if col not in num_cols:
            assert X[col].equals(X_scaled[col]), f"Column {col} should not be scaled"

def test_preprocess_data(sample_data, tmp_path):
    """
    Test the complete preprocessing pipeline.
    """
    # Set models directory to temp path
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    with patch('src.data.preprocess.os.path.join', return_value=str(models_dir / "preprocessors.pkl")):
        # Run preprocessing
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(
            sample_data, test_size=0.4, random_state=42
        )
        
        # Check output shapes
        assert X_train.shape[0] + X_test.shape[0] == len(sample_data)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert len(feature_names) == X_train.shape[1]
        
        # Check that preprocessors file is created
        assert os.path.exists(models_dir / "preprocessors.pkl")