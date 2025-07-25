# src/data/preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats.mstats import winsorize
import logging
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Input data
        
    Returns:
        pandas.DataFrame: Data with handled missing values
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"Found {missing_values.sum()} missing values")
        logger.info(f"Missing values by column:\n{missing_values[missing_values > 0]}")
        
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        # For categorical columns, fill with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def handle_outliers(df, method='iqr', columns=None):
    """
    Handle outliers in the dataset.
    
    Args:
        df (pandas.DataFrame): Input data
        method (str): Method to handle outliers ('iqr' or 'winsorize')
        columns (list): Columns to handle outliers for. If None, uses all numeric columns.
        
    Returns:
        pandas.DataFrame: Data with handled outliers
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # If columns not specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    # Handle outliers based on method
    if method == 'iqr':
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the values
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
    elif method == 'winsorize':
        for col in columns:
            df_clean[col] = winsorize(df_clean[col], limits=[0.05, 0.05])
    
    return df_clean

def encode_categorical(df, columns=None):
    """
    Encode categorical variables.
    
    Args:
        df (pandas.DataFrame): Input data
        columns (list): Columns to encode. If None, uses all object columns.
        
    Returns:
        pandas.DataFrame: Data with encoded categorical variables
        dict: Dictionary of encoders
    """
    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # If columns not specified, use all object columns
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    # Create encoders dictionary
    encoders = {}
    
    # Encode each column
    for col in columns:
        if col in df_encoded.columns:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col])
            encoders[col] = encoder
    
    return df_encoded, encoders

def scale_features(df, columns=None):
    """
    Scale numeric features.
    
    Args:
        df (pandas.DataFrame): Input data
        columns (list): Columns to scale. If None, uses all numeric columns.
        
    Returns:
        pandas.DataFrame: Data with scaled features
        sklearn.preprocessing.StandardScaler: Fitted scaler
    """
    # Make a copy to avoid modifying the original
    df_scaled = df.copy()
    
    # If columns not specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    # Create and fit scaler
    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    return df_scaled, scaler

def preprocess_data(df, target_column='Wind_Speed', test_size=0.2, random_state=42):
    """
    Preprocess the wind turbine dataset.
    
    Args:
        df (pandas.DataFrame): Raw data
        target_column (str): Column to predict
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessors
    """
    logger.info("Starting data preprocessing")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df, method='iqr')
    
    # Encode categorical variables
    df_encoded, encoders = encode_categorical(df)
    
    # Ensure target exists
    if target_column not in df_encoded.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Split features and target
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Create preprocessors dictionary
    preprocessors = {
        'encoders': encoders,
        'scaler': scaler
    }
    
    # Save preprocessors
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'), exist_ok=True)
    joblib.dump(preprocessors, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'preprocessors.pkl'))
    
    logger.info("Data preprocessing completed")
    logger.info(f"Training set shape: {X_train_scaled.shape}")
    logger.info(f"Testing set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessors