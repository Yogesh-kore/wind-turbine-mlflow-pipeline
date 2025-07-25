# src/data/load_data.py

import pandas as pd
import os
from pymongo import MongoClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = 'windturbine'
COLLECTION_NAME = 'Wind_Turbine'

# Column names for the dataset
COLUMN_NAMES = [
    "Nacelle_Position",
    "Wind_direction",
    "Ambient_Air_temp",
    "Bearing_Temp",
    "BladePitchAngle",
    "GearBoxSumpTemp",
    "Generator_Speed",
    "Hub_Speed",
    "Power",
    "Wind_Speed",
    "GearTemp",
    "GeneratorTemp",
    "TurbineName"
]

def load_data_from_csv(file_path=None):
    """
    Load wind turbine data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file. If None, uses default path.
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    if file_path is None:
        # Use the cleaned data from the specific path provided
        file_path = 'C:\\Users\\ADMIN\\Desktop\\Project\\wind-turbine-mlflow-pipeline\\windt_cleaned_iqr.csv'
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            # Try to copy from the original project
            original_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                        'Wind', 'windt_cleaned_iqr.csv')
            if os.path.exists(original_path):
                import shutil
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy(original_path, file_path)
                logger.info(f"Copied data file from {original_path} to {file_path}")
            else:
                logger.error(f"Data file not found at {file_path} or {original_path}")
                raise FileNotFoundError(f"Data file not found at {file_path} or {original_path}")
        
        # Load the data
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # If the data doesn't have column names, add them
        if df.shape[1] == len(COLUMN_NAMES) and all(isinstance(col, int) or col.isdigit() for col in df.columns):
            df.columns = COLUMN_NAMES
            
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_data_to_mongodb(df=None, drop_existing=True):
    """
    Load wind turbine data to MongoDB.
    
    Args:
        df (pandas.DataFrame): Data to load. If None, loads from CSV.
        drop_existing (bool): Whether to drop existing collection.
        
    Returns:
        int: Number of documents inserted
    """
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Drop existing collection if requested
        if drop_existing and COLLECTION_NAME in db.list_collection_names():
            collection.drop()
            logger.info(f"Dropped existing collection {COLLECTION_NAME}")
        
        # Load data if not provided
        if df is None:
            df = load_data_from_csv()
        
        # Convert DataFrame to list of dictionaries and insert
        records = df.to_dict('records')
        result = collection.insert_many(records)
        
        logger.info(f"Inserted {len(result.inserted_ids)} documents to MongoDB")
        client.close()
        return len(result.inserted_ids)
    
    except Exception as e:
        logger.error(f"Error loading data to MongoDB: {str(e)}")
        raise

def load_data_from_mongodb():
    """
    Load wind turbine data from MongoDB.
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Check if collection exists
        if COLLECTION_NAME not in db.list_collection_names():
            logger.warning(f"Collection {COLLECTION_NAME} not found in MongoDB")
            # Try to load data from CSV to MongoDB
            load_data_to_mongodb()
        
        # Load data from MongoDB
        data = list(collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)
        
        logger.info(f"Loaded {len(df)} documents from MongoDB")
        client.close()
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from MongoDB: {str(e)}")
        raise

if __name__ == "__main__":
    # Load data from CSV and insert to MongoDB
    df = load_data_from_csv()
    load_data_to_mongodb(df)