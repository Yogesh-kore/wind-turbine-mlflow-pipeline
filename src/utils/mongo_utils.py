# src/utils/mongo_utils.py

import os
import logging
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = 'windturbine'
DATA_COLLECTION = 'turbine_data'
PREDICTIONS_COLLECTION = 'predictions'
EXPERIMENTS_COLLECTION = 'mlflow_experiments'

def get_mongo_client():
    """
    Get a MongoDB client.
    
    Returns:
        MongoClient: MongoDB client
    """
    try:
        client = MongoClient(MONGO_URI)
        return client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def insert_dataframe(df, collection_name, drop_existing=False):
    """
    Insert a DataFrame into MongoDB.
    
    Args:
        df: DataFrame to insert
        collection_name: Name of the collection
        drop_existing: Whether to drop the existing collection
        
    Returns:
        int: Number of documents inserted
    """
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        
        # Drop existing collection if requested
        if drop_existing and collection_name in db.list_collection_names():
            db[collection_name].drop()
            logger.info(f"Dropped existing collection: {collection_name}")
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert records
        result = db[collection_name].insert_many(records)
        client.close()
        
        logger.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}")
        return len(result.inserted_ids)
    
    except Exception as e:
        logger.error(f"Error inserting data into MongoDB: {str(e)}")
        return 0

def get_dataframe(collection_name, query=None, limit=None):
    """
    Get a DataFrame from MongoDB.
    
    Args:
        collection_name: Name of the collection
        query: MongoDB query
        limit: Maximum number of documents to return
        
    Returns:
        DataFrame: DataFrame containing the data
    """
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            logger.warning(f"Collection {collection_name} does not exist")
            client.close()
            return pd.DataFrame()
        
        # Get data
        collection = db[collection_name]
        if query is None:
            query = {}
        
        if limit:
            cursor = collection.find(query).limit(limit)
        else:
            cursor = collection.find(query)
        
        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))
        client.close()
        
        if df.empty:
            logger.warning(f"No data found in collection {collection_name}")
        else:
            logger.info(f"Retrieved {len(df)} documents from {collection_name}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting data from MongoDB: {str(e)}")
        return pd.DataFrame()

def log_prediction(input_data, prediction, model_name=None):
    """
    Log a prediction to MongoDB.
    
    Args:
        input_data: Input data for the prediction
        prediction: Prediction result
        model_name: Name of the model used
        
    Returns:
        str: MongoDB document ID
    """
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        collection = db[PREDICTIONS_COLLECTION]
        
        # Create log entry
        log_entry = {
            "input": input_data,
            "prediction": prediction,
            "model_name": model_name,
            "timestamp": datetime.now()
        }
        
        # Insert log entry
        result = collection.insert_one(log_entry)
        client.close()
        
        logger.info(f"Prediction logged to MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    except Exception as e:
        logger.error(f"Error logging prediction to MongoDB: {str(e)}")
        return None

def get_collection_stats(collection_name):
    """
    Get statistics for a MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        dict: Collection statistics
    """
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            logger.warning(f"Collection {collection_name} does not exist")
            client.close()
            return {}
        
        # Get collection stats
        stats = db.command("collStats", collection_name)
        client.close()
        
        # Extract relevant stats
        relevant_stats = {
            "count": stats.get("count", 0),
            "size": stats.get("size", 0),
            "avg_obj_size": stats.get("avgObjSize", 0),
            "storage_size": stats.get("storageSize", 0),
            "index_count": len(stats.get("indexSizes", {})),
            "index_size": stats.get("totalIndexSize", 0)
        }
        
        logger.info(f"Retrieved stats for collection {collection_name}")
        return relevant_stats
    
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        return {}