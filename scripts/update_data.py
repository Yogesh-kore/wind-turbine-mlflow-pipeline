#!/usr/bin/env python
# scripts/update_data.py

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_data_from_csv, load_data_to_mongodb
from src.utils.mongo_utils import get_dataframe, get_collection_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'update_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Update Wind Turbine Data')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the new data CSV file')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing data instead of replacing')
    parser.add_argument('--output-path', type=str, default='C:\\Users\\ADMIN\\Desktop\\Project\\wind-turbine-mlflow-pipeline\\data\\windt_updated.csv',
                        help='Path to save the updated data')
    parser.add_argument('--collection-name', type=str, default='turbine_data',
                        help='MongoDB collection name')
    
    return parser.parse_args()

def create_directories():
    """
    Create necessary directories.
    """
    directories = ['logs', 'data']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def update_data():
    """
    Update data from external sources.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create directories
    create_directories()
    
    try:
        # Load new data
        logger.info(f"Loading new data from {args.data_path}")
        new_data = load_data_from_csv(args.data_path)
        
        if new_data is None or new_data.empty:
            logger.error("Failed to load new data. Exiting.")
            return 1
        
        logger.info(f"Loaded new data with shape: {new_data.shape}")
        
        # Get existing data from MongoDB if appending
        if args.append:
            logger.info(f"Getting existing data from MongoDB collection: {args.collection_name}")
            existing_data = get_dataframe(args.collection_name)
            
            if existing_data.empty:
                logger.warning("No existing data found in MongoDB. Will use only new data.")
                updated_data = new_data
            else:
                logger.info(f"Existing data shape: {existing_data.shape}")
                
                # Drop MongoDB _id column if present
                if '_id' in existing_data.columns:
                    existing_data = existing_data.drop('_id', axis=1)
                
                # Concatenate existing and new data
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                
                # Drop duplicates if any
                original_shape = updated_data.shape
                updated_data = updated_data.drop_duplicates()
                if updated_data.shape[0] < original_shape[0]:
                    logger.info(f"Dropped {original_shape[0] - updated_data.shape[0]} duplicate rows")
        else:
            # Use only new data
            updated_data = new_data
        
        logger.info(f"Updated data shape: {updated_data.shape}")
        
        # Save updated data to CSV
        updated_data.to_csv(args.output_path, index=False)
        logger.info(f"Saved updated data to {args.output_path}")
        
        # Load updated data to MongoDB
        logger.info(f"Loading updated data to MongoDB collection: {args.collection_name}")
        load_data_to_mongodb(updated_data, collection_name=args.collection_name, drop_existing=not args.append)
        
        # Get collection stats
        stats = get_collection_stats(args.collection_name)
        logger.info(f"MongoDB collection stats: {stats}")
        
        logger.info("Data update completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(update_data())