# src/api/app.py

from flask import Flask, request, jsonify
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from pymongo import MongoClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.evaluate import load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = 'windturbine'
PREDICTIONS_COLLECTION = 'predictions'

# Create Flask app
app = Flask(__name__)

# Load model and preprocessors
model = None
preprocessors = None

def initialize():
    """
    Initialize the API by loading the model and preprocessors.
    """
    global model, preprocessors
    
    try:
        # Load model
        model = load_model()
        logger.info("Model loaded successfully")
        
        # Load preprocessors
        preprocessors_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models',
            'preprocessors.pkl'
        )
        if os.path.exists(preprocessors_path):
            preprocessors = joblib.load(preprocessors_path)
            logger.info("Preprocessors loaded successfully")
        else:
            logger.warning("Preprocessors not found. Some preprocessing steps may be skipped.")
            preprocessors = None
    
    except Exception as e:
        logger.error(f"Error initializing API: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    
    return jsonify({"status": "ok", "message": "API is healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    """
    try:
        # Get input data
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess input
        if preprocessors:
            # Handle categorical variables
            if 'encoders' in preprocessors:
                for col, encoder in preprocessors['encoders'].items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = encoder.transform(input_df[col])
                        except:
                            # If value not in encoder, use most common value
                            input_df[col] = 0
            
            # Scale features
            if 'scaler' in preprocessors:
                numeric_cols = input_df.select_dtypes(include=['number']).columns
                input_df[numeric_cols] = preprocessors['scaler'].transform(input_df[numeric_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Log prediction to MongoDB
        try:
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            collection = db[PREDICTIONS_COLLECTION]
            
            log_entry = {
                "input": data,
                "prediction": float(prediction),
                "timestamp": datetime.now()
            }
            collection.insert_one(log_entry)
            client.close()
        except Exception as e:
            logger.error(f"Error logging prediction to MongoDB: {str(e)}")
        
        # Return prediction
        return jsonify({
            "prediction": float(prediction),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint.
    """
    try:
        # Get input data
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input must be a list of data points"}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data)
        
        # Preprocess input
        if preprocessors:
            # Handle categorical variables
            if 'encoders' in preprocessors:
                for col, encoder in preprocessors['encoders'].items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = encoder.transform(input_df[col])
                        except:
                            # If value not in encoder, use most common value
                            input_df[col] = 0
            
            # Scale features
            if 'scaler' in preprocessors:
                numeric_cols = input_df.select_dtypes(include=['number']).columns
                input_df[numeric_cols] = preprocessors['scaler'].transform(input_df[numeric_cols])
        
        # Make predictions
        predictions = model.predict(input_df).tolist()
        
        # Return predictions
        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Model information endpoint.
    """
    try:
        # Get best model info
        best_model_info_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models',
            'best_model_info.txt'
        )
        
        info = {}
        if os.path.exists(best_model_info_path):
            with open(best_model_info_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        info[key.strip()] = value.strip()
        
        # Return model info
        return jsonify({
            "model_info": info,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Initialize on startup
initialize()

if __name__ == "__main__":
    # Run the app
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)