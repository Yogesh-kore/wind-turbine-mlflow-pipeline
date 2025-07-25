#!/usr/bin/env python
# scripts/deploy_api.py

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'deploy_api_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Deploy Wind Turbine API')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind the API server')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to bind the API server')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of Gunicorn workers')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Gunicorn worker timeout in seconds')
    parser.add_argument('--mlflow-uri', type=str, default='http://localhost:5000',
                        help='MLflow tracking URI')
    parser.add_argument('--mongodb-uri', type=str, default='mongodb://localhost:27017/',
                        help='MongoDB URI')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (not recommended for production)')
    
    return parser.parse_args()

def create_directories():
    """
    Create necessary directories.
    """
    directories = ['logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def deploy_api():
    """
    Deploy the API server.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create directories
    create_directories()
    
    try:
        # Set environment variables
        os.environ['MLFLOW_TRACKING_URI'] = args.mlflow_uri
        os.environ['MONGODB_URI'] = args.mongodb_uri
        
        logger.info(f"MLflow tracking URI: {args.mlflow_uri}")
        logger.info(f"MongoDB URI: {args.mongodb_uri}")
        
        # Check if models directory exists
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            logger.warning("Models directory is empty. The API may not work correctly.")
        
        # Run in debug mode or with Gunicorn
        if args.debug:
            logger.info(f"Starting API server in debug mode on {args.host}:{args.port}")
            from src.api.app import app
            app.run(host=args.host, port=args.port, debug=True)
        else:
            logger.info(f"Starting API server with Gunicorn on {args.host}:{args.port}")
            
            # Build Gunicorn command
            cmd = [
                'gunicorn',
                '--bind', f"{args.host}:{args.port}",
                '--workers', str(args.workers),
                '--timeout', str(args.timeout),
                '--log-level', 'info',
                'src.api.app:app'
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run Gunicorn
            process = subprocess.Popen(cmd)
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"Gunicorn exited with code {process.returncode}")
                return process.returncode
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error deploying API: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(deploy_api())