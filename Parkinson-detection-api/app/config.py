"""
Configuration settings for the API
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "parkinsons_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# API settings
API_TITLE = "Parkinson's Disease Detection API"
API_DESCRIPTION = """
Machine Learning-based API for detecting Parkinson's disease from voice features.

## Features:
* Real-time prediction from 13 voice features (after correlation filtering)
* High accuracy ML model (Random Forest - 92.31% accuracy)
* RESTful API design
* Automatic API documentation

## Endpoints:
* `/api/v1/predict` - Single prediction
* `/api/v1/predict-batch` - Batch predictions
* `/api/v1/health` - Health check
* `/api/v1/model-info` - Model information
"""
API_VERSION = "1.0.0"

# Server settings
HOST = "127.0.0.1"
PORT = 8000

# Security (Change in production!)
API_KEY = os.getenv("API_KEY", "parkinsons-api-key-2026")

# Model settings - 13 FEATURES (after removing highly correlated features)
# Removed: MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer(dB), 
#          Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, PPE
FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:Shimmer",
    "Shimmer:APQ11",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2"
]

NUM_FEATURES = 13

# Model metadata
MODEL_METADATA = {
    "model_type": "Random Forest Classifier",
    "accuracy": 0.9231,
    "f1_score": 0.9474,
    "roc_auc": 0.9845,
    "training_samples": 156,
    "test_samples": 39,
    "features_count": NUM_FEATURES,
    "dataset": "Parkinson's Disease Voice Dataset",
    "preprocessing": "StandardScaler + SMOTE balancing"
}
