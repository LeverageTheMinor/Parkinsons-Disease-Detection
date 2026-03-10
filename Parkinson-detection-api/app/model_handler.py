"""
Model loading and prediction handler
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from app.config import MODEL_PATH, SCALER_PATH, FEATURE_NAMES, NUM_FEATURES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = FEATURE_NAMES
        self.num_features = NUM_FEATURES
    
    def is_model_loaded(self) -> bool:
        """Check if model and scaler are loaded"""
        return self.model is not None and self.scaler is not None
    
    def load_model(self) -> bool:
        """
        Load the trained model and scaler
        
        Returns:
            bool: True if loading successful
            
        Raises:
            FileNotFoundError: If model or scaler files not found
            Exception: If loading fails
        """
        try:
            # Check if files exist
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            if not Path(SCALER_PATH).exists():
                raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
            
            # Load model and scaler
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            
            logger.info(f"✓ Model loaded successfully from {MODEL_PATH}")
            logger.info(f"✓ Scaler loaded successfully from {SCALER_PATH}")
            logger.info(f"✓ Expected features: {self.num_features}")
            logger.info(f"✓ Feature names: {self.feature_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise
    
    def preprocess(self, features: List[float]) -> np.ndarray:
        """
        Preprocess features using the loaded scaler
        
        Args:
            features: List of 13 feature values
            
        Returns:
            np.ndarray: Scaled features ready for prediction
            
        Raises:
            ValueError: If feature count doesn't match or preprocessing fails
        """
        try:
            # Validate feature count
            if len(features) != self.num_features:
                raise ValueError(
                    f"Expected {self.num_features} features, got {len(features)}. "
                    f"Make sure you're providing all 13 features in correct order."
                )
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.transform(features_array)
            
            logger.debug(f"Features preprocessed successfully")
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise ValueError(f"Preprocessing failed: {str(e)}")
    
    def predict_single(self, voice_features: Dict[str, float]) -> Tuple[int, float, float]:
        """
        Make a single prediction
        
        Args:
            voice_features: Dictionary of voice features with exact feature names
            
        Returns:
            Tuple of (prediction_label, prob_parkinsons, prob_healthy)
            
        Raises:
            ValueError: If required features are missing or prediction fails
        """
        try:
            # Mapping from config feature names to Pydantic schema field names
            feature_name_to_field = {
                "MDVP:Fo(Hz)": "MDVP_Fo_Hz",
                "MDVP:Fhi(Hz)": "MDVP_Fhi_Hz",
                "MDVP:Flo(Hz)": "MDVP_Flo_Hz",
                "MDVP:Jitter(%)": "MDVP_Jitter_percent",
                "MDVP:Jitter(Abs)": "MDVP_Jitter_Abs",
                "MDVP:Shimmer": "MDVP_Shimmer",
                "Shimmer:APQ11": "Shimmer_APQ11",
                "NHR": "NHR",
                "HNR": "HNR",
                "RPDE": "RPDE",
                "DFA": "DFA",
                "spread1": "spread1",
                "spread2": "spread2",
            }
            
            # Extract features in the correct order
            features = []
            missing_features = []
            
            for name in self.feature_names:
                field_name = feature_name_to_field.get(name, name)
                value = voice_features.get(field_name)
                
                if value is None:
                    missing_features.append(name)
                else:
                    features.append(value)
            
            # Check for missing features
            if missing_features:
                raise ValueError(
                    f"Missing required features: {missing_features}. "
                    f"Please provide all 13 features with exact names."
                )
            
            logger.info(f"Processing features: {features}")
            
            # Preprocess features
            scaled_features = self.preprocess(features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            prob_healthy = float(probabilities[0])
            prob_parkinsons = float(probabilities[1])
            
            logger.info(
                f"Prediction: {prediction}, "
                f"Probabilities -> Healthy: {prob_healthy:.4f}, "
                f"Parkinson's: {prob_parkinsons:.4f}"
            )
            
            return int(prediction), prob_parkinsons, prob_healthy
            
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise ValueError(f"Single prediction failed: {str(e)}")
    
    def predict_batch(self, voice_features_list: List[Dict[str, float]]) -> List[Tuple[int, float, float]]:
        """
        Make batch predictions
        
        Args:
            voice_features_list: List of dictionaries of voice features
            
        Returns:
            List of tuples (prediction_label, prob_parkinsons, prob_healthy)
            
        Raises:
            ValueError: If batch prediction fails
        """
        try:
            results = []
            
            for i, features in enumerate(voice_features_list):
                try:
                    # Predict each sample
                    prediction, prob_parkinsons, prob_healthy = self.predict_single(features)
                    results.append((prediction, prob_parkinsons, prob_healthy))
                    
                except Exception as e:
                    logger.error(f"Failed to predict sample {i}: {str(e)}")
                    # Return default values for failed predictions
                    results.append((0, 0.0, 1.0))
            
            logger.info(f"Batch prediction completed for {len(results)} samples")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise ValueError(f"Batch prediction failed: {str(e)}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of required feature names
        
        Returns:
            List of feature names
        """
        return self.feature_names
    
    def validate_features(self, voice_features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate if all required features are present
        
        Args:
            voice_features: Dictionary of voice features
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = []
        
        for name in self.feature_names:
            if name not in voice_features:
                missing.append(name)
        
        is_valid = len(missing) == 0
        
        return is_valid, missing


# Global model handler instance
model_handler = ModelHandler()
