"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class VoiceFeatures(BaseModel):
    """Input schema for voice features (13 features after correlation filtering)"""
    
    MDVP_Fo_Hz: float = Field(..., description="Average vocal fundamental frequency", gt=0)
    MDVP_Fhi_Hz: float = Field(..., description="Maximum vocal fundamental frequency", gt=0)
    MDVP_Flo_Hz: float = Field(..., description="Minimum vocal fundamental frequency", gt=0)
    MDVP_Jitter_percent: float = Field(..., description="Jitter percentage", ge=0)
    MDVP_Jitter_Abs: float = Field(..., description="Absolute jitter", ge=0)
    MDVP_Shimmer: float = Field(..., description="Shimmer", ge=0)
    Shimmer_APQ11: float = Field(..., description="11-point amplitude perturbation quotient", ge=0)
    NHR: float = Field(..., description="Noise-to-harmonics ratio", ge=0)
    HNR: float = Field(..., description="Harmonics-to-noise ratio", ge=0)
    RPDE: float = Field(..., description="Recurrence period density entropy", ge=0)
    DFA: float = Field(..., description="Detrended fluctuation analysis", ge=0)
    spread1: float = Field(..., description="Nonlinear measure of fundamental frequency variation")
    spread2: float = Field(..., description="Nonlinear measure of fundamental frequency variation")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "MDVP_Fo_Hz": 119.992,
                "MDVP_Fhi_Hz": 157.302,
                "MDVP_Flo_Hz": 74.997,
                "MDVP_Jitter_percent": 0.00784,
                "MDVP_Jitter_Abs": 0.00007,
                "MDVP_Shimmer": 0.04374,
                "Shimmer_APQ11": 0.02971,
                "NHR": 0.02211,
                "HNR": 21.033,
                "RPDE": 0.414783,
                "DFA": 0.815285,
                "spread1": -4.813031,
                "spread2": 0.266482
            }
        }
    }


class BatchVoiceFeatures(BaseModel):
    """Input schema for batch predictions"""
    samples: List[VoiceFeatures]
    
    @field_validator('samples')
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100 samples")
        if len(v) == 0:
            raise ValueError("Batch must contain at least 1 sample")
        return v


class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    prediction: str = Field(..., description="'Parkinson's Disease' or 'Healthy'")
    prediction_label: int = Field(..., description="0 for Healthy, 1 for Parkinson's")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probability_parkinsons: float = Field(..., description="Probability of Parkinson's disease")
    probability_healthy: float = Field(..., description="Probability of being healthy")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Parkinson's Disease",
                "prediction_label": 1,
                "confidence": 0.92,
                "probability_parkinsons": 0.92,
                "probability_healthy": 0.08
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    total_samples: int


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    api_version: str
    features_expected: int


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    model_type: str
    features_count: int
    feature_names: List[str]
    training_accuracy: Optional[float]
    f1_score: Optional[float]
    roc_auc: Optional[float]
    version: str
    description: str


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    detail: str
    error_code: Optional[str] = None
