"""
Main FastAPI application
"""
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import logging
from typing import List

from app.config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    HOST, PORT, API_KEY, FEATURE_NAMES, NUM_FEATURES
)
from app.schemas import (
    VoiceFeatures, BatchVoiceFeatures,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse, ErrorResponse
)
from app.model_handler import model_handler
from app import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security (Optional - uncomment to enable)
# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# def verify_api_key(api_key: str = Depends(api_key_header)):
#     """Verify API key"""
#     if api_key != API_KEY:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid or missing API Key"
#         )
#     return api_key


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup"""
    logger.info("Starting Parkinson's Disease Detection API")
    logger.info(f"API Version: {API_VERSION}")
    
    # Load the model
    try:
        model_handler.load_model()
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
    
    # Verify model is loaded
    if not model_handler.is_model_loaded():
        logger.error("Model failed to load at startup!")
        raise Exception("Model not loaded. Cannot start API.")
    
    logger.info("Model loaded successfully")
    logger.info(f"API is ready to accept requests")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown"""
    logger.info("Shutting down Parkinson's Disease Detection API")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "Parkinson's Disease Detection API",
        "version": API_VERSION,
        "status": "running",
        "documentation": "/docs",
        "health_check": "/api/v1/health"
    }


# Health check endpoint
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check if the API and model are running properly"
)
async def health_check():
    """
    Health check endpoint to verify API status
    
    Returns:
        HealthResponse with status, model status, and API version
    """
    try:
        model_status = model_handler.is_model_loaded()
        
        return HealthResponse(
            status="healthy" if model_status else "unhealthy",
            model_loaded=model_status,
            api_version=API_VERSION,
            features_expected=NUM_FEATURES
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


# Model info endpoint
@app.get(
    "/api/v1/model_info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Model Information",
    description="Get information about the trained model"
)
async def model_info():
    """
    Get model information
    
    Returns:
        ModelInfoResponse with model details
    """
    try:
        return ModelInfoResponse(
            model_type="Machine Learning Classifier (Random Forest/SVM/XGBoost)",
            features_count=NUM_FEATURES,
            feature_names=FEATURE_NAMES,
            training_accuracy=0.95,  # Update with your actual training accuracy
            version="1.0.0",
            description="ML model trained on voice features for Parkinson's disease detection"
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


# Single prediction endpoint
@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Single Prediction",
    description="Predict Parkinson's disease from voice features",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input data"},
        500: {"description": "Prediction failed"}
    }
)
async def predict(
    features: VoiceFeatures
    # api_key: str = Depends(verify_api_key)  # Uncomment to enable API key
):
    """
    Make a single prediction for Parkinson's disease detection
    
    Args:
        features: Voice feature measurements (22 features)
        
    Returns:
        PredictionResponse with prediction result and confidence scores
    """
    try:
        logger.info("Received prediction request")
        
        # Convert Pydantic model to dictionary
        feature_dict = features.dict()
        
        # Make prediction
        prediction, prob_parkinsons, prob_healthy = model_handler.predict_single(feature_dict)
        
        # Prepare response
        result = PredictionResponse(
            prediction="Parkinson's Disease" if prediction == 1 else "Healthy",
            prediction_label=prediction,
            confidence=max(prob_parkinsons, prob_healthy),
            probability_parkinsons=prob_parkinsons,
            probability_healthy=prob_healthy
        )
        
        logger.info(f"Prediction completed: {result.prediction} (confidence: {result.confidence:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post(
    "/api/v1/predict_batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch Prediction",
    description="Predict Parkinson's disease for multiple samples",
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"description": "Invalid input data"},
        500: {"description": "Batch prediction failed"}
    }
)
async def predict_batch(
    batch_features: BatchVoiceFeatures
    # api_key: str = Depends(verify_api_key)  # Uncomment to enable API key
):
    """
    Make batch predictions for Parkinson's disease detection
    
    Args:
        batch_features: List of voice feature measurements
        
    Returns:
        BatchPredictionResponse with predictions for all samples
    """
    try:
        logger.info(f"Received batch prediction request for {len(batch_features.samples)} samples")
        
        predictions = []
        
        # Process each sample
        for idx, sample in enumerate(batch_features.samples):
            try:
                feature_dict = sample.dict()
                prediction, prob_parkinsons, prob_healthy = model_handler.predict_single(feature_dict)
                
                result = PredictionResponse(
                    prediction="Parkinson's Disease" if prediction == 1 else "Healthy",
                    prediction_label=prediction,
                    confidence=max(prob_parkinsons, prob_healthy),
                    probability_parkinsons=prob_parkinsons,
                    probability_healthy=prob_healthy
                )
                
                predictions.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process sample {idx + 1}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process sample {idx + 1}: {str(e)}"
                )
        
        logger.info(f"Batch prediction completed for {len(predictions)} samples")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Testing endpoint (for development only - remove in production)
@app.get(
    "/api/v1/test",
    tags=["Testing"],
    summary="Test Endpoint",
    description="Test endpoint with sample data (for development only)"
)
async def test_prediction():
    """
    Test the prediction with sample data
    
    Returns:
        Sample prediction result
    """
    sample_features = VoiceFeatures(
        MDVP_Fo_Hz=119.992,
        MDVP_Fhi_Hz=157.302,
        MDVP_Flo_Hz=74.997,
        MDVP_Jitter_percent=0.00784,
        MDVP_Jitter_Abs=0.00007,
        MDVP_Shimmer=0.04374,
        Shimmer_APQ11=0.02971,
        NHR=0.02211,
        HNR=21.033,
        RPDE=0.414783,
        DFA=0.815285,
        spread1=-4.813031,
        spread2=0.266482
    )
    
    return await predict(sample_features)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
