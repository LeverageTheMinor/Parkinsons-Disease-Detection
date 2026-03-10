"""
API unit tests
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["status"] == "running"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/api/v1/model_info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "features_count" in data
        assert data["features_count"] == 22
        assert len(data["feature_names"]) == 22


class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing"""
        return {
            "MDVP_Fo_Hz": 119.992,
            "MDVP_Fhi_Hz": 157.302,
            "MDVP_Flo_Hz": 74.997,
            "MDVP_Jitter_percent": 0.00784,
            "MDVP_Jitter_Abs": 0.00007,
            "MDVP_RAP": 0.00370,
            "MDVP_PPQ": 0.00554,
            "Jitter_DDP": 0.01109,
            "MDVP_Shimmer": 0.04374,
            "MDVP_Shimmer_dB": 0.426,
            "Shimmer_APQ3": 0.02182,
            "Shimmer_APQ5": 0.03130,
            "MDVP_APQ": 0.02971,
            "Shimmer_DDA": 0.06545,
            "NHR": 0.02211,
            "HNR": 21.033,
            "RPDE": 0.414783,
            "DFA": 0.815285,
            "spread1": -4.813031,
            "spread2": 0.266482,
            "D2": 2.301442,
            "PPE": 0.284654
        }
    
    def test_single_prediction(self, sample_features):
        """Test single prediction endpoint"""
        response = client.post("/api/v1/predict", json=sample_features)
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data
        assert "probability_parkinsons" in data
        assert "probability_healthy" in data
        
        # Check data types
        assert isinstance(data["prediction"], str)
        assert isinstance(data["prediction_label"], int)
        assert isinstance(data["confidence"], float)
        
        # Check value ranges
        assert data["prediction_label"] in [0, 1]
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["probability_parkinsons"] <= 1
        assert 0 <= data["probability_healthy"] <= 1
    
    def test_prediction_with_missing_feature(self, sample_features):
        """Test prediction with missing feature"""
        incomplete_features = sample_features.copy()
        del incomplete_features["MDVP_Fo_Hz"]
        
        response = client.post("/api/v1/predict", json=incomplete_features)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_with_invalid_value(self, sample_features):
        """Test prediction with invalid value (negative where positive expected)"""
        invalid_features = sample_features.copy()
        invalid_features["MDVP_Fo_Hz"] = -100  # Should be positive
        
        response = client.post("/api/v1/predict", json=invalid_features)
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction(self, sample_features):
        """Test batch prediction endpoint"""
        batch_data = {
            "samples": [sample_features, sample_features]
        }
        
        response = client.post("/api/v1/predict_batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "total_samples" in data
        assert data["total_samples"] == 2
        assert len(data["predictions"]) == 2
        
        # Check first prediction structure
        first_pred = data["predictions"][0]
        assert "prediction" in first_pred
        assert "confidence" in first_pred
    
    def test_batch_prediction_empty(self):
        """Test batch prediction with empty samples"""
        batch_data = {
            "samples": []
        }
        
        response = client.post("/api/v1/predict_batch", json=batch_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_exceeds_limit(self, sample_features):
        """Test batch prediction exceeding maximum batch size"""
        batch_data = {
            "samples": [sample_features] * 101  # Exceeds 100 limit
        }
        
        response = client.post("/api/v1/predict_batch", json=batch_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_test_endpoint(self):
        """Test the test endpoint"""
        response = client.get("/api/v1/test")
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "confidence" in data


class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_openapi_json(self):
        """Test OpenAPI JSON schema"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_page(self):
        """Test Swagger UI docs page"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_page(self):
        """Test ReDoc page"""
        response = client.get("/redoc")
        assert response.status_code == 200


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
