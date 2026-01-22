"""
Production FastAPI REST API for XAI Agents
Provides endpoints for model prediction and explanation generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import uvicorn
import logging
import time
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from models.baseline_models import BaselineModel
from xai.xai_methods import SHAPExplainer, LIMEExplainer, XAIMethodSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="XAI Agents API",
    description="Production API for Explainable AI Financial Decision Making",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model store (in production, use proper model registry)
MODEL_STORE = {}
EXPLAINER_CACHE = {}


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for prediction"""

    features: List[float] = Field(..., description="Input features for prediction")
    model_name: str = Field(default="default", description="Model name to use")

    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, -0.3, 1.2, 0.8, -0.5, 0.2, 0.9, -0.1, 0.4, 0.7],
                "model_name": "default",
            }
        }


class ExplanationRequest(BaseModel):
    """Request model for explanation"""

    features: List[float] = Field(..., description="Input features to explain")
    model_name: str = Field(default="default", description="Model name to use")
    method: str = Field(default="SHAP", description="XAI method: SHAP, LIME, IG")
    num_samples: Optional[int] = Field(
        default=100, description="Number of samples for explanation"
    )

    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, -0.3, 1.2, 0.8, -0.5, 0.2, 0.9, -0.1, 0.4, 0.7],
                "model_name": "default",
                "method": "SHAP",
                "num_samples": 100,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""

    features_batch: List[List[float]] = Field(
        ..., description="Batch of input features"
    )
    model_name: str = Field(default="default", description="Model name to use")


class PredictionResponse(BaseModel):
    """Response model for prediction"""

    prediction: float
    prediction_class: int
    confidence: float
    model_name: str
    timestamp: str
    latency_ms: float


class ExplanationResponse(BaseModel):
    """Response model for explanation"""

    prediction: float
    prediction_class: int
    explanation: Dict[str, Any]
    method: str
    model_name: str
    timestamp: str
    computation_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    models_loaded: int
    version: str


class ModelInfo(BaseModel):
    """Model information"""

    name: str
    type: str
    loaded: bool
    features: int


# Startup event - Load default model
@app.on_event("startup")
async def startup_event():
    """Load default model on startup"""
    logger.info("Starting XAI Agents API...")

    # Create a simple default model
    try:
        np.random.seed(42)
        X_train = np.random.randn(1000, 10)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = BaselineModel(model_type="logistic", random_state=42)
        model.train(X_train, y_train)

        MODEL_STORE["default"] = {
            "model": model,
            "type": "logistic",
            "features": 10,
            "background_data": X_train,
        }

        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")


# Health Check Endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint
    Returns system status and loaded models count
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=len(MODEL_STORE),
        version="1.0.0",
    )


# Model Management Endpoints
@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """
    List all available models
    """
    models = []
    for name, info in MODEL_STORE.items():
        models.append(
            ModelInfo(
                name=name, type=info["type"], loaded=True, features=info["features"]
            )
        )
    return models


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str):
    """
    Get information about a specific model
    """
    if model_name not in MODEL_STORE:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    info = MODEL_STORE[model_name]
    return ModelInfo(
        name=model_name, type=info["type"], loaded=True, features=info["features"]
    )


# Prediction Endpoints
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction for given features
    Returns prediction probability and class
    """
    start_time = time.time()

    # Validate model exists
    if request.model_name not in MODEL_STORE:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_name}' not found"
        )

    # Validate features
    model_info = MODEL_STORE[request.model_name]
    if len(request.features) != model_info["features"]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {model_info['features']} features, got {len(request.features)}",
        )

    # Make prediction
    try:
        model = model_info["model"]
        X = np.array(request.features).reshape(1, -1)

        prob = model.predict_proba(X)[0]
        pred_class = int(prob > 0.5)

        latency = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=float(prob),
            prediction_class=pred_class,
            confidence=float(max(prob, 1 - prob)),
            model_name=request.model_name,
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for a batch of feature sets
    """
    if request.model_name not in MODEL_STORE:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_name}' not found"
        )

    model_info = MODEL_STORE[request.model_name]
    model = model_info["model"]

    try:
        X = np.array(request.features_batch)
        probs = model.predict_proba(X)
        pred_classes = (probs > 0.5).astype(int)

        return {
            "predictions": probs.tolist(),
            "prediction_classes": pred_classes.tolist(),
            "count": len(probs),
            "model_name": request.model_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


# Explanation Endpoints
@app.post("/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_prediction(request: ExplanationRequest):
    """
    Generate explanation for a prediction
    Supports SHAP, LIME, and other XAI methods
    """
    start_time = time.time()

    # Validate model exists
    if request.model_name not in MODEL_STORE:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_name}' not found"
        )

    model_info = MODEL_STORE[request.model_name]

    # Validate features
    if len(request.features) != model_info["features"]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {model_info['features']} features, got {len(request.features)}",
        )

    try:
        model = model_info["model"]
        X = np.array(request.features).reshape(1, -1)

        # Make prediction
        prob = model.predict_proba(X)[0]
        pred_class = int(prob > 0.5)

        # Generate explanation
        if request.method.upper() == "SHAP":
            # Initialize or get cached SHAP explainer
            cache_key = f"{request.model_name}_shap"
            if cache_key not in EXPLAINER_CACHE:
                explainer = SHAPExplainer(
                    model, background_data=model_info["background_data"]
                )
                EXPLAINER_CACHE[cache_key] = explainer
            else:
                explainer = EXPLAINER_CACHE[cache_key]

            explanation_result = explainer.explain(X, nsamples=request.num_samples)

        elif request.method.upper() == "LIME":
            cache_key = f"{request.model_name}_lime"
            if cache_key not in EXPLAINER_CACHE:
                feature_names = [f"feature_{i}" for i in range(model_info["features"])]
                explainer = LIMEExplainer(model, feature_names=feature_names)
                EXPLAINER_CACHE[cache_key] = explainer
            else:
                explainer = EXPLAINER_CACHE[cache_key]

            explanation_result = explainer.explain(X, num_samples=request.num_samples)

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported XAI method: {request.method}"
            )

        computation_time = (time.time() - start_time) * 1000

        # Convert numpy arrays to lists for JSON serialization
        explanation_data = {}
        for key, value in explanation_result.items():
            if isinstance(value, np.ndarray):
                explanation_data[key] = value.tolist()
            else:
                explanation_data[key] = value

        return ExplanationResponse(
            prediction=float(prob),
            prediction_class=pred_class,
            explanation=explanation_data,
            method=request.method,
            model_name=request.model_name,
            timestamp=datetime.utcnow().isoformat(),
            computation_time_ms=computation_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/explain/methods", tags=["Explanation"])
async def list_xai_methods():
    """
    List available XAI methods with their characteristics
    """
    return {"methods": XAIMethodSelector.get_method_comparison(), "default": "SHAP"}


@app.post("/explain/recommend", tags=["Explanation"])
async def recommend_xai_method(
    model_name: str, dataset_size: int = 1000, time_budget_ms: float = 1000
):
    """
    Recommend best XAI method based on context
    """
    if model_name not in MODEL_STORE:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    model_info = MODEL_STORE[model_name]

    recommended = XAIMethodSelector.recommend_method(
        model_type=model_info["type"],
        dataset_size=dataset_size,
        feature_count=model_info["features"],
        time_budget_ms=time_budget_ms,
    )

    return {
        "recommended_method": recommended,
        "model_type": model_info["type"],
        "feature_count": model_info["features"],
        "considerations": XAIMethodSelector.get_method_comparison().get(
            recommended, {}
        ),
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
