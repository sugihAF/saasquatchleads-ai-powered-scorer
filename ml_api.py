"""
ML Training and Management API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
import json
import os

# Import ML components
try:
    from ml_scoring import ml_model, create_mock_training_data
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

ml_router = APIRouter()

class FeedbackRequest(BaseModel):
    lead_data: dict
    actual_outcome: int  # 1 for converted, 0 for not converted

class TrainingRequest(BaseModel):
    use_mock_data: bool = True
    num_samples: int = 200

class ModelStatus(BaseModel):
    ml_available: bool
    is_trained: bool
    model_version: str
    training_samples: int
    last_trained: str = None

@ml_router.get("/ml/status", response_model=ModelStatus)
async def get_ml_status():
    """Get ML model status and information"""
    if not ML_AVAILABLE:
        return ModelStatus(
            ml_available=False,
            is_trained=False,
            model_version="N/A",
            training_samples=0
        )
    
    # Count feedback samples
    feedback_file = "data/feedback_data.json"
    training_samples = 0
    last_trained = None
    
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
                training_samples = len(feedback_data)
        except:
            training_samples = 0
    
    # Check if model exists
    last_trained = "Never trained"
    if os.path.exists(ml_model.model_path):
        try:
            import pickle
            with open(ml_model.model_path, 'rb') as f:
                model_data = pickle.load(f)
                last_trained = model_data.get('trained_at', 'Unknown')
        except:
            last_trained = 'Unknown'
    
    return ModelStatus(
        ml_available=True,
        is_trained=ml_model.is_trained,
        model_version=ml_model.model_version,
        training_samples=training_samples,
        last_trained=last_trained
    )

@ml_router.post("/ml/train")
async def train_ml_model(request: TrainingRequest):
    """Train the ML model with data"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML functionality not available")
    
    try:
        if request.use_mock_data:
            # Generate mock training data
            leads_data, outcomes = create_mock_training_data(request.num_samples)
            metrics = ml_model.train_model(leads_data, outcomes)
        else:
            # Use real feedback data
            feedback_file = "data/feedback_data.json"
            if not os.path.exists(feedback_file):
                raise HTTPException(status_code=404, detail="No feedback data available for training")
            
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            if len(feedback_data) < 20:
                raise HTTPException(status_code=400, detail=f"Not enough training data. Need at least 20 samples, got {len(feedback_data)}")
            
            leads_data = [entry['lead_data'] for entry in feedback_data]
            outcomes = [entry['outcome'] for entry in feedback_data]
            metrics = ml_model.train_model(leads_data, outcomes)
        
        return {
            "success": True,
            "message": f"Model trained successfully with {len(leads_data) if 'leads_data' in locals() else request.num_samples} samples",
            "metrics": metrics,
            "model_version": ml_model.model_version
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@ml_router.post("/ml/feedback")
async def add_feedback(request: FeedbackRequest):
    """Add feedback data for model improvement"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML functionality not available")
    
    try:
        result = ml_model.retrain_with_feedback(request.lead_data, request.actual_outcome)
        
        response = {
            "success": True,
            "message": "Feedback recorded successfully"
        }
        
        if result:
            response["retrained"] = True
            response["metrics"] = result
        else:
            response["retrained"] = False
            response["message"] += " (model will retrain when enough data is collected)"
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@ml_router.get("/ml/predictions/{company_name}")
async def get_ml_prediction(company_name: str):
    """Get ML prediction for a specific company (demo endpoint)"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML functionality not available")
    
    if not ml_model.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Create mock lead data for demo
    mock_lead = {
        'company_name': company_name,
        'website': f'https://{company_name.lower().replace(" ", "")}.com',
        'hiring_intent': '✅ Yes (Medium confidence)',
        'score': 45,
        'hiring_details': {
            'has_hiring_intent': True,
            'confidence_level': 'medium',
            'hiring_indicators': ['careers', 'jobs', 'hiring'],
            'careers_page_exists': True,
            'open_positions': ['Sales Manager', 'Marketing Specialist'],
            'urgency_signals': ['growing team']
        },
        'scoring_details': {
            'total_score': 45,
            'base_score': 40,
            'bonus_score': 5,
            'company_keywords': ['software'],
            'risk_factors': []
        }
    }
    
    try:
        prediction = ml_model.predict(mock_lead)
        return {
            "company_name": company_name,
            "ml_score": prediction.ml_score,
            "confidence": prediction.confidence,
            "feature_importance": prediction.feature_importance,
            "explanation": prediction.prediction_explanation,
            "model_version": prediction.model_version
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
