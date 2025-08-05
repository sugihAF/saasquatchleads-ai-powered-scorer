# main.py
import os
from typing import List, Dict

from fastapi import FastAPI
from dotenv import load_dotenv
from urls import router

load_dotenv()
app = FastAPI(
    title="🎯 SaaS Lead Scorer API",
    description="AI-powered lead scoring system with ML capabilities for SaaS companies",
    version="2.0.0"
)

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "🎯 SaaS Lead Scorer API is running",
        "status": "healthy",
        "version": "2.0.0",
        "features": ["lead_scoring", "hiring_intent_analysis", "ml_predictions"]
    }

@app.get("/health")
def health_check():
    """Alternative health check endpoint"""
    return {"status": "healthy"}

# Include main router
app.include_router(router)

# Include ML router
try:
    from ml_api import ml_router
    app.include_router(ml_router, prefix="/api/v1")
    print("🤖 ML API endpoints loaded successfully")
except ImportError as e:
    print(f"⚠️  ML API not available: {e}")
except Exception as e:
    print(f"❌ Error loading ML API: {e}")