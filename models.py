from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any

class ScrapeRequest(BaseModel):
    g2_url: HttpUrl
    company_count: Optional[int] = 5  # Default to 5 companies

class Lead(BaseModel):
    company_name: Optional[str]
    website: Optional[str]
    hiring_intent: str
    score: int
    scoring_details: Optional[Dict[str, Any]] = None
    hiring_details: Optional[Dict[str, Any]] = None
