from fastapi import APIRouter, HTTPException
from typing import List
from models import ScrapeRequest, Lead
from core import scrape_g2_companies, get_hiring_intent, calculate_lead_score

router = APIRouter()

@router.post("/score-leads", response_model=List[Lead])
async def score_leads_endpoint(request: ScrapeRequest):
    """
    Accepts a G2 category URL, scrapes it for companies, analyzes each for
    hiring intent, scores them, and returns a prioritized list of leads.
    """
    company_count = request.company_count if request.company_count else 5
    companies = await scrape_g2_companies(str(request.g2_url), company_count)
    if not companies:
        raise HTTPException(status_code=404, detail="Could not extract any companies from the provided G2 URL.")

    enriched_leads = []
    for company in companies:
        website_url = company.get('website_url')
        company_name = company.get('company_name', '')
        if isinstance(website_url, str):
            # Get detailed hiring intent analysis
            hiring_intent_details = await get_hiring_intent(website_url)
            
            # Calculate detailed score with all factors including hiring details
            score_details = calculate_lead_score(
                has_hiring_intent=hiring_intent_details['has_hiring_intent'],
                company_name=company_name,
                website_url=website_url,
                hiring_indicators=hiring_intent_details.get('hiring_indicators', []),
                careers_page_exists=hiring_intent_details.get('careers_page_exists', False),
                hiring_details=hiring_intent_details  # Pass full hiring details for ML
            )
            
            # Format hiring intent display
            hiring_status = '✅ Yes' if hiring_intent_details['has_hiring_intent'] else '❌ No'
            confidence = hiring_intent_details.get('confidence_level', 'low').title()
            hiring_display = f"{hiring_status} ({confidence} confidence)"
            
            enriched_leads.append({
                'company_name': company_name,
                'website': website_url,
                'hiring_intent': hiring_display,
                'score': int(score_details['total_score']),  # Ensure it's a Python int
                # Additional enriched data for the dashboard
                'scoring_details': score_details,
                'hiring_details': hiring_intent_details
            })
    
    sorted_leads = sorted(enriched_leads, key=lambda x: x['score'], reverse=True)
    return sorted_leads
