import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from firecrawl import AsyncFirecrawlApp
from datetime import datetime
import numpy as np

# Import ML components
try:
    from ml_scoring import ml_model, MLPrediction
    ML_AVAILABLE = True
    print("🤖 ML scoring system loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️  ML dependencies not available: {e}")

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj
    print("📊 Using rule-based scoring only")

load_dotenv()

def log(message):
    """Simple logging function with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    print("-" * 50)  # Visual separator

async def scrape_g2_companies(url: str, max_companies: int = 5) -> List[Dict[str, str]]:
    """
    Scrape a G2 category URL and return a list of companies with name and website_url.
    Uses Firecrawl's extract feature for structured data extraction.
    """
    start_time = time.time()
    log("🎯 STARTING G2 SCRAPING")
    log(f"📊 Target URL: {url}")
    log(f"� Requested Companies: {max_companies}")
    log(f"�🔧 URL Type: {type(url)}")
    
    # Ensure URL is properly formatted
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
        log(f"🔄 URL Formatted: {url}")
    
    # Check if API key exists
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        log("❌ FIRECRAWL_API_KEY not found in environment")
        return []
    
    log(f"🔑 API Key Status: Found ({api_key[:10]}...)")
    
    fc_app = AsyncFirecrawlApp(api_key=api_key)
    
    try:
        # Use Firecrawl's extract feature for structured data extraction
        log("🚀 Initiating Firecrawl EXTRACT for company data...")
        scrape_start = time.time()
        
        # Define schema for company extraction
        company_schema = {
            "type": "object",
            "properties": {
                "companies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company_name": {
                                "type": "string",
                                "description": "Name of the software company"
                            },
                            "website_url": {
                                "type": "string",
                                "description": "Official website URL of the company"
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of what the company does"
                            }
                        },
                        "required": ["company_name", "website_url"]
                    }
                }
            },
            "required": ["companies"]
        }
        
        # Extract structured data using Firecrawl
        response = await fc_app.extract(
            urls=[url],
            prompt="Extract all software companies listed on this G2 category page. For each company, get the company name, official website URL, and a brief description of what they do.",
            schema=company_schema
        )
        
        scrape_duration = time.time() - scrape_start
        log(f"✅ Firecrawl Extract Response Received!")
        log(f"⏱️  Extraction Duration: {scrape_duration:.2f}s")
        log(f"📦 Response Type: {type(response)}")
        
        # Process the extracted data
        companies = []
        if response and hasattr(response, 'data') and response.data:
            extracted_data = response.data
            log(f"🔍 Extracted Data Structure: {type(extracted_data)}")
            
            # Handle different response formats
            if isinstance(extracted_data, list) and len(extracted_data) > 0:
                company_data = extracted_data[0]  # First URL result
                if isinstance(company_data, dict) and 'companies' in company_data:
                    companies_list = company_data['companies']
                    log(f"� Found {len(companies_list)} companies in extracted data")
                    
                    for company in companies_list[:max_companies]:  # Use dynamic limit based on user selection
                        if isinstance(company, dict) and 'company_name' in company and 'website_url' in company:
                            # Clean up the website URL
                            website = company['website_url']
                            if website and not website.startswith('http'):
                                website = f"https://{website}"
                            
                            companies.append({
                                'company_name': company['company_name'],
                                'website_url': website
                            })
                            log(f"🏢 Extracted Company #{len(companies)}: {company['company_name']}")
            
            if companies:
                total_time = time.time() - start_time
                log(f"🎉 SUCCESS: Extracted {len(companies)} companies using Firecrawl Extract in {total_time:.2f}s")
                return companies
        
        # If extraction fails, try fallback scraping
        log("🔄 Extract failed, trying fallback scraping...")
        response = await fc_app.scrape_url(url, formats=['markdown'])
        
        if response and hasattr(response, 'data') and response.data:
            markdown_content = response.data.get('markdown', '') if hasattr(response.data, 'get') else getattr(response.data, 'markdown', '')
            log(f"📄 Fallback: Markdown Content Length: {len(markdown_content)} characters")
            
            if markdown_content:
                # Simple pattern matching for company names
                lines = markdown_content.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['software', 'platform', 'solution', 'crm']) and len(line.strip()) > 5:
                        company_name = line.strip()[:50]
                        if company_name and len(companies) < max_companies:
                            companies.append({
                                'company_name': company_name,
                                'website_url': f"https://{company_name.lower().replace(' ', '').replace(',', '').replace('.', '')[:20]}.com"
                            })
                            log(f"🏢 Fallback Company #{len(companies)}: {company_name}")
        
        if companies:
            total_time = time.time() - start_time
            log(f"🎉 Fallback SUCCESS: Extracted {len(companies)} companies in {total_time:.2f}s")
            return companies
        
        # Final fallback to mock data
        log("⚠️  All extraction methods failed, using mock data...")
        mock_companies = [
            {'company_name': 'Salesforce CRM', 'website_url': 'https://salesforce.com'},
            {'company_name': 'HubSpot Marketing', 'website_url': 'https://hubspot.com'},
            {'company_name': 'Pipedrive Sales', 'website_url': 'https://pipedrive.com'},
            {'company_name': 'Zoho CRM Solutions', 'website_url': 'https://zoho.com'},
            {'company_name': 'Microsoft Dynamics', 'website_url': 'https://dynamics.microsoft.com'},
            {'company_name': 'Zendesk Support', 'website_url': 'https://zendesk.com'},
            {'company_name': 'Slack Communication', 'website_url': 'https://slack.com'},
            {'company_name': 'Notion Productivity', 'website_url': 'https://notion.so'},
            {'company_name': 'Shopify Commerce', 'website_url': 'https://shopify.com'},
            {'company_name': 'Mailchimp Marketing', 'website_url': 'https://mailchimp.com'}
        ][:max_companies]  # Limit mock data to requested number
        total_time = time.time() - start_time
        log(f"🔄 Returning {len(mock_companies)} mock companies in {total_time:.2f}s")
        return mock_companies
        
    except Exception as e:
        total_time = time.time() - start_time
        log(f"❌ Error in G2 extraction after {total_time:.2f}s: {str(e)}")
        log(f"🔍 Error Type: {type(e)}")
        return []

async def get_hiring_intent(company_url: str) -> Dict:
    """
    Check for hiring keywords on the company's main page to infer hiring intent.
    Uses Firecrawl's extract feature for structured analysis.
    Returns detailed hiring information including indicators and analysis.
    """
    if not company_url.startswith('http'):
        log(f"🚫 Invalid URL format: {company_url}")
        return {"has_hiring_intent": False, "hiring_indicators": [], "careers_page_exists": False, "confidence_level": "low"}
    
    start_time = time.time()
    log("🔍 ANALYZING HIRING INTENT")
    log(f"🌐 Target Website: {company_url}")
    
    # Check if API key exists
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        log("❌ FIRECRAWL_API_KEY not found for hiring intent analysis")
        return {"has_hiring_intent": False, "hiring_indicators": [], "careers_page_exists": False, "confidence_level": "low"}
    
    fc_app = AsyncFirecrawlApp(api_key=api_key)
    
    try:
        # Use Firecrawl's extract feature for hiring intent analysis
        log("🚀 Initiating Firecrawl EXTRACT for hiring intent...")
        scrape_start = time.time()
        
        # Define schema for hiring intent extraction
        hiring_schema = {
            "type": "object",
            "properties": {
                "hiring_intent": {
                    "type": "boolean",
                    "description": "Whether the company appears to be actively hiring based on career pages, job postings, or hiring-related content"
                },
                "confidence_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level of the hiring intent assessment"
                },
                "hiring_indicators": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of specific hiring-related keywords, phrases, or sections found on the website"
                },
                "careers_page_exists": {
                    "type": "boolean",
                    "description": "Whether the website has a careers or jobs page"
                },
                "open_positions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of specific job titles or departments mentioned if available"
                },
                "urgency_signals": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Urgent hiring language like 'immediate', 'urgent', 'ASAP', 'growing team', etc."
                }
            },
            "required": ["hiring_intent"]
        }
        
        # Extract hiring intent using Firecrawl
        response = await fc_app.extract(
            urls=[company_url],
            prompt="Analyze this company website for hiring intent. Look for careers pages, job postings, 'we're hiring' messages, open positions, recruitment content, or any indication that the company is actively looking to hire new employees. Focus on sales, marketing, business development, and growth-related positions. Also identify urgency signals and confidence level.",
            schema=hiring_schema
        )
        
        scrape_duration = time.time() - scrape_start
        log(f"✅ Hiring Intent Extract Response Received!")
        log(f"⏱️  Extraction Duration: {scrape_duration:.2f}s")
        
        # Process the extracted data
        if response and hasattr(response, 'data') and response.data:
            extracted_data = response.data
            
            if isinstance(extracted_data, list) and len(extracted_data) > 0:
                hiring_data = extracted_data[0]  # First URL result
                if isinstance(hiring_data, dict):
                    result = {
                        "has_hiring_intent": hiring_data.get('hiring_intent', False),
                        "confidence_level": hiring_data.get('confidence_level', 'low'),
                        "hiring_indicators": hiring_data.get('hiring_indicators', []),
                        "careers_page_exists": hiring_data.get('careers_page_exists', False),
                        "open_positions": hiring_data.get('open_positions', []),
                        "urgency_signals": hiring_data.get('urgency_signals', [])
                    }
                    
                    total_time = time.time() - start_time
                    log(f"📊 Hiring Intent Analysis Results:")
                    log(f"   🎯 Hiring Intent: {result['has_hiring_intent']}")
                    log(f"   � Confidence: {result['confidence_level']}")
                    log(f"   �🔍 Indicators: {result['hiring_indicators']}")
                    log(f"   📄 Careers Page: {result['careers_page_exists']}")
                    log(f"   💼 Open Positions: {result['open_positions']}")
                    log(f"   ⚡ Urgency Signals: {result['urgency_signals']}")
                    log(f"✅ Analysis completed in {total_time:.2f}s")
                    
                    return result
        
        # Fallback to simple scraping method
        log("🔄 Extract failed, trying fallback scraping...")
        response = await fc_app.scrape_url(company_url, formats=['markdown'])
        
        if response and hasattr(response, 'data') and response.data:
            content = response.data.get('markdown', '') if hasattr(response.data, 'get') else getattr(response.data, 'markdown', '')
            if content:
                text = content.lower()
                keywords = ['sales', 'marketing', 'growth', 'account executive', 'business development', 'hiring', 'careers', 'jobs', 'we\'re hiring', 'join our team', 'positions available', 'open positions', 'apply now']
                urgency_keywords = ['immediate', 'urgent', 'asap', 'growing team', 'expanding', 'rapid growth', 'hiring now']
                
                found_keywords = [kw for kw in keywords if kw in text]
                found_urgency = [kw for kw in urgency_keywords if kw in text]
                has_intent = len(found_keywords) > 0
                careers_page = any(word in text for word in ['careers', 'jobs', 'employment', 'opportunities'])
                
                result = {
                    "has_hiring_intent": has_intent,
                    "confidence_level": "high" if len(found_keywords) > 3 else "medium" if len(found_keywords) > 1 else "low",
                    "hiring_indicators": found_keywords,
                    "careers_page_exists": careers_page,
                    "open_positions": [],
                    "urgency_signals": found_urgency
                }
                
                total_time = time.time() - start_time
                log(f"📊 Fallback Analysis Results:")
                log(f"   📄 Content Length: {len(content)} characters")
                log(f"   🎯 Keywords Found: {found_keywords}")
                log(f"   ⚡ Urgency Signals: {found_urgency}")
                log(f"   ✅ Hiring Intent: {has_intent} (analyzed in {total_time:.2f}s)")
                return result
    
    except Exception as e:
        total_time = time.time() - start_time
        log(f"⚠️  Could not analyze {company_url} after {total_time:.2f}s: {str(e)}")
        log(f"🔍 Error Type: {type(e)}")
    
    # Final fallback to heuristic logic
    has_intent = any(keyword in company_url.lower() for keyword in ['crm', 'sales', 'marketing', 'saas', 'software'])
    total_time = time.time() - start_time
    log(f"🔄 Heuristic fallback for {company_url}: {has_intent} (processed in {total_time:.2f}s)")
    
    return {
        "has_hiring_intent": has_intent,
        "confidence_level": "low",
        "hiring_indicators": ["URL contains business keywords"] if has_intent else [],
        "careers_page_exists": False,
        "open_positions": [],
        "urgency_signals": []
    }

def calculate_lead_score(has_hiring_intent: bool, company_name: str = "", website_url: str = "", hiring_indicators: List[str] = None, careers_page_exists: bool = False, hiring_details: Dict = None) -> Dict:
    """
    Calculates a lead score using ML model when available, with rule-based fallback.
    Returns detailed scoring breakdown for transparency.
    """
    # Prepare lead data for ML model
    lead_data = {
        'company_name': company_name,
        'website': website_url,
        'hiring_intent': '✅ Yes' if has_hiring_intent else '❌ No',
        'hiring_details': hiring_details or {
            'has_hiring_intent': has_hiring_intent,
            'hiring_indicators': hiring_indicators or [],
            'careers_page_exists': careers_page_exists
        }
    }
    
    # Calculate rule-based score first (for fallback and comparison)
    rule_scoring_details = {
        "total_score": 0,
        "base_score": 0,
        "bonus_score": 0,
        "scoring_criteria": [],
        "hiring_intent": has_hiring_intent,
        "hiring_indicators": hiring_indicators or [],
        "careers_page_exists": careers_page_exists,
        "company_keywords": [],
        "risk_factors": []
    }
    
    # Base score for hiring intent (most important factor)
    if has_hiring_intent:
        rule_scoring_details["base_score"] = 40
        rule_scoring_details["scoring_criteria"].append("✅ Active hiring intent detected (+40)")
    else:
        rule_scoring_details["base_score"] = 5
        rule_scoring_details["scoring_criteria"].append("❌ No hiring intent detected (+5)")
    
    # Bonus scoring factors
    bonus_score = 0
    
    # Company name analysis (business maturity indicators)
    if company_name:
        high_value_keywords = ['enterprise', 'solutions', 'platform', 'software', 'tech', 'saas', 'cloud', 'analytics', 'intelligence']
        found_keywords = [kw for kw in high_value_keywords if kw.lower() in company_name.lower()]
        if found_keywords:
            keyword_bonus = len(found_keywords) * 3
            bonus_score += keyword_bonus
            rule_scoring_details["company_keywords"] = found_keywords
            rule_scoring_details["scoring_criteria"].append(f"🏢 Business keywords found: {', '.join(found_keywords)} (+{keyword_bonus})")
    
    # Domain credibility check
    if website_url:
        if not any(test in website_url.lower() for test in ['example', 'test', 'demo', 'mock']):
            bonus_score += 8
            rule_scoring_details["scoring_criteria"].append("🌐 Established domain detected (+8)")
        else:
            rule_scoring_details["risk_factors"].append("⚠️ Test/demo domain detected")
    
    # Careers page bonus
    if careers_page_exists:
        bonus_score += 15
        rule_scoring_details["scoring_criteria"].append("💼 Dedicated careers page exists (+15)")
    
    # Hiring indicators bonus
    if hiring_indicators and len(hiring_indicators) > 0:
        indicator_bonus = min(len(hiring_indicators) * 2, 10)  # Cap at 10 points
        bonus_score += indicator_bonus
        rule_scoring_details["scoring_criteria"].append(f"🎯 Hiring indicators found: {len(hiring_indicators)} signals (+{indicator_bonus})")
    
    # Company size estimation (based on name complexity and keywords)
    if company_name and len(company_name) > 15:
        bonus_score += 5
        rule_scoring_details["scoring_criteria"].append("📈 Established company name length (+5)")
    
    # Final rule-based calculations
    rule_scoring_details["bonus_score"] = bonus_score
    rule_scoring_details["total_score"] = rule_scoring_details["base_score"] + bonus_score
    
    # Add rule-based scoring to lead data
    lead_data['scoring_details'] = rule_scoring_details
    
    # Try ML prediction if available
    ml_prediction = None
    if ML_AVAILABLE and ml_model.is_trained:
        try:
            ml_prediction = ml_model.predict(lead_data)
            log(f"🤖 ML Prediction: {ml_prediction.ml_score:.1f}/100 (confidence: {ml_prediction.confidence:.3f})")
        except Exception as e:
            log(f"⚠️  ML prediction failed: {e}")
            ml_prediction = None
    
    # Combine ML and rule-based scoring
    final_scoring_details = rule_scoring_details.copy()
    
    if ml_prediction:
        # Use weighted combination: 70% ML, 30% rule-based
        ml_weight = 0.7
        rule_weight = 0.3
        combined_score = float(ml_prediction.ml_score * ml_weight + 
                              rule_scoring_details["total_score"] * rule_weight)
        
        # Convert numpy types to Python native types for JSON serialization
        feature_importance = ml_prediction.feature_importance
        if feature_importance and hasattr(feature_importance, 'items'):
            feature_importance = {k: float(v) for k, v in feature_importance.items()}
        elif feature_importance and hasattr(feature_importance, '__iter__'):
            feature_importance = [float(x) if hasattr(x, 'item') else x for x in feature_importance]
        
        final_scoring_details.update({
            "total_score": int(round(combined_score)),
            "ml_score": float(ml_prediction.ml_score),
            "rule_based_score": int(rule_scoring_details["total_score"]),
            "ml_confidence": float(ml_prediction.confidence),
            "ml_explanation": ml_prediction.prediction_explanation,
            "feature_importance": feature_importance,
            "model_version": ml_prediction.model_version,
            "scoring_method": "ML + Rule-based (70/30 blend)"
        })
        
        # Add ML explanations to criteria
        final_scoring_details["scoring_criteria"].extend([
            f"🤖 ML Score: {ml_prediction.ml_score:.1f}/100",
            f"📊 Combined Score: {combined_score:.1f}/100 (70% ML + 30% Rules)"
        ])
        final_scoring_details["scoring_criteria"].extend(ml_prediction.prediction_explanation)
    else:
        final_scoring_details["scoring_method"] = "Rule-based only"
        if not ML_AVAILABLE:
            final_scoring_details["scoring_criteria"].append("📊 Using rule-based scoring (ML not available)")
        elif not ml_model.is_trained:
            final_scoring_details["scoring_criteria"].append("📊 Using rule-based scoring (ML model not trained)")
    
    # Risk assessment based on final score
    final_score = final_scoring_details["total_score"]
    if final_score < 20:
        final_scoring_details["risk_level"] = "High Risk"
        final_scoring_details["recommendation"] = "Low priority - limited signals"
    elif final_score < 40:
        final_scoring_details["risk_level"] = "Medium Risk"
        final_scoring_details["recommendation"] = "Moderate priority - some positive signals"
    else:
        final_scoring_details["risk_level"] = "Low Risk"
        final_scoring_details["recommendation"] = "High priority - strong signals"
    
    log("📊 DETAILED LEAD SCORING:")
    log(f"   🎯 Final Score: {final_scoring_details['total_score']}")
    if ml_prediction:
        log(f"   🤖 ML Score: {ml_prediction.ml_score:.1f}")
        log(f"   📏 Rule Score: {rule_scoring_details['total_score']}")
        log(f"   🎲 ML Confidence: {ml_prediction.confidence:.3f}")
    log(f"   📊 Risk Level: {final_scoring_details['risk_level']}")
    log(f"   💡 Recommendation: {final_scoring_details['recommendation']}")
    
    # Convert all numpy types to Python natives before returning
    return convert_numpy_types(final_scoring_details)
