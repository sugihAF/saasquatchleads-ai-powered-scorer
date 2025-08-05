import streamlit as st
import requests
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List

# API Base URL - use service name when running in Docker
API_BASE_URL = "http://api:8501"

# Configure page
st.set_page_config(
    page_title="🎯 SaaS Lead Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Check if the FastAPI server is running"""
    # Try different endpoints for Docker and local environments
    api_endpoints = [
        ("http://api:8501/", "Docker service name"),
        ("http://localhost:8501/", "Local development"),
        ("http://127.0.0.1:8501/", "Alternative local")
    ]
    
    for api_url, description in api_endpoints:
        try:
            # Try GET request to root
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                st.sidebar.write(f"✅ Connected via: {description}")
                return True
            else:
                st.sidebar.write(f"⚠️ {description}: HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            st.sidebar.write(f"⏰ {description}: Timeout (5s)")
        except requests.exceptions.ConnectionError:
            st.sidebar.write(f"🔌 {description}: Connection failed")
        except Exception as e:
            st.sidebar.write(f"❌ {description}: {str(e)[:100]}")
            continue
    
    # Try with score-leads endpoint (should return 405 Method Not Allowed but server is up)  
    for api_url, description in api_endpoints:
        try:
            score_url = api_url.rstrip('/') + '/score-leads'
            response = requests.get(score_url, timeout=5)
            if response.status_code in [405, 422]:  # Method not allowed or validation error means server is up
                st.sidebar.write(f"✅ Connected via: {description} (score-leads endpoint)")
                return True
        except Exception as e:
            continue
    
    return False

def fetch_leads(g2_url: str, company_count: int = 5) -> Dict:
    """Fetch leads from the FastAPI endpoint"""
    # Try different API endpoints for Docker and local environments
    api_urls = [
        "http://api:8501/score-leads",  # Docker service name
        "http://localhost:8501/score-leads",  # Local development
        "http://127.0.0.1:8501/score-leads"  # Alternative local
    ]
    
    payload = {"g2_url": g2_url, "company_count": company_count}
    start_time = time.time()
    
    # Calculate dynamic timeout: 2 minutes per company (120 seconds each)
    timeout_seconds = company_count * 120
    timeout_minutes = timeout_seconds / 60
    
    for api_url in api_urls:
        try:
            with st.spinner(f"🔍 Analyzing {company_count} leads from {g2_url}... This may take up to {int(timeout_minutes)} minutes."):
                # Show real-time progress while waiting
                progress_container = st.empty()
                with progress_container.container():
                    st.info(f"⏳ **Processing {company_count} companies...** Each analysis takes 1-2 minutes.")
                    st.write("🔄 AI is analyzing company websites for hiring signals...")
                    st.write(f"⏱️ Maximum wait time: {int(timeout_minutes)} minutes")
                    st.write("☕ Perfect time for a coffee break!")
                
                response = requests.post(api_url, json=payload, timeout=timeout_seconds)
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                progress_container.empty()  # Clear progress message
                return {
                    "success": True, 
                    "data": response.json(),
                    "duration": duration,
                    "status_code": response.status_code,
                    "api_url": api_url
                }
            else:
                # Log the error for debugging
                st.error(f"API Error from {api_url}: HTTP {response.status_code}")
                st.error(f"Response: {response.text[:500]}")
                continue
                
        except requests.exceptions.Timeout:
            st.error(f"⏰ Timeout after {int(timeout_minutes)} minutes for {api_url}")
            st.error(f"💡 Try reducing the number of companies or check your internet connection")
            st.warning("💡 **Tip:** Try reducing the number of companies by using a more specific G2 category")
            continue
        except requests.exceptions.ConnectionError as e:
            st.error(f"🔌 Connection error to {api_url}: {str(e)[:200]}")
            continue
        except Exception as e:
            st.error(f"❌ Unexpected error with {api_url}: {str(e)[:200]}")
            continue
    
    # If all URLs failed
    duration = time.time() - start_time
    return {
        "success": False,
        "error": "Could not connect to API backend. Please check if the API service is running.",
        "duration": duration,
        "attempted_urls": api_urls
    }

def check_ml_status():
    """Check ML system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/ml/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "available": True,
                "ml_available": data.get("ml_available", False),
                "model_trained": data.get("is_trained", False),
                "training_samples": data.get("training_samples", 0),
                "last_trained": data.get("last_trained", "Never"),
                "model_version": data.get("model_version", "Unknown")
            }
        else:
            return {"available": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"available": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"available": False, "error": f"Unexpected error: {str(e)}"}

def train_ml_model():
    """Train ML model"""
    try:
        # Send proper training request with default parameters
        payload = {
            "use_mock_data": True,
            "num_samples": 200
        }
        response = requests.post(f"{API_BASE_URL}/api/v1/ml/train", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def submit_feedback(company_name: str, predicted_score: float, actual_outcome: str, feedback_score: float):
    """Submit feedback for ML model"""
    try:
        data = {
            "company_name": company_name,
            "predicted_score": predicted_score,
            "actual_outcome": actual_outcome,
            "feedback_score": feedback_score
        }
        response = requests.post(f"{API_BASE_URL}/api/v1/ml/feedback", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def main():
    st.markdown('<h1 class="main-header">🎯 SaaS Lead Scorer Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # API Status Check
    api_status = check_api_status()
    if api_status:
        st.sidebar.markdown('<div class="success-card">✅ API Server: Online</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="error-card">❌ API Server: Offline</div>', unsafe_allow_html=True)
        st.sidebar.markdown("**Start the API server first:**")
        st.sidebar.code("docker-compose up --build")
        st.sidebar.markdown("**Or run FastAPI directly:**")
        st.sidebar.code("uvicorn app:app --host 0.0.0.0 --port 8501")
        return
    
    # Configuration
    st.sidebar.subheader("📊 G2 Category Selection")
    
    # Predefined G2 URLs
    g2_categories = {
        "CRM Software": "https://www.g2.com/categories/crm",
        "Marketing Automation": "https://www.g2.com/categories/marketing-automation", 
        "Sales Software": "https://www.g2.com/categories/sales",
        "Customer Support": "https://www.g2.com/categories/help-desk",
        "Email Marketing": "https://www.g2.com/categories/email-marketing",
        "Project Management": "https://www.g2.com/categories/project-management",
        "Business Intelligence": "https://www.g2.com/categories/business-intelligence",
        "Custom URL": "custom"
    }
    
    selected_category = st.sidebar.selectbox("Choose G2 Category:", list(g2_categories.keys()))
    
    if selected_category == "Custom URL":
        g2_url = st.sidebar.text_input("Enter G2 URL:", "https://www.g2.com/categories/")
    else:
        g2_url = g2_categories[selected_category]
        st.sidebar.write(f"**Selected URL:**")
        st.sidebar.code(g2_url)
    
    # Company count control
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Analysis Settings")
    company_count = st.sidebar.slider(
        "Number of companies to analyze:",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="More companies = longer processing time (~2 minutes per company)"
    )
    
    # Show estimated time based on selection
    estimated_time = company_count * 2  # 2 minutes per company
    if estimated_time < 1:
        time_display = f"{int(estimated_time * 60)} seconds"
    else:
        time_display = f"{estimated_time} minutes"
    
    st.sidebar.info(f"⏱️ Estimated time: {time_display}")
    st.sidebar.markdown("---")
    
    # ML Management Section
    st.sidebar.subheader("🤖 AI/ML Management")
    
    # Check ML status
    ml_status = check_ml_status()
    if ml_status.get("available", False):
        st.sidebar.markdown('<div class="success-card">✅ ML System: Online</div>', unsafe_allow_html=True)
        
        # Show ML model info  
        if ml_status.get("model_trained", False):
            st.sidebar.success(f"🎯 Model trained")
            st.sidebar.info(f"📅 Last trained: {ml_status.get('last_trained', 'Unknown')}")
        else:
            st.sidebar.warning("⚠️ Model not trained yet")
        
        # ML Training Button
        if st.sidebar.button("🎓 Train ML Model", help="Train the machine learning model with sample data"):
            with st.spinner("Training ML model..."):
                train_result = train_ml_model()
                if train_result.get("success", False):
                    st.sidebar.success("✅ ML model trained successfully!")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Training failed: {train_result.get('error', 'Unknown error')}")
    else:
        st.sidebar.markdown('<div class="error-card">❌ ML System: Offline</div>', unsafe_allow_html=True)
        st.sidebar.write(f"Error: {ml_status.get('error', 'Unknown')}")
    
    st.sidebar.markdown("---")
    
    # Validation
    if not g2_url or not g2_url.startswith("http"):
        st.sidebar.error("Please enter a valid G2 URL")
        return
    
    # Main action button
    analyze_button = st.sidebar.button("🚀 Analyze Leads", type="primary", use_container_width=True)
    
    # Add dynamic timing information
    st.sidebar.markdown("---")
    st.sidebar.markdown("⏱️ **Expected Processing Time:**")
    st.sidebar.markdown(f"• {company_count} companies: ~{time_display}")
    st.sidebar.markdown("• Uses AI to analyze each website (2 min/company)")
    st.sidebar.markdown("• Please be patient during analysis")
    st.sidebar.markdown("---")
    
    # Main content area
    if analyze_button or 'leads_data' in st.session_state:
        
        if analyze_button:
            # Clear previous results
            if 'leads_data' in st.session_state:
                del st.session_state['leads_data']
            
            # Show real-time progress
            progress_placeholder = st.empty()
            
            with progress_placeholder.container():
                st.info(f"🎯 **Starting Analysis**")
                st.write(f"📊 Target: {selected_category}")
                st.write(f"🌐 URL: {g2_url}")
                
                # Show progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress updates
                for i in range(1, 4):
                    progress_bar.progress(i * 25)
                    if i == 1:
                        status_text.text("🔍 Extracting companies from G2...")
                    elif i == 2:
                        status_text.text("🌐 Analyzing company websites...")
                    elif i == 3:
                        status_text.text("📊 Calculating lead scores...")
                    time.sleep(0.5)
            
            # Fetch the actual data
            result = fetch_leads(g2_url, company_count)
            
            # Clear progress
            progress_placeholder.empty()
            
            # Store results in session state
            st.session_state['leads_data'] = result
            st.session_state['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state['category'] = selected_category
            st.session_state['url'] = g2_url
            st.session_state['company_count'] = company_count
        
        # Display results from session state
        if 'leads_data' in st.session_state:
            result = st.session_state['leads_data']
            
            if result["success"]:
                leads_data = result["data"]
                
                # Success header
                st.success(f"✅ **Analysis Complete!** Found {len(leads_data)} leads in {result.get('duration', 0):.1f} seconds")
                
                # Convert to DataFrame
                if leads_data:
                    df = pd.DataFrame(leads_data)
                    data = leads_data  # Keep original data for detailed views
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📈 Total Leads", len(df))
                    with col2:
                        hiring_count = len(df[df['hiring_intent'].str.contains('✅')])
                        percentage = (hiring_count / len(df) * 100) if len(df) > 0 else 0
                        st.metric("🎯 With Hiring Intent", hiring_count, f"{percentage:.1f}%")
                    with col3:
                        avg_score = df['score'].mean() if not df.empty else 0
                        st.metric("📊 Average Score", f"{avg_score:.1f}")
                    with col4:
                        max_score = df['score'].max() if not df.empty else 0
                        st.metric("🏆 Highest Score", max_score)
                    
                    # Enhanced Results with Tabs
                    st.subheader("📋 Comprehensive Lead Analysis")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Scoring Details", "💼 Hiring Intelligence", "📄 Raw Data", "🤖 ML Feedback"])
                    
                    with tab1:
                        # Basic overview table
                        df_display = df.copy()
                        df_display['Rank'] = range(1, len(df_display) + 1)
                        df_display = df_display[['Rank', 'company_name', 'website', 'hiring_intent', 'score']]
                        df_display.columns = ['Rank', 'Company Name', 'Website', 'Hiring Intent', 'Score']
                        
                        # Color-code the dataframe
                        def color_score(val):
                            if val >= 40:
                                return 'background-color: #d4edda'  # Green
                            elif val >= 20:
                                return 'background-color: #fff3cd'  # Yellow
                            else:
                                return 'background-color: #f8d7da'  # Red
                        
                        styled_df = df_display.style.applymap(color_score, subset=['Score'])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
                    
                    with tab2:
                        # Detailed scoring breakdown
                        st.write("**🎯 Comprehensive Scoring Analysis**")
                        st.info("💡 Each lead is scored based on hiring intent, company maturity indicators, and business signals.")
                        
                        for idx, lead in enumerate(data):
                            with st.expander(f"#{idx+1} {lead['company_name']} (Score: {lead['score']})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**🏢 Company Information**")
                                    st.write(f"• **Name:** {lead['company_name']}")
                                    st.write(f"• **Website:** {lead['website']}")
                                    st.write(f"• **Final Score:** **{lead['score']}**/100")
                                    
                                    if 'scoring_details' in lead and lead['scoring_details']:
                                        scoring = lead['scoring_details']
                                        risk_color = "🟢" if scoring.get('risk_level') == 'Low Risk' else "🟡" if scoring.get('risk_level') == 'Medium Risk' else "🔴"
                                        st.write(f"• **Risk Level:** {risk_color} {scoring.get('risk_level', 'Unknown')}")
                                        st.write(f"• **Recommendation:** {scoring.get('recommendation', 'N/A')}")
                                
                                with col2:
                                    st.write("**📊 Scoring Breakdown**")
                                    if 'scoring_details' in lead and lead['scoring_details']:
                                        scoring = lead['scoring_details']
                                        st.write(f"• **Base Score:** {scoring.get('base_score', 0)} points")
                                        st.write(f"• **Bonus Score:** {scoring.get('bonus_score', 0)} points")
                                        
                                        if scoring.get('company_keywords'):
                                            st.write(f"• **Business Keywords:** {', '.join(scoring['company_keywords'])}")
                                        
                                        st.write("**🎯 Scoring Factors:**")
                                        for criterion in scoring.get('scoring_criteria', []):
                                            st.write(f"  {criterion}")
                                        
                                        if scoring.get('risk_factors'):
                                            st.write("**⚠️ Risk Factors:**")
                                            for risk in scoring['risk_factors']:
                                                st.write(f"  {risk}")
                    
                    with tab3:
                        # Hiring intent details
                        st.write("**💼 Hiring Intent Intelligence**")
                        st.info("🔍 AI-powered analysis of company websites to detect active hiring signals and urgency levels.")
                        
                        for idx, lead in enumerate(data):
                            intent_status = "🟢" if lead['hiring_intent'].startswith('✅') else "🔴"
                            with st.expander(f"#{idx+1} {lead['company_name']} - {intent_status} {lead['hiring_intent']}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if 'hiring_details' in lead and lead['hiring_details']:
                                        hiring = lead['hiring_details']
                                        st.write("**🎯 Hiring Analysis**")
                                        intent_display = "✅ Active" if hiring.get('has_hiring_intent') else "❌ None detected"
                                        st.write(f"• **Hiring Intent:** {intent_display}")
                                        confidence = hiring.get('confidence_level', 'Unknown').title()
                                        confidence_color = "🟢" if confidence == 'High' else "🟡" if confidence == 'Medium' else "🔴"
                                        st.write(f"• **Confidence:** {confidence_color} {confidence}")
                                        careers_status = "✅ Yes" if hiring.get('careers_page_exists') else "❌ No"
                                        st.write(f"• **Careers Page:** {careers_status}")
                                        
                                        if hiring.get('hiring_indicators'):
                                            st.write("**🔍 Hiring Indicators:**")
                                            for indicator in hiring['hiring_indicators'][:5]:  # Show top 5
                                                st.write(f"  • {indicator}")
                                            if len(hiring['hiring_indicators']) > 5:
                                                st.write(f"  ... and {len(hiring['hiring_indicators']) - 5} more")
                                
                                with col2:
                                    if 'hiring_details' in lead and lead['hiring_details']:
                                        hiring = lead['hiring_details']
                                        
                                        if hiring.get('open_positions'):
                                            st.write("**💼 Open Positions:**")
                                            for position in hiring['open_positions'][:5]:  # Show top 5
                                                st.write(f"  • {position}")
                                            if len(hiring['open_positions']) > 5:
                                                st.write(f"  ... and {len(hiring['open_positions']) - 5} more")
                                        else:
                                            st.write("**💼 Open Positions:** None specifically detected")
                                        
                                        if hiring.get('urgency_signals'):
                                            st.write("**⚡ Urgency Signals:**")
                                            for signal in hiring['urgency_signals']:
                                                st.write(f"  • {signal}")
                                        else:
                                            st.write("**⚡ Urgency Signals:** No immediate urgency detected")
                    
                    with tab4:
                        # Raw data view
                        st.write("**📄 Complete Raw Data**")
                        st.info("🔍 All scraped and processed data for transparency and debugging. This includes the complete API response and internal data structures.")
                        
                        for idx, lead in enumerate(data):
                            with st.expander(f"#{idx+1} {lead['company_name']} - Complete Data Dump"):
                                # Basic company data
                                st.write("**🏢 Basic Company Information**")
                                st.json({
                                    "company_name": lead.get('company_name'),
                                    "website": lead.get('website'),
                                    "hiring_intent": lead.get('hiring_intent'),
                                    "score": lead.get('score')
                                })
                                
                                # Scoring details raw data
                                if 'scoring_details' in lead and lead['scoring_details']:
                                    st.write("**📊 Complete Scoring Data**")
                                    st.json(lead['scoring_details'])
                                
                                # Hiring details raw data
                                if 'hiring_details' in lead and lead['hiring_details']:
                                    st.write("**💼 Complete Hiring Analysis Data**")
                                    st.json(lead['hiring_details'])
                                
                                # Show any additional fields that might exist
                                other_fields = {k: v for k, v in lead.items() 
                                              if k not in ['company_name', 'website', 'hiring_intent', 'score', 'scoring_details', 'hiring_details']}
                                
                                if other_fields:
                                    st.write("**🔧 Additional Fields**")
                                    st.json(other_fields)
                                
                                # Raw data download for individual company
                                company_data = json.dumps(lead, indent=2, default=str)
                                st.download_button(
                                    label=f"📥 Download {lead['company_name']} Raw Data",
                                    data=company_data,
                                    file_name=f"raw_data_{lead['company_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    key=f"download_raw_{idx}"
                                )
                        
                        # Complete dataset download
                        st.markdown("---")
                        st.write("**📦 Complete Dataset**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Full raw data JSON
                            complete_data = json.dumps(data, indent=2, default=str)
                            st.download_button(
                                label="📄 Download All Raw Data (JSON)",
                                data=complete_data,
                                file_name=f"complete_raw_data_{st.session_state.get('category', 'analysis').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col2:
                            # API response metadata
                            api_metadata = {
                                "analysis_time": st.session_state.get('analysis_time'),
                                "category": st.session_state.get('category'),
                                "url": st.session_state.get('url'),
                                "company_count": st.session_state.get('company_count'),
                                "api_url": result.get('api_url'),
                                "duration": result.get('duration'),
                                "status_code": result.get('status_code'),
                                "total_companies_processed": len(data)
                            }
                            metadata_json = json.dumps(api_metadata, indent=2, default=str)
                            st.download_button(
                                label="🔧 Download Analysis Metadata",
                                data=metadata_json,
                                file_name=f"analysis_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    
                    with tab5:
                        # ML Feedback tab
                        st.write("**🤖 Machine Learning Feedback & Training**")
                        st.info("💡 Help improve the AI scoring by providing feedback on actual lead outcomes.")
                        
                        # ML Status display
                        ml_status = check_ml_status()
                        if ml_status.get("available", False):
                            if ml_status.get("model_trained", False):
                                st.success(f"✅ ML Model Status: Trained with {ml_status.get('training_samples', 'N/A')} samples")
                                st.info(f"📅 Last trained: {ml_status.get('last_trained', 'Unknown')}")
                                st.info(f"🔢 Model version: {ml_status.get('model_version', 'Unknown')}")
                            else:
                                st.warning("⚠️ ML Model not trained yet - predictions using rule-based scoring only")
                        else:
                            st.error("❌ ML System not available")
                            st.error(f"Error details: {ml_status.get('error', 'Unknown error')}")
                        
                        # Feedback form
                        st.markdown("---")
                        st.subheader("📝 Provide Lead Outcome Feedback")
                        
                        # Company selection for feedback
                        company_options = [f"{lead['company_name']} (Score: {lead['score']:.1f})" for lead in data]
                        selected_company_str = st.selectbox(
                            "Select a company to provide feedback for:",
                            options=company_options,
                            help="Choose a company from your current analysis results"
                        )
                        
                        if selected_company_str:
                            # Extract company data
                            selected_idx = company_options.index(selected_company_str)
                            selected_lead = data[selected_idx]
                            company_name = selected_lead['company_name']
                            predicted_score = selected_lead['score']
                            
                            st.write(f"**Company:** {company_name}")
                            st.write(f"**AI Predicted Score:** {predicted_score:.1f}/100")
                            
                            # Feedback form
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                actual_outcome = st.select_slider(
                                    "What was the actual outcome?",
                                    options=["Very Poor Lead", "Poor Lead", "Average Lead", "Good Lead", "Excellent Lead"],
                                    value="Average Lead",
                                    help="How did this lead perform in reality?"
                                )
                            
                            with col2:
                                feedback_score = st.slider(
                                    "Actual Performance Score:",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=50.0,
                                    step=1.0,
                                    help="What score would you give this lead based on actual performance?"
                                )
                            
                            # Additional feedback
                            feedback_notes = st.text_area(
                                "Additional Notes (Optional):",
                                placeholder="Any additional context about this lead's performance...",
                                height=100
                            )
                            
                            # Submit feedback
                            if st.button("🚀 Submit Feedback", type="primary"):
                                with st.spinner("Submitting feedback..."):
                                    feedback_result = submit_feedback(
                                        company_name=company_name,
                                        predicted_score=predicted_score,
                                        actual_outcome=actual_outcome,
                                        feedback_score=feedback_score
                                    )
                                    
                                    if feedback_result.get("success", False):
                                        st.success("✅ Feedback submitted successfully! This will help improve future predictions.")
                                        st.balloons()
                                    else:
                                        st.error(f"❌ Failed to submit feedback: {feedback_result.get('error', 'Unknown error')}")
                        
                        # Training section
                        st.markdown("---")
                        st.subheader("🎓 Model Training")
                        st.write("Train the ML model with accumulated feedback and sample data.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Retrain Model", help="Retrain the ML model with latest feedback"):
                                with st.spinner("Training ML model..."):
                                    train_result = train_ml_model()
                                    if train_result.get("success", False):
                                        st.success("✅ Model retrained successfully!")
                                        st.info(f"📊 Training info: {train_result.get('message', 'Model updated')}")
                                    else:
                                        st.error(f"❌ Training failed: {train_result.get('error', 'Unknown error')}")
                        
                        with col2:
                            if st.button("📊 View ML Predictions", help="See detailed ML predictions for current results"):
                                st.write("**🔍 ML Prediction Details:**")
                                for idx, lead in enumerate(data):
                                    if 'ml_prediction' in lead:
                                        with st.expander(f"{lead['company_name']} - ML Analysis"):
                                            ml_pred = lead['ml_prediction']
                                            st.write(f"**ML Score:** {ml_pred.get('ml_score', 'N/A'):.1f}")
                                            st.write(f"**Confidence:** {ml_pred.get('confidence', 'N/A'):.1f}%")
                                            if 'explanation' in ml_pred:
                                                st.write(f"**Reasoning:** {ml_pred['explanation']}")
                    
                    styled_df = df_display.style.applymap(color_score, subset=['Score'])
                    
                    st.dataframe(
                        styled_df, 
                        use_container_width=True, 
                        hide_index=True, 
                        height=400
                    )
                    
                    # Export options
                    st.subheader("📥 Export Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv,
                            file_name=f"leads_{st.session_state.get('category', 'analysis').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Enhanced JSON with all details
                        enhanced_json = json.dumps(data, indent=2, default=str)
                        st.download_button(
                            label="📝 Download Enhanced JSON",
                            data=enhanced_json,
                            file_name=f"enhanced_leads_{st.session_state.get('category', 'analysis').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col3:
                        # Create comprehensive investor report
                        detailed_analysis = []
                        for lead in data:
                            analysis = f"\n## {lead['company_name']} (Score: {lead['score']}/100)\n"
                            analysis += f"**Website:** {lead['website']}\n"
                            analysis += f"**Hiring Status:** {lead['hiring_intent']}\n"
                            
                            if 'scoring_details' in lead and lead['scoring_details']:
                                scoring = lead['scoring_details']
                                analysis += f"**Investment Risk:** {scoring.get('risk_level', 'Unknown')}\n"
                                analysis += f"**Recommendation:** {scoring.get('recommendation', 'N/A')}\n"
                                analysis += f"**Base Score:** {scoring.get('base_score', 0)} points\n"
                                analysis += f"**Bonus Score:** {scoring.get('bonus_score', 0)} points\n"
                                
                                if scoring.get('company_keywords'):
                                    analysis += f"**Business Keywords:** {', '.join(scoring['company_keywords'])}\n"
                                
                                analysis += f"**Scoring Rationale:**\n"
                                for criterion in scoring.get('scoring_criteria', []):
                                    analysis += f"  • {criterion}\n"
                                
                                if scoring.get('risk_factors'):
                                    analysis += f"**Risk Factors:**\n"
                                    for risk in scoring['risk_factors']:
                                        analysis += f"  • {risk}\n"
                            
                            if 'hiring_details' in lead and lead['hiring_details']:
                                hiring = lead['hiring_details']
                                analysis += f"**Hiring Confidence:** {hiring.get('confidence_level', 'Unknown').title()}\n"
                                analysis += f"**Careers Page:** {'Yes' if hiring.get('careers_page_exists') else 'No'}\n"
                                
                                if hiring.get('hiring_indicators'):
                                    analysis += f"**Hiring Signals:** {', '.join(hiring['hiring_indicators'][:3])}\n"
                                if hiring.get('urgency_signals'):
                                    analysis += f"**Urgency Indicators:** {', '.join(hiring['urgency_signals'])}\n"
                                if hiring.get('open_positions'):
                                    analysis += f"**Open Roles:** {', '.join(hiring['open_positions'][:3])}\n"
                            
                            detailed_analysis.append(analysis)
                        
                        investor_report = f"""
# 🎯 AI-Powered Lead Scoring Report for Investors

**Analysis Date:** {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}
**Category:** {st.session_state.get('category', 'Various')}
**Source:** {st.session_state.get('url', 'G2.com Category Analysis')}

## 📈 Executive Summary

**Investment Pipeline Overview:**
• **Total Companies Analyzed:** {len(df)}
• **Active Hiring Intent Detected:** {hiring_count} companies ({percentage:.1f}%)
• **Average Lead Score:** {avg_score:.1f}/100
• **Highest Scoring Opportunity:** {max_score}/100
• **High Priority Investments (40+ score):** {len(df[df['score'] >= 40])} companies
• **Medium Priority (20-39 score):** {len(df[(df['score'] >= 20) & (df['score'] < 40)])} companies
• **Low Priority (<20 score):** {len(df[df['score'] < 20])} companies

## 🔍 Investment Thesis

This analysis leverages AI-powered website intelligence to identify SaaS companies with active hiring intent - a strong indicator of growth, funding runway, and market expansion. Companies actively hiring in sales, marketing, and business development roles represent prime investment opportunities during scale-up phases.

## 🎯 Scoring Methodology

**Base Scoring (0-40 points):**
• Active hiring intent detection: +40 points
• No hiring signals detected: +5 points (baseline)

**Bonus Factors (0-35+ points):**
• Business maturity keywords: +3 per keyword
• Established domain credibility: +8 points  
• Dedicated careers pages: +15 points
• Multiple hiring indicators: +2 per signal (max 10)
• Company name complexity: +5 points

**Risk Assessment:**
• High Priority (40+ score): Low risk, strong growth signals
• Medium Priority (20-39): Moderate risk, some positive indicators  
• Low Priority (<20): High risk, limited growth evidence

## 🏆 Detailed Company Analysis
{''.join(detailed_analysis)}

## 💡 Investment Recommendations

**Immediate Action (High Priority):**
Focus on companies scoring 40+ with multiple hiring signals and established market presence.

**Due Diligence (Medium Priority):** 
Investigate companies scoring 20-39 for additional growth indicators and market position.

**Watch List (Low Priority):**
Monitor companies scoring <20 for future hiring activity and growth catalysts.

---
*This report was generated using AI-powered analysis of company websites and hiring signals. Data should be verified through additional due diligence before making investment decisions.*
"""
                        
                        st.download_button(
                            label="📄 Investor Report",
                            data=investor_report,
                            file_name=f"investor_report_{st.session_state.get('category', 'analysis').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    # Enhanced Analysis insights
                    st.subheader("🔍 Investment Intelligence")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Hiring Intent Analysis**")
                        intent_counts = df['hiring_intent'].value_counts()
                        for intent, count in intent_counts.items():
                            percentage = (count / len(df)) * 100
                            icon = "🟢" if "✅" in intent else "🔴"
                            st.write(f"{icon} {intent}: {count} companies ({percentage:.1f}%)")
                        
                        # Confidence distribution if available
                        if any('hiring_details' in lead for lead in data):
                            st.markdown("**📊 Confidence Levels**")
                            confidence_counts = {}
                            for lead in data:
                                if 'hiring_details' in lead and lead['hiring_details']:
                                    conf = lead['hiring_details'].get('confidence_level', 'unknown')
                                    confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
                            
                            for conf, count in confidence_counts.items():
                                icon = "🟢" if conf == 'high' else "🟡" if conf == 'medium' else "🔴"
                                st.write(f"{icon} {conf.title()}: {count} companies")
                    
                    with col2:
                        st.markdown("**📊 Investment Priority Distribution**")
                        high_score = len(df[df['score'] >= 40])
                        medium_score = len(df[(df['score'] >= 20) & (df['score'] < 40)])
                        low_score = len(df[df['score'] < 20])
                        
                        st.write(f"🟢 **High Priority** (40+): {high_score} companies")
                        st.write(f"🟡 **Medium Priority** (20-39): {medium_score} companies") 
                        st.write(f"🔴 **Low Priority** (<20): {low_score} companies")
                        
                        # Additional insights
                        if any('scoring_details' in lead for lead in data):
                            st.markdown("**🏢 Business Maturity Signals**")
                            keyword_companies = sum(1 for lead in data if 'scoring_details' in lead and lead['scoring_details'] and lead['scoring_details'].get('company_keywords'))
                            careers_companies = sum(1 for lead in data if 'hiring_details' in lead and lead['hiring_details'] and lead['hiring_details'].get('careers_page_exists'))
                            
                            st.write(f"📈 Companies with business keywords: {keyword_companies}")
                            st.write(f"💼 Companies with careers pages: {careers_companies}")
                
                else:
                    st.warning("No leads found in the analysis.")
            
            else:
                st.error(f"❌ **Analysis Failed**")
                st.write(f"**Error:** {result['error']}")
                if result.get('status_code'):
                    st.write(f"**Status Code:** {result['status_code']}")
    
    else:
        # Initial state - show instructions
        st.info("👆 **Select a G2 category from the sidebar and click 'Analyze Leads' to begin**")
        
        st.markdown("### 🎯 How it works:")
        st.markdown("""
        1. **Select Category**: Choose from popular G2 categories or enter a custom URL
        2. **AI Extraction**: Uses Firecrawl's extract feature to get company data from G2
        3. **Hiring Analysis**: Analyzes each company's website for hiring intent
        4. **Lead Scoring**: Calculates scores based on hiring intent and other factors
        5. **Results**: View, sort, and export your scored leads
        """)
        
        st.markdown("### 📊 Features:")
        st.markdown("""
        - **Real-time Processing**: Watch the analysis happen live
        - **Smart Extraction**: AI-powered data extraction from websites
        - **Hiring Intent Detection**: Identifies companies actively hiring
        - **Lead Scoring**: Prioritizes leads based on multiple factors
        - **Export Options**: Download results as CSV, JSON, or text report
        """)

if __name__ == "__main__":
    main()
