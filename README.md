# 🎯 SaaS Lead Scorer - Full Stack Application

A comprehensive lead scoring system that extracts companies from G2 category pages, analyzes their hiring intent, and provides scored leads through a beautiful Streamlit dashboard.

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Start both API and dashboard
docker-compose up --build

# Access the applications:
# Dashboard: http://localhost:8502
# API Docs: http://localhost:8501/docs
```

### Option 2: Local Development
```bash
# Windows
start_app.bat

# Linux/Mac
python start_app.py
```

## 🌟 Features

### 🎨 **Advanced Streamlit Dashboard**
- **5-Tab Interface**: Lead Scoring, ML Training, Analytics, Data Management, Settings
- **Real-time Processing**: Live progress tracking with status updates
- **Interactive Tables**: Sortable, filterable lead displays with pagination
- **ML Model Management**: Train, evaluate, and monitor model performance
- **Export Options**: CSV, JSON, and formatted text reports
- **Analytics Dashboard**: Performance metrics and insights

### 🤖 **Machine Learning Engine**
- **XGBoost Classifier**: Advanced gradient boosting for lead conversion prediction
- **15 Engineered Features**: Comprehensive feature extraction from lead data
- **Hybrid Scoring**: 70% ML predictions + 30% rule-based scoring
- **Model Training**: Interactive ML training with mock and real data
- **Performance Metrics**: Accuracy, precision, recall, F1-score, AUC tracking
- **Feature Importance**: Understand which factors drive predictions
- **Continuous Learning**: Feedback loop for model improvement

### � **AI-Powered Analysis**
- **Firecrawl Integration**: Advanced AI extraction from G2 pages and company websites
- **Hiring Intent Detection**: AI-powered analysis of hiring signals
- **Content Analysis**: Intelligent parsing of careers pages and job postings
- **Risk Assessment**: Automated company credibility evaluation
- **Confidence Scoring**: Uncertainty quantification for predictions

### 🔧 **FastAPI Backend**
- **RESTful API**: Clean, documented endpoints with OpenAPI specs
- **Async Processing**: Non-blocking concurrent operations
- **Type Safety**: Full Pydantic model validation
- **Error Handling**: Comprehensive exception management with fallbacks
- **Performance Monitoring**: Request/response time tracking

## 📊 How to Use

### 1. Start the Application
Choose your preferred method:

**Docker (Both services):**
```bash
docker-compose up --build
```

**Local (Both services):**
```bash
# Windows
start_app.bat

# Linux/Mac  
python start_app.py
```

### 2. Access the Dashboard
Open your browser to: **http://localhost:8502**

### 3. Analyze Leads
1. **Select Category**: Choose from dropdown or enter custom G2 URL
2. **Click Analyze**: Watch real-time processing
3. **View Results**: See scored leads in interactive table
4. **Export Data**: Download CSV, JSON, or text report

### 4. Review Results
The dashboard shows:
- **Lead Rankings**: Sorted by score
- **Hiring Intent**: AI-detected hiring signals
- **Company Details**: Names and websites
- **Score Breakdown**: Understanding the scoring
- **Export Options**: Multiple download formats

## 🎯 G2 Categories Supported

Pre-configured categories include:
- **CRM Software**
- **Marketing Automation** 
- **Sales Software**
- **Customer Support**
- **Email Marketing**
- **Project Management**
- **Business Intelligence**
- **Custom URLs**

## 🔧 API Endpoints

### POST /score-leads
Analyze a G2 category URL and return scored leads.

**Request:**
```json
{
  "g2_url": "https://www.g2.com/categories/crm"
}
```

**Response:**
```json
[
  {
    "company_name": "Salesforce CRM",
    "website": "https://salesforce.com",
    "hiring_intent": "✅ Yes",
    "score": 45
  }
]
```

## 🛠️ Configuration & Requirements

### 📦 **Python Dependencies**
```txt
# Core Framework
fastapi              # High-performance async web framework
uvicorn[standard]    # ASGI server with performance extras
streamlit           # Interactive web dashboard framework

# Machine Learning Stack  
xgboost             # Gradient boosting classifier
scikit-learn        # ML preprocessing & evaluation
numpy               # Numerical computing & arrays
pandas              # Data manipulation & analysis

# AI & Web Services
firecrawl-py        # AI-powered web scraping API
requests            # HTTP client for external APIs

# Configuration
python-dotenv       # Environment variable management
```

### 🌍 **Environment Variables**
Create a `.env` file in the project root:
```env
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

### 🔧 **System Requirements**
- **Python**: 3.8+ (recommended 3.9+)
- **Memory**: 2GB+ RAM (4GB+ recommended for ML training)
- **Storage**: 500MB+ free space (for models and logs)
- **Network**: Internet access for Firecrawl API calls

### ⚙️ **Port Configuration**
- **API Backend**: Port 8501
- **Streamlit Dashboard**: Port 8502
- **Configurable**: Modify `docker-compose.yaml` or startup scripts

## 📈 Advanced Lead Scoring System

### 🤖 **Machine Learning Component (70% Weight)**
- **XGBoost Classifier**: Trained on 15 engineered features
- **Feature Engineering**: 
  - Hiring intent signals (binary and confidence levels)
  - Company characteristics (keywords, domain credibility)
  - Content analysis (careers page, job postings, urgency signals)
  - Risk factors and quality indicators
- **Prediction Pipeline**: Feature extraction → normalization → XGBoost → probability score
- **Confidence Metrics**: Model confidence in predictions (0-1 scale)

### ⚙️ **Rule-Based Component (30% Weight)**
- **Base Score Calculation**:
  - Hiring Intent Detected: 40 points
  - No Clear Intent: 5 points
- **Intelligent Bonus System**:
  - Enterprise Keywords: +3 per keyword (enterprise, solutions, platform, software, tech, saas, cloud, analytics)
  - Careers Page Exists: +15 points
  - Hiring Indicators: +2 per indicator (max 10 points)
  - Open Positions Found: +1 per position (max 5 points)
  - Urgency Signals: +3 per signal (max 12 points)
  - Quality Domain: +10 points (non-test/demo domains)
- **Risk Assessment**:
  - Risk Factors Detected: -5 per factor
  - Low Confidence: Reduced final score
  - Suspicious Patterns: Additional penalties

### 🎯 **Hybrid Scoring Formula**
```
Final Score = (ML_Prediction × 0.7) + (Rule_Based_Score × 0.3)
```

### 📊 **Scoring Ranges & Interpretation**
- **🟢 High Priority (60-100)**: Strong ML confidence + established company with clear hiring signals
- **🟡 Medium Priority (30-59)**: Moderate indicators with some positive signals
- **🔴 Low Priority (0-29)**: Limited hiring signals or high uncertainty
- **⚪ Review Required**: Conflicting signals between ML and rules

### 🔍 **Confidence Levels**
- **High Confidence**: Clear hiring signals, established company, strong ML prediction
- **Medium Confidence**: Some positive indicators, moderate ML confidence  
- **Low Confidence**: Uncertain signals, conflicting data, low ML confidence

## 🔍 Technical Architecture & Technology Stack

### 🎨 **Frontend Technologies**
- **Streamlit 1.28+**: Modern web framework for data applications
  - Multi-tab interface with real-time updates
  - Interactive data tables with sorting and filtering
  - File upload/download capabilities
  - Progress bars and status indicators
  - Session state management for user workflows

### 🚀 **Backend Technologies**
- **FastAPI 0.104+**: High-performance async web framework
  - Automatic OpenAPI/Swagger documentation
  - Type hints with Pydantic models
  - Async/await support for concurrent processing
  - Built-in validation and serialization
  - CORS middleware for cross-origin requests

### 🤖 **Machine Learning Stack**
- **XGBoost**: Gradient boosting classifier for lead conversion prediction
  - 15 engineered features from lead data
  - Binary classification with probability scores
  - Feature importance analysis
  - Cross-validation and performance metrics
- **Scikit-learn**: ML preprocessing and evaluation
  - StandardScaler for feature normalization
  - Train/test splitting with stratification
  - Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- **NumPy & Pandas**: Data manipulation and numerical computing
- **Hybrid Scoring**: 70% ML predictions + 30% rule-based scoring

### 🧠 **AI & Data Processing**
- **Firecrawl API**: Advanced web scraping and content extraction
  - AI-powered content extraction from web pages
  - Structured data parsing from G2 listings
  - Hiring intent detection from company websites
  - Rate limiting and error handling
- **JSON Processing**: Structured data handling throughout pipeline
- **Data Validation**: Type checking and error recovery

### 🗄️ **Data Management**
- **Pickle**: Model serialization and persistence
- **JSON**: Configuration and data exchange format
- **CSV Export**: Structured data export for analysis
- **File-based Storage**: Feedback data and model artifacts
- **Memory Management**: Efficient data processing for large datasets

### 🐳 **DevOps & Deployment**
- **Docker**: Containerized application deployment
  - Multi-stage builds for optimized images
  - Docker Compose for service orchestration
  - Environment-specific configurations
- **Cross-platform Support**: Windows, Linux, macOS compatibility
- **Port Management**: Configurable service ports (8501 API, 8502 Dashboard)

### 📊 **Architecture Patterns**
- **Microservices**: Separate API and Dashboard services
- **Async Processing**: Non-blocking I/O operations
- **Error Handling**: Graceful degradation with fallbacks
- **Type Safety**: Full type annotations with Pydantic
- **Separation of Concerns**: Modular codebase organization

### 🔧 **Core Components**

#### ML Scoring Engine (`ml_scoring.py`)
- **XGBoost Classifier**: Binary classification model
- **Feature Engineering**: 15 numerical features extracted from lead data
- **Model Persistence**: Automatic save/load with versioning
- **Prediction Pipeline**: Feature extraction → scaling → prediction
- **Feedback Loop**: Continuous learning from user feedback

#### Rule-Based Scoring (`core.py`)  
- **Multi-factor Algorithm**: Base score + bonus factors
- **Risk Assessment**: Company credibility evaluation
- **Keyword Analysis**: Industry-specific signal detection
- **Confidence Levels**: Uncertainty quantification (low/medium/high)

#### API Layer (`ml_api.py`)
- **RESTful Endpoints**: Standard HTTP methods and status codes
- **Request Validation**: Pydantic models for type safety  
- **Response Serialization**: JSON with numpy type conversion
- **Error Handling**: Comprehensive exception management

#### Dashboard Interface (`dashboard.py`)
- **5-Tab Layout**: Lead Scoring, ML Training, Analytics, Data Management, Settings
- **Real-time Updates**: Live progress tracking during processing
- **Interactive Tables**: Sortable, filterable lead displays
- **Export Functions**: Multiple format support (CSV, JSON, TXT)

### 🌊 **Data Flow Architecture**
1. **User Input**: G2 URL via Streamlit interface
2. **API Request**: Dashboard sends POST to FastAPI backend
3. **G2 Extraction**: Firecrawl AI extracts company listings
4. **Parallel Processing**: Concurrent hiring intent analysis
5. **Feature Engineering**: Extract 15 ML features per lead
6. **Hybrid Scoring**: ML prediction (70%) + rule-based (30%)
7. **Result Aggregation**: Sort, filter, and format leads
8. **Response**: JSON with lead scores and explanations
9. **Display**: Interactive dashboard with export options
10. **Feedback**: User corrections feed back to ML model

### 🎯 **Performance Optimizations**
- **Async Operations**: Concurrent API calls to Firecrawl
- **Connection Pooling**: Efficient HTTP client management
- **Memory Efficiency**: Streaming data processing
- **Caching**: Model loading and feature computation
- **Error Recovery**: Graceful handling of API failures

## 📁 Project Structure

```
saasquatchleads/
├── 🚀 Core Application
│   ├── app.py              # Main FastAPI application
│   ├── dashboard.py        # Streamlit dashboard (5 tabs)
│   ├── start_app.py       # Application launcher
│   └── start_app.bat      # Windows launcher
│
├── 🤖 Machine Learning
│   ├── ml_scoring.py      # XGBoost ML engine & training
│   ├── ml_api.py         # ML API endpoints
│   └── models/           # Trained model storage
│
├── ⚙️ Business Logic  
│   ├── core.py           # Rule-based scoring & lead processing
│   ├── models.py         # Data models & schemas
│   └── urls.py           # URL routing & validation
│
├── 📊 Monitoring & Analytics
│   ├── monitor.py        # Performance monitoring
│   ├── log_viewer.py     # Log analysis interface
│   ├── run_monitor.py    # Monitoring runner
│   └── logs/            # Application logs
│
├── 🐳 Deployment
│   ├── docker-compose.yaml  # Service orchestration
│   ├── Dockerfile          # Container configuration
│   └── requirements.txt    # Python dependencies
│
└── 📚 Documentation
    ├── README.md           # This comprehensive guide
    └── ML_PROPOSAL.md     # ML system architecture
```

### 🔑 **Key Components**

| File | Purpose | Technology |
|------|---------|------------|
| `ml_scoring.py` | ML training & prediction engine | XGBoost, Scikit-learn |
| `core.py` | Rule-based scoring & data processing | Python, JSON validation |
| `dashboard.py` | Multi-tab web interface | Streamlit, async operations |
| `app.py` | RESTful API backend | FastAPI, Pydantic |
| `monitor.py` | Performance & error monitoring | Logging, metrics tracking |

## 🚨 Troubleshooting

### Common Issues

**Dashboard won't load:**
- Check if port 8502 is free
- Ensure API backend is running on port 8501
- Verify Docker services are up: `docker-compose ps`

**API errors:**
- Check `.env` file has valid `FIRECRAWL_API_KEY`
- Verify Firecrawl API key has sufficient credits
- Check Docker logs: `docker-compose logs api`

**No results returned:**
- Verify G2 URL is valid and accessible
- Check if Firecrawl service is operational
- Review backend logs for extraction errors

### Debug Commands
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs api
docker-compose logs dashboard

# Test API directly
curl -X POST "http://localhost:8501/score-leads" \
  -H "Content-Type: application/json" \
  -d '{"g2_url": "https://www.g2.com/categories/crm"}'

# Access API documentation
open http://localhost:8501/docs
```

## 📱 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:8502 | Main user interface |
| **API Docs** | http://localhost:8501/docs | Interactive API documentation |
| **API Endpoint** | http://localhost:8501/score-leads | Direct API access |

## 🎉 Success Metrics

A successful analysis will show:
- ✅ **Companies Extracted**: 5-10+ companies from G2 page
- ✅ **Hiring Intent Detected**: AI finds hiring signals
- ✅ **Leads Scored**: Multi-factor scoring applied
- ✅ **Results Exported**: Data available for download
- ✅ **Performance**: Analysis completed in <60 seconds

Start analyzing your leads now with the power of AI! 🚀
