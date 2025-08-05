"""
Machine Learning Lead Scoring System
Uses XGBoost to predict lead conversion probability
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from dataclasses import dataclass

@dataclass
class MLPrediction:
    """Structure for ML prediction results"""
    ml_score: float  # 0-100 scale
    confidence: float  # 0-1 scale
    feature_importance: Dict[str, float]
    prediction_explanation: List[str]
    model_version: str

class LeadScoringML:
    """
    Machine Learning Lead Scoring System using XGBoost
    """
    
    def __init__(self, model_path: str = "models/lead_scoring_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_version = "1.0.0"
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def extract_features(self, lead_data: Dict) -> Dict[str, float]:
        """
        Extract numerical features from lead data for ML model
        """
        features = {}
        
        # Basic hiring intent features
        hiring_details = lead_data.get('hiring_details', {})
        scoring_details = lead_data.get('scoring_details', {})
        
        # Hiring intent features (0/1)
        features['has_hiring_intent'] = 1.0 if hiring_details.get('has_hiring_intent', False) else 0.0
        features['careers_page_exists'] = 1.0 if hiring_details.get('careers_page_exists', False) else 0.0
        
        # Confidence level encoding (0=low, 1=medium, 2=high)
        confidence_map = {'low': 0, 'medium': 1, 'high': 2}
        features['confidence_level'] = confidence_map.get(hiring_details.get('confidence_level', 'low'), 0)
        
        # Count-based features
        features['hiring_indicators_count'] = len(hiring_details.get('hiring_indicators', []))
        features['open_positions_count'] = len(hiring_details.get('open_positions', []))
        features['urgency_signals_count'] = len(hiring_details.get('urgency_signals', []))
        features['company_keywords_count'] = len(scoring_details.get('company_keywords', []))
        
        # Scoring components
        features['base_score'] = scoring_details.get('base_score', 0)
        features['bonus_score'] = scoring_details.get('bonus_score', 0)
        features['rule_based_score'] = scoring_details.get('total_score', 0)
        
        # Company characteristics
        company_name = lead_data.get('company_name', '')
        website = lead_data.get('website', '')
        
        features['company_name_length'] = len(company_name)
        features['website_credibility'] = 1.0 if website and not any(test in website.lower() for test in ['example', 'test', 'demo', 'mock']) else 0.0
        
        # Business keywords presence
        high_value_keywords = ['enterprise', 'solutions', 'platform', 'software', 'tech', 'saas', 'cloud', 'analytics', 'intelligence']
        features['has_enterprise_keywords'] = 1.0 if any(kw in company_name.lower() for kw in high_value_keywords) else 0.0
        
        # Risk factors
        risk_factors = scoring_details.get('risk_factors', [])
        features['risk_factors_count'] = len(risk_factors)
        features['has_risk_factors'] = 1.0 if len(risk_factors) > 0 else 0.0
        
        return features
    
    def prepare_training_data(self, leads_data: List[Dict], outcomes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector for training
        """
        feature_list = []
        
        for lead in leads_data:
            features = self.extract_features(lead)
            feature_list.append(features)
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(feature_list)
        
        # Store feature names
        self.feature_names = list(df.columns)
        
        # Convert to numpy arrays
        X = df.values
        y = np.array(outcomes)
        
        return X, y
    
    def train_model(self, leads_data: List[Dict], outcomes: List[int], test_size: float = 0.2):
        """
        Train XGBoost model on historical lead data
        
        Args:
            leads_data: List of lead data dictionaries
            outcomes: List of outcomes (1 for converted, 0 for not converted)
            test_size: Fraction of data to use for testing
        """
        print(f"🤖 Training ML model with {len(leads_data)} samples...")
        
        # Prepare data
        X, y = self.prepare_training_data(leads_data, outcomes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Configure XGBoost
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        print(f"📊 Model Performance:")
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.3f}")
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return metrics
    
    def predict(self, lead_data: Dict) -> MLPrediction:
        """
        Predict lead score using ML model
        """
        if not self.is_trained or self.model is None:
            # Fallback to rule-based scoring if model not available
            rule_score = lead_data.get('scoring_details', {}).get('total_score', 0)
            return MLPrediction(
                ml_score=float(rule_score),
                confidence=0.5,
                feature_importance={},
                prediction_explanation=["Using rule-based scoring (ML model not trained)"],
                model_version="rule-based"
            )
        
        # Extract features
        features = self.extract_features(lead_data)
        feature_vector = np.array([list(features.values())])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get prediction
        probability = self.model.predict_proba(feature_vector_scaled)[0][1]
        ml_score = probability * 100  # Convert to 0-100 scale
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Generate explanation
        explanation = self._generate_explanation(features, feature_importance, probability)
        
        # Convert numpy types to Python natives for JSON serialization
        return MLPrediction(
            ml_score=float(ml_score),
            confidence=float(probability),
            feature_importance={k: float(v) for k, v in feature_importance.items()},
            prediction_explanation=explanation,
            model_version=self.model_version
        )
    
    def _generate_explanation(self, features: Dict, importance: Dict, probability: float) -> List[str]:
        """
        Generate human-readable explanation for the prediction
        """
        explanations = []
        
        # Top 3 most important features
        top_features = list(importance.items())[:3]
        
        for feature_name, importance_value in top_features:
            feature_value = features.get(feature_name, 0)
            
            if feature_name == 'has_hiring_intent' and feature_value > 0:
                explanations.append(f"✅ Active hiring intent detected (importance: {importance_value:.3f})")
            elif feature_name == 'hiring_indicators_count':
                explanations.append(f"🔍 Found {int(feature_value)} hiring indicators (importance: {importance_value:.3f})")
            elif feature_name == 'careers_page_exists' and feature_value > 0:
                explanations.append(f"💼 Dedicated careers page exists (importance: {importance_value:.3f})")
            elif feature_name == 'confidence_level':
                confidence_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
                explanations.append(f"📊 Hiring confidence: {confidence_labels.get(int(feature_value), 'Unknown')} (importance: {importance_value:.3f})")
            elif feature_name == 'company_keywords_count':
                explanations.append(f"🏢 {int(feature_value)} business keywords found (importance: {importance_value:.3f})")
        
        # Overall confidence
        if probability > 0.7:
            explanations.append("🎯 High confidence prediction")
        elif probability > 0.4:
            explanations.append("🟡 Medium confidence prediction")
        else:
            explanations.append("🔴 Low confidence prediction")
        
        return explanations
    
    def save_model(self):
        """Save trained model and scaler"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_version': self.model_version,
                'is_trained': self.is_trained,
                'trained_at': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model and scaler"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.model_version = model_data.get('model_version', '1.0.0')
                self.is_trained = model_data.get('is_trained', False)
                
                print(f"📁 Model loaded from {self.model_path}")
                print(f"🏷️  Model version: {self.model_version}")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
        return False
    
    def retrain_with_feedback(self, lead_data: Dict, actual_outcome: int):
        """
        Add new feedback data and retrain model
        """
        # For now, we'll save feedback data to a file
        # In production, this would go to a database
        feedback_file = "data/feedback_data.json"
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'lead_data': lead_data,
            'outcome': actual_outcome
        }
        
        # Load existing feedback data
        feedback_data = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        
        # Add new entry
        feedback_data.append(feedback_entry)
        
        # Save updated feedback data
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        print(f"💡 Feedback recorded. Total feedback entries: {len(feedback_data)}")
        
        # Retrain if we have enough data (e.g., 50+ samples)
        if len(feedback_data) >= 50:
            print("🔄 Retraining model with new feedback data...")
            leads = [entry['lead_data'] for entry in feedback_data]
            outcomes = [entry['outcome'] for entry in feedback_data]
            return self.train_model(leads, outcomes)
        
        return None

def create_mock_training_data(num_samples: int = 200) -> Tuple[List[Dict], List[int]]:
    """
    Create mock training data for demonstration
    In production, this would come from actual historical lead data
    """
    np.random.seed(42)
    
    leads_data = []
    outcomes = []
    
    for i in range(num_samples):
        # Create mock lead data
        has_hiring = np.random.choice([True, False], p=[0.6, 0.4])
        careers_page = np.random.choice([True, False], p=[0.7 if has_hiring else 0.3, 0.3 if has_hiring else 0.7])
        confidence = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3] if has_hiring else [0.6, 0.3, 0.1])
        
        hiring_indicators = np.random.randint(0, 8 if has_hiring else 3)
        open_positions = np.random.randint(0, 5 if has_hiring else 2)
        urgency_signals = np.random.randint(0, 4 if has_hiring else 1)
        company_keywords = np.random.randint(0, 6)
        
        # Calculate rule-based score
        base_score = 40 if has_hiring else 5
        bonus_score = (company_keywords * 3 + 
                      (15 if careers_page else 0) + 
                      min(hiring_indicators * 2, 10))
        total_score = base_score + bonus_score
        
        lead_data = {
            'company_name': f"Company_{i}",
            'website': f"https://company{i}.com",
            'hiring_intent': '✅ Yes (High confidence)' if has_hiring else '❌ No',
            'score': total_score,
            'hiring_details': {
                'has_hiring_intent': has_hiring,
                'confidence_level': confidence,
                'hiring_indicators': [f'indicator_{j}' for j in range(hiring_indicators)],
                'careers_page_exists': careers_page,
                'open_positions': [f'position_{j}' for j in range(open_positions)],
                'urgency_signals': [f'signal_{j}' for j in range(urgency_signals)]
            },
            'scoring_details': {
                'total_score': total_score,
                'base_score': base_score,
                'bonus_score': bonus_score,
                'company_keywords': [f'keyword_{j}' for j in range(company_keywords)],
                'risk_factors': []
            }
        }
        
        # Simulate conversion probability based on features
        conversion_prob = (
            0.4 * (1 if has_hiring else 0) +
            0.2 * (1 if careers_page else 0) +
            0.1 * min(hiring_indicators / 5, 1) +
            0.1 * min(company_keywords / 3, 1) +
            0.1 * ({'low': 0, 'medium': 0.5, 'high': 1}[confidence]) +
            0.1 * min(total_score / 60, 1)
        )
        
        # Add some noise
        conversion_prob += np.random.normal(0, 0.1)
        conversion_prob = max(0, min(1, conversion_prob))
        
        # Generate outcome
        outcome = 1 if np.random.random() < conversion_prob else 0
        
        leads_data.append(lead_data)
        outcomes.append(outcome)
    
    print(f"📊 Generated {num_samples} mock training samples")
    print(f"   Conversion rate: {sum(outcomes)/len(outcomes):.2%}")
    
    return leads_data, outcomes

# Global ML model instance
ml_model = LeadScoringML()

if __name__ == "__main__":
    # Demo: Train model with mock data
    print("🚀 ML Lead Scoring Demo")
    
    # Generate mock training data
    leads_data, outcomes = create_mock_training_data(200)
    
    # Train model
    metrics = ml_model.train_model(leads_data, outcomes)
    
    # Test prediction on a sample lead
    sample_lead = leads_data[0]
    prediction = ml_model.predict(sample_lead)
    
    print(f"\n🎯 Sample Prediction:")
    print(f"   ML Score: {prediction.ml_score:.1f}/100")
    print(f"   Confidence: {prediction.confidence:.3f}")
    print(f"   Explanations:")
    for explanation in prediction.prediction_explanation:
        print(f"     • {explanation}")
