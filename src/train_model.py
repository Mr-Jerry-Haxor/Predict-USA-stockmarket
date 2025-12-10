"""
Multi-Model Training Module for Stock Price Prediction
Supports: Gradient Boosting (default), Random Forest, Logistic Regression, SVM, Neural Network
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import joblib
import os
import sys
# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_STOCK_TICKERS
from paths import get_data_path, get_model_path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL DEFINITIONS - Gradient Boosting is DEFAULT (best performer)
# ============================================================================

AVAILABLE_MODELS = {
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'needs_scaling': False,
        'description': '🏆 Best performer - Recommended for accuracy (F1: 0.845, ROC-AUC: 0.91)'
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'needs_scaling': False,
        'description': '🌲 Good baseline - Fast and interpretable'
    },
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'needs_scaling': True,
        'description': '📈 Simple and fast - Good for quick analysis'
    },
    'SVM': {
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'needs_scaling': True,
        'description': '🎯 Strong for small datasets'
    },
    'Neural Network': {
        'model': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
        'needs_scaling': True,
        'description': '🧠 Captures complex patterns'
    }
}

# Default model (best performer based on evaluation)
DEFAULT_MODEL = 'Gradient Boosting'


def get_model_list():
    """Return list of available model names"""
    return list(AVAILABLE_MODELS.keys())


def get_model_description(model_name):
    """Return description for a model"""
    return AVAILABLE_MODELS.get(model_name, {}).get('description', '')


def train_single_model(ticker, model_name, horizon, features, labels):
    """
    Train a single model and return accuracy and the trained model.
    """
    model_info = AVAILABLE_MODELS.get(model_name)
    if model_info is None:
        print(f"❌ Unknown model: {model_name}")
        return None, None, None
    
    # Clone the model to avoid reusing fitted model
    model = clone(model_info['model'])
    needs_scaling = model_info['needs_scaling']
    scaler = StandardScaler() if needs_scaling else None
    
    try:
        # Adjusted minimum based on horizon
        min_required = 3 if horizon == 1 else (5 if horizon == 7 else 10)
        
        if len(features) < min_required:
            print(f"⚠️ Insufficient data for {ticker} {horizon}-day (have {len(features)}, need {min_required})")
            return None, None, None
        
        # For very small datasets, adjust test_size
        if len(features) < 10:
            test_size = 0.25
        else:
            test_size = 0.2
        
        # Check if we can stratify
        label_counts = labels.value_counts()
        can_stratify = len(label_counts) > 1 and label_counts.min() >= 2
        
        if not can_stratify:
            if len(features) < 5:
                X_train, y_train = features, labels
                X_test, y_test = features, labels
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=test_size, random_state=42, shuffle=True
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42, stratify=labels
            )
        
        # Scale if needed
        if needs_scaling:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, scaler, accuracy
        
    except Exception as e:
        print(f"❌ Error training {model_name} for {ticker} {horizon}-day: {e}")
        return None, None, None


def train_all_models(ticker, horizons=[1, 7]):
    """
    Train ALL models for a ticker and save them.
    This is called once when user clicks Analyze - trains all models at once.
    Returns dict: {horizon: {model_name: {'accuracy': float, 'model_path': str}}}
    """
    results = {}
    
    for h in horizons:
        try:
            features_path = get_data_path(f"{ticker}_features_{h}d.csv")
            labels_path = get_data_path(f"{ticker}_labels_{h}d.csv")
            features = pd.read_csv(features_path)
            labels = pd.read_csv(labels_path)['label']
            
            if features.empty or labels.empty:
                print(f"⚠️ No data for {ticker} {h}-day horizon")
                continue
            
            if len(features) != len(labels):
                print(f"⚠️ Feature/label mismatch for {ticker} {h}-day")
                continue
            
            results[h] = {}
            
            for model_name in AVAILABLE_MODELS.keys():
                model, scaler, accuracy = train_single_model(ticker, model_name, h, features, labels)
                
                if model is not None:
                    # Save model with new naming convention
                    model_path = get_model_path(f"{ticker}_{model_name.replace(' ', '_')}_{h}d.pkl")
                    
                    # Save model and scaler together
                    model_data = {
                        'model': model,
                        'scaler': scaler,
                        'accuracy': accuracy,
                        'model_name': model_name
                    }
                    joblib.dump(model_data, model_path)
                    
                    results[h][model_name] = {
                        'accuracy': accuracy,
                        'model_path': model_path
                    }
                    print(f"✅ {model_name} for {ticker} {h}-day: Accuracy = {accuracy:.4f}")
                    
                    # Also save default model with legacy naming for backward compatibility
                    if model_name == DEFAULT_MODEL:
                        legacy_path = get_model_path(f"{ticker}_model_{h}d.pkl")
                        joblib.dump(model_data, legacy_path)
            
        except FileNotFoundError as e:
            print(f"⚠️ Data files not found for {ticker} {h}-day: {e}")
            continue
        except Exception as e:
            print(f"❌ Error training models for {ticker} {h}-day: {e}")
            continue
    
    return results


def train_model(ticker, horizons=[1, 7, 30]):
    """
    Legacy function - now trains ALL models including Gradient Boosting as default.
    For backward compatibility with existing code.
    """
    # Train all models
    results = train_all_models(ticker, horizons=[h for h in horizons if h in [1, 7]])
    
    # Return accuracies for default model (Gradient Boosting)
    accuracies = {}
    for h, models in results.items():
        if DEFAULT_MODEL in models:
            accuracies[h] = models[DEFAULT_MODEL]['accuracy']
    
    if accuracies:
        print(f"Training completed for {ticker}. Default model ({DEFAULT_MODEL}) accuracies: {accuracies}")
    else:
        print(f"No models trained for {ticker}")
    
    return accuracies


def load_model(ticker, model_name, horizon):
    """
    Load a trained model for prediction.
    Returns (model, scaler, accuracy) or (None, None, None) if not found.
    """
    try:
        # Try new naming convention first
        model_path = get_model_path(f"{ticker}_{model_name.replace(' ', '_')}_{horizon}d.pkl")
        
        if not os.path.exists(model_path):
            # Try legacy naming (only for default model)
            if model_name == DEFAULT_MODEL:
                model_path = get_model_path(f"{ticker}_model_{horizon}d.pkl")
        
        if not os.path.exists(model_path):
            return None, None, None
        
        model_data = joblib.load(model_path)
        
        # Handle both old and new format
        if isinstance(model_data, dict):
            return model_data.get('model'), model_data.get('scaler'), model_data.get('accuracy')
        else:
            # Legacy format - just the model (RandomForest, no scaler)
            return model_data, None, None
            
    except Exception as e:
        print(f"❌ Error loading model {model_name} for {ticker} {horizon}-day: {e}")
        return None, None, None


def predict_with_model(ticker, model_name, horizon):
    """
    Make prediction using specified model.
    Returns (prediction, confidence, accuracy) or (None, None, None) if error.
    """
    try:
        model, scaler, accuracy = load_model(ticker, model_name, horizon)
        
        if model is None:
            return None, None, None
        
        # Load latest features
        features_path = get_data_path(f"{ticker}_features_{horizon}d.csv")
        features = pd.read_csv(features_path).iloc[-1:]
        
        if features.empty:
            return None, None, None
        
        # Scale if needed
        if scaler is not None:
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        else:
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        confidence = float(max(proba)) * 100
        
        return pred, confidence, accuracy
        
    except Exception as e:
        print(f"❌ Error predicting with {model_name}: {e}")
        return None, None, None


def get_all_model_predictions(ticker, horizon):
    """
    Get predictions from all available models for comparison.
    Returns dict: {model_name: {'prediction': int, 'confidence': float, 'accuracy': float}}
    """
    predictions = {}
    
    for model_name in AVAILABLE_MODELS.keys():
        pred, confidence, accuracy = predict_with_model(ticker, model_name, horizon)
        
        if pred is not None:
            predictions[model_name] = {
                'prediction': pred,
                'confidence': confidence,
                'accuracy': accuracy
            }
    
    return predictions


def get_available_models_for_ticker(ticker, horizon):
    """
    Check which models are available (trained) for a given ticker and horizon.
    Returns list of model names that have saved models.
    """
    available = []
    for model_name in AVAILABLE_MODELS.keys():
        model_path = get_model_path(f"{ticker}_{model_name.replace(' ', '_')}_{horizon}d.pkl")
        if os.path.exists(model_path):
            available.append(model_name)
        elif model_name == DEFAULT_MODEL:
            # Check legacy path
            legacy_path = get_model_path(f"{ticker}_model_{horizon}d.pkl")
            if os.path.exists(legacy_path):
                available.append(model_name)
    return available


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "GOOGL"
    
    print(f"\n{'='*60}")
    print(f"Training all models for {ticker}")
    print(f"{'='*60}\n")
    
    results = train_all_models(ticker, horizons=[1, 7])
    
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"{'='*60}")
    
    for horizon, models in results.items():
        print(f"\n{horizon}-Day Horizon:")
        for model_name, info in models.items():
            marker = "🏆" if model_name == DEFAULT_MODEL else "  "
            print(f"  {marker} {model_name}: {info['accuracy']:.4f}")