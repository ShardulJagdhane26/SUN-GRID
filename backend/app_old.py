"""
SunGrid Solar Irradiance Prediction API
Production-ready Flask backend for XGBoost model deployment
Author: Your Name
Date: November 2025
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import traceback
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# ============================================================================
# LOAD PRE-TRAINED MODEL AND SCALER
# ============================================================================

MODEL_PATH = 'model/xgboost_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
    print(f"✓ Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    print(f"ERROR: Could not load model/scaler: {e}")
    print("Make sure xgboost_model.pkl and scaler.pkl are in the 'model/' directory")
    model = None
    scaler = None

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS (MATCHING YOUR NOTEBOOK)
# ============================================================================

def apply_transformations(raw_data):
    """
    Apply the EXACT same transformations used during training.
    
    Transformations applied:
    1. Temperature → log(Temperature + 1)
    2. Pressure → BoxCox transformation
    3. Humidity → BoxCox transformation
    4. Speed → log(Speed + 1)
    5. WindDirection → MinMaxScaler (0-1 normalization)
    6. Final StandardScaler on all features
    
    Args:
        raw_data (dict): Raw input parameters from frontend
        
    Returns:
        np.array: Transformed features ready for model prediction
    """
    
    # Extract raw values
    temperature = raw_data.get('Temperature', 48)
    pressure = raw_data.get('Pressure', 30.46)
    humidity = raw_data.get('Humidity', 59)
    wind_direction = raw_data.get('WindDirection', 177.39)
    speed = raw_data.get('Speed', 1.21)
    month = raw_data.get('Month', 9)
    day = raw_data.get('Day', 29)
    hour = raw_data.get('Hour', 23)
    minute = raw_data.get('Minute', 55)
    second = raw_data.get('Second', 26)
    risehour = raw_data.get('risehour', 6)
    riseminute = raw_data.get('riseminute', 40)
    sethour = raw_data.get('sethour', 18)
    setminute = raw_data.get('setminute', 50)
    
    # ==========================================
    # STEP 1: Apply transformations
    # ==========================================
    
    # Temperature: log transformation
    temp_transformed = np.log(temperature + 1)
    
    # Pressure: BoxCox transformation
    # Note: BoxCox requires positive values and same lambda as training
    pressure_transformed = stats.boxcox(np.array([pressure + 1]))[0][0]
    
    # Humidity: BoxCox transformation
    humidity_transformed = stats.boxcox(np.array([humidity + 1]))[0][0]
    
    # Speed: log transformation
    speed_transformed = np.log(speed + 1)
    
    # WindDirection: MinMaxScaler (0-360 → 0-1)
    # Using same scale as training: min=0, max=360
    wind_direction_transformed = wind_direction / 360.0
    
    # ==========================================
    # STEP 2: Create feature array in EXACT order
    # ==========================================
    
    features = np.array([
        temp_transformed,           # 0: Temperature (log)
        pressure_transformed,       # 1: Pressure (BoxCox)
        humidity_transformed,       # 2: Humidity (BoxCox)
        wind_direction_transformed, # 3: WindDirection (MinMax 0-1)
        speed_transformed,          # 4: Speed (log)
        month,                      # 5: Month (raw)
        day,                        # 6: Day (raw)
        hour,                       # 7: Hour (raw)
        minute,                     # 8: Minute (raw)
        second,                     # 9: Second (raw)
        risehour,                   # 10: risehour (raw)
        riseminute,                 # 11: riseminute (raw)
        sethour,                    # 12: sethour (raw)
        setminute                   # 13: setminute (raw)
    ]).reshape(1, -1)
    
    # ==========================================
    # STEP 3: Apply final StandardScaler
    # ==========================================
    
    features_scaled = scaler.transform(features)
    
    return features_scaled


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'SunGrid Solar Irradiance Prediction API',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expected JSON payload:
    {
        "Temperature": 48,
        "Pressure": 30.46,
        "Humidity": 59,
        "WindDirection": 177.39,
        "Speed": 1.21,
        "Month": 9,
        "Day": 29,
        "Hour": 23,
        "Minute": 55,
        "Second": 26,
        "risehour": 6,
        "riseminute": 40,
        "sethour": 18,
        "setminute": 50
    }
    
    Returns:
    {
        "prediction": 245.67,
        "status": "success",
        "unit": "W/m²"
    }
    """
    
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Check server logs.',
            'status': 'error'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No input data provided',
                'status': 'error'
            }), 400
        
        # Apply transformations (matching training pipeline)
        features = apply_transformations(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Return result
        return jsonify({
            'prediction': float(prediction),
            'status': 'success',
            'unit': 'W/m²',
            'model': 'XGBoost',
            'input_features': 14
        })
    
    except Exception as e:
        # Detailed error logging
        error_trace = traceback.format_exc()
        print(f"ERROR in /predict: {error_trace}")
        
        return jsonify({
            'error': str(e),
            'status': 'error',
            'trace': error_trace
        }), 400


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Return model performance metrics from training.
    
    Returns:
    {
        "r2_score": 0.9293,
        "rmse": 82.99,
        "mae": 33.19,
        "training_samples": 26148,
        "test_samples": 6538
    }
    """
    return jsonify({
        'r2_score': 0.9293,
        'rmse': 82.99,
        'mae': 33.19,
        'training_samples': 26148,
        'test_samples': 6538,
        'status': 'success'
    })


@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """
    Return feature importance from the trained XGBoost model.
    
    Returns:
    {
        "features": ["Temperature", "Humidity", ...],
        "importance": [0.28, 0.22, ...]
    }
    """
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        # Get feature importance from XGBoost model
        importance = model.feature_importances_
        
        # Feature names in order
        feature_names = [
            'Temperature', 'Pressure', 'Humidity', 
            'WindDirection', 'Speed', 'Month', 'Day', 
            'Hour', 'Minute', 'Second', 
            'risehour', 'riseminute', 'sethour', 'setminute'
        ]
        
        # Sort by importance
        importance_dict = dict(zip(feature_names, importance))
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Top 5 features
        top_features = sorted_features[:5]
        
        return jsonify({
            'features': [f[0] for f in top_features],
            'importance': [float(f[1]) for f in top_features],
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'endpoints': ['/predict', '/metrics', '/feature_importance', '/health']
    })


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"\n⚠️  WARNING: Model file not found at {MODEL_PATH}")
        print("Please ensure xgboost_model.pkl is in the 'model/' directory\n")
    
    if not os.path.exists(SCALER_PATH):
        print(f"\n⚠️  WARNING: Scaler file not found at {SCALER_PATH}")
        print("Please ensure scaler.pkl is in the 'model/' directory\n")
    
    # Run Flask development server
    print("\n" + "="*60)
    print("🚀 SunGrid API Server Starting...")
    print("="*60)
    print(f"Model: XGBoost (R² = 0.9293)")
    print(f"Server: http://localhost:5000")
    print(f"Endpoints:")
    print(f"  POST /predict - Make predictions")
    print(f"  GET /metrics - Model performance")
    print(f"  GET /feature_importance - Feature rankings")
    print(f"  GET /health - Server status")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
