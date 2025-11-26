"""
SunGrid Solar Irradiance Prediction API with AI Analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from scipy import stats
import traceback
import os
import requests

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'model/xgboost_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

# Your Gemini API Key
GEMINI_API_KEY = "AIzaSyDyD0VJqK4whjuJFiUaE6NBAhO_rJloyC4"

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"ERROR: {e}")
    model = None
    scaler = None

# ============================================================================
# TRANSFORMATION FUNCTION
# ============================================================================

def apply_transformations(raw_data):
    """Apply feature transformations"""
    
    temperature = float(raw_data.get('Temperature', 48))
    pressure = float(raw_data.get('Pressure', 30.46))
    humidity = float(raw_data.get('Humidity', 59))
    wind_direction = float(raw_data.get('WindDirection', 177.39))
    speed = float(raw_data.get('Speed', 1.21))
    month = int(raw_data.get('Month', 9))
    day = int(raw_data.get('Day', 29))
    hour = int(raw_data.get('Hour', 23))
    minute = int(raw_data.get('Minute', 55))
    second = int(raw_data.get('Second', 26))
    risehour = int(raw_data.get('risehour', 6))
    riseminute = int(raw_data.get('riseminute', 40))
    sethour = int(raw_data.get('sethour', 18))
    setminute = int(raw_data.get('setminute', 50))
    
    temp_transformed = np.log(temperature + 1)
    pressure_transformed = np.log(pressure + 1)
    humidity_transformed = np.log(humidity + 1)
    speed_transformed = np.log(speed + 1)
    wind_direction_transformed = wind_direction / 360.0
    
    features = np.array([
        temp_transformed, pressure_transformed, humidity_transformed,
        wind_direction_transformed, speed_transformed,
        month, day, hour, minute, second,
        risehour, riseminute, sethour, setminute
    ]).reshape(1, -1)
    
    return scaler.transform(features)

# ============================================================================
# AI ANALYSIS FUNCTION - REST API VERSION (WORKS RELIABLY)
# ============================================================================

def generate_ai_analysis(inputs, irradiance):
    """Generate AI analysis using Gemini REST API"""
    
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return "AI analysis unavailable. Configure GEMINI_API_KEY"
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        prompt = f"""Act as a senior solar energy engineer. Analyze these weather conditions and predicted solar irradiance.

Input Data:
- Temperature: {inputs.get('Temperature')}°F
- Pressure: {inputs.get('Pressure')} inHg
- Humidity: {inputs.get('Humidity')}%
- Wind Direction: {inputs.get('WindDirection')}°
- Wind Speed: {inputs.get('Speed')} mph
- Time: {inputs.get('Hour')}:{str(inputs.get('Minute')).zfill(2)}
- Month: {inputs.get('Month')}

Predicted Solar Irradiance: {irradiance:.2f} W/m²

Provide 3-4 sentences analyzing:
1. Are conditions optimal for solar generation?
2. How do humidity and wind affect efficiency?
3. Grid operator recommendations?

Professional tone, no markdown."""

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                text = data['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
        else:
            print(f"⚠️  API Error {response.status_code}: {response.text}")
        
        return f"Prediction: {irradiance:.2f} W/m² under current conditions."
    
    except Exception as e:
        print(f"⚠️  Gemini Error: {e}")
        return f"AI analysis unavailable. Prediction: {irradiance:.2f} W/m²."

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'SunGrid Solar Irradiance Prediction API',
        'model_loaded': model is not None,
        'ai_enabled': GEMINI_API_KEY != "YOUR_API_KEY_HERE",
        'version': '2.1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with AI analysis"""
    
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data', 'status': 'error'}), 400
        
        # Prediction
        features = apply_transformations(data)
        prediction = model.predict(features)[0]
        
        # AI Analysis
        ai_analysis = generate_ai_analysis(data, prediction)
        
        return jsonify({
            'prediction': float(prediction),
            'ai_analysis': ai_analysis,
            'status': 'success',
            'unit': 'W/m²',
            'model': 'XGBoost'
        })
    
    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'ai_enabled': GEMINI_API_KEY != "YOUR_API_KEY_HERE",
        'version': '2.1.0'
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 SunGrid API Server v2.1.0 - AI Analysis (REST API)")
    print("="*70)
    print(f"✓ Server starting at http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
