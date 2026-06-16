"""
Stock Price Prediction - Flask Web App
=======================================
Web interface to make stock price predictions using the trained model
- Load trained model from saved_model.pkl
- Accept stock price data input
- Make predictions
- Display results
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Global variables to store model and scaler
model_dict = None
best_model = None
scaler = None
results = None


def load_model():
    global model_dict, best_model, scaler, results
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'saved_model.pkl')
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            
        best_model = model_dict['best_model']
        scaler = model_dict['scaler']
        results = model_dict['results']
        print(f"✅ Model loaded! Best model: {best_model}")
        return True
    else:
        print(f"❌ Model not found at: {model_path}")
        return False


@app.route('/')
def home():
    """Home page with prediction form"""
    if model_dict is None:
        return "Error: Model not loaded. Run train.py first.", 500
    
    return render_template('index.html', best_model=best_model, results=results)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    
    Expects JSON with stock data features:
    {
        'ma_5': float,
        'ma_20': float,
        'ma_50': float,
        'price_lag_1': float,
        'price_lag_5': float,
        'price_lag_10': float,
        'daily_return': float,
        'volume_norm': float
    }
    """
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = np.array([
            data['ma_5'],
            data['ma_20'],
            data['ma_50'],
            data['price_lag_1'],
            data['price_lag_5'],
            data['price_lag_10'],
            data['daily_return'],
            data['volume_norm']
        ]).reshape(1, -1)
        
        # Scale features using the trained scaler
        features_scaled = scaler.transform(features)
        
        # Make prediction based on best model
        if best_model == 'Linear Regression':
            prediction = model_dict['linear_regression'].predict(features_scaled)[0]
        elif best_model == 'Random Forest':
            prediction = model_dict['random_forest'].predict(features_scaled)[0]
        elif best_model == 'LSTM':
            # Reshape for LSTM (add sequence dimension)
            prediction = model_dict['lstm'].predict(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'model_used': best_model,
            'message': f'Predicted S&P 500 price: ${prediction:.2f}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/models', methods=['GET'])
def get_models():
    """Get model comparison results"""
    if results is None:
        return jsonify({'error': 'Models not trained yet'}), 500
    
    return jsonify({
        'best_model': best_model,
        'results': results
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_dict is not None,
        'best_model': best_model
    })


if __name__ == '__main__':
    print("="*60)
    print("STOCK PRICE PREDICTION - FLASK APP")
    print("="*60)
    
    # Load model before starting server
    if load_model():
        print("\n🚀 Starting Flask server...")
        print("📍 Open browser: http://localhost:5000")
        print("="*60)
        
        # Run Flask app
        # debug=True: auto-reload on code changes
        # host='0.0.0.0': accessible from other machines
        if __name__ == '__main__':
            if load_model():
                app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
            else:
                print("❌ Cannot start app without trained model!")
                print("Run: python train.py")
    else:
        print("\n❌ Cannot start app without trained model!")
        print("Run: python train.py")