import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model path from environment variable or use default relative path
MODEL_PATH = os.getenv("MODEL_PATH", "xgb_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    app.logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    model = None

@app.route("/")
def home():
    return "Hello from Flask on Render! Product Pricing Optimization Model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}")

        # Check if data is a dict (single record) or list (batch)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Expecting list of dicts
            if all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            else:
                return jsonify({"error": "All items in the list must be JSON objects"}), 400
        else:
            return jsonify({"error": "Input data must be a JSON object or a list of JSON objects"}), 400

        prediction = model.predict(df)

        # If single record, return single prediction, else list
        if len(prediction) == 1:
            result = prediction.tolist()[0]
        else:
            result = prediction.tolist()

        return jsonify({"prediction": result})

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Use environment variable FLASK_DEBUG or default to False
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
