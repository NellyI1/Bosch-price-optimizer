<<<<<<< HEAD
import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
=======
from flask import Flask, request, jsonify  # Flask: web framework, request to get data, jsonify to send JSON response
import joblib  # joblib: load/save trained ML models efficiently
import pandas as pd  # pandas: work with tabular data easily
import numpy as np  # numpy: numerical operations, random numbers, arrays
import os  # os: interact with file system and paths safely
import logging  # logging: keep track of app events and errors for debugging
>>>>>>> b0956ec (Updated predict_app.py with logging)

app = Flask(__name__)  # Create Flask app instance

<<<<<<< HEAD
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
=======
# Setup logging to show time, level, and message, and show info or above messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
>>>>>>> b0956ec (Updated predict_app.py with logging)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.joblib")  # Safe path to model file

try:
    model = joblib.load(MODEL_PATH)  # Load trained ML model from file
    logging.info(f"Model loaded successfully from {MODEL_PATH}")  # Log success of model loading
except FileNotFoundError:
    model = None  # If model missing, set to None so app doesn't crash
    logging.error(f"Model not found at {MODEL_PATH}")  # Log error if model file is missing

def simulate_competitor_price(row):
    # Create competitor price randomly between 90% and 110% of input price
    return row["price"] * np.random.uniform(0.9, 1.1)

@app.route("/")  # Define route for home page
def home():
<<<<<<< HEAD
    return "Hello from Flask on Render! Product Pricing Optimization Model is running!"
=======
    logging.info("Home endpoint accessed")  # Log when home page is accessed
    return "Flask API is running. Send POST to /predict with product features."  # Simple status message
>>>>>>> b0956ec (Updated predict_app.py with logging)

@app.route("/predict", methods=["POST"])  # Define route to handle prediction requests
def predict():
    if model is None:
<<<<<<< HEAD
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
=======
        logging.error("Prediction requested but model not loaded")  # Log error if prediction requested but model missing
        return jsonify({"error": "Model not loaded"}), 500  # Return error if model not loaded
    try:
        data = request.get_json()  # Parse JSON input from POST request
        logging.info(f"Received data for prediction: {data}")  # Log received input data

        df = pd.DataFrame([data])  # Convert JSON data to pandas DataFrame (table)

        if "competitor_price" not in df.columns:  # If competitor price not included
            df["competitor_price"] = df.apply(simulate_competitor_price, axis=1)  # Simulate competitor price
            logging.info(f"Simulated competitor_price: {df['competitor_price'].values[0]}")  # Log simulated competitor price

        prediction = model.predict(df)  # Predict demand using the loaded model
        logging.info(f"Prediction result: {prediction[0]}")  # Log prediction output

        # Simple price optimization: increase price by 5% if predicted demand > current demand; else decrease by 5%
        optimized_price = df["price"].values[0] * 1.05 if prediction[0] > df["demand"].values[0] else df["price"].values[0] * 0.95

        # Return JSON response with predicted demand, optimized price, and competitor price used
        return jsonify({
            "predicted_demand": float(prediction[0]),
            "optimized_price": round(optimized_price, 2),
            "competitor_price_used": round(df["competitor_price"].values[0], 2)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")  # Log any errors during prediction
        return jsonify({"error": str(e)}), 400  # Return error message if something goes wrong

if __name__ == "__main__":
    app.run(debug=True)  # Run app in debug mode for development (auto-reloads if code changes)
>>>>>>> b0956ec (Updated predict_app.py with logging)
