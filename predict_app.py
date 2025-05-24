from flask import Flask, request, jsonify, send_file  # Flask: web framework, request to get data, jsonify to send JSON response, send_file to allow CSV download
import joblib  # joblib: load/save trained ML models efficiently
import pandas as pd  # pandas: work with tabular data easily
import numpy as np  # numpy: numerical operations, random numbers, arrays
import os  # os: interact with file system and paths safely
import logging  # logging: keep track of app events and errors for debugging
from io import BytesIO  # io: handle in-memory files for CSV download

app = Flask(__name__)  # Create Flask app instance

# Setup logging to show time, level, and message, and show info or above messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.joblib")  # Safe path to model file

# Path to cleaned dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "cleaned_amazon_15000.csv") 

try:
    model = joblib.load(MODEL_PATH)  # Load trained ML model from file
    logging.info(f"Model loaded successfully from {MODEL_PATH}")  # Log success of model loading
except FileNotFoundError:
    model = None  # If model missing, set to None so app doesn't crash
    logging.error(f"Model not found at {MODEL_PATH}")  # Log error if model file is missing

try:
    dataset_df = pd.read_csv(DATASET_PATH)  # Load dataset at startup
    logging.info(f"Cleaned Amazon dataset loaded with {len(dataset_df)} records from {DATASET_PATH}")  # Log dataset load
except FileNotFoundError:
    dataset_df = None
    logging.warning("Cleaned dataset not found at provided path")  # Warn if file not found

def simulate_competitor_price(row):
    # Create competitor price randomly between 90% and 110% of input price
    return row["price"] * np.random.uniform(0.9, 1.1)

def preprocess_input(df):
    """
    Preprocess input DataFrame to match model's expected features:
    - Drop extra columns not used by model
    - Fill missing columns with defaults if necessary
    """
    expected_features = ['stars', 'reviews', 'category', 'isBestSeller', 'boughtInLastMonth', 'Price_per_Review', 'High_Rating', 'Log_Price']
    # Keep only expected columns (drop extras like uid, asin, title, price, competitor_price if present)
    df_clean = df[expected_features].copy()
    return df_clean

@app.route("/")  # Define route for home page
def home():
    logging.info("Home endpoint accessed")  # Log when home page is accessed
    return "Flask API is running. Send POST to /predict with product features or GET /predict_batch to download batch predictions."  # Simple status message

@app.route("/predict", methods=["POST"])  # Define route to handle prediction requests
def predict():
    if model is None:
        logging.error("Prediction requested but model not loaded")  # Log error if prediction requested but model missing
        return jsonify({"error": "Model not loaded"}), 500  # Return error if model not loaded
    try:
        data = request.get_json()  # Parse JSON input from POST request
        logging.info(f"Received data for prediction: {data}")  # Log received input data

        df = pd.DataFrame([data])  # Convert JSON data to pandas DataFrame (table)

        if "competitor_price" not in df.columns:  # If competitor price not included
            df["competitor_price"] = df.apply(simulate_competitor_price, axis=1)  # Simulate competitor price
            logging.info(f"Simulated competitor_price: {df['competitor_price'].values[0]}")  # Log simulated competitor price

        # Preprocess to match model input
        df_processed = preprocess_input(df)

        prediction = model.predict(df_processed)  # Predict demand using the loaded model
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

@app.route("/predict_batch", methods=["GET"])  # Define route to handle batch predictions using preloaded dataset
def predict_batch():
    if model is None or dataset_df is None:
        logging.error("Batch prediction requested but model or dataset not loaded")  # Log if resources are missing
        return jsonify({"error": "Model or dataset not loaded"}), 500  # Return error if missing

    try:
        df = dataset_df.copy()  # Work on a copy to avoid altering original data

        if "competitor_price" not in df.columns:  # If competitor price not present
            df["competitor_price"] = df.apply(simulate_competitor_price, axis=1)  # Simulate competitor price

        # Preprocess dataset to match model input features
        df_processed = preprocess_input(df)

        predictions = model.predict(df_processed)  # Predict demand for all rows
        df["predicted_demand"] = predictions  # Add predictions to DataFrame

        # Calculate optimized price for each row based on predicted demand
        df["optimized_price"] = df.apply(
            lambda row: row["price"] * 1.05 if row["predicted_demand"] > row["demand"] else row["price"] * 0.95,
            axis=1
        )
        df["optimized_price"] = df["optimized_price"].round(2)
        df["competitor_price"] = df["competitor_price"].round(2)

        output = BytesIO()  # Create in-memory output stream for CSV
        df.to_csv(output, index=False)  # Write DataFrame to CSV in memory
        output.seek(0)  # Reset stream position

        logging.info("Batch prediction completed and CSV prepared for download")  # Log completion

        return send_file(output, mimetype='text/csv', download_name="batch_predictions.csv", as_attachment=True)  # Send CSV file
    except Exception as e:
        logging.error(f"Error during batch prediction: {e}")  # Log errors
        return jsonify({"error": str(e)}), 400  # Return error message if something goes wrong

if __name__ == "__main__":
    app.run(debug=True)  # Run app in debug mode for development (auto-reloads if code changes)
