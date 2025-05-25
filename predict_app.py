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

def simulate_competitor_rating(row):
    # Create competitor rating randomly between 0.5 and 5.0
    return round(np.random.uniform(0.5, 5.0), 1)

def preprocess_input(df):
    """
    Preprocess input DataFrame to match model's expected features:
    - Drop extra columns not used by model
    - Fill missing columns with defaults if necessary
    """
    expected_features = ['stars', 'reviews', 'category', 'isBestSeller', 'boughtInLastMonth', 'Price_per_Review', 'High_Rating', 'Log_Price']
    df_clean = df[expected_features].copy()
    return df_clean

@app.route("/")  # Define route for home page
def home():
    logging.info("Home endpoint accessed")  # Log when home page is accessed
    return "Flask API is running. Send POST to /predict with product features, GET /predict_batch for CSV, or GET /suggest_products for recommendations."  # Simple status message

@app.route("/predict", methods=["POST"])  # Define route to handle prediction requests
def predict():
    if model is None:
        logging.error("Prediction requested but model not loaded")  # Log error if prediction requested but model missing
        return jsonify({"error": "Model not loaded"}), 500  # Return error if model not loaded
    try:
        data = request.get_json()  # Parse JSON input from POST request
        logging.info(f"Received data for prediction: {data}")  # Log received input data

        df = pd.DataFrame([data])  # Convert JSON data to pandas DataFrame

        if "competitor_rating" not in df.columns:  # If competitor rating not included
            df["competitor_rating"] = df.apply(simulate_competitor_rating, axis=1)  # Simulate competitor rating
            logging.info(f"Simulated competitor_rating: {df['competitor_rating'].values[0]}")  # Log simulated rating

        df_processed = preprocess_input(df)  # Preprocess to match model input
        prediction = model.predict(df_processed)  # Predict demand using the model
        logging.info(f"Prediction result: {prediction[0]}")  # Log prediction

        optimized_price = df["price"].values[0] * 1.05 if prediction[0] > df["demand"].values[0] else df["price"].values[0] * 0.95

        return jsonify({
            "predicted_demand": float(prediction[0]),
            "optimized_price": round(optimized_price, 2),
            "competitor_rating_used": df["competitor_rating"].values[0]
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")  # Log any errors
        return jsonify({"error": str(e)}), 400  # Return error response

@app.route("/predict_batch", methods=["GET"])  # Route for batch predictions
def predict_batch():
    if model is None or dataset_df is None:
        logging.error("Batch prediction requested but model or dataset not loaded")  # Log if resources are missing
        return jsonify({"error": "Model or dataset not loaded"}), 500

    try:
        df = dataset_df.copy()

        if "competitor_rating" not in df.columns:
            df["competitor_rating"] = df.apply(simulate_competitor_rating, axis=1)  # Simulate competitor rating

        df_processed = preprocess_input(df)
        predictions = model.predict(df_processed)
        df["predicted_demand"] = predictions

        df["optimized_price"] = df.apply(
            lambda row: row["price"] * 1.05 if row["predicted_demand"] > row["demand"] else row["price"] * 0.95,
            axis=1
        )
        df["optimized_price"] = df["optimized_price"].round(2)
        df["competitor_rating"] = df["competitor_rating"].round(1)

        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        logging.info("Batch prediction completed and CSV prepared for download")

        return send_file(output, mimetype='text/csv', download_name="batch_predictions.csv", as_attachment=True)
    except Exception as e:
        logging.error(f"Error during batch prediction: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/suggest_products", methods=["GET"])  # New endpoint to suggest top 5 products
def suggest_products():
    if dataset_df is None:
        logging.warning("Recommendation requested but dataset not loaded")
        return jsonify({"error": "Dataset not loaded"}), 500

    try:
        top_products = dataset_df.sort_values(by=["stars", "reviews"], ascending=[False, False]).head(5)
        suggestions = top_products[["title", "stars", "reviews", "price"]].to_dict(orient="records")

        logging.info(f"Suggested top {len(suggestions)} products based on rating and reviews")

        return jsonify({"suggested_products": suggestions})
    except Exception as e:
        logging.error(f"Error during product recommendation: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)  # Run app in debug mode
