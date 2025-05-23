from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("/Users/ifeomaigbokwe/Desktop/NEXFORD MSC/BAN 6800/customer_segmentation-dataset/milestone 2/xgb_model.joblib")  # Ensure this path is correct

@app.route("/")
def home():
    return "Product Pricing Optimization Model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)
        
        return jsonify({
            "prediction": prediction.tolist()[0]
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
