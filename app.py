from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# ✅ Path to the model file (stored in Git LFS)
MODEL_PATH = "optimized_phishing_model_fixed.pkl"

# ✅ Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found! Make sure it is in your repository.")

# ✅ Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ Dummy feature extraction function (modify this based on your actual logic)
def extract_features(url):
    return {
        "url_length": len(url),
        "https": 1 if url.startswith("https") else 0
    }

@app.route('/')
def home():
    return "Phishing Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    # Extract features
    features = extract_features(url)
    feature_df = pd.DataFrame([features])

    # Make prediction
    prediction = model.predict(feature_df)[0]
    result = "Phishing" if prediction == 1 else "Legitimate"

    return jsonify({"url": url, "prediction": result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
