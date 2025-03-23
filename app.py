from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
import re
import numpy as np

app = Flask(__name__)

# ✅ Path to the model file
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

# ✅ Feature extraction function (updated with necessary features)
def extract_features(url):
    # Count digits in URL
    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    digit_letter_ratio = digit_count / max(letter_count, 1)  # Avoid division by zero

    # Count hyphens in domain
    domain_hyphens = url.count("-")

    # Extract domain and compute length
    domain = re.sub(r"https?://", "", url).split("/")[0]
    domain_length = len(domain)

    # Entropy of domain
    def calculate_entropy(s):
        prob = [s.count(c)/len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in prob)

    domain_entropy = calculate_entropy(domain)

    return {
        "url_length": len(url),
        "https": 1 if url.startswith("https") else 0,
        "digit_letter_ratio": digit_letter_ratio,
        "domain_hyphens": domain_hyphens,
        "domain_length": domain_length,
        "domain_entropy": domain_entropy
    }

@app.route('/')
def home():
    return "Phishing Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
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

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)

