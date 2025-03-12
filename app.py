from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

# Load the trained model
with open("npi_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# ✅ Allow ALL origins for debugging (change "*" to a specific domain for security)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON input
        input_time = int(data["hour"])  # Extract hour from request

        # Create input DataFrame for the model
        input_data = pd.DataFrame([[input_time, 0]], columns=["Hour of Login", "Day of Week"])  # Assume Monday

        # Predict NPIs
        prediction = model.predict(input_data)

        return jsonify({"recommended_NPIs": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
