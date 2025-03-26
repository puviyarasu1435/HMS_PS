from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import joblib

app = Flask(__name__)

# Enable CORS for the specific frontend URL
CORS(app, supports_credentials=True, origins=["https://vite-app-str4.onrender.com"])

# Load the trained model
filename = 'random_forest_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Initialize scaler
scaler = StandardScaler()

# Define the numerical features
numerical_features = [
    'Patient ID', 'Age', 'HeartRate', 'OxygenSaturation',
    'RespiratoryRate', 'Temperature', 'PainLevel', 'UrineOutput',
    'SystolicBP', 'DiastolicBP'
]

classification_mapping = {0: 'Need Improvement', 1: 'Healing', 2: 'Recovered'}

# Reference dataset for fitting the scaler
reference_data = pd.DataFrame({
    "Patient ID": [100, 101, 102, 103, 104],
    "Age": [30, 40, 50, 60, 70],
    "HeartRate": [70, 75, 80, 85, 90],
    "OxygenSaturation": [95, 96, 97, 98, 99],
    "RespiratoryRate": [12, 14, 16, 18, 20],
    "Temperature": [36.5, 37, 37.5, 38, 38.5],
    "PainLevel": [1, 2, 3, 4, 5],
    "UrineOutput": [1000, 1200, 1400, 1600, 1800],
    "SystolicBP": [110, 115, 120, 125, 130],
    "DiastolicBP": [70, 75, 80, 85, 90]
})

# Fit the scaler on reference data
scaler.fit(reference_data[numerical_features])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        new_patient_data = pd.DataFrame([data])

        # Apply the same scaling as reference data
        new_patient_data[numerical_features] = scaler.transform(new_patient_data[numerical_features])

        # Make a prediction
        prediction = loaded_model.predict(new_patient_data)
        print(f"Raw prediction output: {prediction}")

        # Apply threshold-based classification
        if prediction[0] < 0.5:
            discrete_prediction = 0
        elif prediction[0] < 1.5:
            discrete_prediction = 1
        else:
            discrete_prediction = 2

        predicted_class = classification_mapping.get(discrete_prediction, "Unknown")

        return jsonify({"classification": predicted_class, "raw_prediction": float(prediction[0])}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
