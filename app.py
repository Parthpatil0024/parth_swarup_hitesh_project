from flask import Flask, request, jsonify
import pickle
import mlflow
import pandas as pd
from train_model import load_data, train_and_log_model

app = Flask(__name__)

# Load the best model
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

@app.route('/best_model', methods=['GET'])
def get_best_model():
    try:
        with open('best_params.txt', 'r') as f:
            params = f.read()
        return jsonify({"best_parameters": eval(params)})
    except FileNotFoundError:
        return jsonify({"error": "Model parameters not found. Train the model first."}), 404

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 500
    data = request.get_json()
    if 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    message = data['message']
    prediction = model.predict([message])[0]
    label = 'spam' if prediction == 1 else 'ham'
    return jsonify({"message": message, "prediction": label})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)