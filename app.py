import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify
import pickle
import mlflow
from spam_detection import preprocess_text, load_data, train_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load trained model and vectorizer
try:
    model = pickle.load(open('best_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except Exception as e:
    logging.error(f"Error loading model or vectorizer: {str(e)}")
    raise

# GET endpoint for best model parameters
@app.route('/best_model_parameters', methods=['GET'])
def get_best_parameters():
    logging.debug("Received request for /best_model_parameters")
    experiment = mlflow.get_experiment_by_name("spam_detection_experiment")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs.empty:
        latest_run = runs.iloc[0]
        params = {key: value for key, value in latest_run.items() if key.startswith('params.')}
        params = {k.replace('params.', ''): v for k, v in params.items()}
        return jsonify({"best_parameters": params})
    return jsonify({"error": "No runs found"}), 404

# POST endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Received request for /predict")
    try:
        data = request.get_json(force=True)
        logging.debug(f"Received JSON: {data}")
        if 'message' not in data:
            logging.error("Missing 'message' field in JSON")
            return jsonify({"error": "Message field required"}), 400
        message = preprocess_text(data['message'])
        logging.debug(f"Preprocessed message: {message}")
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        label = 'spam' if prediction == 1 else 'ham'
        logging.debug(f"Prediction: {label}")
        return jsonify({"prediction": label})
    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# POST endpoint for training
@app.route('/train', methods=['POST'])
def train():
    logging.debug("Received request for /train")
    try:
        train_model()
        return jsonify({"status": "Training completed"})
    except Exception as e:
        logging.error(f"Error in train endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)