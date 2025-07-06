from flask import Flask, request, jsonify
import pickle
import mlflow
from train_model import preprocess_text, load_data, train_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open('best_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# GET endpoint for best model parameters
@app.route('/best_model_parameters', methods=['GET'])
def get_best_parameters():
    # Retrieve the latest MLflow run
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
    data = request.get_json()
    if 'message' not in data:
        return jsonify({"error": "Message field required"}), 400
    
    message = preprocess_text(data['message'])
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    label = 'spam' if prediction == 1 else 'ham'
    return jsonify({"prediction": label})

# POST endpoint for training
@app.route('/train', methods=['POST'])
def train():
    try:
        train_model()
        return jsonify({"status": "Training completed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)