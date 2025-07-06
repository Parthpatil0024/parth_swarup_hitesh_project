# Email Spam Detection

This project implements a machine learning model to classify SMS messages as spam or not using Natural Language Processing (NLP) techniques. It includes hyper-parameter tuning with MLflow, a Flask-based REST API, and Docker containerization for deployment. The project is developed by a team of three contributors: Parth, Swarup, and Hitesh.

## Project Structure
- `train_model.py`: Script for data preprocessing, model training (Random Forest with TF-IDF), and logging experiments to MLflow.
- `app.py`: Flask API with endpoints for best model parameters (GET), prediction (POST), and training (POST).
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Docker configuration for containerizing the API.
- `spam.csv`: Dataset from [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset).
- `vectorizer.pkl`: Serialized TF-IDF vectorizer for text preprocessing.
- `best_model.pkl`: Serialized trained Random Forest model.
- `mlruns/`: MLflow experiment logs, including parameters, metrics, and models.
- `README.md`: This file.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ml-spam-detection.git
   cd ml-spam-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Resources**:
   ```bash
   python -m nltk.downloader stopwords punkt punkt_tab
   ```

4. **Place Dataset**:
   - Download `spam.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset) and place it in the project root (`ml-spam-detection/`).

## Usage
1. **Train the Model**:
   - Run the training script to preprocess the dataset, train a Random Forest model with hyper-parameter tuning, and log results to MLflow:
     ```bash
     python train_model.py
     ```
   - Outputs: `mlruns/` folder, `vectorizer.pkl`, and `best_model.pkl`.

2. **View MLflow Experiments**:
   - Start the MLflow UI to view experiment logs:
     ```bash
     mlflow ui
     ```
   - Access at `http://localhost:5000`.

3. **Run the Flask API**:
   - Start the Flask server:
     ```bash
     python app.py
     ```
   - API Endpoints:
     - **GET `/best_model_parameters`**:
       ```bash
       curl http://localhost:5000/best_model_parameters
       ```
       Returns the best hyper-parameters from the latest MLflow run.
     - **POST `/predict`**:
       ```bash
       curl -X POST -H "Content-Type: application/json" -d '{"message":"Free offer! Click now!"}' http://localhost:5000/predict
       ```
       Returns `{"prediction":"spam"}` or `{"prediction":"ham"}`.
     - **POST `/train`**:
       ```bash
       curl -X POST http://localhost:5000/train
       ```
       Triggers model retraining and logs a new MLflow run.

## Deployment
1. **Build Docker Image**:
   ```bash
   docker build -t yourusername/spam-detection:latest .
   ```

2. **Run Docker Container Locally**:
   ```bash
   docker run -p 5000:5000 yourusername/spam-detection:latest
   ```

3. **Push to Docker Hub**:
   ```bash
   docker login
   docker push yourusername/spam-detection:latest
   ```

4. **Docker Hub Repository**:
   - Access the image at: `https://hub.docker.com/r/yourusername/spam-detection`

## Contributors
- **Parth**: Implemented data preprocessing and model training (`train_model.py`), MLflow integration.
- **Swarup**: Developed Flask API (`app.py`) and managed dependencies (`requirements.txt`).
- **Hitesh**: Created Docker configuration (`Dockerfile`) and deployment instructions.

## Notes
- Ensure `spam.csv` is in the project root with columns `sms` and `label`.
- The `mlruns/` folder contains MLflow experiment logs, visible in the GitHub repository.
- Replace `yourusername` with your Docker Hub username in Docker commands and the `README.md` link.