import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    if 'sms' in df.columns and 'label' in df.columns:
        df = df[['sms', 'label']].rename(columns={'sms': 'message', 'label': 'label'})
    else:
        raise ValueError("Expected columns 'sms' and 'label' not found in dataset")
    df['message'] = df['message'].apply(preprocess_text)
    return df

def train_model():
    mlflow.set_experiment("spam_detection_experiment")
    df = load_data('spam.csv')

# Main training function
def train_model():
    # Set MLflow experiment
    mlflow.set_experiment("spam_detection_experiment")
    
    # Load data
    df = load_data('spam.csv')
    X = df['message']
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform GridSearchCV
    with mlflow.start_run():
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_vec, y_train)
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_accuracy", grid_search.best_score_)
        
        # Evaluate on test set
        y_pred = grid_search.predict(X_test_vec)
        test_accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        
        # Log model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
        
        # Save best model
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Accuracy: {grid_search.best_score_}")
        print(f"Test Accuracy: {test_accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()