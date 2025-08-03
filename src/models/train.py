import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow_auth():
    """Setup MLflow authentication"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
    mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')

    mlflow.set_tracking_uri(mlflow_uri)

    if mlflow_username and mlflow_password:
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    else:
        print("Warning: No MLflow authentication credentials found in environment variables")

    return mlflow_uri


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42):
    """Train Logistic Regression model"""
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'accuracy': accuracy,
        'classification_report': report
    }


def save_model_and_metrics(model, metrics, model_path, metrics_path):
    """Save model and metrics to files"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def train_and_log_model(train_path, test_path, model_path, metrics_path,
                        experiment_name, run_name, algorithm_type, dataset_name):
    """Complete training pipeline with MLflow logging"""

    # Setup MLflow authentication
    setup_mlflow_auth()

    # Read data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Prepare features and targets
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Set MLflow experiment
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting MLflow experiment: {e}")
        # Create experiment if it doesn't exist
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as create_error:
            print(f"Error creating MLflow experiment: {create_error}")
            raise

    with mlflow.start_run(run_name=run_name):
        try:
            # Train model based on algorithm type
            if algorithm_type == 'random_forest':
                if dataset_name == 'breast_cancer':
                    model = train_random_forest(X_train, y_train, n_estimators=150, max_depth=10)
                    mlflow.log_param("n_estimators", 150)
                    mlflow.log_param("max_depth", 10)
                else:
                    model = train_random_forest(X_train, y_train)
                    mlflow.log_param("n_estimators", 100)
            elif algorithm_type == 'logistic_regression':
                model = train_logistic_regression(X_train, y_train)
                mlflow.log_param("max_iter", 1000)

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)

            # Log parameters and metrics
            mlflow.log_param("algorithm", algorithm_type)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", metrics['accuracy'])

            # Log model
            mlflow.sklearn.log_model(model, f"{algorithm_type}_model")

            print(f"Successfully logged to MLflow: {run_name}")

        except Exception as e:
            print(f"Error logging to MLflow: {e}")
            print("Continuing with local model saving...")

        # Save locally (always do this regardless of MLflow success)
        save_model_and_metrics(model, metrics, model_path, metrics_path)

        return model, metrics