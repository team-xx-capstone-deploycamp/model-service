import luigi
import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, load_wine, load_breast_cancer
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()


class GenerateData(luigi.Task):
    """Generate sample dataset"""

    def output(self):
        return luigi.LocalTarget('data/raw/dataset.csv')

    def run(self):
        # Generate sample classification dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )

        # Create DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        # Save to CSV
        df.to_csv(self.output().path, index=False)


class LoadWineDataset(luigi.Task):
    """Load Wine dataset from sklearn"""

    def output(self):
        return luigi.LocalTarget('data/raw/wine_dataset.csv')

    def run(self):
        # Load wine dataset
        wine = load_wine()

        # Create DataFrame
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        # Save to CSV
        df.to_csv(self.output().path, index=False)


class LoadBreastCancerDataset(luigi.Task):
    """Load Breast Cancer dataset from sklearn"""

    def output(self):
        return luigi.LocalTarget('data/raw/breast_cancer_dataset.csv')

    def run(self):
        # Load breast cancer dataset
        cancer = load_breast_cancer()

        # Create DataFrame
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        # Save to CSV
        df.to_csv(self.output().path, index=False)


class PreprocessData(luigi.Task):
    """Preprocess the raw data"""

    def requires(self):
        return GenerateData()

    def output(self):
        return [
            luigi.LocalTarget('data/processed/train.csv'),
            luigi.LocalTarget('data/processed/test.csv')
        ]

    def run(self):
        # Read raw data
        df = pd.read_csv(self.input().path)

        # Simple preprocessing - split into train/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Ensure directory exists
        os.makedirs('data/processed', exist_ok=True)

        # Save processed data
        train_df.to_csv(self.output()[0].path, index=False)
        test_df.to_csv(self.output()[1].path, index=False)


class PreprocessWineData(luigi.Task):
    """Preprocess the wine dataset"""

    def requires(self):
        return LoadWineDataset()

    def output(self):
        return [
            luigi.LocalTarget('data/processed/wine_train.csv'),
            luigi.LocalTarget('data/processed/wine_test.csv')
        ]

    def run(self):
        # Read raw data
        df = pd.read_csv(self.input().path)

        # Simple preprocessing - split into train/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Ensure directory exists
        os.makedirs('data/processed', exist_ok=True)

        # Save processed data
        train_df.to_csv(self.output()[0].path, index=False)
        test_df.to_csv(self.output()[1].path, index=False)


class PreprocessBreastCancerData(luigi.Task):
    """Preprocess the breast cancer dataset"""

    def requires(self):
        return LoadBreastCancerDataset()

    def output(self):
        return [
            luigi.LocalTarget('data/processed/cancer_train.csv'),
            luigi.LocalTarget('data/processed/cancer_test.csv')
        ]

    def run(self):
        # Read raw data
        df = pd.read_csv(self.input().path)

        # Simple preprocessing - split into train/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Ensure directory exists
        os.makedirs('data/processed', exist_ok=True)

        # Save processed data
        train_df.to_csv(self.output()[0].path, index=False)
        test_df.to_csv(self.output()[1].path, index=False)


class TrainModel(luigi.Task):
    """Train ML model and log to MLflow"""

    def requires(self):
        return PreprocessData()

    def output(self):
        return [
            luigi.LocalTarget('models/model.pkl'),
            luigi.LocalTarget('models/metrics.json')
        ]

    def run(self):
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment("synthetic_data_experiment")

        # Read processed data
        train_df = pd.read_csv(self.input()[0].path)
        test_df = pd.read_csv(self.input()[1].path)

        # Prepare features and targets
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # Start MLflow run
        with mlflow.start_run(run_name="RandomForest_Synthetic"):
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log parameters and metrics to MLflow
            mlflow.log_param("algorithm", "RandomForest")
            mlflow.log_param("dataset", "synthetic")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)

            # Save model locally
            with open(self.output()[0].path, 'wb') as f:
                pickle.dump(model, f)

            # Save metrics
            metrics = {
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }

            with open(self.output()[1].path, 'w') as f:
                json.dump(metrics, f, indent=2)


class TrainWineModel(luigi.Task):
    """Train ML model on Wine dataset and log to MLflow"""

    def requires(self):
        return PreprocessWineData()

    def output(self):
        return [
            luigi.LocalTarget('models/wine_model.pkl'),
            luigi.LocalTarget('models/wine_metrics.json')
        ]

    def run(self):
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment("wine_classification_experiment")

        # Read processed data
        train_df = pd.read_csv(self.input()[0].path)
        test_df = pd.read_csv(self.input()[1].path)

        # Prepare features and targets
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # Start MLflow run
        with mlflow.start_run(run_name="LogisticRegression_Wine"):
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log parameters and metrics to MLflow
            mlflow.log_param("algorithm", "LogisticRegression")
            mlflow.log_param("dataset", "wine")
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "logistic_regression_model")

            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)

            # Save model locally
            with open(self.output()[0].path, 'wb') as f:
                pickle.dump(model, f)

            # Save metrics
            metrics = {
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }

            with open(self.output()[1].path, 'w') as f:
                json.dump(metrics, f, indent=2)


class TrainBreastCancerModel(luigi.Task):
    """Train ML model on Breast Cancer dataset and log to MLflow"""

    def requires(self):
        return PreprocessBreastCancerData()

    def output(self):
        return [
            luigi.LocalTarget('models/cancer_model.pkl'),
            luigi.LocalTarget('models/cancer_metrics.json')
        ]

    def run(self):
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment("breast_cancer_classification_experiment")

        # Read processed data
        train_df = pd.read_csv(self.input()[0].path)
        test_df = pd.read_csv(self.input()[1].path)

        # Prepare features and targets
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # Start MLflow run
        with mlflow.start_run(run_name="RandomForest_BreastCancer"):
            # Train model
            model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log parameters and metrics to MLflow
            mlflow.log_param("algorithm", "RandomForest")
            mlflow.log_param("dataset", "breast_cancer")
            mlflow.log_param("n_estimators", 150)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)

            # Save model locally
            with open(self.output()[0].path, 'wb') as f:
                pickle.dump(model, f)

            # Save metrics
            metrics = {
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }

            with open(self.output()[1].path, 'w') as f:
                json.dump(metrics, f, indent=2)


class MLPipeline(luigi.WrapperTask):
    """Complete ML Pipeline"""

    def requires(self):
        return [
            TrainModel(),
            TrainWineModel(),
            TrainBreastCancerModel()
        ]