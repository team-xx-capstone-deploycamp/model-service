import pandas as pd
import joblib
import os
import io
import luigi
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from minio import Minio

# Configuration placeholders
# These can be set via environment variables or Luigi configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio.capstone.pebrisulistiyo.com")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "YOUR_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "YOUR_SECRET_KEY")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "bucket")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "car_price_prediction")
MLFLOW_USERNAME = os.getenv("MLFLOW_USERNAME", "")
MLFLOW_PASSWORD = os.getenv("MLFLOW_PASSWORD", "")

# Initialize MinIO client
def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True
    )

# Initialize MLflow
def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_USERNAME and MLFLOW_PASSWORD:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

    # Create experiment if it doesn't exist
    try:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    except:
        pass
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Luigi Task for loading data from MinIO
class LoadDataTask(luigi.Task):
    dataset_filename = luigi.Parameter(default="used_car_cleaned.csv")
    output_path = luigi.Parameter(default="/tmp/car_data.csv")

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        print(f"📥 Mengunduh dataset '{self.dataset_filename}' dari bucket '{BUCKET_NAME}'...")
        client = get_minio_client()
        response = client.get_object(BUCKET_NAME, self.dataset_filename)
        df = pd.read_csv(io.BytesIO(response.read()))
        response.close()
        response.release_conn()

        # Save to local file for next task
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print("✅ Dataset berhasil diunduh.")

# Luigi Task for preprocessing data
class PreprocessDataTask(luigi.Task):
    input_path = luigi.Parameter(default="/tmp/car_data.csv")
    output_path = luigi.Parameter(default="/tmp/preprocessed_data")

    def requires(self):
        return LoadDataTask(output_path=self.input_path)

    def output(self):
        return {
            "X_train": luigi.LocalTarget(f"{self.output_path}/X_train.csv"),
            "X_test": luigi.LocalTarget(f"{self.output_path}/X_test.csv"),
            "y_train": luigi.LocalTarget(f"{self.output_path}/y_train.csv"),
            "y_test": luigi.LocalTarget(f"{self.output_path}/y_test.csv"),
            "encoders": luigi.LocalTarget(f"{self.output_path}/encoders.pkl")
        }

    def run(self):
        print("⚙️ Memproses data...")
        df = pd.read_csv(self.input_path)

        # Check if price column exists
        if "price" not in df.columns:
            raise ValueError("Kolom 'price' tidak ditemukan di dataset.")

        # Split features and target
        X = df.drop(columns=["price"])
        y = df["price"]

        # Label encoding for categorical columns
        label_encoders = {}
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Save preprocessed data
        X_train.to_csv(self.output()["X_train"].path, index=False)
        X_test.to_csv(self.output()["X_test"].path, index=False)
        pd.DataFrame(y_train).to_csv(self.output()["y_train"].path, index=False)
        pd.DataFrame(y_test).to_csv(self.output()["y_test"].path, index=False)

        # Save encoders
        with open(self.output()["encoders"].path, 'wb') as f:
            joblib.dump(label_encoders, f)

        print(f"✅ Data siap. Total data latih: {len(X_train)}, data uji: {len(X_test)}")

# Luigi Task for training the model
class TrainModelTask(luigi.Task):
    input_path = luigi.Parameter(default="/tmp/preprocessed_data")
    output_path = luigi.Parameter(default="/tmp/model.pkl")

    # XGBoost parameters
    n_estimators = luigi.IntParameter(default=300)
    learning_rate = luigi.FloatParameter(default=0.1)
    max_depth = luigi.IntParameter(default=6)
    subsample = luigi.FloatParameter(default=0.8)
    colsample_bytree = luigi.FloatParameter(default=0.8)
    random_state = luigi.IntParameter(default=42)

    def requires(self):
        return PreprocessDataTask(output_path=self.input_path)

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        print("🚀 Memulai training model XGBoost...")

        # Load preprocessed data
        X_train = pd.read_csv(self.input()["X_train"].path)
        y_train = pd.read_csv(self.input()["y_train"].path).iloc[:, 0]
        X_test = pd.read_csv(self.input()["X_test"].path)
        y_test = pd.read_csv(self.input()["y_test"].path).iloc[:, 0]

        # Initialize MLflow
        setup_mlflow()

        # Start MLflow run
        with mlflow.start_run(run_name="xgboost_car_price_model"):
            # Log parameters
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("max_depth", self.max_depth)
            mlflow.log_param("subsample", self.subsample)
            mlflow.log_param("colsample_bytree", self.colsample_bytree)
            mlflow.log_param("random_state", self.random_state)

            # Train model
            model = XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log model
            mlflow.xgboost.log_model(model, "model")

            print(f"Model evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

            # Save model locally
            with open(self.output_path, 'wb') as f:
                joblib.dump(model, f)

        print("✅ Model berhasil dilatih dan dievaluasi.")

# Luigi Task for saving model to MinIO
class SaveModelTask(luigi.Task):
    model_path = luigi.Parameter(default="/tmp/model.pkl")
    encoders_path = luigi.Parameter(default="/tmp/preprocessed_data/encoders.pkl")
    output_filename = luigi.Parameter(default="car_price_model.pkl")

    def requires(self):
        return {
            "model": TrainModelTask(output_path=self.model_path),
            "preprocessed": PreprocessDataTask(output_path=os.path.dirname(self.encoders_path))
        }

    def output(self):
        # This is a marker file to indicate the task has completed
        return luigi.LocalTarget(f"/tmp/model_saved_to_minio_{self.output_filename}")

    def run(self):
        print(f"💾 Menyimpan model ke MinIO sebagai '{self.output_filename}'...")

        # Load model and encoders
        with open(self.model_path, 'rb') as f:
            model = joblib.load(f)

        with open(self.encoders_path, 'rb') as f:
            encoders = joblib.load(f)

        # Save model and encoders to MinIO
        buffer = io.BytesIO()
        joblib.dump({"model": model, "encoders": encoders}, buffer)
        buffer.seek(0)

        client = get_minio_client()
        client.put_object(
            BUCKET_NAME, 
            self.output_filename, 
            buffer, 
            length=buffer.getbuffer().nbytes
        )

        # Create marker file
        with open(self.output().path, 'w') as f:
            f.write(f"Model saved to MinIO as {self.output_filename}")

        print("✅ Model berhasil disimpan ke MinIO.")

# Main pipeline task
class CarPricePredictionPipeline(luigi.Task):
    dataset_filename = luigi.Parameter(default="used_car_cleaned.csv")
    model_filename = luigi.Parameter(default="car_price_model.pkl")

    def requires(self):
        return SaveModelTask(output_filename=self.model_filename)

    def output(self):
        return luigi.LocalTarget("/tmp/pipeline_complete.txt")

    def run(self):
        with open(self.output().path, 'w') as f:
            f.write("Pipeline completed successfully!")
        print("🎯 Pipeline selesai.")

if __name__ == "__main__":
    # Run the Luigi pipeline
    luigi.build([CarPricePredictionPipeline()], local_scheduler=True)
