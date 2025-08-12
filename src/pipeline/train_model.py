import pandas as pd
import joblib
import os
import io
import luigi
import mlflow
import numpy as np
import subprocess
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder
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
MINIO_SECURE_ENDPOINT = os.getenv("MINIO_SECURE_ENDPOINT", True)

# Initialize MinIO client
def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE_ENDPOINT
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

# Luigi Task for loading data from local file, DVC, or MinIO
class LoadDataTask(luigi.Task):
    dataset_filename = luigi.Parameter(default="CarPrice_Assignment.csv")
    output_path = luigi.Parameter(default="/tmp/car_data.csv")
    data_dir = luigi.Parameter(default="data")

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        # Try to load from local file first
        local_path = os.path.join(self.data_dir, self.dataset_filename)
        if os.path.exists(local_path):
            print(f"📄 Loading dataset from local file: {local_path}")
            df = pd.read_csv(local_path)
        else:
            # Fallback to MinIO
            print(f"⚠️ Local file not found. Falling back to MinIO...")
            client = get_minio_client()
            try:
                response = client.get_object(BUCKET_NAME, self.dataset_filename)
                df = pd.read_csv(io.BytesIO(response.read()))
                response.close()
                response.release_conn()
                print(f"📥 Dataset loaded from MinIO bucket: {BUCKET_NAME}")
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from MinIO: {str(e)}")

        # Clean the data (remove outliers)
        print("🧹 Cleaning data and removing outliers...")
        df = self._clean_data(df)

        # Save to local file for next task
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print("✅ Dataset successfully loaded and cleaned.")

    def _clean_data(self, df):
        # Remove rows with price = 0
        df = df[df['price'] != 0]

        # Detect outliers using IQR
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        outlier_indices = []

        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].index
            outlier_indices.extend(outliers)

        # Count frequency of indices in outlier list
        outlier_count = Counter(outlier_indices)

        # Remove rows that are outliers in multiple columns
        common_outliers = [idx for idx, count in outlier_count.items() if count >= 2]
        df = df.drop(index=common_outliers).reset_index(drop=True)

        print(f"Removed {len(common_outliers)} outliers. Remaining rows: {len(df)}")
        return df

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
            "preprocessor": luigi.LocalTarget(f"{self.output_path}/preprocessor.pkl")
        }

    def run(self):
        print("⚙️ Processing data...")
        df = pd.read_csv(self.input_path)

        # Check if price column exists
        if "price" not in df.columns:
            raise ValueError("Price column not found in dataset.")

        # Split features and target
        X = df.drop(columns=["price"])
        y = df["price"]

        # Define column types for preprocessing
        onehot_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 
                      'drivewheel', 'enginelocation']
        binary_cols = ['CarName', 'enginetype', 'cylindernumber', 'fuelsystem']

        # Create column transformer for preprocessing
        preprocessor = ColumnTransformer([
            ('OneHot', OneHotEncoder(drop='first', sparse_output=False), onehot_cols),
            ('Binary', BinaryEncoder(), binary_cols)
        ], remainder='passthrough')

        # Fit preprocessor on all data
        preprocessor.fit(X)

        # Split dataset with 70:30 ratio as in notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Save preprocessed data
        X_train.to_csv(self.output()["X_train"].path, index=False)
        X_test.to_csv(self.output()["X_test"].path, index=False)
        pd.DataFrame(y_train).to_csv(self.output()["y_train"].path, index=False)
        pd.DataFrame(y_test).to_csv(self.output()["y_test"].path, index=False)

        # Save preprocessor
        with open(self.output()["preprocessor"].path, 'wb') as f:
            joblib.dump(preprocessor, f)

        print(f"✅ Data ready. Training data: {len(X_train)}, test data: {len(X_test)}")

# Luigi Task for training the model
class TrainModelTask(luigi.Task):
    input_path = luigi.Parameter(default="/tmp/preprocessed_data")
    output_path = luigi.Parameter(default="/tmp/model.pkl")
    run_name = luigi.Parameter(default="xgboost_car_price_model")
    model_name = luigi.Parameter(default="model")

    # Hyperparameter tuning settings
    n_iter = luigi.IntParameter(default=50)  # Number of parameter settings sampled
    cv_folds = luigi.IntParameter(default=5)  # Number of cross-validation folds
    random_state = luigi.IntParameter(default=42)

    def requires(self):
        return PreprocessDataTask(output_path=self.input_path)

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        print("🚀 Starting XGBoost model training with hyperparameter tuning...")

        # Load preprocessed data
        X_train = pd.read_csv(self.input()["X_train"].path)
        y_train = pd.read_csv(self.input()["y_train"].path).iloc[:, 0]
        X_test = pd.read_csv(self.input()["X_test"].path)
        y_test = pd.read_csv(self.input()["y_test"].path).iloc[:, 0]

        # Load preprocessor
        with open(self.input()["preprocessor"].path, 'rb') as f:
            preprocessor = joblib.load(f)

        # Initialize MLflow
        setup_mlflow()

        # Start MLflow run
        with mlflow.start_run(run_name=self.run_name):
            # Define hyperparameter space for XGBoost
            param_space = {
                'model__max_depth': list(range(2, 30)),
                'model__learning_rate': [i / 100 for i in range(1, 100)],
                'model__n_estimators': list(range(100, 201)),
                'model__subsample': [i / 10 for i in range(1, 10)],
                'model__colsample_bytree': [i / 10 for i in range(1, 10)],
                'model__reg_alpha': list(np.logspace(-3, 1, 10))
            }

            # Create pipeline with preprocessor and model
            xgb = XGBRegressor(random_state=self.random_state, verbosity=0)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb)
            ])

            # Set up RandomizedSearchCV
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_space,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )

            # Log hyperparameter tuning settings
            mlflow.log_param("n_iter", self.n_iter)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("random_state", self.random_state)

            # Fit the model with hyperparameter tuning
            print("Performing hyperparameter tuning...")
            X_train_transformed = preprocessor.transform(X_train)
            random_search.fit(X_train_transformed, y_train)

            # Log best parameters
            best_params = random_search.best_params_
            for param, value in best_params.items():
                mlflow.log_param(param, value)

            # Get best model
            best_model = random_search.best_estimator_

            # Evaluate model
            X_test_transformed = preprocessor.transform(X_test)
            y_pred = best_model.predict(X_test_transformed)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape, "r2": r2})
            mlflow.log_metric("best_cv_score", -random_search.best_score_)

            # Log model
            mlflow.sklearn.log_model(best_model, self.model_name)

            print(f"Best parameters: {best_params}")
            print(f"Model evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.4f}, R²: {r2:.4f}")

            # Save model locally
            with open(self.output_path, 'wb') as f:
                joblib.dump(best_model, f)

        print("✅ Model successfully trained and evaluated with hyperparameter tuning.")

# Luigi Task for saving model to MinIO
class SaveModelTask(luigi.Task):
    model_path = luigi.Parameter(default="/tmp/model.pkl")
    preprocessor_path = luigi.Parameter(default="/tmp/preprocessed_data/preprocessor.pkl")
    output_filename = luigi.Parameter(default="car_price_model.pkl")

    def requires(self):
        return {
            "model": TrainModelTask(output_path=self.model_path),
            "preprocessed": PreprocessDataTask(output_path=os.path.dirname(self.preprocessor_path))
        }

    def output(self):
        # This is a marker file to indicate the task has completed
        return luigi.LocalTarget(f"/tmp/model_saved_to_minio_{self.output_filename}")

    def run(self):
        print(f"💾 Saving model to MinIO as '{self.output_filename}'...")

        # Load model and preprocessor
        with open(self.model_path, 'rb') as f:
            model = joblib.load(f)

        with open(self.preprocessor_path, 'rb') as f:
            preprocessor = joblib.load(f)

        # Save model and preprocessor to MinIO
        buffer = io.BytesIO()
        joblib.dump({"model": model, "preprocessor": preprocessor}, buffer)
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

        print("✅ Model successfully saved to MinIO.")

# Main pipeline task
class CarPricePredictionPipeline(luigi.Task):
    dataset_filename = luigi.Parameter(default="CarPrice_Assignment.csv")
    model_filename = luigi.Parameter(default="car_price_model.pkl")

    def requires(self):
        return SaveModelTask(output_filename=self.model_filename)

    def output(self):
        return luigi.LocalTarget("/tmp/pipeline_complete.txt")

    def run(self):
        with open(self.output().path, 'w') as f:
            f.write("Pipeline completed successfully!")
        print("🎯 Pipeline completed.")

if __name__ == "__main__":
    # Run the Luigi pipeline
    luigi.build([CarPricePredictionPipeline()], local_scheduler=True)
