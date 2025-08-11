import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import io
import joblib
import luigi
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.model_selection import RandomizedSearchCV
from src.pipeline.train_model import (
    LoadDataTask,
    PreprocessDataTask,
    TrainModelTask,
    SaveModelTask,
    CarPricePredictionPipeline,
    get_minio_client,
    setup_mlflow
)

class TestHelperFunctions(unittest.TestCase):

    @patch('src.pipeline.train_model.Minio')
    def test_get_minio_client(self, mock_minio):
        # Setup mock
        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        # Call function
        client = get_minio_client()

        # Assertions
        mock_minio.assert_called_once()
        self.assertEqual(client, mock_client)

    @patch('src.pipeline.train_model.mlflow')
    @patch('src.pipeline.train_model.os')
    def test_setup_mlflow(self, mock_os, mock_mlflow):
        # Call function
        setup_mlflow()

        # Assertions
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()

class TestLoadDataTask(unittest.TestCase):

    def test_output(self):
        # Create task
        task = LoadDataTask(output_path="/tmp/test_data.csv")

        # Check output
        output = task.output()
        self.assertIsInstance(output, luigi.LocalTarget)
        self.assertEqual(output.path, "/tmp/test_data.csv")

    @patch('src.pipeline.train_model.os.path.exists')
    @patch('src.pipeline.train_model.pd.read_csv')
    @patch('src.pipeline.train_model.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_local_file(self, mock_file, mock_makedirs, mock_read_csv, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df

        # Mock _clean_data
        mock_df_cleaned = MagicMock()

        # Create task with mocked _clean_data
        task = LoadDataTask(dataset_filename="test.csv", output_path="/tmp/test_data.csv")
        task._clean_data = MagicMock(return_value=mock_df_cleaned)

        # Run task
        task.run()

        # Assertions
        mock_exists.assert_called_once()
        mock_read_csv.assert_called_once()
        task._clean_data.assert_called_once_with(mock_df)
        mock_df_cleaned.to_csv.assert_called_once()

    @patch('src.pipeline.train_model.os.path.exists')
    @patch('src.pipeline.train_model.subprocess.run')
    @patch('src.pipeline.train_model.pd.read_csv')
    @patch('src.pipeline.train_model.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_dvc_pull(self, mock_file, mock_makedirs, mock_read_csv, mock_subprocess, mock_exists):
        # Setup mocks
        # First call to exists returns False (local file doesn't exist)
        # Second call to exists returns True (file exists after DVC pull)
        mock_exists.side_effect = [False, True]

        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df

        # Mock _clean_data
        mock_df_cleaned = MagicMock()

        # Create task with mocked _clean_data
        task = LoadDataTask(dataset_filename="test.csv", output_path="/tmp/test_data.csv")
        task._clean_data = MagicMock(return_value=mock_df_cleaned)

        # Run task
        task.run()

        # Assertions
        self.assertEqual(mock_exists.call_count, 2)
        mock_subprocess.assert_called_once()
        mock_read_csv.assert_called_once()
        task._clean_data.assert_called_once_with(mock_df)
        mock_df_cleaned.to_csv.assert_called_once()

    @patch('src.pipeline.train_model.os.path.exists')
    @patch('src.pipeline.train_model.subprocess.run')
    @patch('src.pipeline.train_model.get_minio_client')
    @patch('src.pipeline.train_model.pd.read_csv')
    @patch('src.pipeline.train_model.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_minio_fallback(self, mock_file, mock_makedirs, mock_read_csv, mock_get_client, mock_subprocess, mock_exists):
        # Setup mocks
        # Local file doesn't exist and DVC pull fails
        mock_exists.return_value = False
        mock_subprocess.side_effect = subprocess.SubprocessError("DVC pull failed")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_client.get_object.return_value = mock_response

        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df

        # Mock _clean_data
        mock_df_cleaned = MagicMock()

        # Create task with mocked _clean_data
        task = LoadDataTask(dataset_filename="test.csv", output_path="/tmp/test_data.csv")
        task._clean_data = MagicMock(return_value=mock_df_cleaned)

        # Run task
        task.run()

        # Assertions
        mock_exists.assert_called_once()
        mock_subprocess.assert_called_once()
        mock_get_client.assert_called_once()
        mock_client.get_object.assert_called_once()
        mock_read_csv.assert_called_once()
        task._clean_data.assert_called_once_with(mock_df)
        mock_df_cleaned.to_csv.assert_called_once()

    def test_clean_data(self):
        # Create test data with outliers
        data = {
            'price': [10000, 20000, 30000, 100000, 0],  # 0 should be removed
            'symboling': [1, 2, 3, 20, 2],  # 20 is an outlier
            'wheelbase': [100, 110, 120, 200, 110]  # 200 is an outlier
        }
        df = pd.DataFrame(data)

        # Create task
        task = LoadDataTask()

        # Call _clean_data
        cleaned_df = task._clean_data(df)

        # Assertions
        self.assertEqual(len(cleaned_df), 3)  # Should remove rows with price=0 and the row with multiple outliers
        self.assertNotIn(0, cleaned_df['price'].values)  # price=0 should be removed

class TestPreprocessDataTask(unittest.TestCase):

    def test_requires(self):
        # Create task
        task = PreprocessDataTask(input_path="/tmp/test_data.csv")

        # Check requires
        requires = task.requires()
        self.assertIsInstance(requires, LoadDataTask)
        self.assertEqual(requires.output_path, "/tmp/test_data.csv")

    def test_output(self):
        # Create task
        task = PreprocessDataTask(output_path="/tmp/test_output")

        # Check output
        output = task.output()
        self.assertIsInstance(output, dict)
        self.assertEqual(len(output), 5)  # X_train, X_test, y_train, y_test, encoders

    @patch('src.pipeline.train_model.pd.read_csv')
    @patch('src.pipeline.train_model.train_test_split')
    @patch('src.pipeline.train_model.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.pipeline.train_model.joblib.dump')
    @patch('src.pipeline.train_model.ColumnTransformer')
    @patch('src.pipeline.train_model.OneHotEncoder')
    @patch('src.pipeline.train_model.BinaryEncoder')
    def test_run(self, mock_binary_encoder, mock_onehot_encoder, mock_column_transformer, 
                 mock_dump, mock_file, mock_makedirs, mock_split, mock_read_csv):
        # Setup mocks
        mock_df = pd.DataFrame({
            'CarName': ['Toyota Camry', 'Honda Civic'],
            'fueltype': ['gas', 'diesel'],
            'aspiration': ['std', 'turbo'],
            'doornumber': ['four', 'two'],
            'carbody': ['sedan', 'hatchback'],
            'drivewheel': ['fwd', 'rwd'],
            'enginelocation': ['front', 'rear'],
            'enginetype': ['ohc', 'dohc'],
            'cylindernumber': ['four', 'six'],
            'fuelsystem': ['mpfi', 'idi'],
            'price': [25000, 22000]
        })
        mock_read_csv.return_value = mock_df

        # Create expected X and y dataframes
        X_cols = [col for col in mock_df.columns if col != 'price']
        mock_X = mock_df[X_cols]
        mock_y = mock_df['price']

        mock_X_train = mock_X.iloc[[0]]
        mock_X_test = mock_X.iloc[[1]]
        mock_y_train = mock_y.iloc[[0]]
        mock_y_test = mock_y.iloc[[1]]
        mock_split.return_value = (mock_X_train, mock_X_test, mock_y_train, mock_y_test)

        # Mock preprocessor
        mock_preprocessor = MagicMock()
        mock_column_transformer.return_value = mock_preprocessor

        # Mock encoder instances
        mock_onehot_instance = MagicMock()
        mock_onehot_encoder.return_value = mock_onehot_instance

        mock_binary_instance = MagicMock()
        mock_binary_encoder.return_value = mock_binary_instance

        # Create task with mocked output
        task = PreprocessDataTask(input_path="/tmp/test_data.csv", output_path="/tmp/test_output")
        task.output = MagicMock(return_value={
            "X_train": MagicMock(path="/tmp/test_output/X_train.csv"),
            "X_test": MagicMock(path="/tmp/test_output/X_test.csv"),
            "y_train": MagicMock(path="/tmp/test_output/y_train.csv"),
            "y_test": MagicMock(path="/tmp/test_output/y_test.csv"),
            "preprocessor": MagicMock(path="/tmp/test_output/preprocessor.pkl")
        })

        # Run task
        task.run()

        # Assertions
        mock_read_csv.assert_called_once()
        mock_split.assert_called_once()
        mock_makedirs.assert_called_once()

        # Check that OneHotEncoder was created with correct parameters
        mock_onehot_encoder.assert_called_once_with(drop='first', sparse_output=False)

        # Check that ColumnTransformer was created with correct transformers
        mock_column_transformer.assert_called_once()
        # Extract the call arguments
        args, kwargs = mock_column_transformer.call_args
        transformers = args[0]

        # Check that there are two transformers: OneHot and Binary
        self.assertEqual(len(transformers), 2)
        self.assertEqual(transformers[0][0], 'OneHot')
        self.assertEqual(transformers[1][0], 'Binary')

        # Check that OneHot transformer is applied to the correct columns
        onehot_cols = transformers[0][2]
        self.assertIn('fueltype', onehot_cols)
        self.assertIn('aspiration', onehot_cols)
        self.assertIn('doornumber', onehot_cols)
        self.assertIn('carbody', onehot_cols)
        self.assertIn('drivewheel', onehot_cols)
        self.assertIn('enginelocation', onehot_cols)

        # Check that Binary transformer is applied to the correct columns
        binary_cols = transformers[1][2]
        self.assertIn('CarName', binary_cols)
        self.assertIn('enginetype', binary_cols)
        self.assertIn('cylindernumber', binary_cols)
        self.assertIn('fuelsystem', binary_cols)

        # Check that preprocessor was fit
        mock_preprocessor.fit.assert_called_once_with(mock_X)

        # Check that preprocessor was saved
        self.assertEqual(mock_dump.call_count, 1)  # Called for preprocessor

class TestTrainModelTask(unittest.TestCase):

    def test_requires(self):
        # Create task
        task = TrainModelTask(input_path="/tmp/test_preprocessed")

        # Check requires
        requires = task.requires()
        self.assertIsInstance(requires, PreprocessDataTask)
        self.assertEqual(requires.output_path, "/tmp/test_preprocessed")

    def test_output(self):
        # Create task
        task = TrainModelTask(output_path="/tmp/test_model.pkl")

        # Check output
        output = task.output()
        self.assertIsInstance(output, luigi.LocalTarget)
        self.assertEqual(output.path, "/tmp/test_model.pkl")

    @patch('src.pipeline.train_model.pd.read_csv')
    @patch('src.pipeline.train_model.setup_mlflow')
    @patch('src.pipeline.train_model.mlflow.start_run')
    @patch('src.pipeline.train_model.mlflow.log_param')
    @patch('src.pipeline.train_model.mlflow.log_metric')
    @patch('src.pipeline.train_model.mlflow.sklearn.log_model')
    @patch('src.pipeline.train_model.XGBRegressor')
    @patch('src.pipeline.train_model.RandomizedSearchCV')
    @patch('src.pipeline.train_model.joblib.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.pipeline.train_model.joblib.dump')
    @patch('src.pipeline.train_model.Pipeline')
    @patch('src.pipeline.train_model.StandardScaler')
    def test_run(self, mock_scaler, mock_pipeline, mock_dump, mock_file, mock_load, 
                 mock_random_search, mock_xgb, mock_log_model, mock_log_metric, 
                 mock_log_param, mock_start_run, mock_setup, mock_read_csv):
        # Setup mocks
        mock_X_train = pd.DataFrame({
            'CarName': [0, 1], 
            'fueltype': [0, 1], 
            'aspiration': [0, 1], 
            'doornumber': [0, 1], 
            'carbody': [0, 1],
            'drivewheel': [0, 1],
            'enginelocation': [0, 1],
            'enginetype': [0, 1],
            'cylindernumber': [0, 1],
            'fuelsystem': [0, 1]
        })
        mock_y_train = pd.Series([25000, 22000])
        mock_X_test = pd.DataFrame({
            'CarName': [2], 
            'fueltype': [2], 
            'aspiration': [2], 
            'doornumber': [2], 
            'carbody': [2],
            'drivewheel': [2],
            'enginelocation': [2],
            'enginetype': [2],
            'cylindernumber': [2],
            'fuelsystem': [2]
        })
        mock_y_test = pd.Series([20000])

        mock_read_csv.side_effect = [mock_X_train, mock_y_train, mock_X_test, mock_y_test]

        # Mock preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.side_effect = [
            np.array([[1, 2, 3], [4, 5, 6]]),  # X_train_transformed
            np.array([[7, 8, 9]])              # X_test_transformed
        ]
        mock_load.return_value = mock_preprocessor

        # Mock pipeline and random search
        mock_scaler_instance = MagicMock()
        mock_scaler.return_value = mock_scaler_instance

        mock_xgb_instance = MagicMock()
        mock_xgb.return_value = mock_xgb_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_best_estimator = MagicMock()
        mock_random_search_instance = MagicMock()
        mock_random_search_instance.best_estimator_ = mock_best_estimator
        mock_random_search_instance.best_score_ = -1500  # Negative RMSE
        mock_random_search_instance.best_params_ = {
            'model__max_depth': 6, 
            'model__learning_rate': 0.1,
            'model__n_estimators': 150,
            'model__subsample': 0.8,
            'model__colsample_bytree': 0.8,
            'model__reg_alpha': 0.01
        }
        mock_random_search.return_value = mock_random_search_instance

        # Mock prediction and metrics
        mock_best_estimator.predict.return_value = np.array([21000])

        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context

        # Create task with mocked input
        task = TrainModelTask(
            input_path="/tmp/test_preprocessed", 
            output_path="/tmp/test_model.pkl",
            n_iter=50,
            cv_folds=5,
            random_state=42
        )
        task.input = MagicMock(return_value={
            "X_train": MagicMock(path="/tmp/test_preprocessed/X_train.csv"),
            "X_test": MagicMock(path="/tmp/test_preprocessed/X_test.csv"),
            "y_train": MagicMock(path="/tmp/test_preprocessed/y_train.csv"),
            "y_test": MagicMock(path="/tmp/test_preprocessed/y_test.csv"),
            "preprocessor": MagicMock(path="/tmp/test_preprocessed/preprocessor.pkl")
        })

        # Run task
        task.run()

        # Assertions
        self.assertEqual(mock_read_csv.call_count, 4)  # Called for X_train, y_train, X_test, y_test
        mock_setup.assert_called_once()
        mock_start_run.assert_called_once()

        # Check that preprocessor was loaded and used
        mock_load.assert_called_once()
        self.assertEqual(mock_preprocessor.transform.call_count, 2)  # Called for X_train and X_test

        # Check that pipeline was created correctly
        mock_scaler.assert_called_once()
        mock_xgb.assert_called_once_with(random_state=42, verbosity=0)
        mock_pipeline.assert_called_once_with([
            ('scaler', mock_scaler_instance),
            ('model', mock_xgb_instance)
        ])

        # Check that RandomizedSearchCV was created with correct parameters
        mock_random_search.assert_called_once()
        args, kwargs = mock_random_search.call_args
        self.assertEqual(kwargs['n_iter'], 50)
        self.assertEqual(kwargs['cv'], 5)
        self.assertEqual(kwargs['scoring'], 'neg_root_mean_squared_error')
        self.assertEqual(kwargs['random_state'], 42)

        # Check that hyperparameter space includes all required parameters
        param_space = kwargs['param_distributions']
        self.assertIn('model__max_depth', param_space)
        self.assertIn('model__learning_rate', param_space)
        self.assertIn('model__n_estimators', param_space)
        self.assertIn('model__subsample', param_space)
        self.assertIn('model__colsample_bytree', param_space)
        self.assertIn('model__reg_alpha', param_space)

        # Check that RandomizedSearchCV was fit with transformed data
        mock_random_search_instance.fit.assert_called_once()

        # Check that best parameters were logged
        self.assertGreaterEqual(mock_log_param.call_count, 6)  # At least one call for each hyperparameter

        # Check that metrics were logged
        self.assertGreaterEqual(mock_log_metric.call_count, 4)  # At least one call for each metric (rmse, mae, mape, r2)

        # Check that model was logged
        mock_log_model.assert_called_once_with(mock_best_estimator, "model")

        # Check that model was saved
        mock_dump.assert_called_once_with(mock_best_estimator, mock_file())

class TestSaveModelTask(unittest.TestCase):

    def test_requires(self):
        # Create task
        task = SaveModelTask(model_path="/tmp/test_model.pkl", preprocessor_path="/tmp/test_preprocessed/preprocessor.pkl")

        # Check requires
        requires = task.requires()
        self.assertIsInstance(requires, dict)
        self.assertIsInstance(requires["model"], TrainModelTask)
        self.assertIsInstance(requires["preprocessed"], PreprocessDataTask)

    def test_output(self):
        # Create task
        task = SaveModelTask(output_filename="test_model.pkl")

        # Check output
        output = task.output()
        self.assertIsInstance(output, luigi.LocalTarget)
        self.assertTrue("test_model.pkl" in output.path)

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.pipeline.train_model.joblib.load')
    @patch('src.pipeline.train_model.joblib.dump')
    @patch('src.pipeline.train_model.io.BytesIO')
    @patch('src.pipeline.train_model.get_minio_client')
    def test_run(self, mock_get_client, mock_bytesio, mock_dump, mock_load, mock_file):
        # Setup mocks
        mock_model = MagicMock()
        mock_preprocessor = MagicMock()
        mock_load.side_effect = [mock_model, mock_preprocessor]

        mock_buffer = MagicMock()
        mock_bytesio.return_value = mock_buffer

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create task
        task = SaveModelTask(
            model_path="/tmp/test_model.pkl", 
            preprocessor_path="/tmp/test_preprocessed/preprocessor.pkl",
            output_filename="test_model.pkl"
        )

        # Run task
        task.run()

        # Assertions
        self.assertEqual(mock_load.call_count, 2)  # Called for model and preprocessor
        mock_dump.assert_called_once_with(
            {"model": mock_model, "preprocessor": mock_preprocessor}, 
            mock_buffer
        )
        mock_client.put_object.assert_called_once()

class TestCarPricePredictionPipeline(unittest.TestCase):

    def test_requires(self):
        # Create task
        task = CarPricePredictionPipeline(model_filename="test_model.pkl")

        # Check requires
        requires = task.requires()
        self.assertIsInstance(requires, SaveModelTask)
        self.assertEqual(requires.output_filename, "test_model.pkl")

    def test_output(self):
        # Create task
        task = CarPricePredictionPipeline()

        # Check output
        output = task.output()
        self.assertIsInstance(output, luigi.LocalTarget)
        self.assertEqual(output.path, "/tmp/pipeline_complete.txt")

    @patch('builtins.open', new_callable=mock_open)
    def test_run(self, mock_file):
        # Create task
        task = CarPricePredictionPipeline()

        # Run task
        task.run()

        # Assertions
        mock_file.assert_called_once()
        mock_file().write.assert_called_once_with("Pipeline completed successfully!")

if __name__ == "__main__":
    unittest.main()
