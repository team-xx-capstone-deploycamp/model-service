import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import io
import joblib
import luigi
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

    @patch('src.pipeline.train_model.get_minio_client')
    @patch('src.pipeline.train_model.pd.read_csv')
    @patch('src.pipeline.train_model.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_run(self, mock_file, mock_makedirs, mock_read_csv, mock_get_client):
        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_client.get_object.return_value = mock_response

        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df

        # Create task
        task = LoadDataTask(dataset_filename="test.csv", output_path="/tmp/test_data.csv")

        # Run task
        task.run()

        # Assertions
        mock_get_client.assert_called_once()
        mock_client.get_object.assert_called_once()
        mock_read_csv.assert_called_once()
        mock_df.to_csv.assert_called_once()

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
    def test_run(self, mock_dump, mock_file, mock_makedirs, mock_split, mock_read_csv):
        # Setup mocks
        mock_df = pd.DataFrame({
            'brand': ['Toyota', 'Honda'],
            'model': ['Camry', 'Civic'],
            'price': [25000, 22000]
        })
        mock_read_csv.return_value = mock_df

        mock_X_train = pd.DataFrame({'brand': [0], 'model': [0]})
        mock_X_test = pd.DataFrame({'brand': [1], 'model': [1]})
        mock_y_train = pd.Series([25000])
        mock_y_test = pd.Series([22000])
        mock_split.return_value = (mock_X_train, mock_X_test, mock_y_train, mock_y_test)

        # Create task with mocked output
        task = PreprocessDataTask(input_path="/tmp/test_data.csv", output_path="/tmp/test_output")
        task.output = MagicMock(return_value={
            "X_train": MagicMock(path="/tmp/test_output/X_train.csv"),
            "X_test": MagicMock(path="/tmp/test_output/X_test.csv"),
            "y_train": MagicMock(path="/tmp/test_output/y_train.csv"),
            "y_test": MagicMock(path="/tmp/test_output/y_test.csv"),
            "encoders": MagicMock(path="/tmp/test_output/encoders.pkl")
        })

        # Run task
        task.run()

        # Assertions
        mock_read_csv.assert_called_once()
        mock_split.assert_called_once()
        mock_makedirs.assert_called_once()
        self.assertEqual(mock_dump.call_count, 1)  # Called for encoders

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
    @patch('src.pipeline.train_model.XGBRegressor')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.pipeline.train_model.joblib.dump')
    def test_run(self, mock_dump, mock_file, mock_xgb, mock_start_run, mock_setup, mock_read_csv):
        # Setup mocks
        mock_X_train = pd.DataFrame({'brand': [0, 1], 'model': [0, 1]})
        mock_y_train = pd.Series([25000, 22000])
        mock_X_test = pd.DataFrame({'brand': [2], 'model': [2]})
        mock_y_test = pd.Series([20000])

        mock_read_csv.side_effect = [mock_X_train, mock_y_train, mock_X_test, mock_y_test]

        mock_model = MagicMock()
        mock_xgb.return_value = mock_model
        mock_model.predict.return_value = np.array([21000])

        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context

        # Create task with mocked input
        task = TrainModelTask(input_path="/tmp/test_preprocessed", output_path="/tmp/test_model.pkl")
        task.input = MagicMock(return_value={
            "X_train": MagicMock(path="/tmp/test_preprocessed/X_train.csv"),
            "X_test": MagicMock(path="/tmp/test_preprocessed/X_test.csv"),
            "y_train": MagicMock(path="/tmp/test_preprocessed/y_train.csv"),
            "y_test": MagicMock(path="/tmp/test_preprocessed/y_test.csv")
        })

        # Run task
        task.run()

        # Assertions
        self.assertEqual(mock_read_csv.call_count, 4)  # Called for X_train, y_train, X_test, y_test
        mock_setup.assert_called_once()
        mock_start_run.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_model.predict.assert_called_once()
        mock_dump.assert_called_once()

class TestSaveModelTask(unittest.TestCase):

    def test_requires(self):
        # Create task
        task = SaveModelTask(model_path="/tmp/test_model.pkl", encoders_path="/tmp/test_preprocessed/encoders.pkl")

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
        mock_encoders = MagicMock()
        mock_load.side_effect = [mock_model, mock_encoders]

        mock_buffer = MagicMock()
        mock_bytesio.return_value = mock_buffer

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create task
        task = SaveModelTask(
            model_path="/tmp/test_model.pkl", 
            encoders_path="/tmp/test_preprocessed/encoders.pkl",
            output_filename="test_model.pkl"
        )

        # Run task
        task.run()

        # Assertions
        self.assertEqual(mock_load.call_count, 2)  # Called for model and encoders
        mock_dump.assert_called_once()
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
