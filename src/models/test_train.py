import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import pickle
import json
from unittest.mock import patch, MagicMock
from src.models.train import (
    setup_mlflow_auth,
    train_random_forest,
    train_logistic_regression,
    evaluate_model,
    save_model_and_metrics,
    train_and_log_model
)

class TestTrain(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data for testing
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Create sample dataframes for testing
        train_df = pd.DataFrame(
            np.hstack([X_train, y_train.reshape(-1, 1)]),
            columns=[f'feature_{i}' for i in range(X_train.shape[1])] + ['target']
        )
        test_df = pd.DataFrame(
            np.hstack([X_test, y_test.reshape(-1, 1)]),
            columns=[f'feature_{i}' for i in range(X_test.shape[1])] + ['target']
        )
        
        # Save dataframes to CSV files
        self.train_path = os.path.join(self.test_dir, 'train.csv')
        self.test_path = os.path.join(self.test_dir, 'test.csv')
        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)
        
        # Paths for model and metrics
        self.model_path = os.path.join(self.test_dir, 'model.pkl')
        self.metrics_path = os.path.join(self.test_dir, 'metrics.json')
    
    @patch('src.models.train.mlflow')
    def test_setup_mlflow_auth(self, mock_mlflow):
        # Test with no environment variables
        with patch.dict(os.environ, {}, clear=True):
            uri = setup_mlflow_auth()
            self.assertEqual(uri, 'http://localhost:5000')
            mock_mlflow.set_tracking_uri.assert_called_once_with('http://localhost:5000')
        
        # Test with environment variables
        with patch.dict(os.environ, {
            'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000',
            'MLFLOW_TRACKING_USERNAME': 'user',
            'MLFLOW_TRACKING_PASSWORD': 'pass'
        }, clear=True):
            mock_mlflow.reset_mock()
            uri = setup_mlflow_auth()
            self.assertEqual(uri, 'http://mlflow-server:5000')
            mock_mlflow.set_tracking_uri.assert_called_once_with('http://mlflow-server:5000')
            self.assertEqual(os.environ['MLFLOW_TRACKING_USERNAME'], 'user')
            self.assertEqual(os.environ['MLFLOW_TRACKING_PASSWORD'], 'pass')
    
    def test_train_random_forest(self):
        # Test training a random forest model
        model = train_random_forest(self.X_train, self.y_train)
        
        # Check that the model is a RandomForestClassifier
        self.assertEqual(model.__class__.__name__, 'RandomForestClassifier')
        
        # Check that the model has the correct parameters
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, 42)
        
        # Test with custom parameters
        model = train_random_forest(self.X_train, self.y_train, n_estimators=50, max_depth=5)
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 5)
    
    def test_train_logistic_regression(self):
        # Test training a logistic regression model
        model = train_logistic_regression(self.X_train, self.y_train)
        
        # Check that the model is a LogisticRegression
        self.assertEqual(model.__class__.__name__, 'LogisticRegression')
        
        # Check that the model has the correct parameters
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.random_state, 42)
        
        # Test with custom parameters
        model = train_logistic_regression(self.X_train, self.y_train, max_iter=500)
        self.assertEqual(model.max_iter, 500)
    
    def test_evaluate_model(self):
        # Train a model for evaluation
        model = train_random_forest(self.X_train, self.y_train)
        
        # Test model evaluation
        metrics = evaluate_model(model, self.X_test, self.y_test)
        
        # Check that the metrics dictionary has the expected keys
        self.assertIn('accuracy', metrics)
        self.assertIn('classification_report', metrics)
        
        # Check that the accuracy is a float between 0 and 1
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
        # Check that the classification report is a dictionary
        self.assertIsInstance(metrics['classification_report'], dict)
    
    def test_save_model_and_metrics(self):
        # Train a model
        model = train_random_forest(self.X_train, self.y_train)
        
        # Evaluate the model
        metrics = evaluate_model(model, self.X_test, self.y_test)
        
        # Test saving model and metrics
        save_model_and_metrics(model, metrics, self.model_path, self.metrics_path)
        
        # Check that the files exist
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.metrics_path))
        
        # Check that the model can be loaded
        with open(self.model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        self.assertEqual(loaded_model.__class__.__name__, 'RandomForestClassifier')
        
        # Check that the metrics can be loaded
        with open(self.metrics_path, 'r') as f:
            loaded_metrics = json.load(f)
        self.assertIn('accuracy', loaded_metrics)
        self.assertIn('classification_report', loaded_metrics)
    
    @patch('src.models.train.mlflow')
    def test_train_and_log_model(self, mock_mlflow):
        # Mock MLflow functions
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Test training and logging a model
        model, metrics = train_and_log_model(
            self.train_path,
            self.test_path,
            self.model_path,
            self.metrics_path,
            'test_experiment',
            'test_run',
            'random_forest',
            'synthetic'
        )
        
        # Check that the model is a RandomForestClassifier
        self.assertEqual(model.__class__.__name__, 'RandomForestClassifier')
        
        # Check that the metrics dictionary has the expected keys
        self.assertIn('accuracy', metrics)
        self.assertIn('classification_report', metrics)
        
        # Check that the files exist
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.metrics_path))
        
        # Check that MLflow functions were called
        mock_mlflow.set_experiment.assert_called_once_with('test_experiment')
        mock_mlflow.log_param.assert_any_call('algorithm', 'random_forest')
        mock_mlflow.log_param.assert_any_call('dataset', 'synthetic')
        mock_mlflow.log_metric.assert_called_once_with('accuracy', metrics['accuracy'])
        mock_mlflow.sklearn.log_model.assert_called_once()

if __name__ == '__main__':
    unittest.main()