import unittest
import os
import tempfile
import pandas as pd
import luigi
from unittest.mock import patch, MagicMock
from src.pipeline.tasks import (
    GenerateData,
    LoadWineDataset,
    LoadBreastCancerDataset,
    PreprocessData,
    PreprocessWineData,
    PreprocessBreastCancerData,
    TrainModel,
    TrainWineModel,
    TrainBreastCancerModel,
    MLPipeline
)

class TestTasks(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create paths for test outputs
        self.raw_data_path = os.path.join(self.test_dir, 'raw_data.csv')
        self.train_data_path = os.path.join(self.test_dir, 'train_data.csv')
        self.test_data_path = os.path.join(self.test_dir, 'test_data.csv')
        self.model_path = os.path.join(self.test_dir, 'model.pkl')
        self.metrics_path = os.path.join(self.test_dir, 'metrics.json')
    
    def test_generate_data_task(self):
        # Create a GenerateData task with a custom output path
        task = GenerateData()
        task.output = lambda: luigi.LocalTarget(self.raw_data_path)
        
        # Run the task
        task.run()
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(self.raw_data_path))
        
        # Check that the output file is a valid CSV
        df = pd.read_csv(self.raw_data_path)
        
        # Check that the dataframe has the expected columns
        self.assertIn('target', df.columns)
        
        # Check that the dataframe has the expected number of rows
        self.assertEqual(len(df), 1000)
    
    def test_load_wine_dataset_task(self):
        # Create a LoadWineDataset task with a custom output path
        task = LoadWineDataset()
        task.output = lambda: luigi.LocalTarget(self.raw_data_path)
        
        # Run the task
        task.run()
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(self.raw_data_path))
        
        # Check that the output file is a valid CSV
        df = pd.read_csv(self.raw_data_path)
        
        # Check that the dataframe has the expected columns
        self.assertIn('target', df.columns)
    
    def test_load_breast_cancer_dataset_task(self):
        # Create a LoadBreastCancerDataset task with a custom output path
        task = LoadBreastCancerDataset()
        task.output = lambda: luigi.LocalTarget(self.raw_data_path)
        
        # Run the task
        task.run()
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(self.raw_data_path))
        
        # Check that the output file is a valid CSV
        df = pd.read_csv(self.raw_data_path)
        
        # Check that the dataframe has the expected columns
        self.assertIn('target', df.columns)
    
    def test_preprocess_data_task(self):
        # Create a GenerateData task with a custom output path
        generate_task = GenerateData()
        generate_task.output = lambda: luigi.LocalTarget(self.raw_data_path)
        
        # Run the GenerateData task
        generate_task.run()
        
        # Create a PreprocessData task with custom input and output paths
        preprocess_task = PreprocessData()
        preprocess_task.requires = lambda: generate_task
        preprocess_task.output = lambda: [
            luigi.LocalTarget(self.train_data_path),
            luigi.LocalTarget(self.test_data_path)
        ]
        
        # Run the PreprocessData task
        preprocess_task.run()
        
        # Check that the output files exist
        self.assertTrue(os.path.exists(self.train_data_path))
        self.assertTrue(os.path.exists(self.test_data_path))
        
        # Check that the output files are valid CSVs
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        
        # Check that the dataframes have the expected columns
        self.assertIn('target', train_df.columns)
        self.assertIn('target', test_df.columns)
        
        # Check that the split sizes are correct (80% train, 20% test)
        raw_df = pd.read_csv(self.raw_data_path)
        expected_train_size = int(len(raw_df) * 0.8)
        expected_test_size = len(raw_df) - expected_train_size
        self.assertEqual(len(train_df), expected_train_size)
        self.assertEqual(len(test_df), expected_test_size)
    
    def test_preprocess_wine_data_task(self):
        # Create a LoadWineDataset task with a custom output path
        load_task = LoadWineDataset()
        load_task.output = lambda: luigi.LocalTarget(self.raw_data_path)
        
        # Run the LoadWineDataset task
        load_task.run()
        
        # Create a PreprocessWineData task with custom input and output paths
        preprocess_task = PreprocessWineData()
        preprocess_task.requires = lambda: load_task
        preprocess_task.output = lambda: [
            luigi.LocalTarget(self.train_data_path),
            luigi.LocalTarget(self.test_data_path)
        ]
        
        # Run the PreprocessWineData task
        preprocess_task.run()
        
        # Check that the output files exist
        self.assertTrue(os.path.exists(self.train_data_path))
        self.assertTrue(os.path.exists(self.test_data_path))
        
        # Check that the output files are valid CSVs
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        
        # Check that the dataframes have the expected columns
        self.assertIn('target', train_df.columns)
        self.assertIn('target', test_df.columns)
    
    def test_preprocess_breast_cancer_data_task(self):
        # Create a LoadBreastCancerDataset task with a custom output path
        load_task = LoadBreastCancerDataset()
        load_task.output = lambda: luigi.LocalTarget(self.raw_data_path)
        
        # Run the LoadBreastCancerDataset task
        load_task.run()
        
        # Create a PreprocessBreastCancerData task with custom input and output paths
        preprocess_task = PreprocessBreastCancerData()
        preprocess_task.requires = lambda: load_task
        preprocess_task.output = lambda: [
            luigi.LocalTarget(self.train_data_path),
            luigi.LocalTarget(self.test_data_path)
        ]
        
        # Run the PreprocessBreastCancerData task
        preprocess_task.run()
        
        # Check that the output files exist
        self.assertTrue(os.path.exists(self.train_data_path))
        self.assertTrue(os.path.exists(self.test_data_path))
        
        # Check that the output files are valid CSVs
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        
        # Check that the dataframes have the expected columns
        self.assertIn('target', train_df.columns)
        self.assertIn('target', test_df.columns)
    
    @patch('src.pipeline.tasks.mlflow')
    def test_train_model_task(self, mock_mlflow):
        # Create sample train and test data
        train_df = pd.DataFrame({
            'feature_1': [0.1, 0.2, 0.3, 0.4],
            'feature_2': [0.5, 0.6, 0.7, 0.8],
            'target': [0, 1, 0, 1]
        })
        test_df = pd.DataFrame({
            'feature_1': [0.9, 1.0],
            'feature_2': [1.1, 1.2],
            'target': [1, 0]
        })
        
        # Save dataframes to CSV files
        train_df.to_csv(self.train_data_path, index=False)
        test_df.to_csv(self.test_data_path, index=False)
        
        # Create a PreprocessData task with custom output paths
        preprocess_task = PreprocessData()
        preprocess_task.output = lambda: [
            luigi.LocalTarget(self.train_data_path),
            luigi.LocalTarget(self.test_data_path)
        ]
        
        # Create a TrainModel task with custom input and output paths
        train_task = TrainModel()
        train_task.requires = lambda: preprocess_task
        train_task.output = lambda: [
            luigi.LocalTarget(self.model_path),
            luigi.LocalTarget(self.metrics_path)
        ]
        
        # Mock MLflow functions
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Run the TrainModel task
        train_task.run()
        
        # Check that the output files exist
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.metrics_path))
        
        # Check that MLflow functions were called
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with("synthetic_data_experiment")
        mock_mlflow.log_param.assert_any_call("algorithm", "RandomForest")
        mock_mlflow.log_param.assert_any_call("dataset", "synthetic")
        mock_mlflow.log_metric.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_called_once()
    
    @patch('src.pipeline.tasks.mlflow')
    def test_ml_pipeline_task(self, mock_mlflow):
        # Create an MLPipeline task
        pipeline_task = MLPipeline()
        
        # Check that the requires method returns a list of tasks
        required_tasks = pipeline_task.requires()
        
        # Check that the required tasks are of the expected types
        self.assertEqual(len(required_tasks), 3)
        self.assertIsInstance(required_tasks[0], TrainModel)
        self.assertIsInstance(required_tasks[1], TrainWineModel)
        self.assertIsInstance(required_tasks[2], TrainBreastCancerModel)

if __name__ == '__main__':
    unittest.main()