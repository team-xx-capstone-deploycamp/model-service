import unittest
import os
import tempfile
import pandas as pd
import luigi
from src.pipeline.tasks import GenerateData, LoadWineDataset, LoadBreastCancerDataset

class TestPipelineTasks(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

    def test_generate_data(self):
        # Test that GenerateData task creates a valid output file
        task = GenerateData()
        # Override the output path to use our test directory
        task.output = lambda: luigi.LocalTarget(os.path.join(self.test_dir, 'synthetic_data.csv'))
        task.run()

        # Check that the output file exists and is a valid CSV
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'synthetic_data.csv')))
        df = pd.read_csv(os.path.join(self.test_dir, 'synthetic_data.csv'))
        self.assertGreater(len(df), 0)

    def test_load_wine_dataset(self):
        # Test that LoadWineDataset task creates a valid output file
        task = LoadWineDataset()
        # Override the output path to use our test directory
        task.output = lambda: luigi.LocalTarget(os.path.join(self.test_dir, 'wine_data.csv'))
        task.run()

        # Check that the output file exists and is a valid CSV
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'wine_data.csv')))
        df = pd.read_csv(os.path.join(self.test_dir, 'wine_data.csv'))
        self.assertGreater(len(df), 0)

    def test_load_breast_cancer_dataset(self):
        # Test that LoadBreastCancerDataset task creates a valid output file
        task = LoadBreastCancerDataset()
        # Override the output path to use our test directory
        task.output = lambda: luigi.LocalTarget(os.path.join(self.test_dir, 'breast_cancer_data.csv'))
        task.run()

        # Check that the output file exists and is a valid CSV
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'breast_cancer_data.csv')))
        df = pd.read_csv(os.path.join(self.test_dir, 'breast_cancer_data.csv'))
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()
