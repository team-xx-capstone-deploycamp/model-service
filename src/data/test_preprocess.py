import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from src.data.preprocess import (
    generate_synthetic_data,
    load_wine_data,
    load_breast_cancer_data,
    split_data,
    preprocess_data
)

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
    def test_generate_synthetic_data(self):
        # Test that synthetic data is generated correctly
        df = generate_synthetic_data(n_samples=100, n_features=10)
        
        # Check that the dataframe has the correct shape
        self.assertEqual(df.shape, (100, 11))  # 10 features + 1 target column
        
        # Check that the feature columns are named correctly
        for i in range(10):
            self.assertIn(f'feature_{i}', df.columns)
        
        # Check that the target column exists
        self.assertIn('target', df.columns)
        
        # Check that the target values are binary (0 or 1)
        self.assertTrue(set(df['target'].unique()).issubset({0, 1}))
    
    def test_load_wine_data(self):
        # Test that wine data is loaded correctly
        df = load_wine_data()
        
        # Check that the dataframe is not empty
        self.assertGreater(len(df), 0)
        
        # Check that the target column exists
        self.assertIn('target', df.columns)
    
    def test_load_breast_cancer_data(self):
        # Test that breast cancer data is loaded correctly
        df = load_breast_cancer_data()
        
        # Check that the dataframe is not empty
        self.assertGreater(len(df), 0)
        
        # Check that the target column exists
        self.assertIn('target', df.columns)
    
    def test_split_data(self):
        # Create a sample dataframe
        df = pd.DataFrame({
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Test splitting with default parameters
        train_df, test_df = split_data(df)
        
        # Check that the split sizes are correct (80% train, 20% test)
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)
        
        # Test splitting with custom test_size
        train_df, test_df = split_data(df, test_size=0.3)
        
        # Check that the split sizes are correct (70% train, 30% test)
        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(test_df), 30)
    
    def test_preprocess_data(self):
        # Create a sample dataframe and save it to a temporary file
        df = pd.DataFrame({
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        input_path = os.path.join(self.test_dir, 'input.csv')
        output_train_path = os.path.join(self.test_dir, 'train.csv')
        output_test_path = os.path.join(self.test_dir, 'test.csv')
        
        df.to_csv(input_path, index=False)
        
        # Test preprocessing
        train_df, test_df = preprocess_data(input_path, output_train_path, output_test_path, test_size=0.2)
        
        # Check that the output files exist
        self.assertTrue(os.path.exists(output_train_path))
        self.assertTrue(os.path.exists(output_test_path))
        
        # Check that the split sizes are correct (80% train, 20% test)
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)
        
        # Check that the output files can be read as dataframes
        train_df_read = pd.read_csv(output_train_path)
        test_df_read = pd.read_csv(output_test_path)
        
        self.assertEqual(len(train_df_read), 80)
        self.assertEqual(len(test_df_read), 20)

if __name__ == '__main__':
    unittest.main()