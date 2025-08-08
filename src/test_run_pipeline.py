import unittest
import os
from unittest.mock import patch, MagicMock

class TestRunPipeline(unittest.TestCase):
    @patch('src.run_pipeline.luigi')
    @patch('src.run_pipeline.os')
    def test_run_pipeline(self, mock_os, mock_luigi):
        # Import the module after patching
        import src.run_pipeline
        
        # Check that the environment variable was set correctly
        mock_os.environ.__setitem__.assert_called_once_with('LUIGI_CONFIG_PATH', 'config/luigi.cfg')
        
        # Check that luigi.run was called with the correct arguments
        mock_luigi.run.assert_called_once_with(['MLPipeline', '--workers=1', '--local-scheduler'])

if __name__ == '__main__':
    unittest.main()