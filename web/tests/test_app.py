import unittest
from unittest.mock import patch, MagicMock
import json
import os
from flask import session
from app import app, User, check_credentials, get_postgres_connection, get_luigi_status, get_task_history


class TestApp(unittest.TestCase):
    def setUp(self):
        """Set up test client and environment variables"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        
        # Set test environment variables
        os.environ['FLASK_SECRET_KEY'] = 'test-secret-key'
        os.environ['FLASK_ADMIN_USERNAME'] = 'testadmin'
        os.environ['FLASK_ADMIN_PASSWORD'] = 'testpassword'
        os.environ['LUIGI_DB_HOST'] = 'testhost'
        os.environ['LUIGI_DB_NAME'] = 'testdb'
        os.environ['LUIGI_DB_USER'] = 'testuser'
        os.environ['LUIGI_DB_PASSWORD'] = 'testpass'
        os.environ['LUIGI_HOST'] = 'testluigi'

    def tearDown(self):
        """Clean up after tests"""
        # Remove test environment variables
        for var in ['FLASK_SECRET_KEY', 'FLASK_ADMIN_USERNAME', 'FLASK_ADMIN_PASSWORD',
                   'LUIGI_DB_HOST', 'LUIGI_DB_NAME', 'LUIGI_DB_USER', 'LUIGI_DB_PASSWORD',
                   'LUIGI_HOST']:
            if var in os.environ:
                del os.environ[var]

    def test_user_class(self):
        """Test User class initialization"""
        user = User('testuser')
        self.assertEqual(user.id, 'testuser')

    def test_check_credentials(self):
        """Test credential checking function"""
        # Valid credentials
        self.assertTrue(check_credentials('testadmin', 'testpassword'))
        
        # Invalid credentials
        self.assertFalse(check_credentials('testadmin', 'wrongpassword'))
        self.assertFalse(check_credentials('wronguser', 'testpassword'))

    @patch('app.psycopg2.connect')
    def test_get_postgres_connection(self, mock_connect):
        """Test PostgreSQL connection function"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        conn = get_postgres_connection()
        
        # Check that connect was called with correct parameters
        mock_connect.assert_called_once_with(
            host='testhost',
            database='testdb',
            user='testuser',
            password='testpass',
            port=5432
        )
        
        self.assertEqual(conn, mock_conn)

    @patch('app.requests.get')
    def test_get_luigi_status(self, mock_get):
        """Test Luigi status check function"""
        # Test when Luigi is running
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        self.assertTrue(get_luigi_status())
        mock_get.assert_called_with('http://testluigi:8082/api/graph', timeout=5)
        
        # Test when Luigi is not running
        mock_response.status_code = 500
        self.assertFalse(get_luigi_status())
        
        # Test when request raises an exception
        mock_get.side_effect = Exception("Connection error")
        self.assertFalse(get_luigi_status())

    @patch('app.get_postgres_connection')
    def test_get_task_history(self, mock_get_conn):
        """Test task history retrieval function"""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock cursor fetchall result
        mock_cursor.fetchall.return_value = [
            ('task1', 'host1', '2023-01-01 12:00:00', 'SUCCESS'),
            ('task2', 'host2', '2023-01-01 13:00:00', 'FAILURE')
        ]
        
        task_history = get_task_history()
        
        # Check that the function returns the expected task list
        self.assertEqual(len(task_history), 2)
        self.assertEqual(task_history[0]['name'], 'task1')
        self.assertEqual(task_history[0]['host'], 'host1')
        self.assertEqual(task_history[0]['event_name'], 'SUCCESS')
        self.assertEqual(task_history[1]['name'], 'task2')
        self.assertEqual(task_history[1]['event_name'], 'FAILURE')
        
        # Test exception handling
        mock_get_conn.side_effect = Exception("Database error")
        self.assertEqual(get_task_history(), [])

    def test_login_page(self):
        """Test login page rendering"""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_login_success(self):
        """Test successful login"""
        response = self.client.post('/login', data={
            'username': 'testadmin',
            'password': 'testpassword'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        # Should be redirected to dashboard
        self.assertIn(b'Dashboard', response.data)

    def test_login_failure(self):
        """Test failed login"""
        response = self.client.post('/login', data={
            'username': 'testadmin',
            'password': 'wrongpassword'
        })
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid credentials', response.data)

    def test_logout(self):
        """Test logout functionality"""
        # First login
        self.client.post('/login', data={
            'username': 'testadmin',
            'password': 'testpassword'
        })
        
        # Then logout
        response = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    @patch('app.get_luigi_status')
    @patch('app.get_task_history')
    def test_dashboard(self, mock_task_history, mock_luigi_status):
        """Test dashboard rendering"""
        # Mock dependencies
        mock_luigi_status.return_value = True
        mock_task_history.return_value = [
            {'name': 'task1', 'host': 'host1', 'ts': '2023-01-01 12:00:00', 'event_name': 'SUCCESS'}
        ]
        
        # Login first
        self.client.post('/login', data={
            'username': 'testadmin',
            'password': 'testpassword'
        })
        
        # Access dashboard
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)
        self.assertIn(b'task1', response.data)

    @patch('app.get_luigi_status')
    @patch('app.get_task_history')
    def test_task_status_api(self, mock_task_history, mock_luigi_status):
        """Test task status API endpoint"""
        # Mock dependencies
        mock_luigi_status.return_value = True
        mock_task_history.return_value = [
            {'name': 'task1', 'host': 'host1', 'ts': '2023-01-01 12:00:00', 'event_name': 'SUCCESS'}
        ]
        
        # Login first
        self.client.post('/login', data={
            'username': 'testadmin',
            'password': 'testpassword'
        })
        
        # Access API endpoint
        response = self.client.get('/api/task_status')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['luigi_running'])
        self.assertEqual(len(data['recent_tasks']), 1)
        self.assertEqual(data['recent_tasks'][0]['name'], 'task1')

    @patch('app.subprocess.run')
    def test_run_pipeline_api(self, mock_run):
        """Test run pipeline API endpoint"""
        # Mock subprocess.run
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Pipeline executed successfully"
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        # Login first
        self.client.post('/login', data={
            'username': 'testadmin',
            'password': 'testpassword'
        })
        
        # Call API endpoint
        response = self.client.post('/api/run_pipeline')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['stdout'], "Pipeline executed successfully")
        
        # Test failure case
        mock_process.returncode = 1
        mock_process.stderr = "Error executing pipeline"
        
        response = self.client.post('/api/run_pipeline')
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['stderr'], "Error executing pipeline")


if __name__ == '__main__':
    unittest.main()