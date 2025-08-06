from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import requests
import os
import subprocess
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logging
log_dir = os.getenv('LOG_DIR', '.')
log_file = os.path.join(log_dir, 'app.log')

# Ensure log directory exists
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file}")

load_dotenv()


class User(UserMixin):
    def __init__(self, username):
        self.id = username


app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this')

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


def check_credentials(username, password):
    """Simple credential check - replace with your auth system"""
    admin_user = os.getenv('FLASK_ADMIN_USERNAME', 'admin')
    admin_pass = os.getenv('FLASK_ADMIN_PASSWORD', 'admin123')
    is_valid = username == admin_user and password == admin_pass
    logger.info(f"Authentication attempt for user {username}: {'success' if is_valid else 'failed'}")
    return is_valid


def get_postgres_connection():
    """Get PostgreSQL connection"""
    host = os.getenv('LUIGI_DB_HOST', 'localhost')
    database = os.getenv('LUIGI_DB_NAME', 'luigi_db')
    user = os.getenv('LUIGI_DB_USER', 'luigi_user')
    port = os.getenv('LUIGI_DB_PORT', 5432)

    logger.info(f"Connecting to PostgreSQL database {database} on {host}:{port} as {user}")
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=os.getenv('LUIGI_DB_PASSWORD', 'luigi_password'),
            port=port
        )
        logger.info("Successfully connected to PostgreSQL database")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL database: {e}")
        raise


def get_luigi_status():
    """Get Luigi scheduler status"""
    luigi_host = os.getenv('LUIGI_HOST', 'localhost')
    logger.info(f"Checking Luigi scheduler status at {luigi_host}:8082")
    try:
        response = requests.get(f'http://{luigi_host}:8082/api/graph', timeout=5)
        is_running = response.status_code == 200
        logger.info(f"Luigi scheduler status: {'running' if is_running else 'not running'}")
        return is_running
    except Exception as e:
        logger.error(f"Error checking Luigi scheduler status: {e}")
        return False


def get_task_history():
    """Get task history from PostgreSQL"""
    logger.info("Retrieving task history from database")
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        cur.execute("""
                    SELECT * FROM (SELECT DISTINCT ON (t.id, t.name, t.host)
                        t.name,
                        t.host,
                        te.ts,
                        te.event_name
                    FROM tasks t
                    LEFT JOIN task_events te ON t.id = te.task_id
                    ORDER BY t.id, t.name, t.host, te.ts DESC NULLS LAST) AS r ORDER BY r.ts DESC
                    """)
        tasks = cur.fetchall()
        conn.close()

        task_list = [{
            'name': task[0],
            'host': task[1],
            'ts': task[2],
            'event_name': task[3]
        } for task in tasks]

        logger.info(f"Retrieved {len(task_list)} task history records")
        return task_list
    except Exception as e:
        logger.error(f"Error getting task history: {e}")
        return []


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        logger.info(f"Login attempt for user: {username}")

        if check_credentials(username, password):
            user = User(username)
            login_user(user)
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for('dashboard'))
        else:
            logger.warning(f"Failed login attempt for user: {username}")
            return render_template('login.html', error='Invalid credentials')

    logger.info("Login page accessed")
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logger.info(f"User {current_user.id} logged out")
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def dashboard():
    logger.info(f"Dashboard accessed by user {current_user.id}")
    luigi_status = get_luigi_status()
    task_history = get_task_history()

    logger.info(f"Rendering dashboard with Luigi status: {luigi_status}, task count: {len(task_history)}")
    return render_template('dashboard.html',
                           luigi_status=luigi_status,
                           tasks=task_history,
                           current_time=datetime.now())


@app.route('/luigi')
@login_required
def luigi_ui():
    """Proxy to Luigi UI with authentication"""
    return render_template('luigi_frame.html')


@app.route('/api/luigi/<path:path>')
@login_required
def luigi_proxy(path):
    """Proxy API calls to Luigi"""
    try:
        luigi_host = os.getenv('LUIGI_HOST', 'localhost')
        luigi_url = f'http://{luigi_host}:8082/{path}'
        params = request.args.to_dict()

        if request.method == 'GET':
            response = requests.get(luigi_url, params=params)
        else:
            response = requests.post(luigi_url, data=request.data, params=params)

        return response.content, response.status_code, response.headers.items()
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run_pipeline', methods=['POST'])
@login_required
def run_pipeline():
    """Trigger pipeline execution"""
    logger.info(f"Pipeline execution triggered by user {current_user.id}")
    try:
        # Run Luigi pipeline
        logger.info("Executing pipeline script")
        result = subprocess.run([
            'python', 'run_pipeline.py'
        ], capture_output=True, text=True, cwd='..')

        success = result.returncode == 0
        if success:
            logger.info("Pipeline execution completed successfully")
        else:
            logger.error(f"Pipeline execution failed with return code {result.returncode}")
            logger.error(f"Pipeline stderr: {result.stderr}")

        return jsonify({
            'success': success,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except Exception as e:
        logger.exception(f"Exception during pipeline execution: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/task_status')
@login_required
def task_status():
    """Get current task status"""
    logger.info(f"Task status API called by user {current_user.id}")
    task_history = get_task_history()
    luigi_status = get_luigi_status()

    logger.info(f"Returning task status: Luigi running: {luigi_status}, recent tasks: {len(task_history[:10])}")
    return jsonify({
        'luigi_running': luigi_status,
        'recent_tasks': task_history[:10]
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
