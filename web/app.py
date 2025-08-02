from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import requests
import os
import subprocess
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import json

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
    return username == admin_user and password == admin_pass


def get_postgres_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        database=os.getenv('POSTGRES_DB', 'luigi_db'),
        user=os.getenv('POSTGRES_USER', 'luigi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'luigi_password'),
        port=os.getenv('POSTGRES_PORT', 5432)
    )


def get_luigi_status():
    """Get Luigi scheduler status"""
    try:
        response = requests.get('http://localhost:8082/api/graph', timeout=5)
        return response.status_code == 200
    except:
        return False


def get_task_history():
    """Get task history from PostgreSQL"""
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        cur.execute("""
                    SELECT task_family, task_id, status, host, start_time, finish_time
                    FROM task_events
                    ORDER BY start_time DESC LIMIT 50
                    """)
        tasks = cur.fetchall()
        conn.close()

        return [{
            'family': task[0],
            'id': task[1],
            'status': task[2],
            'host': task[3],
            'start_time': task[4],
            'finish_time': task[5]
        } for task in tasks]
    except Exception as e:
        print(f"Error getting task history: {e}")
        return []


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if check_credentials(username, password):
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def dashboard():
    luigi_status = get_luigi_status()
    task_history = get_task_history()

    return render_template('dashboard.html',
                           luigi_status=luigi_status,
                           task_history=task_history,
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
        luigi_url = f'http://localhost:8082/{path}'
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
    try:
        # Run Luigi pipeline
        result = subprocess.run([
            'python', 'run_pipeline.py'
        ], capture_output=True, text=True, cwd='..')

        return jsonify({
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/task_status')
@login_required
def task_status():
    """Get current task status"""
    task_history = get_task_history()
    luigi_status = get_luigi_status()

    return jsonify({
        'luigi_running': luigi_status,
        'recent_tasks': task_history[:10]
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)