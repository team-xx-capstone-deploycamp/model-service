# Model Service

[![CI/CD Pipeline For Luigi Scheduler](https://github.com/pebrisulistiyo/model-service/actions/workflows/workflow-scheduler.yml/badge.svg)](https://github.com/pebrisulistiyo/model-service/actions/workflows/workflow-scheduler.yml)
[![CI/CD Pipeline For Luigi Wrapper](https://github.com/pebrisulistiyo/model-service/actions/workflows/workflow-wrapper.yml/badge.svg)](https://github.com/pebrisulistiyo/model-service/actions/workflows/workflow-wrapper.yml)
[![Model Training Pipeline](https://github.com/pebrisulistiyo/model-service/actions/workflows/model-training.yml/badge.svg)](https://github.com/pebrisulistiyo/model-service/actions/workflows/model-training.yml)

A machine learning model service with data processing and training pipelines.

## Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.11
- DVC for data version control

### Installation

1. Clone the repository
2. Install dependencies:
```bash
# Using uv (recommended)
pip install uv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# Or using pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development

### Running the Pipeline
```bash
# Run only the pipeline
python app.py --pipeline

# Run only the web application
python app.py --web

# Run both (default)
python app.py
```

### Running with Docker

#### Development Environment
```bash
# Run the service with Docker Compose for development
docker compose -f docker-compose.dev.yml up

# Run the service with specific services
docker compose -f docker-compose.dev.yml up app postgres luigi-scheduler mlflow

# Run in detached mode
docker compose -f docker-compose.dev.yml up -d

# Access the web interface
# Open http://localhost:5001 in your browser
# Default credentials: admin/admin123
```

#### Production Environment
```bash
# Create a .env file with the necessary environment variables
# Example:
# POSTGRES_HOST=production-db-host
# POSTGRES_DB=luigi_db
# POSTGRES_USER=luigi_user
# POSTGRES_PASSWORD=secure_password
# MLFLOW_TRACKING_URI=http://production-mlflow:5000
# LUIGI_HOST=production-luigi-scheduler

# Run the service with Docker Compose for production
docker compose -f docker-compose.ci.yml up -d

# Access the web interface
# Open http://localhost:5001 in your browser
```

### Running Tests
```bash
# Run tests locally with uv
uv pip install -r requirements-dev.txt
pytest

# Run tests locally with pip
pip install -r requirements-dev.txt
pytest

# Run tests with Docker Compose
docker compose -f docker-compose.dev.yml up test

# Run a specific test with Docker Compose
docker compose -f docker-compose.dev.yml run test pytest tests/test_pipeline.py -v
```

## CI/CD Workflow

The project uses GitHub Actions for CI/CD:

1. **Feature Branches**: Run tests and code quality checks
2. **Dev Branch**: Run tests, build and deploy to development environment
3. **Main Branch**: Run tests, security scans, and deploy to production

## Project Structure

- `src/`: Source code
  - `data/`: Data processing modules
  - `models/`: Model training modules
  - `pipeline/`: Luigi pipeline tasks
- `tests/`: Test files
- `config/`: Configuration files
- `web/`: Web interface

## Docker Services

### Development Environment (docker-compose.dev.yml)

The development environment includes the following services:

- **app**: The main application service that runs the web interface and pipeline
- **test**: A service for running tests
- **postgres**: A PostgreSQL database for storing task history and application data
- **luigi-scheduler**: A Luigi scheduler for managing pipeline tasks
- **mlflow**: An MLflow server for tracking machine learning experiments

### Production Environment (docker-compose.prod.yml)

The production environment includes only the main application service:

- **app**: The main application service that runs the web interface and pipeline

In production, the PostgreSQL, Luigi scheduler, and MLflow services are expected to be provided externally, with their connection details specified through environment variables.

### Service Ports

- Web Interface: http://localhost:5001
- Luigi Scheduler: http://localhost:8082
- MLflow: http://localhost:5000
- PostgreSQL: localhost:5432
