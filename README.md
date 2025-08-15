# Model Service

[![CI/CD Pipeline For Luigi Scheduler](https://github.com/team-xx-capstone-deploycamp/model-service/actions/workflows/workflow-scheduler.yml/badge.svg)](https://github.com/team-xx-capstone-deploycamp/model-service/actions/workflows/workflow-scheduler.yml)
[![CI/CD Pipeline For Luigi Wrapper](https://github.com/team-xx-capstone-deploycamp/model-service/actions/workflows/workflow-wrapper.yml/badge.svg)](https://github.com/team-xx-capstone-deploycamp/model-service/actions/workflows/workflow-wrapper.yml)
[![Model Training Pipeline](https://github.com/team-xx-capstone-deploycamp/model-service/actions/workflows/model-training.yml/badge.svg)](https://github.com/team-xx-capstone-deploycamp/model-service/actions/workflows/model-training.yml)

A machine learning model service for car price prediction with data processing and training pipelines. The service uses Luigi for task orchestration, MLflow for experiment tracking, and DVC for data version control.

## What This Service Does

This service provides an end-to-end machine learning pipeline for car price prediction:

1. **Data Loading & Cleaning**: Loads car price data from local files or MinIO storage and performs data cleaning
2. **Data Preprocessing**: Handles feature engineering, encoding categorical variables, and train-test splitting
3. **Model Training**: Trains an XGBoost regression model with hyperparameter tuning
4. **Model Evaluation**: Calculates metrics like RMSE, MAE, MAPE, and R²
5. **Model Storage**: Saves trained models to MinIO and logs experiments to MLflow
6. **Web Interface**: Provides a web dashboard for monitoring and managing the pipeline

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
uv pip install -r docker/luigi/requirements.txt
uv pip install -r docker/wrapper/requirements.txt
uv pip install -r requirements-dev.txt

# Or using pip
pip install -r docker/luigi/requirements.txt
pip install -r docker/wrapper/requirements.txt
pip install -r requirements-dev.txt
```

## Development

### Running the Service with Docker (Recommended)

The easiest way to run the service is using Docker Compose:

```bash
# Run the service with Docker Compose for development
docker compose -f docker-compose.dev.yml up

# Run specific services only
docker compose -f docker-compose.dev.yml up luigi wrapper postgres

# Run in detached mode
docker compose -f docker-compose.dev.yml up -d
```

### Accessing the Services

Once running, you can access:
- **Web Interface**: http://localhost:5001 (Default credentials: admin/admin123)
- **Luigi Scheduler**: http://localhost:8082
- **PostgreSQL**: localhost:5432 (User: luigi, Password: luigipass)

### Running Tests
```bash
# Run tests locally with uv
uv pip install -r requirements-dev.txt
pytest

# Run tests with Docker Compose
docker compose -f docker-compose.dev.yml up test

# Run a specific test with Docker Compose
docker compose -f docker-compose.dev.yml run test pytest src/tests/test_train_model.py -v
```

## CI/CD Workflow

The project uses GitHub Actions for CI/CD with three main workflows:

1. **Luigi Wrapper Workflow** (.github/workflows/workflow-wrapper.yml):
   - Triggered on pushes to main/prod branches and PRs to prod
   - Runs code quality checks and tests for the web interface
   - Builds and pushes Docker images to GitHub Container Registry
   - Performs security scanning with Trivy
   - Deploys to production VPS when merged to prod

2. **Luigi Scheduler Workflow** (.github/workflows/workflow-scheduler.yml):
   - Similar to the wrapper workflow but for the scheduler component
   - Can perform full rebuilds or just update source files based on changes

3. **Model Training Workflow** (.github/workflows/model-training.yml):
   - Triggered after scheduler deployment or when model code changes
   - Pulls data using DVC
   - Runs the model training pipeline on the production environment

## Project Structure

- `src/`: Source code
  - `pipeline/`: Luigi pipeline tasks for data processing and model training
  - `tests/`: Test files
- `config/`: Configuration files for Luigi and other services
- `data/`: Data files (version controlled with DVC)
- `web/`: Web interface for wrapper Luigi dashboard
- `docker/`: Docker configuration files
  - `luigi/`: Luigi scheduler Dockerfile and configuration
  - `wrapper/`: Web interface Dockerfile and configuration

## Docker Services

### Development Environment (docker-compose.dev.yml)

- **luigi**: Luigi scheduler for orchestrating pipeline tasks
- **wrapper**: Web interface for monitoring and managing the pipeline
- **postgres**: PostgreSQL database for storing task history

### Production Environment

In production, the services are deployed to a VPS with:
- Environment variables for configuration
- Secure connections to external services
- Automated deployments via GitHub Actions
