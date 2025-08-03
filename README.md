# Model Service

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
python web/app.py --pipeline

# Run only the web application
python web/app.py --web

# Run both (default)
python web/app.py
```

### Running with Docker
```bash
# Run the service with Docker Compose
docker-compose up
```

### Running Tests
```bash
# Run tests locally with uv
uv pip install -r requirements-dev.txt
pytest

# Run tests locally with pip
pip install -r requirements-dev.txt
pytest
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
