<div align="center">

# AirCast

</div>

### AI-Powered Air Quality Forecasting and Monitoring Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2%20%7C%20ECR-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> An end-to-end MLOps platform that forecasts air pollution levels across Indian cities — from raw data ingestion to real-time predictions, model versioning, and live observability.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [ML Workflow](#ml-workflow)
- [API Reference](#api-reference)
- [Setup & Installation](#setup--installation)
- [Docker Deployment](#docker-deployment)
- [Monitoring](#monitoring)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

Air pollution is a critical public health challenge in urban India. Predicting pollutant concentrations in advance enables authorities and citizens to take timely preventive action.

AirCast builds a production-grade ML system that predicts pollutant levels (PM2.5, PM10, NO2) from historical environmental data, surfaces those predictions via a REST API and interactive dashboard, and tracks every model experiment for full reproducibility.

---

## Key Features

| Component | Description |
|---|---|
| **Data Pipeline** | Automated ingestion, cleaning, and preprocessing of air quality datasets |
| **Feature Engineering** | Temporal signals (hour, month, season) and lag-based environmental features |
| **Model Training** | XGBoost and Scikit-learn regression models with k-fold cross-validation |
| **Experiment Tracking** | MLflow integration — metrics, parameters, and artifacts logged per run |
| **Model Registry** | Versioned model lifecycle management (Staging to Production) via MLflow |
| **Prediction API** | FastAPI REST endpoint for real-time pollution forecasting |
| **Dashboard** | Streamlit app for interactive pollution trend visualization |
| **Containerization** | Docker with optional Kubernetes deployment |
| **CI/CD** | GitHub Actions for automated testing and model deployment |
| **Monitoring** | Prometheus metrics and Grafana dashboards for model observability |

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         AirCast — MLOps Architecture                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                            DATA LAYER                                   │
  │                                                                         │
  │   Raw CSV Dataset  ──►  ingest.py  ──►  preprocess.py                  │
  │   (Air Quality India)    (S3/local)      (clean + normalize)            │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        FEATURE ENGINEERING                              │
  │                                                                         │
  │   feature_engineering.py                                                │
  │   • Temporal features   (hour, day, month, season)                     │
  │   • Lag features        (rolling averages, past N hours)               │
  │   • City-level encoding                                                 │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          MODEL TRAINING                                 │
  │                                                                         │
  │   train.py + evaluate.py                                                │
  │   • Algorithms : XGBoost, RandomForest, LinearRegression               │
  │   • Metrics    : RMSE | MAE | R²                                       │
  │   • Validation : k-fold cross-validation                               │
  └────────────────────┬────────────────────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │     MLflow Tracking     │
          │   • Params & Metrics   │
          │   • Artifacts          │
          │   • Model Registry     │
          │   Staging → Production │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────────────────────────────┐
          │                 SERVING LAYER                    │
          │                                                  │
          │  ┌──────────────────┐   ┌─────────────────────┐ │
          │  │  FastAPI Server  │   │  Streamlit Dashboard│ │
          │  │  POST /predict   │   │  Pollution Trends   │ │
          │  │  GET  /health    │   │  City Comparisons   │ │
          │  │  GET  /metrics   │   │  Forecast Charts    │ │
          │  └────────┬─────────┘   └─────────────────────┘ │
          └───────────┼──────────────────────────────────────┘
                      │
          ┌───────────▼──────────────────────┐
          │          MONITORING LAYER        │
          │                                  │
          │   Prometheus ──► Grafana         │
          │   • Prediction latency           │
          │   • Request throughput           │
          │   • Model drift signals          │
          └──────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          INFRASTRUCTURE                                 │
  │                                                                         │
  │   Docker  |  Kubernetes (optional)  |  AWS S3 + EC2 + ECR              │
  │   GitHub Actions CI/CD                                                  │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Machine Learning

| Library | Purpose |
|---|---|
| `Python 3.10+` | Core language |
| `Pandas` | Data manipulation |
| `NumPy` | Numerical computation |
| `Scikit-learn` | Preprocessing, baseline models, evaluation metrics |
| `XGBoost` | Primary gradient-boosted regression model |

### MLOps

| Tool | Purpose |
|---|---|
| `MLflow` | Experiment tracking, model registry, artifact storage |
| `Docker` | Containerization for reproducible environments |
| `GitHub Actions` | CI/CD — automated test, train, and deploy pipelines |

### Backend / API

| Tool | Purpose |
|---|---|
| `FastAPI` | REST API for predictions |
| `Uvicorn` | ASGI server |
| `Pydantic` | Request and response schema validation |

### Visualization

| Tool | Purpose |
|---|---|
| `Streamlit` | Interactive pollution monitoring dashboard |
| `Plotly` | Interactive charts |
| `Matplotlib` | Static plots and EDA |

### Infrastructure

| Service | Purpose |
|---|---|
| `AWS S3` | Raw and processed data storage |
| `AWS EC2` | Model training and API hosting |
| `AWS ECR` | Docker image registry |
| `Kubernetes` | Optional scalable orchestration |

### Monitoring

| Tool | Purpose |
|---|---|
| `Prometheus` | Metrics scraping and time-series storage |
| `Grafana` | Real-time dashboards and alerting |

---

## ML Workflow

```
1. Data Ingestion        Load raw CSV data (local or S3)
        │
        ▼
2. Preprocessing         Handle nulls, outliers, encode categoricals
        │
        ▼
3. Feature Engineering   Hour, month, season, lag features, city encoding
        │
        ▼
4. Model Training        XGBoost regressor with k-fold CV
        │
        ▼
5. Evaluation            RMSE, MAE, R² logged to MLflow
        │
        ▼
6. Model Registry        Promote best run to Production stage
        │
        ▼
7. API Deployment        FastAPI loads model from registry for inference
```

---

## API Reference

### `POST /predict`

Predict the pollution level for a given city, pollutant, and time.

**Request**
```json
{
  "city": "Delhi",
  "pollutant": "pm25",
  "hour": 10,
  "month": 5
}
```

**Response**
```json
{
  "predicted_pollution_level": 168.3
}
```

---

### `GET /health`

Returns API health status and active model version.

```json
{
  "status": "ok",
  "model_version": "1.3.0",
  "uptime_seconds": 3820
}
```

---

### `GET /metrics`

Exposes Prometheus-compatible metrics for scraping.

```
# HELP prediction_latency_seconds Time taken for a single prediction
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.05"} 412
```

---

## Setup & Installation

**Prerequisites:** Python 3.10+, Docker, AWS CLI configured

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aircast.git
cd aircast
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

```env
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_BUCKET_NAME=aircast-data
MODEL_NAME=aircast-xgboost
```

### 5. Run the Data Pipeline

```bash
python src/data/ingest.py
python src/data/preprocess.py
python src/features/feature_engineering.py
```

### 6. Train the Model

```bash
python src/models/train.py
```

MLflow UI available at `http://localhost:5000`

### 7. Start the Prediction API

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs at `http://localhost:8000/docs`

### 8. Launch the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard available at `http://localhost:8501`

---

## Monitoring

Start the observability stack via Docker Compose:

```bash
docker compose -f docker/docker-compose.monitoring.yml up -d
```

| Service | URL | Purpose |
|---|---|---|
| Prometheus | `http://localhost:9090` | Metrics collection |
| Grafana | `http://localhost:3000` | Dashboards and alerts |

**Tracked Metrics**

| Metric | Description |
|---|---|
| `prediction_latency_seconds` | Inference time per request |
| `prediction_requests_total` | Cumulative request count |
| `model_drift_score` | Feature distribution shift signal |
| `api_error_rate` | 4xx / 5xx error rate |

---

## Roadmap

- [x] Real-time air quality data ingestion via public APIs
- [ ] Time-series forecasting with LSTM and Prophet
- [ ] Automated retraining pipeline triggered by drift detection
- [ ] Multi-city, multi-pollutant forecasting
- [ ] Airflow / Prefect DAG orchestration
- [ ] Feature Store integration (Feast)

---

## Author

**Digpal Singh Rathore** — Cloud · DevOps · ML Systems

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat-square&logo=github)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)

---

## License

This project is licensed under the [MIT License](LICENSE).