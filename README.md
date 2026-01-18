# SBA Loan Default Prediction - MLOps Platform

A production-ready machine learning platform for predicting Small Business Administration (SBA) loan defaults with explainable AI. Features real-time risk assessment through a FastAPI backend and Streamlit dashboard with SHAP-based explanations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start (Local)](#quick-start-local)
- [Project Structure](#project-structure)
- [Running the Application](#running-the-application)
- [Model Performance](#model-performance)
- [AWS Deployment](#aws-deployment)
- [API Documentation](#api-documentation)
- [Testing](#testing)

---

## Overview

This platform analyzes SBA 7(a) loan applications and predicts default probability using an XGBoost model trained on historical loan data.

**Key Features:**
- **Real-time predictions** via REST API
- **Explainable AI** with SHAP waterfall plots
- **Banker-friendly UI** for loan officers
- **Cloud deployment** on AWS ECS Fargate with ALB
- **Training-inference consistency** through shared feature engineering

**Business Impact:**
- Quantify default probability for loan applications
- Understand which features drive risk predictions
- Provide transparent, auditable risk assessments
- Recommend APPROVE/REJECT based on optimized risk thresholds (28%)

---

## Quick Start (Local)

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (optional, for containerized deployment)
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd ml-eng-lr
```

### Step 2: Download Data

Download the SBA FOIA dataset from:
**https://data.sba.gov/dataset/7-a-504-foia**

1. Visit the above URL
2. Download the **"7(a) FY2020 - Present"** CSV file
3. Rename it to: `foia-7a-fy2020-present-asof-250930.csv`
4. Place it in: `data/raw/`

```bash
# Create data directory
mkdir -p data/raw

# Move downloaded file (adjust filename as needed)
mv ~/Downloads/7a_*.csv data/raw/foia-7a-fy2020-present-asof-250930.csv
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run ML Pipeline

Generate the trained model and feature encoders:

```bash
# Run full pipeline (preprocessing + training)
python run_pipeline.py

# Or with fewer trials for faster training (for testing)
python run_pipeline.py --n-trials 10
```

**Expected Output:**
```
✓ PIPELINE COMPLETE
Local artifacts:
  Model:       models/trained/xgb_tuned.joblib
  Encoder:     models/encoders/frequency_map.pkl
  Data:        data/feature/processed_data.parquet
```

### Step 5: Run the Application

**Option A: Using Docker Compose (Recommended)**

```bash
# Build and start both API and UI services
docker-compose up --build

# Access:
# - Streamlit UI: http://localhost:8501
# - FastAPI: http://localhost:8000
# - API Docs: http://localhost:8000/api/docs
```

**Option B: Using Python Directly**

Open two terminal windows:

**Terminal 1 - FastAPI Backend:**
```bash
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 - Streamlit Frontend:**
```bash
source venv/bin/activate
streamlit run app.py
```

**Access:**
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

---

## Project Structure

```
ml-eng-lr/
│
├── data/
│   ├── raw/                               # Raw CSV from SBA FOIA
│   └── feature/                           # Processed features (generated)
│
├── models/
│   ├── trained/                           # Trained XGBoost models (generated)
│   └── encoders/                          # Feature encoders (generated)
│
├── src/
│   ├── feature_pipeline/                  # Data preprocessing
│   │   ├── load.py                        # Load CSV, filter loans
│   │   ├── cleaning.py                    # Handle missing values
│   │   └── engineering.py                 # Create features
│   │
│   ├── training_pipeline/                 # Model training
│   │   ├── train_baseline.py              # Baseline XGBoost
│   │   ├── tune_optuna.py                 # Hyperparameter tuning
│   │   └── evaluation.py                  # Metrics (ROC-AUC, KS, etc.)
│   │
│   ├── inference_pipeline/
│   │   └── predict.py                     # LoanPredictor (predict + explain)
│   │
│   ├── api/
│   │   └── main.py                        # FastAPI REST API
│   │
│   └── utils/
│       ├── feature_engineering.py         # Shared feature engineering
│       └── s3_manager.py                  # S3 artifact management
│
├── infrastructure/                        # AWS ECS deployment configs
│   ├── ecs-task-api.json
│   └── ecs-task-ui.json
│
├── app.py                                 # Streamlit dashboard
├── run_pipeline.py                        # ⭐ Pipeline orchestrator
├── docker-compose.yml                     # Local multi-container setup
├── Dockerfile.api                         # API container
├── Dockerfile.streamlit                   # UI container
└── requirements.txt
```

---

## Running the Application

### Local Development

**1. Run ML Pipeline**
```bash
# Full pipeline
python run_pipeline.py

# Skip preprocessing if data exists
python run_pipeline.py --skip-preprocessing

# Upload to S3 (requires AWS credentials)
python run_pipeline.py --upload
```

**2. Start Services**

**Option A: Docker Compose**
```bash
docker-compose up --build
```

**Option B: Python**
```bash
# Terminal 1: API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: UI
streamlit run app.py
```

### Making Predictions

**Via Streamlit UI:**
1. Open http://localhost:8501
2. Fill in loan details
3. Click "Assess Risk"
4. View prediction + SHAP explanation

**Via REST API:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "GrossApproval": 50000,
    "SBAGuaranteedApproval": 37500,
    "InitialInterestRate": 6.5,
    "ApprovalFY": 2020,
    "RevolverStatus": 0,
    "JobsSupported": 5,
    "NAICSCode": "441110",
    "BusinessType": "CORPORATION",
    "BusinessAge": "Existing or more than 2 years old",
    "ProjectState": "CA",
    "BankState": "CA",
    "LocationID": 12345.0,
    "ApprovalDate": "2020-03-15",
    "BankNCUANumber": null,
    "FranchiseCode": null,
    "FixedorVariableInterestRate": "F",
    "CollateralInd": "Y"
  }'
```

**Via Python:**
```python
from src.inference_pipeline.predict import LoanPredictor
import pandas as pd

predictor = LoanPredictor()

loan = pd.DataFrame([{
    "GrossApproval": 50000,
    "SBAGuaranteedApproval": 37500,
    "ApprovalFY": 2020,
    "InitialInterestRate": 6.5,
    "RevolverStatus": 0,
    "JobsSupported": 5,
    "ApprovalDate": "2020-03-15",
    "NAICSCode": "441110",
    "BusinessType": "CORPORATION",
    "BusinessAge": "Existing or more than 2 years old",
    "ProjectState": "CA",
    "BankState": "CA",
    "LocationID": 12345.0,
    "BankNCUANumber": None,
    "FranchiseCode": None,
    "FixedorVariableInterestRate": "F",
    "CollateralInd": "Y",
}])

# Predict
prob = predictor.predict(loan)[0]
print(f"Default probability: {prob:.2%}")

# Explain
shap_values = predictor.explain(loan)
```

---

## Model Performance

**Model:** XGBoost with Optuna hyperparameter tuning
**Test Set:** 11,167 samples (92.55% Good, 7.45% Default)
**Threshold:** 28% (optimized for recall)

| Metric              | Value  | Interpretation |
|---------------------|--------|----------------|
| **ROC-AUC**         | 0.8452 | Excellent discrimination |
| **Recall**          | 0.8341 | Catches 83% of defaults |
| **Precision**       | 0.2689 | 27% of flagged loans default |
| **F1 Score**        | 0.3834 | Balanced performance |
| **KS Statistic**    | 0.5407 | Excellent separation |

**Risk Thresholds:**

| Probability | Risk      | Action |
|-------------|-----------|--------|
| ≥ 28%       | 🔴 HIGH   | REJECT or require collateral |
| 15-27%      | 🟡 MEDIUM | APPROVE with monitoring |
| < 15%       | 🟢 LOW    | APPROVE |

**Why 28% threshold?**
- Optimized to catch 83% of defaults (high recall)
- Cost-asymmetric: Missing a default ($50k loss) >> Rejecting good loan ($3k opportunity cost)
- Based on precision-recall analysis and business cost function

**Top Features:**
1. GrossApproval (loan amount)
2. InitialInterestRate
3. LocationIDCount
4. IsCovidEra
5. SBAGuaranteedApproval

---

## AWS Deployment

The platform deploys on **AWS ECS Fargate** with an **Application Load Balancer** for production workloads.

### Architecture

```
┌─────────────────┐
│ Application     │  ← sba-loan-alb-*.ap-south-2.elb.amazonaws.com
│ Load Balancer   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌─────────┐
│ API     │ │ UI      │  ← ECS Fargate Tasks
│ Service │ │ Service │
└─────────┘ └─────────┘
    │
    ▼
┌─────────────┐
│ S3 Bucket   │  ← Model artifacts
│ (ap-south-2)│
└─────────────┘
```

### Deployment Steps

**1. Build and Push Docker Images**
```bash
# Authenticate to ECR
aws ecr get-login-password --region ap-south-2 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-2.amazonaws.com

# Build and push API image
docker build --platform linux/amd64 -t sba-api -f Dockerfile.api .
docker tag sba-api:latest <account>.dkr.ecr.ap-south-2.amazonaws.com/sba-api:latest
docker push <account>.dkr.ecr.ap-south-2.amazonaws.com/sba-api:latest

# Build and push UI image
docker build --platform linux/amd64 -t sba-ui -f Dockerfile.streamlit .
docker tag sba-ui:latest <account>.dkr.ecr.ap-south-2.amazonaws.com/sba-ui:latest
docker push <account>.dkr.ecr.ap-south-2.amazonaws.com/sba-ui:latest
```

**2. Upload Artifacts to S3**
```bash
# Run pipeline with S3 upload
python run_pipeline.py --skip-preprocessing --skip-training --upload

# Or manually
aws s3 cp models/trained/xgb_tuned.joblib \
  s3://sba-credit-risk-artifacts-sagar/models/trained/xgb_tuned.joblib

aws s3 cp models/encoders/frequency_map.pkl \
  s3://sba-credit-risk-artifacts-sagar/models/encoders/frequency_map.pkl
```

**3. Deploy to ECS**

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed AWS setup instructions including:
- ECS cluster creation
- Task definitions
- Service creation
- ALB configuration
- CI/CD with GitHub Actions

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | FastAPI endpoint | `http://localhost:8000` |
| `SYNC_FROM_S3` | Download artifacts from S3 | `false` (local), `true` (cloud) |
| `S3_BUCKET_NAME` | S3 bucket for artifacts | `sba-credit-risk-artifacts-sagar` |
| `AWS_REGION` | AWS region | `ap-south-2` |

---

## API Documentation

### Endpoints

**`GET /api`**
- Returns API information

**`GET /api/health`**
- Health check endpoint
- Response: `{"status": "healthy", "predictor_loaded": true}`

**`POST /api/predict`**
- Predict default probability
- Request: Loan application JSON
- Response: `{"default_probability": 0.12, "risk_category": "LOW", "recommendation": "APPROVE"}`

**`POST /api/explain`**
- Generate SHAP explanation
- Request: Same as `/predict`
- Response: `{"shap_values": [...], "feature_names": [...], "base_value": 0.075}`

### Interactive Docs
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

---

## Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# Feature consistency tests
pytest tests/test_feature_consistency.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage
- COVID-19 indicator creation
- NAICS sector extraction
- Binary indicators (credit union, franchise, fixed rate, collateral)
- Frequency encoding (training/inference modes)
- One-hot encoding
- Column alignment and data types

---

## Tech Stack

**ML:** scikit-learn, XGBoost, Optuna, SHAP
**Data:** pandas, numpy, pyarrow
**Web:** FastAPI, Streamlit, Uvicorn
**MLOps:** MLflow, pytest
**Cloud:** AWS (ECS, S3, ALB, ECR), boto3
**Viz:** matplotlib, seaborn

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add feature'`)
5. Push (`git push origin feature/name`)
6. Open Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- SBA FOIA data source: https://data.sba.gov/dataset/7-a-504-foia
- XGBoost, SHAP, FastAPI, and Streamlit communities
