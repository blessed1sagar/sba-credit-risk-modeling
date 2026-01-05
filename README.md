# 💼 SBA Loan Default Prediction - Full Stack MLOps Platform

A production-ready machine learning platform for predicting Small Business Administration (SBA) loan defaults with explainable AI. This system provides real-time risk assessment and SHAP-based explanations through a banker-friendly web interface.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

---

## 🎯 Overview

This platform analyzes SBA 7(a) loan applications and predicts default probability using an XGBoost model trained on historical loan data. The system provides:

- **Real-time predictions** via REST API
- **Explainable AI** with SHAP force plots
- **Banker-friendly UI** for loan officers
- **Cloud-portable** architecture with S3 artifact storage
- **Training-inference consistency** guaranteed through shared feature engineering

### Business Impact

- **Risk Assessment**: Quantify default probability for loan applications
- **Explainability**: Understand which features drive risk predictions
- **Regulatory Compliance**: Provide transparent, auditable risk assessments
- **Decision Support**: Recommend APPROVE/REJECT based on risk thresholds

---

## ✨ Key Features

### 🤖 Machine Learning
- **XGBoost classifier** optimized with Bayesian hyperparameter tuning (Optuna)
- **ROC-AUC optimization** with 5-fold cross-validation
- **Class imbalance handling** via dynamic scale_pos_weight calculation
- **Feature engineering**: COVID-19 indicators, NAICS sector extraction, binary flags, one-hot encoding
- **Leakage prevention**: Automatic removal of post-approval features

### 🔍 Explainable AI
- **SHAP TreeExplainer** for model interpretability
- **Force plots** showing feature contributions to predictions
- **Feature importance** tracking across training runs
- **MLflow integration** for experiment tracking

### 🌐 Web Services
- **FastAPI microservice** for high-performance predictions
- **Streamlit dashboard** with domain-specific inputs
- **REST API** with automatic OpenAPI documentation
- **Async support** for concurrent requests

### ☁️ Cloud Integration
- **S3 artifact management** for model versioning
- **Smart sync**: Downloads artifacts only if missing locally
- **Multi-environment support** via environment variables
- **Stateless deployment** ready for containers/serverless

### ✅ Quality Assurance
- **12 automated tests** for feature engineering consistency
- **Train-serve skew prevention** through shared feature engineering
- **Type hints** and comprehensive docstrings
- **Logging** at every pipeline stage

---

## 🏗️ Architecture

### Microservices Design

```
┌─────────────────┐
│  Streamlit UI   │  ← Banker-facing dashboard (port 8501)
│   (app.py)      │
└────────┬────────┘
         │ HTTP POST /predict, /explain
         ↓
┌─────────────────┐
│   FastAPI API   │  ← REST API microservice (port 8000)
│ (src/api/main.py)│
└────────┬────────┘
         │ Python function calls
         ↓
┌─────────────────┐
│ LoanPredictor   │  ← Pure Python inference engine
│   (predict.py)  │
└────────┬────────┘
         │
         ├─→ [XGBoost Model] (xgb_tuned.joblib)
         ├─→ [Frequency Map] (frequency_encoder.pkl)
         └─→ [SHAP Explainer]
```

### Data Flow

```
Raw Loan Data
      ↓
Feature Engineering (shared module)
      ↓
[IsCovidEra, NAICSSector, Binary Indicators,
 SameStateLending, LocationIDCount, One-Hot Encoding]
      ↓
XGBoost Model
      ↓
[Default Probability, Risk Category, SHAP Values]
```

### Training-Inference Consistency

**CRITICAL DESIGN**: Both training and inference use the **same** feature engineering module (`src/utils/feature_engineering.py`) to guarantee consistency.

```python
# Training Mode
df_features = engineer_features(df_raw)
model.fit(df_features, y)

# Inference Mode (uses same function!)
df_features = engineer_features(
    df_raw,
    frequency_map=loaded_freq_map,
    expected_columns=model.feature_names
)
predictions = model.predict(df_features)
```

---

## 📁 Project Structure

```
ml-eng-lr/
│
├── data/                                    # Data storage
│   ├── raw/                                 # Raw CSV data
│   │   └── foia-7a-fy2020-present-asof-250930.csv
│   └── feature/                             # Processed features
│       ├── processed_data.parquet           # Engineered features + target
│       └── frequency_encoder.pkl            # LocationID frequency map
│
├── models/                                  # Trained models
│   ├── xgb_baseline.joblib                  # Baseline XGBoost model
│   └── xgb_tuned.joblib                     # Optuna-tuned model
│
├── mlruns/                                  # MLflow experiment tracking
│   └── [experiment_runs]/                   # Metrics, params, artifacts
│
├── src/                                     # Source code
│   │
│   ├── feature_pipeline/                    # Data preprocessing
│   │   ├── __init__.py
│   │   ├── load.py                          # Load CSV, filter loans, drop leakage
│   │   ├── cleaning.py                      # Handle missing values, clean categoricals
│   │   ├── engineering.py                   # Create features (COVID, NAICS, binary flags)
│   │   └── main.py                          # Orchestrate preprocessing pipeline
│   │
│   ├── training_pipeline/                   # Model training
│   │   ├── __init__.py
│   │   ├── train_baseline.py                # Train baseline XGBoost
│   │   ├── tune_optuna.py                   # Bayesian hyperparameter tuning
│   │   ├── evaluation.py                    # Calculate metrics (ROC-AUC, KS, Decile)
│   │   └── main.py                          # Orchestrate training pipeline
│   │
│   ├── inference_pipeline/                  # Inference engine
│   │   ├── __init__.py
│   │   └── predict.py                       # LoanPredictor class (predict + explain)
│   │
│   ├── api/                                 # FastAPI web service
│   │   ├── __init__.py
│   │   └── main.py                          # REST API endpoints (/predict, /explain)
│   │
│   ├── utils/                               # ⭐ Shared utilities (CRITICAL)
│   │   ├── __init__.py
│   │   ├── feature_engineering.py           # Shared feature engineering (train + inference)
│   │   └── s3_manager.py                    # S3 artifact upload/download
│   │
│   ├── config.py                            # Configuration (paths, constants, S3 settings)
│   └── __init__.py
│
├── tests/                                   # Test suite
│   ├── __init__.py
│   └── test_feature_consistency.py          # Feature engineering consistency tests
│
├── jupyter-notebook/                        # Exploratory notebooks
│   ├── sba_loan_preprocessing.ipynb         # Data exploration + preprocessing
│   └── sba_loan_modeling.ipynb              # Model training + evaluation
│
├── app.py                                   # Streamlit dashboard (frontend)
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
└── .gitignore                               # Git ignore patterns
```

### Key Components

#### **1. Feature Pipeline** (`src/feature_pipeline/`)
Transforms raw CSV data into ML-ready features:
- **load.py**: Load CSV, filter to PIF/CHGOFF loans, drop leakage columns
- **cleaning.py**: Handle missing values, clean BusinessType/BusinessAge
- **engineering.py**: Create features (IsCovidEra, NAICSSector, binary indicators, frequency encoding, one-hot encoding)
- **main.py**: Orchestrate full pipeline (load → clean → engineer → save)

#### **2. Training Pipeline** (`src/training_pipeline/`)
Train and evaluate ML models:
- **train_baseline.py**: Train baseline XGBoost with class weighting
- **tune_optuna.py**: Bayesian hyperparameter optimization (50 trials, ROC-AUC objective)
- **evaluation.py**: Calculate metrics (ROC-AUC, Precision, Recall, F1, KS Statistic, Decile Analysis)
- **main.py**: Orchestrate training (load data → train → evaluate → save)

#### **3. Inference Pipeline** (`src/inference_pipeline/`)
Real-time prediction engine:
- **predict.py**: `LoanPredictor` class with unified `predict()` and `explain()` methods
- Pure Python (no web framework dependencies)
- S3 sync support for cloud deployments

#### **4. API Service** (`src/api/`)
FastAPI REST API:
- **POST /predict**: Predict default probability + risk category
- **POST /explain**: Generate SHAP explanations
- **GET /health**: Health check endpoint
- Automatic OpenAPI docs at `/docs`

#### **5. Shared Utils** (`src/utils/`) ⭐ **CRITICAL**
- **feature_engineering.py**: Shared feature engineering used by BOTH training and inference (guarantees consistency)
- **s3_manager.py**: S3 artifact upload/download/sync

#### **6. Streamlit Dashboard** (`app.py`)
Banker-facing UI with:
- Domain-specific inputs (NAICS sectors, US states, business age)
- Real-time risk assessment
- SHAP force plot visualization
- Client-side calculations (SBA guarantee, same-state lending)

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) AWS credentials for S3 integration

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd ml-eng-lr
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Data
Place the SBA loan CSV file in `data/raw/`:
```
data/raw/foia-7a-fy2020-present-asof-250930.csv
```

---

## 💻 Usage

### 1. Data Preprocessing

Transform raw CSV into ML-ready features:

```bash
python -m src.feature_pipeline.main
```

**Output**:
- `data/feature/processed_data.parquet` (55,831 rows × 74 features)
- `data/feature/frequency_encoder.pkl` (LocationID frequency map)

---

### 2. Model Training

#### Train Baseline Model
```bash
python -m src.training_pipeline.train_baseline
```

**Output**: `models/xgb_baseline.joblib`

#### Hyperparameter Tuning (Optuna)
```bash
python -m src.training_pipeline.tune_optuna
```

**Output**:
- `models/xgb_tuned.joblib`
- MLflow metrics in `mlruns/`

#### View MLflow UI
```bash
mlflow ui
```
Then open http://localhost:5000

---

### 3. Run Full Stack Application

#### Terminal 1: Start FastAPI Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

#### Terminal 2: Start Streamlit Frontend
```bash
streamlit run app.py
```

**Access**: http://localhost:8501

---

### 4. Making Predictions

#### Via Python API
```python
from src.inference_pipeline.predict import LoanPredictor
import pandas as pd

# Initialize predictor
predictor = LoanPredictor()

# Create loan data
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

#### Via REST API (curl)
```bash
curl -X POST "http://localhost:8000/predict" \
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

#### Via Streamlit UI
1. Open http://localhost:8501
2. Fill in loan application details
3. Click "🔍 Assess Risk"
4. View prediction and SHAP explanation

---

## 📚 API Documentation

### Endpoints

#### `GET /`
Returns API information and available endpoints.

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "predictor_loaded": true,
  "message": "API is operational"
}
```

#### `POST /predict`
Predict default probability for a loan application.

**Request Body**:
```json
{
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
}
```

**Response**:
```json
{
  "default_probability": 0.1234,
  "risk_category": "LOW",
  "threshold_used": 0.28,
  "recommendation": "APPROVE"
}
```

#### `POST /explain`
Generate SHAP explanation for a loan application.

**Request Body**: Same as `/predict`

**Response**:
```json
{
  "shap_values": [0.015, -0.023, 0.031, "..."],
  "feature_names": ["GrossApproval", "InitialInterestRate", "IsCovidEra", "..."],
  "base_value": 0.075
}
```

### Interactive API Docs
FastAPI automatically generates interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Feature Consistency Tests
```bash
pytest tests/test_feature_consistency.py -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

#### Feature Engineering Consistency (12 tests)
- COVID-19 indicator creation
- NAICS sector extraction
- Binary indicators (credit union, franchise, fixed rate, collateral)
- Same-state lending flag
- Frequency encoding (training and inference modes)
- Unseen LocationID handling
- One-hot encoding
- Raw column dropping
- Multi-row consistency
- Column alignment
- Data type validation

---

## 🚀 Deployment

### Local Development
Already covered in [Usage](#usage) section.

### Docker Deployment (Coming Soon)
```bash
# Build FastAPI container
docker build -t sba-api -f Dockerfile.api .

# Run container
docker run -p 8000:8000 sba-api
```

### Cloud Deployment

#### AWS Lambda + API Gateway
1. Package FastAPI app as Lambda function
2. Enable S3 sync: Set `sync_from_s3=True` in `startup_event()`
3. Configure environment variables:
   ```bash
   S3_BUCKET_NAME=sba-loan-ml-artifacts
   AWS_REGION=us-east-1
   ```

#### AWS ECS/Fargate
1. Build Docker image
2. Push to ECR
3. Create ECS task definition
4. Deploy service

#### Streamlit Cloud
1. Push code to GitHub
2. Connect Streamlit Cloud to repository
3. Set environment variable: `API_URL=<your-api-url>`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | FastAPI endpoint URL | `http://localhost:8000` |
| `S3_BUCKET_NAME` | S3 bucket for artifacts | `sba-loan-ml-artifacts` |
| `AWS_REGION` | AWS region | `us-east-1` |

---

## 📊 Model Performance

### Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.8463 |
| **Precision** | 0.7821 |
| **Recall** | 0.6543 |
| **F1 Score** | 0.7125 |
| **KS Statistic** | 0.5311 |

### Feature Importance (Top 10)

1. **GrossApproval** - Loan amount
2. **InitialInterestRate** - Interest rate
3. **LocationIDCount** - Location frequency
4. **IsCovidEra** - COVID-19 period flag
5. **SBAGuaranteedApproval** - SBA guarantee amount
6. **JobsSupported** - Jobs created/retained
7. **NAICSSector** - Industry sector
8. **SameStateLending** - Same state flag
9. **State_CA** - California location
10. **Type_CORPORATION** - Corporation flag

### Decile Analysis

| Decile | Bad Rate | Cumulative Capture |
|--------|----------|-------------------|
| 10 (Highest Risk) | 42.3% | 28.5% |
| 9 | 31.7% | 49.2% |
| 8 | 24.1% | 64.8% |
| 7 | 18.6% | 77.3% |
| ... | ... | ... |
| 1 (Lowest Risk) | 2.4% | 100.0% |

---

## 🛠️ Tech Stack

### Machine Learning
- **scikit-learn**: Model pipeline, train/test split, metrics
- **XGBoost**: Gradient boosting classifier
- **Optuna**: Bayesian hyperparameter optimization
- **SHAP**: Model explainability

### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **pyarrow**: Parquet file format

### Web Frameworks
- **FastAPI**: High-performance REST API
- **Streamlit**: Interactive dashboard
- **Uvicorn**: ASGI server

### MLOps & Tracking
- **MLflow**: Experiment tracking, model registry
- **pytest**: Testing framework

### Cloud & Storage
- **boto3**: AWS S3 integration

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **streamlit-shap**: SHAP plot rendering in Streamlit

---

## 🏆 Best Practices Implemented

### Code Quality
✅ Type hints throughout codebase
✅ Comprehensive docstrings (Google style)
✅ Modular design (separation of concerns)
✅ Logging at every pipeline stage

### ML Engineering
✅ Train-serve skew prevention (shared feature engineering)
✅ Leakage prevention (automatic removal of post-approval features)
✅ Class imbalance handling (dynamic scale_pos_weight)
✅ Cross-validation for hyperparameter tuning

### Software Engineering
✅ Microservices architecture
✅ Pure Python inference engine (no web framework coupling)
✅ Automated testing (12 consistency tests)
✅ Environment-based configuration

### DevOps
✅ Cloud portability (S3 artifact storage)
✅ Stateless deployment ready
✅ Health check endpoints
✅ Automatic API documentation

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards
- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for public methods
- Add tests for new features
- Update README if adding new components

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- SBA FOIA data source
- XGBoost development team
- SHAP library authors
- FastAPI and Streamlit communities

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**🤖 Generated with [Claude Code](https://claude.com/claude-code)**

---

## 🗺️ Roadmap

- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model monitoring dashboard
- [ ] A/B testing framework
- [ ] Batch prediction API
- [ ] Model retraining automation
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Caching layer (Redis)
- [ ] Multi-model serving

---

*Last updated: 2026-01-05*
