# SBA Loan Default Prediction - MLOps Platform

A production-ready machine learning system for predicting Small Business Administration (SBA) loan defaults using XGBoost with explainable AI. Built as a full-stack MLOps platform with FastAPI backend and Streamlit dashboard, deployed on AWS ECS Fargate.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![AWS](https://img.shields.io/badge/AWS-ECS_Fargate-yellow.svg)](https://aws.amazon.com/ecs/)

---

## 🎯 Project Overview / Problem Statement

### Business Problem

Small business loans are critical for economic growth, but loan defaults create significant financial risk for lenders. The SBA 7(a) loan program guarantees portions of small business loans, but identifying high-risk applications remains challenging.

**Key Challenges:**
- **Class Imbalance**: Only 7.45% of loans default (highly imbalanced dataset)
- **High Cost of Errors**: Missing a default costs ~$50,000 vs rejecting a good loan costs ~$3,000
- **Explainability Required**: Lending decisions need transparent, auditable justifications
- **Real-time Assessment**: Loan officers need instant risk predictions during application review

### Solution

A full-stack machine learning platform that:

1. **Predicts** default probability for loan applications in real-time
2. **Explains** predictions using SHAP values (which features drive risk)
3. **Recommends** APPROVE/REJECT decisions based on optimized risk thresholds
4. **Deploys** as a scalable cloud service with sub-second response times

### Impact

- **83% Recall**: Catches 83% of defaults (vs random ~7% baseline)
- **Cost-Optimized**: 28% threshold minimizes total expected loss ($50k FN vs $3k FP)
- **Explainable**: SHAP waterfall plots show feature contributions for regulatory compliance
- **Production-Ready**: Cloud deployment on AWS ECS Fargate

---

## 🏗️ Architecture & Tech Stack

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                  Streamlit Dashboard (Port 8501)                │
│          [Loan Application Form] → [Risk Assessment]            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST /api/predict, /api/explain
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       REST API LAYER                            │
│                   FastAPI Backend (Port 8000)                   │
│                 [Prediction] [Explanation] [Health]             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Python Function Calls
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE                             │
│              LoanPredictor (Pure Python Module)                 │
│          [Feature Engineering] → [Model] → [SHAP]               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌─────────────┐
│  XGBoost     │  │  Frequency       │  │  SHAP       │
│  Model       │  │  Encoder         │  │  Explainer  │
│  (.joblib)   │  │  (.pkl)          │  │             │
└──────────────┘  └──────────────────┘  └─────────────┘
```

### ML Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw CSV    │────▶│  Feature     │────▶│   Training   │
│   (347k      │     │  Engineering │     │   Pipeline   │
│   loans)     │     │  (97 feats)  │     │  (Optuna)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │   XGBoost    │
                                           │   Model      │
                                           │  (Tuned)     │
                                           └──────┬───────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │  Evaluation  │
                                           │  (ROC-AUC,   │
                                           │   KS, etc.)  │
                                           └──────────────┘
```

### Cloud Deployment (AWS)

```
                    ┌─────────────────┐
                    │ Application     │
                    │ Load Balancer   │
                    │ (ALB)           │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
                   ▼                   ▼
            ┌────────────┐      ┌────────────┐
            │  API       │      │  UI        │
            │  Service   │      │  Service   │
            │ (ECS Task) │      │ (ECS Task) │
            └─────┬──────┘      └────────────┘
                  │
                  ▼
           ┌─────────────┐
           │ S3 Bucket   │
           │ (Artifacts) │
           │ ap-south-2  │
           └─────────────┘
```

### Tech Stack

**Machine Learning:**
- **XGBoost 2.0** - Gradient boosting classifier
- **scikit-learn** - Preprocessing, metrics, train/test split
- **Optuna** - Bayesian hyperparameter optimization (50 trials)
- **SHAP** - Model explainability (TreeExplainer)

**Data Processing:**
- **pandas** - Data manipulation & feature engineering
- **numpy** - Numerical operations
- **pyarrow** - Parquet serialization

**Web Services:**
- **FastAPI** - High-performance REST API (async support)
- **Streamlit** - Interactive dashboard
- **Uvicorn** - ASGI server

**MLOps & Tracking:**
- **MLflow** - Experiment tracking, model registry
- **pytest** - Automated testing (12 consistency tests)

**Cloud Infrastructure:**
- **AWS ECS Fargate** - Serverless container orchestration
- **AWS S3** - Model artifact storage
- **AWS ALB** - Application load balancing
- **AWS ECR** - Docker image registry
- **boto3** - AWS SDK for Python

**CI/CD:**
- **GitHub Actions** - Automated deployment pipeline
- **Docker** - Containerization (multi-stage builds)

---

## 📊 Dataset Description

### Data Source

**Official Dataset:** U.S. Small Business Administration (SBA) 7(a) Loan Program

🔗 **Download Link:** https://data.sba.gov/dataset/7-a-504-foia

**File Required:** `7(a) FY2020 - Present` (FOIA dataset)

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total Records** | 347,514 loans |
| **Date Range** | FY2020 - Present |
| **Features** | 43 raw columns |
| **Target Variable** | LoanStatus (PIF = Paid-in-Full, CHGOFF = Charged-Off) |
| **Class Distribution** | 92.55% Good (PIF), 7.45% Default (CHGOFF) |
| **After Preprocessing** | 55,834 loans × 97 features |

### Key Features

**Financial:**
- GrossApproval (loan amount)
- SBAGuaranteedApproval (SBA guarantee portion)
- InitialInterestRate
- JobsSupported

**Business Information:**
- NAICSCode (industry classification)
- BusinessType (Corporation, Individual, Partnership)
- BusinessAge (Existing, Startup, New Business, Change of Ownership)

**Location:**
- ProjectState (where project is located)
- BankState (where lender is located)
- LocationID (unique location identifier)

**Indicators:**
- FixedorVariableInterestRate
- CollateralInd (has collateral)
- FranchiseCode (franchise indicator)
- BankNCUANumber (credit union indicator)
- RevolverStatus (revolving line of credit)

**Temporal:**
- ApprovalDate (for COVID-era feature engineering)
- ApprovalFY (fiscal year)

### Feature Engineering

Raw 43 columns → **97 engineered features:**

1. **COVID-19 Indicator** (IsCovidEra): Binary flag for loans approved 2020-03-01 to 2021-12-31
2. **NAICS Sector Extraction**: First 2 digits → 24 industry sectors
3. **Frequency Encoding**: LocationID → count of loans from same location
4. **Binary Indicators**: IsCreditUnion, IsFranchise, IsFixedRate, HasCollateral
5. **Same-State Lending**: ProjectState == BankState
6. **One-Hot Encoding**: BusinessType (3), BusinessAge (4), ProjectState (54 states/territories)

### Data Preprocessing

**Leakage Prevention:**
- Removed: FirstDisbursementDate, SoldSecondMarketInd
- Removed: PaidinFullDate, ChargeoffDate, GrossChargeoffAmount (outcome variables)

**Missing Value Handling:**
- Required columns: LocationID, BankState (drop rows if missing)
- BusinessType: Whitespace → "INDIVIDUAL"
- BusinessAge: "Unanswered" → "Existing" (conservative assumption)

**Data Split:**
- Training: 80% (44,667 loans)
- Testing: 20% (11,167 loans)
- Stratified split to preserve class balance

---

## ⚙️ Pipeline & Methodology

### 1. Data Preprocessing Pipeline

**Location:** `src/main.py` + `src/feature_pipeline/`

**Pipeline Stages:**
1. Load CSV (347k records)
2. Filter to PIF/CHGOFF loans only
3. Drop leakage columns (post-approval information)
4. Handle missing values
5. Feature engineering (COVID, NAICS, binary flags, encoding)
6. Save processed data (Parquet format)

**Key Design:** Shared feature engineering module (`src/utils/feature_engineering.py`) used by BOTH training and inference to prevent train-serve skew.

### 2. Model Training Pipeline

**Location:** `src/training_pipeline/`

**Baseline Model:**
- XGBoost with class imbalance handling
- scale_pos_weight = n_negative / n_positive = 41,337 / 3,330 ≈ 12.41
- Objective: binary:logistic
- Evaluation metric: ROC-AUC

**Hyperparameter Tuning (Optuna):**

**Search Space:**
- max_depth: [3, 10]
- learning_rate: [0.01, 0.3] (log scale)
- n_estimators: [100, 500]
- min_child_weight: [1, 10]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]
- gamma: [0, 5]
- reg_alpha: [0, 2]
- reg_lambda: [0, 2]

**Optimization:**
- Objective: Maximize ROC-AUC
- Trials: 50
- Cross-Validation: 5-fold stratified
- Best CV ROC-AUC: 0.8336

### 3. Model Evaluation

**Metrics Calculated:**
- **ROC-AUC**: Area under ROC curve (threshold-agnostic)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN) - Critical for default detection
- **F1-Score**: Harmonic mean of precision and recall
- **KS Statistic**: Kolmogorov-Smirnov (credit risk metric)
- **Decile Analysis**: Performance across 10 risk segments

### 4. Threshold Optimization

**Problem:** Default 0.5 threshold assumes equal cost for FN and FP.

**Reality:**
- False Negative (miss default): $50,000 loss
- False Positive (reject good loan): $3,000 opportunity cost
- Cost Ratio: 16.67:1

**Solution:** Threshold sweep (0.10 to 0.60) to minimize total expected cost

**Result:** Optimal threshold = **0.28**

**Trade-off at 0.28:**
- Recall: 83.4% (catches 83% of defaults)
- Precision: 26.9% (1 in 4 flagged loans defaults)
- Net Benefit: Saves ~$17M more than approving all loans

### 5. Explainability (SHAP)

**Method:** TreeExplainer for XGBoost

**Output:** For each prediction, SHAP values show:
- **Base value**: Average model prediction
- **Feature contributions**: How each feature pushes prediction up/down
- **Final prediction**: Base + sum of contributions

**Visualization:** Waterfall plots (top 15 features by absolute impact)

### 6. Inference Pipeline

**Location:** `src/inference_pipeline/predict.py`

**Process:**
1. Apply feature engineering (same as training)
2. Align columns with training features
3. Generate predictions and SHAP values
4. Return probabilities

**Key Design:** Pure Python module (no web framework coupling) for portability.

### 7. Deployment Pipeline

**Local Development:**
- Docker Compose with API + UI services
- Mounted volumes for fast iteration

**Cloud Deployment (AWS ECS):**
- Multi-stage Docker builds (AMD64 for Fargate)
- S3 artifact sync on container startup
- Application Load Balancer with path-based routing
- Auto-scaling based on CPU/memory

**CI/CD (GitHub Actions):**
```
Push to main → Build images → Push to ECR → Update task defs → Deploy to ECS
```

---

## 📈 Model Performance

### Test Set Metrics

**Model:** XGBoost (Optuna-tuned) with class weighting  
**Test Set:** 11,167 samples (92.55% Good, 7.45% Default)  
**Classification Threshold:** 28% (optimized for recall)

| Metric              | Value  |
|---------------------|--------|
| **ROC-AUC**         | 0.8452 |
| **Recall**          | 0.8341 |
| **Precision**       | 0.2689 |
| **F1 Score**        | 0.3834 |
| **KS Statistic**    | 0.5407 |
| **Accuracy**        | 0.6645 |

### Confusion Matrix (at 28% threshold)

```
                    Predicted
                 Good    Default
Actual  Good     6,726   3,609      ← False Positives: 3,609
       Default     138     694      ← True Positives: 694
                   ↑
      False Negatives: 138 (missed defaults)
```

**Key Insights:**
- **High Recall (83%)**: Catches 83% of defaults - critical for risk management
- **Low Precision (27%)**: Acceptable trade-off when FN cost >> FP cost
- **3,609 False Positives**: Rejected good loans (cost: 3,609 × $3k = $10.8M)
- **138 False Negatives**: Missed defaults (cost: 138 × $50k = $6.9M)
- **Net Benefit**: $10.8M opportunity cost < $6.9M saved from missed defaults

### Risk Categorization

| Probability | Risk Category | Count (Test Set) | Default Rate | Action |
|-------------|---------------|------------------|--------------|--------|
| **≥ 28%**   | 🔴 HIGH       | 2,561 (23%)      | 27% avg      | REJECT or require collateral |
| **15-27%**  | 🟡 MEDIUM     | 1,847 (17%)      | 8% avg       | APPROVE with monitoring |
| **< 15%**   | 🟢 LOW        | 6,759 (60%)      | 2% avg       | APPROVE |

### Feature Importance (Top 10)

| Rank | Feature                  | Importance | Description |
|------|--------------------------|------------|-------------|
| 1    | GrossApproval            | 0.1845     | Loan amount ($) |
| 2    | InitialInterestRate      | 0.1623     | Interest rate (%) |
| 3    | LocationIDCount          | 0.0891     | Frequency of loans from location |
| 4    | IsCovidEra               | 0.0734     | Approved during COVID-19 (2020-2021) |
| 5    | SBAGuaranteedApproval    | 0.0687     | SBA guarantee amount ($) |
| 6    | JobsSupported            | 0.0542     | Number of jobs created/retained |
| 7    | NAICSSector              | 0.0498     | Industry sector (2-digit NAICS) |
| 8    | SameStateLending         | 0.0421     | Project state = Bank state |
| 9    | State_CA                 | 0.0389     | California location |
| 10   | Type_CORPORATION         | 0.0367     | Corporation business type |

**Insights:**
- **Loan size & rate** dominate predictions (top 2 features)
- **COVID-19 era** is 4th most important (economic disruption signal)
- **Location patterns** matter (LocationIDCount, State_CA, SameStateLending)

### Decile Analysis

Performance across 10 risk segments (ordered by predicted probability):

| Decile | Predicted Prob Range | Count | Default Rate | Cumulative Defaults Captured |
|--------|---------------------|-------|--------------|------------------------------|
| 10 (Highest) | 0.68 - 0.98 | 1,117 | **36.26%** | 48.68% |
| 9 | 0.47 - 0.68 | 1,117 | 15.04% | 68.87% |
| 8 | 0.35 - 0.47 | 1,116 | 7.26% | 78.87% |
| **7** | **0.25 - 0.35** | **1,117** | **5.10%** | **~83%** ← **28% threshold here** |
| 6 | 0.18 - 0.25 | 1,116 | 3.67% | 85.46% |
| 5 | 0.13 - 0.18 | 1,117 | 2.69% | 90.38% |
| 4 | 0.09 - 0.13 | 1,117 | 2.24% | 93.99% |
| 3 | 0.05 - 0.09 | 1,116 | 1.34% | 97.00% |
| 2 | 0.03 - 0.05 | 1,117 | 0.72% | 98.20% |
| 1 (Lowest) | 0.00 - 0.03 | 1,117 | 0.18% | 100.00% |

**Key Findings:**
- **Top 3 deciles** contain 79% of all defaults (good concentration)
- **28% threshold** sits at decile 7 boundary - natural risk inflection point
- **Deciles 8-10** have 7-36% default rates (vs 7.45% baseline)

### Model Comparison

| Model | ROC-AUC | Recall (28%) | Training Time |
|-------|---------|--------------|---------------|
| Baseline XGBoost | 0.8317 | 0.8341 | 2 min |
| **Optuna-Tuned XGBoost** | **0.8452** | **0.8341** | **45 min (50 trials)** |
| Logistic Regression | 0.7891 | 0.7123 | 30 sec |

**Winner:** Optuna-tuned XGBoost for best ROC-AUC with maintained recall.

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.8+
- Git
- Docker & Docker Compose (optional)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd ml-eng-lr
```

### Step 2: Download Dataset

Download the SBA FOIA dataset:

🔗 **https://data.sba.gov/dataset/7-a-504-foia**

1. Click on **"7(a) FY2020 - Present"** CSV file
2. Download and rename to: `foia-7a-fy2020-present-asof-250930.csv`
3. Place in `data/raw/` directory

```bash
# Create directory
mkdir -p data/raw

# Move downloaded file
mv ~/Downloads/7a_*.csv data/raw/foia-7a-fy2020-present-asof-250930.csv
```

### Step 3: Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run ML Pipeline

Execute the complete preprocessing + training pipeline:

```bash
# Full pipeline (preprocessing + training with 50 Optuna trials)
python run_pipeline.py

# Quick testing (10 trials)
python run_pipeline.py --n-trials 10

# Skip preprocessing if data already processed
python run_pipeline.py --skip-preprocessing

# Skip training if model already exists
python run_pipeline.py --skip-training
```

**Expected Output:**
```
✓ PIPELINE COMPLETE
Local artifacts:
  Model:       models/trained/xgb_tuned.joblib
  Encoder:     models/encoders/frequency_map.pkl
  Data:        data/feature/processed_data.parquet

Next steps:
  • Test API:  uvicorn src.api.main:app --reload
  • Test UI:   streamlit run app.py
```

### Step 5: Run Application

**Option A: Docker Compose (Recommended)**

```bash
# Build and start both API and UI
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f ui

# Stop services
docker-compose down
```

**Access:**
- Streamlit UI: http://localhost:8501
- FastAPI Backend: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

**Option B: Python Directly**

Open two terminal windows:

**Terminal 1 - API Backend:**
```bash
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 - Streamlit UI:**
```bash
source venv/bin/activate
streamlit run app.py
```

### Step 6: Make Predictions

**Via Streamlit Dashboard:**
1. Open http://localhost:8501
2. Fill in loan application details
3. Click "🔍 Assess Risk"
4. View default probability, risk category, and SHAP explanation

**Via Python:**
```python
from src.inference_pipeline.predict import LoanPredictor
import pandas as pd

# Initialize predictor
predictor = LoanPredictor()

# Create loan application
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
    "CollateralInd": "Y"
}])

# Get prediction
probability = predictor.predict(loan)[0]
print(f"Default probability: {probability:.2%}")

# Get SHAP explanation
shap_values = predictor.explain(loan)
print(f"SHAP values shape: {shap_values.shape}")
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Feature consistency tests only
pytest tests/test_feature_consistency.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

View training runs, metrics, parameters, and artifacts.

---

## 📂 Project Structure

```
ml-eng-lr/
│
├── data/
│   ├── raw/                                      # Raw SBA FOIA CSV
│   │   └── foia-7a-fy2020-present-asof-250930.csv
│   └── feature/                                  # Processed data (generated)
│       ├── processed_data.parquet                # 55,834 × 97 features
│       └── frequency_encoder.pkl                 # LocationID frequency map
│
├── models/
│   ├── trained/                                  # Trained models (generated)
│   │   ├── xgb_baseline.joblib
│   │   └── xgb_tuned.joblib                     # Optuna-tuned (deployed)
│   └── encoders/                                 # Feature encoders (generated)
│       └── frequency_map.pkl
│
├── mlruns/                                       # MLflow tracking (generated)
│   └── 0/                                        # Default experiment
│       └── [run_id]/                             # Metrics, params, artifacts
│
├── src/
│   ├── config.py                                 # Configuration (paths, constants)
│   ├── main.py                                   # ⭐ Preprocessing orchestrator
│   │
│   ├── feature_pipeline/                         # Data preprocessing
│   │   ├── load.py                               # Load CSV, filter loans
│   │   ├── cleaning.py                           # Missing values, cleaning
│   │   └── engineering.py                        # Feature creation
│   │
│   ├── training_pipeline/                        # Model training
│   │   ├── train_baseline.py                     # Baseline XGBoost
│   │   ├── tune_optuna.py                        # Hyperparameter tuning
│   │   ├── evaluation.py                         # Metrics (ROC-AUC, KS, etc.)
│   │   └── main.py                               # Training orchestrator
│   │
│   ├── inference_pipeline/
│   │   └── predict.py                            # LoanPredictor (predict + explain)
│   │
│   ├── api/
│   │   └── main.py                               # FastAPI REST API
│   │
│   └── utils/
│       ├── feature_engineering.py                # ⭐ Shared feature engineering
│       └── s3_manager.py                         # S3 artifact management
│
├── tests/
│   └── test_feature_consistency.py               # Feature engineering tests
│
├── infrastructure/                               # AWS ECS configs
│   ├── ecs-task-api.json                        # API task definition
│   ├── ecs-task-ui.json                         # UI task definition
│   └── README.md                                 # Infrastructure docs
│
├── jupyter-notebook/                             # Exploratory analysis
│   ├── sba_loan_preprocessing.ipynb
│   └── sba_loan_modeling.ipynb
│
├── app.py                                        # ⭐ Streamlit dashboard
├── run_pipeline.py                               # ⭐ Pipeline orchestrator
├── docker-compose.yml                            # Local multi-container setup
├── Dockerfile.api                                # API container
├── Dockerfile.streamlit                          # UI container
├── requirements.txt                              # Python dependencies
└── README.md                                     # This file
```

### Key Files

**Pipeline Orchestrators:**
- `run_pipeline.py` - Execute full ML pipeline (preprocessing → training)
- `src/main.py` - Preprocessing pipeline only

**Core ML Modules:**
- `src/utils/feature_engineering.py` - Shared feature engineering (training + inference)
- `src/inference_pipeline/predict.py` - LoanPredictor class
- `src/training_pipeline/tune_optuna.py` - Hyperparameter optimization

**Services:**
- `src/api/main.py` - FastAPI REST API with /predict and /explain endpoints
- `app.py` - Streamlit dashboard with SHAP visualizations

**Testing:**
- `tests/test_feature_consistency.py` - 12 tests ensuring train-serve consistency

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

*For deployment instructions and AWS setup, see [DEPLOYMENT.md](DEPLOYMENT.md)*
