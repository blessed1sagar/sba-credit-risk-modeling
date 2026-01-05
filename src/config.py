"""
Configuration module for SBA Loan preprocessing pipeline.

Contains all constants, file paths, column names, and mappings used throughout
the feature engineering pipeline.
"""
from pathlib import Path
from typing import List, Dict

# ============================================================================
# FILE PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "foia-7a-fy2020-present-asof-250930.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "feature" / "processed_data.parquet"
FREQUENCY_ENCODER_PATH = PROJECT_ROOT / "data" / "feature" / "frequency_encoder.pkl"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODEL_PATH = MODELS_DIR / "xgb_baseline.joblib"
TUNED_MODEL_PATH = MODELS_DIR / "xgb_tuned.joblib"
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"

# S3 Configuration
import os
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'sba-loan-ml-artifacts')
S3_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_MODEL_KEY = 'models/xgb_tuned.joblib'
S3_ENCODER_KEY = 'artifacts/frequency_encoder.pkl'

# ============================================================================
# TARGET DEFINITION
# ============================================================================
TARGET_COLUMN = "LoanStatus"
VALID_LOAN_STATUSES = ["PIF", "CHGOFF"]
TARGET_MAPPING = {"PIF": 0, "CHGOFF": 1}  # 0=Good, 1=Default

# ============================================================================
# LEAKAGE COLUMNS - DROP IMMEDIATELY AFTER LOADING
# ============================================================================
# These columns contain information not available at loan origination
# or were proven to be noisy/leaky through A/B testing
LEAKAGE_COLUMNS = [
    "FirstDisbursementDate",  # Future information - no TimeToDisbursementDays
    "TerminMonths",           # A/B testing showed TermInYears is leaky/noisy
    "SoldSecondMarketInd"     # Secondary market status - not predictive at origination
]

# ============================================================================
# DATA CLEANING
# ============================================================================
# Columns that must not have missing values (will drop rows)
# Note: FirstDisbursementDate removed from this list as it's now dropped entirely
REQUIRED_COLUMNS = ["LocationID", "BankState"]

# BusinessType cleaning
BUSINESS_TYPE_COLUMN = "BusinessType"
BUSINESS_TYPE_WHITESPACE_VALUE = "        "  # 8 spaces
BUSINESS_TYPE_DEFAULT = "INDIVIDUAL"

# BusinessAge mapping
BUSINESS_AGE_COLUMN = "BusinessAge"
BUSINESS_AGE_CLEAN_COLUMN = "BusinessAge_Clean"
BUSINESS_AGE_MAPPING: Dict[str, str] = {
    "Existing or more than 2 years old": "Existing",
    "Startup, Loan Funds will Open Business": "Startup",
    "New Business or 2 years or less": "NewBusiness",
    "Change of Ownership": "ChangeOfOwnership",
    "Unanswered": "Existing",  # Conservative assumption
}
BUSINESS_AGE_DEFAULT = "Existing"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Date columns
APPROVAL_DATE_COLUMN = "ApprovalDate"

# COVID-19 period definition
COVID_START_DATE = "2020-03-01"
COVID_END_DATE = "2021-12-31"

# NAICS sector extraction
NAICS_CODE_COLUMN = "NAICSCode"
NAICS_SECTOR_DIGITS = 2

# Binary indicator source columns
CREDIT_UNION_INDICATOR_COLUMN = "BankNCUANumber"
FRANCHISE_INDICATOR_COLUMN = "FranchiseCode"
INTEREST_RATE_TYPE_COLUMN = "FixedorVariableInterestRate"
FIXED_RATE_VALUE = "F"
COLLATERAL_INDICATOR_COLUMN = "CollateralInd"
COLLATERAL_YES_VALUE = "Y"

# State columns for same-state lending
BANK_STATE_COLUMN = "BankState"
PROJECT_STATE_COLUMN = "ProjectState"

# Location ID for frequency encoding
LOCATION_ID_COLUMN = "LocationID"

# ============================================================================
# CATEGORICAL ENCODING
# ============================================================================
# Columns to one-hot encode
BUSINESS_TYPE_ENCODE_COLUMNS = ["BusinessType", "BusinessAge_Clean"]
BUSINESS_TYPE_PREFIXES = ["Type", "Age"]

STATE_ENCODE_COLUMN = "ProjectState"
STATE_PREFIX = "State"

# ============================================================================
# FEATURE SELECTION
# ============================================================================
# Columns to drop before creating final feature set
# These are administrative, PII, or columns already processed into features
DROP_COLUMNS: List[str] = [
    # Administrative/Identifier columns (PII, not predictive)
    "AsOfDate", "Program", "BorrName", "BorrStreet", "BorrCity", "BorrState", "BorrZip",
    "BankName", "BankStreet", "BankCity", "BankState", "BankZip",
    "FranchiseName", "ProjectCounty", "SBADistrictOffice", "CongressionalDistrict",
    "ProcessingMethod", "Subprogram",

    # Outcome columns (data leakage - known after loan outcome)
    "PaidinFullDate", "ChargeoffDate", "GrossChargeoffAmount",

    # Raw columns already encoded/engineered
    "FranchiseCode", "BankNCUANumber", "BankFDICNumber",
    "NAICSCode", "NAICSDescription",
    "BusinessAge",
    "FixedorVariableInterestRate", "CollateralInd",
    "LocationID",
    "RevolverStatus",  # Not using this feature

    # Date columns (ApprovalDate dropped after creating IsCovidEra)
    # Note: ApprovalFY is kept as a feature
]

# ============================================================================
# EXPECTED DIMENSIONS (for validation)
# ============================================================================
RAW_DATA_ROWS = 347_514
RAW_DATA_COLUMNS = 43

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = None  # No timeout (use n_trials instead)

# XGBoost baseline parameters
BASELINE_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0
}
