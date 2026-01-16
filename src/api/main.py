"""
FastAPI microservice for SBA Loan Risk prediction.

Thin HTTP adapter layer over LoanPredictor. Provides REST API endpoints
for loan default prediction and SHAP-based explanations.

Run with: uvicorn src.api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import logging
import os

from src.inference_pipeline.predict import LoanPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SBA Loan Risk API",
    description="Predict default probability and explain risk factors for SBA loan applications",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Global predictor (initialized on startup)
predictor: Optional[LoanPredictor] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize predictor on application startup.

    Uses environment variable SYNC_FROM_S3 to control artifact loading:
      - SYNC_FROM_S3=true:  Always try S3 sync (cloud deployment)
      - SYNC_FROM_S3=false: Use local artifacts only (local development)

    Environment Variables:
      - SYNC_FROM_S3: 'true'|'false' (default: 'false')
      - AWS_REGION: AWS region (default: 'ap-south-2')
      - S3_BUCKET_NAME: S3 bucket name (default: 'sba-credit-risk-artifacts-sagar')
    """
    global predictor

    logger.info("=" * 80)
    logger.info("STARTING SBA LOAN RISK API")
    logger.info("=" * 80)

    # Determine if we should try S3 sync from environment variable
    sync_from_s3 = os.getenv('SYNC_FROM_S3', 'false').lower() in ('true', '1', 'yes')

    logger.info(f"Environment: {'Cloud (S3 sync enabled)' if sync_from_s3 else 'Local (S3 disabled)'}")
    logger.info(f"Region: {os.getenv('AWS_REGION', 'ap-south-2')}")

    try:
        # Initialize predictor with smart sync
        predictor = LoanPredictor(sync_from_s3=sync_from_s3)
        logger.info("✓ Predictor initialized successfully")
    except FileNotFoundError as e:
        logger.error(f"❌ Failed to initialize predictor - missing artifacts: {e}")
        logger.error("\nTo generate artifacts:")
        logger.error("  1. Run locally: python run_pipeline.py")
        logger.error("  2. (Optional) Upload to S3: python scripts/push_artifacts.py")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        raise

    logger.info("✓ API ready to accept requests")


# ============================================================================
# PYDANTIC SCHEMAS (Raw Input - NOT Engineered Features!)
# ============================================================================

class LoanApplicationRequest(BaseModel):
    """
    Loan application data for prediction (raw format, not pre-engineered).

    Note: Input should contain raw column values as they appear in the original
    data. Feature engineering is handled internally by the API.
    """
    # Financials
    GrossApproval: float = Field(..., description="Gross loan amount approved ($)", gt=0)
    SBAGuaranteedApproval: float = Field(..., description="SBA guaranteed amount ($)", gt=0)
    InitialInterestRate: float = Field(..., description="Initial interest rate (%)", gt=0)
    ApprovalFY: int = Field(..., description="Approval fiscal year", ge=2010, le=2030)
    RevolverStatus: int = Field(default=0, description="Revolver status (0 or 1)")
    JobsSupported: int = Field(..., description="Number of jobs supported", ge=0)

    # Business Information
    NAICSCode: str = Field(..., description="NAICS industry code (e.g., '441110')")
    BusinessType: str = Field(..., description="Business type (e.g., 'CORPORATION', 'INDIVIDUAL')")
    BusinessAge: str = Field(..., description="Business age category")

    # Location
    ProjectState: str = Field(..., description="Project state abbreviation (e.g., 'CA')")
    BankState: str = Field(..., description="Bank state abbreviation (e.g., 'CA')")
    LocationID: float = Field(..., description="Location ID")

    # Dates (for feature engineering)
    ApprovalDate: str = Field(..., description="Approval date (YYYY-MM-DD) - used for IsCovidEra")

    # Flags (raw values)
    BankNCUANumber: Optional[str] = Field(default=None, description="Bank NCUA number (for credit union indicator)")
    FranchiseCode: Optional[str] = Field(default=None, description="Franchise code (for franchise indicator)")
    FixedorVariableInterestRate: str = Field(..., description="Interest rate type ('F' or 'V')")
    CollateralInd: str = Field(..., description="Collateral indicator ('Y' or 'N')")

    class Config:
        schema_extra = {
            "example": {
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
                "BankNCUANumber": None,
                "FranchiseCode": None,
                "FixedorVariableInterestRate": "F",
                "CollateralInd": "Y"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response with default probability and risk category."""
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_category: str = Field(..., description="Risk category: HIGH, MEDIUM, or LOW")
    threshold_used: float = Field(default=0.28, description="Classification threshold used")
    recommendation: str = Field(..., description="Loan recommendation: APPROVE or REJECT")

    class Config:
        schema_extra = {
            "example": {
                "default_probability": 0.1234,
                "risk_category": "LOW",
                "threshold_used": 0.28,
                "recommendation": "APPROVE"
            }
        }


class ExplanationResponse(BaseModel):
    """SHAP explanation response with feature contributions."""
    shap_values: List[float] = Field(..., description="SHAP values for each feature")
    feature_names: List[str] = Field(..., description="Feature names corresponding to SHAP values")
    base_value: float = Field(..., description="SHAP base value (expected value)")

    class Config:
        schema_extra = {
            "example": {
                "shap_values": [0.015, -0.023, 0.031, "..."],
                "feature_names": ["GrossApproval", "InitialInterestRate", "IsCovidEra", "..."],
                "base_value": 0.075
            }
        }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SBA Loan Risk API",
        "version": "1.0.0",
        "description": "Predict default probability and explain risk factors for SBA loan applications",
        "endpoints": {
            "/api/predict": "POST - Predict default probability for a loan application",
            "/api/explain": "POST - Get SHAP explanation for a loan application",
            "/api/health": "GET - Check API health status"
        }
    }


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        API health status and predictor availability.
    """
    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None,
        "message": "API is operational" if predictor is not None else "Predictor not loaded"
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_default(request: LoanApplicationRequest):
    """
    Predict default probability for a loan application.

    Applies complete feature engineering pipeline and returns:
    - Default probability (0-1)
    - Risk category (HIGH/MEDIUM/LOW)
    - Recommendation (APPROVE/REJECT)

    Args:
        request: Loan application data (raw format).

    Returns:
        Prediction response with probability, risk category, and recommendation.

    Raises:
        HTTPException: If prediction fails.

    Example:
        curl -X POST "http://localhost:8000/predict" \\
             -H "Content-Type: application/json" \\
             -d '{"GrossApproval": 50000, "ApprovalDate": "2020-03-15", ...}'
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    try:
        # Convert request to DataFrame (1 row)
        loan_df = pd.DataFrame([request.dict()])

        # Call unified predict() method
        probs = predictor.predict(loan_df)
        prob = float(probs[0])

        # Categorize risk
        if prob >= 0.28:
            risk_category = "HIGH"
            recommendation = "REJECT"
        elif prob >= 0.15:
            risk_category = "MEDIUM"
            recommendation = "APPROVE"  # May require additional review
        else:
            risk_category = "LOW"
            recommendation = "APPROVE"

        logger.info(
            f"Prediction complete - Probability: {prob:.4f}, "
            f"Risk: {risk_category}, Recommendation: {recommendation}"
        )

        return PredictionResponse(
            default_probability=prob,
            risk_category=risk_category,
            threshold_used=0.28,
            recommendation=recommendation
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/explain", response_model=ExplanationResponse)
async def explain_prediction(request: LoanApplicationRequest):
    """
    Generate SHAP explanation for a loan application.

    Applies complete feature engineering and calculates SHAP values
    showing how each feature contributes to the prediction.

    Args:
        request: Loan application data (raw format).

    Returns:
        SHAP values, feature names, and base value for force plot visualization.

    Raises:
        HTTPException: If explanation generation fails or SHAP not available.

    Example:
        curl -X POST "http://localhost:8000/explain" \\
             -H "Content-Type: application/json" \\
             -d '{"GrossApproval": 50000, "ApprovalDate": "2020-03-15", ...}'
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    if predictor.explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer not available. Install with: pip install shap"
        )

    try:
        # Convert request to DataFrame (1 row)
        loan_df = pd.DataFrame([request.dict()])

        # Call unified explain() method
        shap_values = predictor.explain(loan_df)

        if shap_values is None:
            raise HTTPException(status_code=500, detail="SHAP calculation failed")

        # Extract SHAP values for the single row
        shap_values_list = shap_values[0].tolist()

        logger.info("SHAP explanation generated successfully")

        return ExplanationResponse(
            shap_values=shap_values_list,
            feature_names=predictor.feature_names,
            base_value=float(predictor.explainer.expected_value)
        )

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
