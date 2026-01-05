"""
Inference module for SBA Loan default prediction.

Provides LoanPredictor class for real-time predictions and SHAP-based explanations.

CRITICAL: Uses shared feature engineering module (src.utils.feature_engineering)
to guarantee training-inference consistency.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Optional
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)


class LoanPredictor:
    """
    Loan default predictor with SHAP explanations.

    Pure Python class with NO web framework dependencies. Loads trained model
    and frequency encoder, performs predictions, and generates SHAP-based
    explanations for individual loans.

    Attributes:
        model: Trained XGBoost model.
        frequency_map: LocationID frequency encoding map.
        explainer: SHAP TreeExplainer (loaded once for performance).
        feature_names: List of expected feature names in correct order.

    Example:
        >>> predictor = LoanPredictor()
        >>> loan_df = pd.DataFrame([{
        ...     "GrossApproval": 50000,
        ...     "ApprovalDate": "2020-03-15",
        ...     "NAICSCode": "441110",
        ...     "BusinessType": "CORPORATION",
        ...     "ProjectState": "CA",
        ...     "BankState": "CA",
        ...     "LocationID": 12345.0,
        ...     # ... other required fields
        ... }])
        >>> probs = predictor.predict(loan_df)
        >>> print(f"Default probability: {probs[0]:.2%}")
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        frequency_encoder_path: Optional[Path] = None,
        sync_from_s3: bool = False
    ):
        """
        Initialize LoanPredictor.

        Args:
            model_path: Path to trained model. If None, uses config.TUNED_MODEL_PATH.
            frequency_encoder_path: Path to frequency encoder. If None, uses config.FREQUENCY_ENCODER_PATH.
            sync_from_s3: If True, sync artifacts from S3 before loading.

        Raises:
            FileNotFoundError: If model or encoder file not found.

        Example:
            >>> # Local mode
            >>> predictor = LoanPredictor()
            >>> # S3 sync mode (for cloud deployment)
            >>> predictor = LoanPredictor(sync_from_s3=True)
        """
        logger.info("Initializing LoanPredictor...")

        # S3 sync if requested
        if sync_from_s3:
            logger.info("Syncing artifacts from S3...")
            try:
                from src.utils.s3_manager import S3ArtifactManager
                manager = S3ArtifactManager(config.S3_BUCKET_NAME, config.S3_REGION)
                manager.sync_inference_artifacts(config.PROJECT_ROOT)
            except Exception as e:
                logger.warning(f"S3 sync failed: {e}. Attempting to use local artifacts...")

        # Set default paths
        if model_path is None:
            model_path = config.TUNED_MODEL_PATH
        if frequency_encoder_path is None:
            frequency_encoder_path = config.FREQUENCY_ENCODER_PATH

        # Load model
        try:
            self.model = joblib.load(model_path)
            logger.info(f"✓ Model loaded from: {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise

        # Load frequency encoder
        try:
            self.frequency_map = joblib.load(frequency_encoder_path)
            logger.info(f"✓ Frequency encoder loaded from: {frequency_encoder_path}")
        except FileNotFoundError:
            logger.error(f"Frequency encoder file not found: {frequency_encoder_path}")
            raise

        # Get feature names from model
        if hasattr(self.model, 'get_booster'):
            # XGBoost model
            self.feature_names = self.model.get_booster().feature_names
        else:
            # Fallback
            self.feature_names = None
            logger.warning("Could not extract feature names from model")

        # Initialize SHAP explainer (load once for performance)
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("✓ SHAP TreeExplainer initialized")
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
            self.explainer = None
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None

        logger.info("✓ LoanPredictor initialized successfully")

    def _preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply COMPLETE feature engineering to raw data.

        CRITICAL: Uses shared engineer_features() function to match training
        pipeline exactly. This ensures training-inference consistency.

        Args:
            data: Raw DataFrame with columns like ApprovalDate, NAICSCode,
                  BankState, ProjectState, BusinessType, LocationID, etc.

        Returns:
            Preprocessed DataFrame ready for prediction (features only).

        Example:
            >>> raw_df = pd.DataFrame([{
            ...     "ApprovalDate": "2020-03-15",
            ...     "NAICSCode": "441110",
            ...     # ... other fields
            ... }])
            >>> processed_df = predictor._preprocess_input(raw_df)
            >>> print(processed_df.shape)
            (1, 74)
        """
        from src.utils.feature_engineering import engineer_features

        # Apply complete feature engineering (matches training pipeline)
        df_engineered = engineer_features(
            data,
            frequency_map=self.frequency_map,
            expected_columns=self.feature_names
        )

        # Remove LoanStatus if present (inference data shouldn't have it, but just in case)
        if config.TARGET_COLUMN in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=[config.TARGET_COLUMN])

        # Ensure column order matches model features
        if self.feature_names is not None:
            # Verify all required features are present
            missing_features = set(self.feature_names) - set(df_engineered.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Reorder columns
            df_engineered = df_engineered[self.feature_names]
        else:
            logger.warning("Feature names not available. Using engineered column order.")

        return df_engineered

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict default probabilities for loans.

        Unified prediction method handling ANY DataFrame size (1 row or 10,000 rows).
        No separate single/batch logic needed.

        Args:
            data: Raw DataFrame with loan features (one row per loan).
                  Required columns: ApprovalDate, NAICSCode, BankState,
                  ProjectState, BusinessType, LocationID, etc.

        Returns:
            Array of default probabilities (length = len(data)).

        Example (Single prediction):
            >>> loan_df = pd.DataFrame([{
            ...     "GrossApproval": 50000,
            ...     "ApprovalDate": "2020-03-15",
            ...     # ... other fields
            ... }])
            >>> prob = predictor.predict(loan_df)[0]
            >>> print(f"Default probability: {prob:.2%}")
            Default probability: 12.34%

        Example (Batch prediction):
            >>> loans_df = pd.DataFrame([loan1, loan2, loan3])
            >>> probs = predictor.predict(loans_df)
            >>> print(probs)
            [0.1234, 0.5678, 0.0912]
        """
        # Preprocess input (applies complete feature engineering)
        df_processed = self._preprocess_input(data)

        # Get probabilities (probability of class 1 = default)
        probabilities = self.model.predict_proba(df_processed)[:, 1]

        logger.info(f"Predictions complete for {len(data)} loan(s)")

        return probabilities

    def explain(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Generate SHAP explanations for predictions.

        Unified explanation method handling ANY DataFrame size.

        Args:
            data: Raw DataFrame with loan features (one row per loan).

        Returns:
            SHAP values array (shape: [n_samples, n_features]).
            Returns None if SHAP explainer not available.

        Example (Single explanation):
            >>> shap_values = predictor.explain(loan_df)
            >>> # Frontend can use shap_values to render force plot
            >>> # shap.force_plot(explainer.expected_value, shap_values[0], ...)

        Example (Batch explanation):
            >>> shap_values = predictor.explain(loans_df)
            >>> print(shap_values.shape)
            (3, 74)
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available")
            return None

        # Preprocess input (applies complete feature engineering)
        df_processed = self._preprocess_input(data)

        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(df_processed)
            logger.info(f"SHAP values calculated for {len(data)} loan(s)")
            return shap_values

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize predictor
    predictor = LoanPredictor()

    # Example loan data (minimal - would need all required fields)
    example_loan = pd.DataFrame([{
        "GrossApproval": 50000,
        "SBAGuaranteedApproval": 37500,
        "ApprovalFY": 2020,
        "InitialInterestRate": 6.5,
        "RevolverStatus": 0,
        "JobsSupported": 5,
        "ApprovalDate": "2020-03-15",
        "NAICSCode": "441110",
        "BankNCUANumber": None,
        "FranchiseCode": None,
        "FixedorVariableInterestRate": "F",
        "CollateralInd": "Y",
        "BankState": "CA",
        "ProjectState": "CA",
        "BusinessType": "CORPORATION",
        "BusinessAge": "Existing or more than 2 years old",
        "LocationID": 12345.0,
    }])

    # Predict
    try:
        prob = predictor.predict(example_loan)[0]
        print(f"\nDefault probability: {prob:.2%}")

        # Explain
        shap_values = predictor.explain(example_loan)
        if shap_values is not None:
            print(f"SHAP values shape: {shap_values.shape}")
    except Exception as e:
        logger.error(f"Example prediction failed: {e}")
        logger.info("This is expected if not all required features are provided")
