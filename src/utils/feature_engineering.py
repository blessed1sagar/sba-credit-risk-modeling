"""
Shared feature engineering module for SBA Loan ML pipeline.

CRITICAL: This module centralizes ALL feature engineering logic to guarantee
training-inference consistency. Both training_pipeline and inference_pipeline
MUST use this module.

Provides the engineer_features() function that applies the complete
feature engineering pipeline matching feature_pipeline/engineering.py.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from src import config

logger = logging.getLogger(__name__)


def _create_covid_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create COVID-19 period indicator feature and drop ApprovalDate.

    Features created:
    - IsCovidEra: 1 if loan approved between 2020-03-01 and 2021-12-31, else 0

    IMPORTANT: Drops ApprovalDate after creating indicator (prevents leakage).

    Args:
        df: Input DataFrame with ApprovalDate column.

    Returns:
        DataFrame with IsCovidEra feature (ApprovalDate removed).
    """
    df = df.copy()

    # Convert ApprovalDate to datetime if not already
    df[config.APPROVAL_DATE_COLUMN] = pd.to_datetime(
        df[config.APPROVAL_DATE_COLUMN], errors='coerce'
    )

    # Create COVID indicator
    df['IsCovidEra'] = (
        (df[config.APPROVAL_DATE_COLUMN] >= config.COVID_START_DATE) &
        (df[config.APPROVAL_DATE_COLUMN] <= config.COVID_END_DATE)
    ).astype(int)

    # Drop ApprovalDate to prevent leakage
    df = df.drop(columns=[config.APPROVAL_DATE_COLUMN])

    return df


def _create_naics_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract NAICS sector from NAICSCode.

    Feature created:
    - NAICSSector: First 2 digits of NAICSCode (industry sector)

    Args:
        df: Input DataFrame with NAICSCode column.

    Returns:
        DataFrame with NAICSSector feature.
    """
    df = df.copy()

    df['NAICSSector'] = (
        df[config.NAICS_CODE_COLUMN]
        .astype(str)
        .str[:config.NAICS_SECTOR_DIGITS]
    )

    return df


def _create_binary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary indicator features.

    Features created:
    - IsCreditUnion: 1 if BankNCUANumber is not null
    - IsFranchise: 1 if FranchiseCode is not null
    - IsFixedRate: 1 if FixedorVariableInterestRate == 'F'
    - HasCollateral: 1 if CollateralInd == 'Y'

    Args:
        df: Input DataFrame with source indicator columns.

    Returns:
        DataFrame with binary indicator features.
    """
    df = df.copy()

    df['IsCreditUnion'] = df[config.CREDIT_UNION_INDICATOR_COLUMN].notna().astype(int)
    df['IsFranchise'] = df[config.FRANCHISE_INDICATOR_COLUMN].notna().astype(int)
    df['IsFixedRate'] = (
        df[config.INTEREST_RATE_TYPE_COLUMN] == config.FIXED_RATE_VALUE
    ).astype(int)
    df['HasCollateral'] = (
        df[config.COLLATERAL_INDICATOR_COLUMN] == config.COLLATERAL_YES_VALUE
    ).astype(int)

    return df


def _create_same_state_lending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create SameStateLending interaction feature.

    Feature created:
    - SameStateLending: 1 if BankState == ProjectState

    Args:
        df: Input DataFrame with BankState and ProjectState.

    Returns:
        DataFrame with SameStateLending feature.
    """
    df = df.copy()

    df['SameStateLending'] = (
        df[config.BANK_STATE_COLUMN] == df[config.PROJECT_STATE_COLUMN]
    ).astype(int)

    return df


def _apply_frequency_encoding(
    df: pd.DataFrame,
    frequency_map: Optional[Dict[float, int]] = None
) -> Tuple[pd.DataFrame, Dict[float, int]]:
    """
    Apply frequency encoding to LocationID.

    Feature created:
    - LocationIDCount: Number of loans associated with each LocationID

    TRAINING MODE (frequency_map=None):
    - Computes frequency map from the data
    - Returns both DataFrame and frequency_map

    INFERENCE MODE (frequency_map provided):
    - Uses provided frequency_map
    - For unseen LocationIDs, uses minimum frequency from training
    - Returns DataFrame and original frequency_map

    Args:
        df: Input DataFrame with LocationID column.
        frequency_map: Optional pre-computed frequency map for inference.

    Returns:
        Tuple of (DataFrame with LocationIDCount feature, frequency_map dict).
    """
    df = df.copy()

    if frequency_map is None:
        # TRAINING MODE: Compute frequency map
        frequency_map = df[config.LOCATION_ID_COLUMN].value_counts().to_dict()
        df['LocationIDCount'] = df[config.LOCATION_ID_COLUMN].map(frequency_map)
        logger.info(
            f"Created LocationIDCount (training mode) - "
            f"range: {df['LocationIDCount'].min()} to {df['LocationIDCount'].max()}"
        )
    else:
        # INFERENCE MODE: Use provided frequency map
        min_frequency = min(frequency_map.values()) if frequency_map else 1

        def get_frequency(location_id):
            if location_id in frequency_map:
                return frequency_map[location_id]
            else:
                # Unseen LocationID - use minimum frequency from training
                logger.warning(
                    f"Unseen LocationID {location_id}. Using minimum frequency = {min_frequency}"
                )
                return min_frequency

        df['LocationIDCount'] = df[config.LOCATION_ID_COLUMN].apply(get_frequency)
        logger.info(
            f"Applied LocationIDCount (inference mode) - "
            f"range: {df['LocationIDCount'].min()} to {df['LocationIDCount'].max()}"
        )

    return df, frequency_map


def _one_hot_encode_categoricals(
    df: pd.DataFrame,
    expected_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    One-hot encode categorical variables.

    Encodes:
    - BusinessType → Type_*
    - BusinessAge_Clean → Age_*
    - ProjectState → State_*

    TRAINING MODE (expected_columns=None):
    - Creates one-hot columns based on data
    - drop_first=False for interpretability

    INFERENCE MODE (expected_columns provided):
    - Creates one-hot columns based on data
    - Ensures all training columns are present (fills missing with 0)
    - Removes extra columns not seen in training

    Args:
        df: Input DataFrame with categorical columns.
        expected_columns: Optional list of expected column names after encoding.

    Returns:
        DataFrame with one-hot encoded features.
    """
    df = df.copy()

    # Encode BusinessType and BusinessAge_Clean
    df = pd.get_dummies(
        df,
        columns=config.BUSINESS_TYPE_ENCODE_COLUMNS,
        prefix=config.BUSINESS_TYPE_PREFIXES,
        dtype=int,
        drop_first=False
    )

    # Encode ProjectState
    df = pd.get_dummies(
        df,
        columns=[config.STATE_ENCODE_COLUMN],
        prefix=config.STATE_PREFIX,
        dtype=int,
        drop_first=False
    )

    if expected_columns is not None:
        # INFERENCE MODE: Ensure columns match training
        current_cols = set(df.columns)
        expected_cols = set(expected_columns)

        # Add missing columns (fill with 0)
        missing_cols = expected_cols - current_cols
        if missing_cols:
            logger.warning(f"Adding {len(missing_cols)} missing columns (filled with 0)")
            for col in missing_cols:
                df[col] = 0

        # Remove extra columns
        extra_cols = current_cols - expected_cols
        if extra_cols:
            logger.warning(f"Removing {len(extra_cols)} extra columns not in training")
            df = df.drop(columns=list(extra_cols))

        # Reorder columns to match training
        df = df[expected_columns]
        logger.info(f"Columns aligned with training ({len(expected_columns)} features)")
    else:
        # TRAINING MODE
        logger.info(f"Categorical encoding complete. Shape: {df.shape}")

    return df


def _drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw/intermediate columns that were used for feature engineering.

    Drops columns from config.DROP_COLUMNS that exist in the DataFrame.

    Args:
        df: Input DataFrame with engineered features.

    Returns:
        DataFrame with raw columns removed.
    """
    df = df.copy()

    # Drop only columns that exist
    existing_cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        logger.info(f"Dropped {len(existing_cols_to_drop)} raw/intermediate columns")

    return df


def engineer_features(
    df: pd.DataFrame,
    frequency_map: Optional[Dict[float, int]] = None,
    expected_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline to raw data.

    CRITICAL: This function must exactly match feature_pipeline.engineering.create_features()
    to guarantee training-inference consistency.

    Transformations (in order):
    1. Create IsCovidEra from ApprovalDate → drop ApprovalDate
    2. Extract NAICSSector from NAICSCode (first 2 digits)
    3. Create binary indicators:
       - IsCreditUnion (BankNCUANumber not null)
       - IsFranchise (FranchiseCode not null)
       - IsFixedRate (FixedorVariableInterestRate == 'F')
       - HasCollateral (CollateralInd == 'Y')
    4. Create SameStateLending (BankState == ProjectState)
    5. Apply LocationID frequency encoding → LocationIDCount
    6. One-hot encode BusinessType → Type_*
    7. One-hot encode BusinessAge_Clean → Age_*
    8. One-hot encode ProjectState → State_*
    9. Drop raw/intermediate columns
    10. Ensure column order matches training (inference only)

    TRAINING MODE (frequency_map=None, expected_columns=None):
    - Computes frequency_map from data
    - Creates all one-hot columns based on data
    - Returns DataFrame ready for model training

    INFERENCE MODE (frequency_map provided, expected_columns provided):
    - Uses provided frequency_map (unseen LocationIDs get min frequency)
    - Ensures all training columns present (missing filled with 0)
    - Removes columns not seen in training
    - Reorders columns to match training exactly
    - Returns DataFrame ready for model prediction

    Args:
        df: Raw DataFrame with columns like ApprovalDate, NAICSCode, etc.
        frequency_map: Optional LocationID frequency map (required for inference).
        expected_columns: Optional list of expected columns (required for inference).

    Returns:
        DataFrame with all engineered features, ready for model input.

    Example (Training):
        >>> df_engineered = engineer_features(df_raw)
        >>> df_engineered.shape
        (55831, 74)

    Example (Inference):
        >>> df_engineered = engineer_features(
        ...     df_raw,
        ...     frequency_map=loaded_freq_map,
        ...     expected_columns=model_feature_names
        ... )
        >>> df_engineered.shape
        (1, 74)
    """
    logger.info("Starting feature engineering pipeline")

    # Step 1: COVID indicator (drops ApprovalDate)
    df = _create_covid_indicator(df)

    # Step 2: NAICS sector
    df = _create_naics_sector(df)

    # Step 3: Binary indicators
    df = _create_binary_indicators(df)

    # Step 4: SameStateLending
    df = _create_same_state_lending(df)

    # Step 5: Frequency encoding
    df, _ = _apply_frequency_encoding(df, frequency_map)

    # Step 6-8: One-hot encoding
    df = _one_hot_encode_categoricals(df, expected_columns)

    # Step 9: Drop raw columns
    df = _drop_raw_columns(df)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")

    return df
