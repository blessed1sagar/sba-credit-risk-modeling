"""
Feature engineering module for SBA Loan preprocessing pipeline.

Handles feature creation, categorical encoding, and final dataset preparation.
All engineered features use PascalCase naming convention.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

from src import config

logger = logging.getLogger(__name__)


def create_covid_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create COVID-19 period indicator feature.

    Features created:
    - IsCovidEra: 1 if loan approved between 2020-03-01 and 2021-12-31, else 0

    IMPORTANT: This function also DROPS ApprovalDate after creating the indicator,
    as the raw date should not be used for modeling (prevents leakage).

    Args:
        df: Input DataFrame with ApprovalDate column.

    Returns:
        DataFrame with IsCovidEra feature (ApprovalDate removed).

    Example:
        >>> df_eng = create_covid_indicator(df)
        >>> print(df_eng['IsCovidEra'].value_counts())
        1    29537  # COVID loans
        0    26294  # Non-COVID loans
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

    covid_count = df['IsCovidEra'].sum()
    logger.info(f"Created IsCovidEra feature ({covid_count:,} COVID loans)")

    # Drop ApprovalDate to prevent leakage
    df = df.drop(columns=[config.APPROVAL_DATE_COLUMN])
    logger.info(f"Dropped {config.APPROVAL_DATE_COLUMN} to prevent leakage")

    return df


def create_naics_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract NAICS sector from NAICSCode.

    Feature created:
    - NAICSSector: First 2 digits of NAICSCode (industry sector)

    Args:
        df: Input DataFrame with NAICSCode column.

    Returns:
        DataFrame with NAICSSector feature.

    Example:
        >>> df_eng = create_naics_sector(df)
        >>> print(df_eng['NAICSSector'].nunique())
        24  # unique sectors
    """
    df = df.copy()

    df['NAICSSector'] = (
        df[config.NAICS_CODE_COLUMN]
        .astype(str)
        .str[:config.NAICS_SECTOR_DIGITS]
    )

    logger.info(
        f"Created NAICSSector feature "
        f"({df['NAICSSector'].nunique()} unique sectors)"
    )

    return df


def create_binary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary indicator features.

    Features created (all use PascalCase):
    - IsCreditUnion: 1 if BankNCUANumber is not null
    - IsFranchise: 1 if FranchiseCode is not null
    - IsFixedRate: 1 if FixedorVariableInterestRate == 'F'
    - HasCollateral: 1 if CollateralInd == 'Y'

    Note: SoldSecondaryMarket is NOT created (SoldSecondMarketInd already dropped
    as a leakage column).

    Args:
        df: Input DataFrame with source indicator columns.

    Returns:
        DataFrame with binary indicator features.

    Example:
        >>> df_eng = create_binary_indicators(df)
        >>> print(df_eng[['IsCreditUnion', 'IsFranchise']].sum())
        IsCreditUnion    1829
        IsFranchise      7218
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

    logger.info("Created binary indicator features (IsCreditUnion, IsFranchise, IsFixedRate, HasCollateral)")

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between existing columns.

    Features created:
    - SameStateLending: 1 if BankState == ProjectState

    Args:
        df: Input DataFrame with BankState and ProjectState.

    Returns:
        DataFrame with interaction features.

    Example:
        >>> df_eng = create_interaction_features(df)
        >>> print(df_eng['SameStateLending'].value_counts())
        0    34133  # 61.14% cross-state
        1    21698  # 38.86% same-state
    """
    df = df.copy()

    df['SameStateLending'] = (
        df[config.BANK_STATE_COLUMN] == df[config.PROJECT_STATE_COLUMN]
    ).astype(int)

    same_state_count = df['SameStateLending'].sum()
    same_state_pct = (same_state_count / len(df)) * 100

    logger.info(
        f"Created SameStateLending feature "
        f"({same_state_count:,} same-state loans, {same_state_pct:.2f}%)"
    )

    return df


def create_frequency_encoding(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[float, int]]:
    """
    Create frequency encoding for LocationID.

    Features created:
    - LocationIDCount: Number of loans associated with each LocationID

    IMPORTANT: This function returns BOTH the DataFrame AND the frequency_map
    dictionary. The frequency_map must be saved for the inference pipeline to
    use the same encoding.

    Args:
        df: Input DataFrame with LocationID column.

    Returns:
        Tuple of (DataFrame with LocationIDCount feature, frequency_map dict).

    Example:
        >>> df_eng, freq_map = create_frequency_encoding(df)
        >>> print(df_eng['LocationIDCount'].describe())
        # Range: 1 to 4948
    """
    df = df.copy()

    # Create frequency map (LocationID -> count)
    frequency_map = df[config.LOCATION_ID_COLUMN].value_counts().to_dict()

    # Map to DataFrame
    df['LocationIDCount'] = df[config.LOCATION_ID_COLUMN].map(frequency_map)

    logger.info(
        f"Created LocationIDCount frequency encoding "
        f"(range: {df['LocationIDCount'].min()} to {df['LocationIDCount'].max()})"
    )

    return df, frequency_map


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical variables.

    Encodes:
    - BusinessType → Type_CORPORATION, Type_INDIVIDUAL, Type_PARTNERSHIP
    - BusinessAge_Clean → Age_ChangeOfOwnership, Age_Existing, Age_NewBusiness, Age_Startup
    - NAICSSector → Sector_11, Sector_21, ..., Sector_99 (NAICS 2-digit sectors)
    - ProjectState → State_AK, State_AL, ..., State_WY (54 states/territories)

    Args:
        df: Input DataFrame with categorical columns.

    Returns:
        DataFrame with one-hot encoded features.

    Example:
        >>> df_enc = encode_categoricals(df)
        >>> type_cols = [col for col in df_enc.columns if col.startswith('Type_')]
        >>> print(type_cols)
        ['Type_CORPORATION', 'Type_INDIVIDUAL', 'Type_PARTNERSHIP']
    """
    df = df.copy()

    # Encode BusinessType and BusinessAge_Clean
    df = pd.get_dummies(
        df,
        columns=config.BUSINESS_TYPE_ENCODE_COLUMNS,
        prefix=config.BUSINESS_TYPE_PREFIXES,
        dtype=int,
        drop_first=False  # Keep all categories for interpretability
    )

    # Encode NAICSSector (2-digit industry code)
    df = pd.get_dummies(
        df,
        columns=['NAICSSector'],
        prefix='Sector',
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

    logger.info(f"Categorical encoding complete. Shape: {df.shape}")

    return df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[float, int]]:
    """
    Execute full feature engineering pipeline.

    Orchestrates all feature creation steps:
    1. COVID-19 indicator (and drop ApprovalDate)
    2. NAICS sector
    3. Binary indicators
    4. Interaction features
    5. Frequency encoding (returns frequency_map)
    6. Categorical encoding

    IMPORTANT: Returns both the DataFrame AND the frequency_map for LocationID.
    The frequency_map must be saved for inference pipeline reproducibility.

    Args:
        df: Input DataFrame from cleaning module.

    Returns:
        Tuple of (DataFrame with all engineered features, frequency_map dict).

    Example:
        >>> df_features, freq_map = create_features(df_clean)
        >>> print(df_features.shape)
        (55831, varies)
    """
    logger.info("Starting feature engineering pipeline")

    df = create_covid_indicator(df)
    df = create_naics_sector(df)
    df = create_binary_indicators(df)
    df = create_interaction_features(df)
    df, frequency_map = create_frequency_encoding(df)
    df = encode_categoricals(df)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")

    return df, frequency_map


def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant columns and prepare final dataset.

    Steps:
    1. Drop columns defined in config.DROP_COLUMNS
    2. Keep LoanStatus column (train.py will handle X/y split)
    3. Validate no missing values

    IMPORTANT: This function does NOT split X and y. The final DataFrame
    contains all features plus the LoanStatus column. The training pipeline
    will handle the train/test split and X/y separation.

    Args:
        df: Input DataFrame with all engineered features.

    Returns:
        Final processed DataFrame with all features + LoanStatus column.

    Raises:
        ValueError: If any features contain missing values.

    Example:
        >>> df_final = prepare_final_dataset(df_features)
        >>> assert 'LoanStatus' in df_final.columns
        >>> assert df_final.isnull().sum().sum() == 0
    """
    df = df.copy()

    # Drop irrelevant columns (only those that exist)
    existing_cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    logger.info(f"Dropped {len(existing_cols_to_drop)} irrelevant columns")

    # Validate LoanStatus is present
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"{config.TARGET_COLUMN} column missing from dataset")

    # Validate no missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        missing_summary = df[missing_cols].isnull().sum()
        raise ValueError(
            f"Missing values found in columns:\n{missing_summary[missing_summary > 0]}"
        )

    logger.info(f"Final dataset prepared - Shape: {df.shape}")
    logger.info(
        f"Target distribution: {df[config.TARGET_COLUMN].value_counts().to_dict()}"
    )

    return df
