"""
Data cleaning module for SBA Loan preprocessing pipeline.

Handles missing values and categorical variable standardization.
"""
import pandas as pd
import logging

from src import config

logger = logging.getLogger(__name__)


def drop_missing_required(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values in required columns.

    Removes rows missing critical fields needed for feature engineering:
    - LocationID: Required for frequency encoding (120 rows typically)
    - BankState: Required for same-state lending feature (120 rows typically)

    Note: FirstDisbursementDate is no longer in REQUIRED_COLUMNS since it's
    dropped entirely as a leakage column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with required columns guaranteed non-null (55,831 rows typically).

    Example:
        >>> df_clean = drop_missing_required(df_filtered)
        >>> assert df_clean[config.REQUIRED_COLUMNS].isnull().sum().sum() == 0
    """
    rows_before = len(df)
    df_clean = df.dropna(subset=config.REQUIRED_COLUMNS).copy()
    rows_dropped = rows_before - len(df_clean)

    logger.info(
        f"Dropped {rows_dropped:,} rows with missing values in {config.REQUIRED_COLUMNS}"
    )

    return df_clean


def clean_business_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean BusinessType column by removing whitespace and filling NaN.

    Strategy:
    - Replace '        ' (8 spaces) with 'INDIVIDUAL'
    - Fill remaining NaN with 'INDIVIDUAL' (most common for small businesses)

    Args:
        df: Input DataFrame with BusinessType column.

    Returns:
        DataFrame with cleaned BusinessType (no missing values).

    Example:
        >>> df_clean = clean_business_type(df)
        >>> print(df_clean['BusinessType'].value_counts())
        CORPORATION    51746
        INDIVIDUAL      3452
        PARTNERSHIP      633
    """
    df = df.copy()

    # Replace whitespace-only values
    df[config.BUSINESS_TYPE_COLUMN] = df[config.BUSINESS_TYPE_COLUMN].replace(
        config.BUSINESS_TYPE_WHITESPACE_VALUE,
        config.BUSINESS_TYPE_DEFAULT
    )

    # Fill remaining NaN
    df[config.BUSINESS_TYPE_COLUMN] = df[config.BUSINESS_TYPE_COLUMN].fillna(
        config.BUSINESS_TYPE_DEFAULT
    )

    logger.info(f"Cleaned {config.BUSINESS_TYPE_COLUMN}")

    return df


def clean_business_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map BusinessAge to standardized categories.

    Creates new column BusinessAge_Clean with simplified categories:
    - 'Existing or more than 2 years old' → 'Existing'
    - 'Startup, Loan Funds will Open Business' → 'Startup'
    - 'New Business or 2 years or less' → 'NewBusiness'
    - 'Change of Ownership' → 'ChangeOfOwnership'
    - 'Unanswered' / NaN → 'Existing' (conservative assumption)

    Args:
        df: Input DataFrame with BusinessAge column.

    Returns:
        DataFrame with new BusinessAge_Clean column.

    Example:
        >>> df_clean = clean_business_age(df)
        >>> print(df_clean['BusinessAge_Clean'].value_counts())
        Existing             32620
        NewBusiness           9016
        Startup               8521
        ChangeOfOwnership     5674
    """
    df = df.copy()

    df[config.BUSINESS_AGE_CLEAN_COLUMN] = (
        df[config.BUSINESS_AGE_COLUMN]
        .map(config.BUSINESS_AGE_MAPPING)
        .fillna(config.BUSINESS_AGE_DEFAULT)
    )

    logger.info(f"Created {config.BUSINESS_AGE_CLEAN_COLUMN}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute full data cleaning pipeline.

    Orchestrates all cleaning steps:
    1. Drop rows with missing required columns
    2. Clean BusinessType
    3. Clean BusinessAge

    Args:
        df: Input DataFrame from load module (after filtering and leakage drop).

    Returns:
        Cleaned DataFrame (55,831 rows typically).

    Example:
        >>> df_clean = clean_data(df_filtered)
        >>> print(df_clean.shape)
        (55831, varies)
    """
    logger.info("Starting data cleaning pipeline")

    df = drop_missing_required(df)
    df = clean_business_type(df)
    df = clean_business_age(df)

    logger.info(f"Cleaning complete. Final shape: {df.shape}")

    return df
