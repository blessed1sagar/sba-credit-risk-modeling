"""
Data loading module for SBA Loan preprocessing pipeline.

Handles loading raw CSV data, filtering to relevant loan statuses, and dropping
leakage columns that should not be used for modeling.
"""
import pandas as pd
from typing import Optional
import logging

from src import config

logger = logging.getLogger(__name__)


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw SBA loan data from CSV file.

    Args:
        file_path: Path to raw CSV file. If None, uses default from config.

    Returns:
        DataFrame with raw loan data (347,514 rows × 43 columns).

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        pd.errors.EmptyDataError: If CSV file is empty.

    Example:
        >>> df = load_raw_data()
        >>> print(df.shape)
        (347514, 43)
    """
    if file_path is None:
        file_path = config.RAW_DATA_PATH

    logger.info(f"Loading raw data from: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {file_path}")
        raise

    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    return df


def filter_by_loan_status(
    df: pd.DataFrame,
    valid_statuses: Optional[list] = None
) -> pd.DataFrame:
    """
    Filter dataset to keep only loans with specified statuses.

    Filters to completed loans (PIF = Paid in Full, CHGOFF = Charged Off) to
    create clear binary classification target. Removes intermediate statuses
    like EXEMPT, CANCLD, COMMIT.

    Args:
        df: Input DataFrame with LoanStatus column.
        valid_statuses: List of loan statuses to keep. If None, uses config default.

    Returns:
        Filtered DataFrame (55,954 rows × 43 columns before cleaning).

    Raises:
        KeyError: If LoanStatus column doesn't exist.
        ValueError: If no rows match the valid statuses.

    Example:
        >>> df_filtered = filter_by_loan_status(df_raw)
        >>> print(df_filtered['LoanStatus'].unique())
        ['PIF' 'CHGOFF']
    """
    if valid_statuses is None:
        valid_statuses = config.VALID_LOAN_STATUSES

    if config.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Column '{config.TARGET_COLUMN}' not found in DataFrame")

    rows_before = len(df)
    df_filtered = df[df[config.TARGET_COLUMN].isin(valid_statuses)].copy()
    rows_after = len(df_filtered)

    if rows_after == 0:
        raise ValueError(f"No rows match statuses: {valid_statuses}")

    logger.info(
        f"Filtered to {valid_statuses}: "
        f"{rows_after:,} rows kept, {rows_before - rows_after:,} removed "
        f"({(rows_before - rows_after) / rows_before * 100:.2f}%)"
    )

    return df_filtered


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that contain leakage or future information.

    Removes columns that either:
    1. Contain information not available at loan origination
    2. Were proven through A/B testing to be noisy or leaky
    3. Relate to post-approval events

    IMPORTANT: This function MUST be called before any feature engineering
    to ensure downstream code doesn't accidentally use these columns.

    Columns dropped:
    - FirstDisbursementDate: Future information (prevents TimeToDisbursementDays)
    - TerminMonths: A/B testing showed TermInYears is leaky (prevents TermInYears)
    - SoldSecondMarketInd: Secondary market status (prevents SoldSecondaryMarket)

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with leakage columns removed (55,954 rows × 40 columns).

    Example:
        >>> df_clean = drop_leakage_columns(df_filtered)
        >>> assert 'FirstDisbursementDate' not in df_clean.columns
        >>> assert 'TerminMonths' not in df_clean.columns
    """
    rows_before = len(df)
    cols_before = len(df.columns)

    # Only drop columns that exist in the dataframe
    existing_leakage_cols = [c for c in config.LEAKAGE_COLUMNS if c in df.columns]

    df_clean = df.drop(columns=existing_leakage_cols).copy()

    cols_after = len(df_clean.columns)

    logger.info(
        f"Dropped {len(existing_leakage_cols)} leakage columns: {existing_leakage_cols}"
    )
    logger.info(f"Shape: ({rows_before:,} × {cols_before}) → ({len(df_clean):,} × {cols_after})")

    return df_clean
