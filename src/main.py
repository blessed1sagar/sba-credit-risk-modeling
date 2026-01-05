"""
SBA Loan Preprocessing - Main Pipeline Orchestration

Example script demonstrating how to execute the full preprocessing pipeline
from raw CSV to final processed parquet file.
"""
import logging
import joblib

from src.feature_pipeline import (
    load_raw_data,
    filter_by_loan_status,
    drop_leakage_columns,
    clean_data,
    create_features,
    prepare_final_dataset,
)
from src import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(save_outputs: bool = True) -> None:
    """
    Execute complete preprocessing pipeline.

    Pipeline stages:
    1. Load raw CSV data
    2. Filter to PIF/CHGOFF loan statuses
    3. Drop leakage columns (FirstDisbursementDate, TerminMonths, SoldSecondMarketInd)
    4. Clean data (handle missing values, standardize categoricals)
    5. Engineer features (create derived features, encode categoricals)
    6. Prepare final dataset (drop irrelevant columns, validate)
    7. Save outputs (parquet file + frequency encoder)

    Args:
        save_outputs: If True, save processed data and frequency encoder to disk.

    Outputs (if save_outputs=True):
        - data/feature/processed_data.parquet: Final dataset with LoanStatus
        - data/feature/frequency_encoder.pkl: LocationID frequency mapping
    """
    logger.info("=" * 80)
    logger.info("SBA LOAN PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    # Stage 1: Load raw data
    logger.info("\n[1/7] Loading raw data...")
    df_raw = load_raw_data()

    # Stage 2: Filter to target loan statuses
    logger.info("\n[2/7] Filtering to PIF/CHGOFF loans...")
    df_filtered = filter_by_loan_status(df_raw)

    # Stage 3: Drop leakage columns
    logger.info("\n[3/7] Dropping leakage columns...")
    df_no_leakage = drop_leakage_columns(df_filtered)

    # Stage 4: Clean data
    logger.info("\n[4/7] Cleaning data...")
    df_clean = clean_data(df_no_leakage)

    # Stage 5: Engineer features
    logger.info("\n[5/7] Engineering features...")
    df_features, frequency_map = create_features(df_clean)

    # Stage 6: Prepare final dataset
    logger.info("\n[6/7] Preparing final dataset...")
    df_final = prepare_final_dataset(df_features)

    # Stage 7: Save outputs
    if save_outputs:
        logger.info("\n[7/7] Saving outputs...")

        # Ensure output directory exists
        config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save processed data as parquet (preserves dtypes)
        df_final.to_parquet(config.PROCESSED_DATA_PATH, index=False)
        logger.info(f"✓ Saved processed data to: {config.PROCESSED_DATA_PATH}")

        # Save frequency encoder for inference pipeline
        joblib.dump(frequency_map, config.FREQUENCY_ENCODER_PATH)
        logger.info(f"✓ Saved frequency encoder to: {config.FREQUENCY_ENCODER_PATH}")
    else:
        logger.info("\n[7/7] Skipping save (save_outputs=False)")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final shape: {df_final.shape}")
    logger.info(f"Features: {df_final.shape[1] - 1} (+ LoanStatus target)")
    logger.info(f"Samples: {df_final.shape[0]:,}")
    logger.info(
        f"Target distribution: "
        f"{df_final[config.TARGET_COLUMN].value_counts().to_dict()}"
    )

    if save_outputs:
        logger.info(f"\nOutputs saved:")
        logger.info(f"  - {config.PROCESSED_DATA_PATH}")
        logger.info(f"  - {config.FREQUENCY_ENCODER_PATH}")


if __name__ == "__main__":
    run_preprocessing_pipeline(save_outputs=True)
