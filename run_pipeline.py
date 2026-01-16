"""
Unified ML pipeline orchestrator for SBA Loan default prediction.

Executes complete pipeline: preprocessing → training → S3 upload (optional).

Usage:
    python run_pipeline.py              # Run locally (no S3 upload)
    python run_pipeline.py --upload     # Run and upload to S3
    python run_pipeline.py --skip-preprocessing  # Skip if data already processed
    python run_pipeline.py --skip-training       # Skip if model already trained
    python run_pipeline.py --n-trials 100        # Custom number of Optuna trials
"""
import argparse
import logging
import sys
from pathlib import Path

from src import config
from src.main import run_preprocessing_pipeline
from src.training_pipeline.tune_optuna import run_hyperparameter_tuning
from src.utils.s3_manager import S3ArtifactManager

logger = logging.getLogger(__name__)


def run_full_pipeline(
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    upload_to_s3: bool = False,
    n_trials: int = None
) -> None:
    """
    Execute complete ML pipeline.

    Stages:
    1. Preprocessing (creates processed_data.parquet + frequency_encoder.pkl)
    2. Training (creates xgb_tuned.joblib)
    3. S3 Upload (optional - uploads model + encoder)

    Args:
        skip_preprocessing: Skip preprocessing if data already processed.
        skip_training: Skip training if model already trained.
        upload_to_s3: Upload artifacts to S3 after training.
        n_trials: Number of Optuna trials (None = use config default).
    """
    logger.info("=" * 80)
    logger.info("SBA LOAN ML PIPELINE ORCHESTRATOR")
    logger.info("=" * 80)

    # Stage 1: Preprocessing
    if skip_preprocessing:
        logger.info("\n[Stage 1/3] SKIPPING PREPROCESSING (--skip-preprocessing)")
        if not config.PROCESSED_DATA_PATH.exists():
            logger.error(f"ERROR: Cannot skip preprocessing - processed data not found!")
            logger.error(f"Expected: {config.PROCESSED_DATA_PATH}")
            logger.error(f"Run without --skip-preprocessing to generate data")
            sys.exit(1)
        logger.info(f"✓ Using existing processed data: {config.PROCESSED_DATA_PATH}")
    else:
        logger.info("\n[Stage 1/3] PREPROCESSING")
        logger.info(f"Input: {config.RAW_DATA_PATH}")
        logger.info(f"Output: {config.PROCESSED_DATA_PATH}")
        try:
            run_preprocessing_pipeline(save_outputs=True)
            logger.info("✓ Preprocessing complete")
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            sys.exit(1)

    # Stage 2: Training
    if skip_training:
        logger.info("\n[Stage 2/3] SKIPPING TRAINING (--skip-training)")
        if not config.TUNED_MODEL_PATH.exists():
            logger.error(f"ERROR: Cannot skip training - model not found!")
            logger.error(f"Expected: {config.TUNED_MODEL_PATH}")
            logger.error(f"Run without --skip-training to train model")
            sys.exit(1)
        logger.info(f"✓ Using existing model: {config.TUNED_MODEL_PATH}")
    else:
        logger.info("\n[Stage 2/3] HYPERPARAMETER TUNING")
        logger.info(f"Method: Optuna with TPE sampler")
        logger.info(f"Trials: {n_trials if n_trials else config.OPTUNA_N_TRIALS}")
        logger.info(f"Objective: Maximize ROC-AUC")
        try:
            run_hyperparameter_tuning(
                n_trials=n_trials,
                save_model_flag=True,
                log_mlflow=True
            )
            logger.info("✓ Training complete")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            sys.exit(1)

    # Stage 3: S3 Upload
    logger.info("\n[Stage 3/3] ARTIFACT MANAGEMENT")

    if upload_to_s3:
        logger.info("Uploading artifacts to S3...")
        s3_manager = S3ArtifactManager(config.S3_BUCKET_NAME, config.S3_REGION)

        if not s3_manager.is_active:
            logger.warning("⚠️  S3 not available. Artifacts saved locally only.")
            logger.warning("To enable S3 upload:")
            logger.warning("  1. Configure AWS credentials: aws configure")
            logger.warning("  2. Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        else:
            results = s3_manager.upload_training_artifacts(
                model_path=config.TUNED_MODEL_PATH,
                encoder_path=config.FREQUENCY_ENCODER_PATH
            )

            if results['success']:
                logger.info("✓ Artifacts uploaded to S3 successfully")
            else:
                logger.warning("⚠️  Pipeline complete but S3 upload had errors")
    else:
        logger.info("✓ Artifacts saved locally (use --upload to push to S3)")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nLocal artifacts:")
    logger.info(f"  Model:       {config.TUNED_MODEL_PATH}")
    logger.info(f"  Encoder:     {config.FREQUENCY_ENCODER_PATH}")
    logger.info(f"  Data:        {config.PROCESSED_DATA_PATH}")

    if upload_to_s3 and s3_manager.is_active:
        logger.info(f"\nS3 artifacts:")
        logger.info(f"  Bucket:      s3://{config.S3_BUCKET_NAME}/")
        logger.info(f"  Region:      {config.S3_REGION}")
        logger.info(f"  Model:       {config.S3_MODEL_KEY}")
        logger.info(f"  Encoder:     {config.S3_ENCODER_KEY}")

    logger.info(f"\nNext steps:")
    logger.info(f"  • Test API:  uvicorn src.api.main:app --reload")
    logger.info(f"  • Test UI:   streamlit run app.py")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SBA Loan ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline locally (no S3)
  python run_pipeline.py

  # Run and upload to S3
  python run_pipeline.py --upload

  # Skip preprocessing if data already processed
  python run_pipeline.py --skip-preprocessing

  # Quick training with fewer trials
  python run_pipeline.py --n-trials 10

  # Use existing data and model, just upload to S3
  python run_pipeline.py --skip-preprocessing --skip-training --upload
        """
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing if data already processed'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training if model already exists'
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload artifacts to S3 after training'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=None,
        help=f'Number of Optuna trials (default: {config.OPTUNA_N_TRIALS})'
    )

    args = parser.parse_args()

    run_full_pipeline(
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training,
        upload_to_s3=args.upload,
        n_trials=args.n_trials
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
