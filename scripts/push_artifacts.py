"""
Manual artifact upload script for pushing "golden state" to S3.

Uploads trained model and frequency encoder to S3 bucket for cloud deployment.

Usage:
    python scripts/push_artifacts.py              # Upload current artifacts
    python scripts/push_artifacts.py --dry-run    # Preview without uploading
    python scripts/push_artifacts.py --force      # Overwrite existing S3 artifacts
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.utils.s3_manager import S3ArtifactManager

logger = logging.getLogger(__name__)


def verify_artifacts_exist() -> dict:
    """
    Verify all required artifacts exist locally.

    Returns:
        dict: Mapping of artifact names to paths.

    Raises:
        FileNotFoundError: If any required artifacts are missing.
    """
    artifacts = {
        'model': config.TUNED_MODEL_PATH,
        'encoder': config.FREQUENCY_ENCODER_PATH
    }

    missing = []
    for name, path in artifacts.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError(
            f"Cannot upload - missing artifacts:\n" + "\n".join(f"  • {m}" for m in missing) +
            f"\n\nTo generate artifacts, run: python run_pipeline.py"
        )

    return artifacts


def push_artifacts(dry_run: bool = False, force: bool = False) -> None:
    """
    Push local artifacts to S3.

    Args:
        dry_run: If True, preview without uploading.
        force: If True, overwrite existing S3 artifacts without prompting.
    """
    logger.info("=" * 80)
    logger.info("PUSH ARTIFACTS TO S3")
    logger.info("=" * 80)

    # Verify artifacts exist
    logger.info("\nVerifying local artifacts...")
    try:
        artifacts = verify_artifacts_exist()
        logger.info("✓ All artifacts found locally:")
        for name, path in artifacts.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"  • {name:10s}: {path.name:30s} ({size_mb:.2f} MB)")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)

    # Initialize S3 manager
    logger.info(f"\nInitializing S3 connection...")
    logger.info(f"  Bucket: {config.S3_BUCKET_NAME}")
    logger.info(f"  Region: {config.S3_REGION}")

    s3_manager = S3ArtifactManager(config.S3_BUCKET_NAME, config.S3_REGION)

    if not s3_manager.is_active:
        logger.error("❌ S3 not available. Please configure AWS credentials:")
        logger.error("  1. Run: aws configure")
        logger.error("  2. Or set environment variables:")
        logger.error("       export AWS_ACCESS_KEY_ID=your_key")
        logger.error("       export AWS_SECRET_ACCESS_KEY=your_secret")
        logger.error("       export AWS_REGION=ap-south-2")
        sys.exit(1)

    logger.info("✓ S3 connection active")

    # Check if artifacts already exist in S3
    logger.info("\nChecking S3 for existing artifacts...")
    existing_in_s3 = []
    artifact_keys = {
        'model': config.S3_MODEL_KEY,
        'encoder': config.S3_ENCODER_KEY
    }

    for artifact_name, s3_key in artifact_keys.items():
        if s3_manager.file_exists_in_s3(s3_key):
            existing_in_s3.append(f"{artifact_name} ({s3_key})")
            logger.info(f"  ⚠️  {artifact_name}: Already exists in S3")
        else:
            logger.info(f"  ✓ {artifact_name}: Not in S3 (will be new upload)")

    # Prompt for confirmation if artifacts exist and not forced
    if existing_in_s3 and not force and not dry_run:
        logger.warning("\n⚠️  The following artifacts already exist in S3:")
        for item in existing_in_s3:
            logger.warning(f"  • {item}")

        response = input("\nOverwrite existing artifacts? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Upload cancelled by user")
            sys.exit(0)

    # Dry run preview
    if dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 DRY RUN - Would upload:")
        logger.info("=" * 80)
        logger.info(f"  Model:   {config.TUNED_MODEL_PATH}")
        logger.info(f"      →    s3://{config.S3_BUCKET_NAME}/{config.S3_MODEL_KEY}")
        logger.info(f"  Encoder: {config.FREQUENCY_ENCODER_PATH}")
        logger.info(f"      →    s3://{config.S3_BUCKET_NAME}/{config.S3_ENCODER_KEY}")
        logger.info("\nRun without --dry-run to perform actual upload")
        return

    # Perform upload
    logger.info("\n" + "=" * 80)
    logger.info("UPLOADING ARTIFACTS")
    logger.info("=" * 80)

    results = s3_manager.upload_training_artifacts(
        model_path=config.TUNED_MODEL_PATH,
        encoder_path=config.FREQUENCY_ENCODER_PATH
    )

    # Summary
    logger.info("\n" + "=" * 80)
    if results['success']:
        logger.info("✓ UPLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nBucket:  s3://{config.S3_BUCKET_NAME}/")
        logger.info(f"Region:  {config.S3_REGION}")
        logger.info(f"\nArtifacts uploaded:")
        logger.info(f"  • Model:   {config.S3_MODEL_KEY}")
        logger.info(f"  • Encoder: {config.S3_ENCODER_KEY}")
        logger.info(f"\nCloud deployment ready!")
        logger.info(f"Set SYNC_FROM_S3=true in your deployment environment to use these artifacts.")
    else:
        logger.error("❌ UPLOAD FAILED")
        logger.error("=" * 80)
        logger.error("Check logs above for errors")
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Push ML artifacts to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload artifacts (with confirmation)
  python scripts/push_artifacts.py

  # Preview without uploading
  python scripts/push_artifacts.py --dry-run

  # Force upload without confirmation
  python scripts/push_artifacts.py --force

  # Check what would be uploaded
  python scripts/push_artifacts.py --dry-run
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without uploading'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing S3 artifacts without prompting'
    )

    args = parser.parse_args()

    push_artifacts(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
