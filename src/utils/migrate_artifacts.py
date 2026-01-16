"""
One-time migration script to move artifacts to new directory structure.

Migrates from old structure:
  - models/xgb_tuned.joblib
  - models/xgb_baseline.joblib
  - data/feature/frequency_encoder.pkl

To new structure:
  - models/trained/xgb_tuned.joblib
  - models/trained/xgb_baseline.joblib
  - models/encoders/frequency_map.pkl

Usage:
    python -m src.utils.migrate_artifacts
"""
import shutil
from pathlib import Path
import logging

from src import config

logger = logging.getLogger(__name__)


# Define old and new paths
OLD_PATHS = {
    'tuned_model': config.PROJECT_ROOT / "models" / "xgb_tuned.joblib",
    'baseline_model': config.PROJECT_ROOT / "models" / "xgb_baseline.joblib",
    'encoder': config.PROJECT_ROOT / "data" / "feature" / "frequency_encoder.pkl"
}

NEW_PATHS = {
    'tuned_model': config.TUNED_MODEL_PATH,
    'baseline_model': config.BASELINE_MODEL_PATH,
    'encoder': config.FREQUENCY_ENCODER_PATH
}


def migrate_artifacts():
    """
    Migrate artifacts from old to new directory structure.

    Creates new directories, copies files, and provides status report.
    Safe to run multiple times (checks if files already exist).
    """
    logger.info("=" * 80)
    logger.info("ARTIFACT MIGRATION TO NEW DIRECTORY STRUCTURE")
    logger.info("=" * 80)

    # Create new directories
    logger.info("\nCreating new directories...")
    config.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Created: {config.TRAINED_MODELS_DIR}")
    config.ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Created: {config.ENCODERS_DIR}")

    # Migrate each artifact
    logger.info("\nMigrating artifacts...")
    migrated = []
    skipped = []
    not_found = []

    for artifact_type in ['tuned_model', 'baseline_model', 'encoder']:
        old_path = OLD_PATHS[artifact_type]
        new_path = NEW_PATHS[artifact_type]

        if new_path.exists():
            logger.info(f"⏭️  {artifact_type}: Already at new location: {new_path.name}")
            skipped.append(artifact_type)
        elif old_path.exists():
            shutil.copy2(old_path, new_path)
            logger.info(f"✓ {artifact_type}: Copied {old_path.name} → {new_path}")
            migrated.append(artifact_type)
        else:
            logger.warning(f"⚠️  {artifact_type}: Not found at old location: {old_path}")
            not_found.append(artifact_type)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Migrated: {len(migrated)} artifacts")
    logger.info(f"Skipped (already migrated): {len(skipped)} artifacts")
    logger.info(f"Not found: {len(not_found)} artifacts")

    if migrated:
        logger.info(f"\nMigrated artifacts:")
        for artifact in migrated:
            logger.info(f"  - {artifact}")

    if not_found:
        logger.info(f"\nMissing artifacts (run training to generate):")
        for artifact in not_found:
            logger.info(f"  - {artifact}")

    logger.info("\n✓ Migration complete!")
    logger.info(f"\nNew artifact locations:")
    logger.info(f"  Models: {config.TRAINED_MODELS_DIR}")
    logger.info(f"  Encoders: {config.ENCODERS_DIR}")

    if not_found:
        logger.info(f"\n💡 To generate missing artifacts, run:")
        logger.info(f"   python run_pipeline.py")


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    migrate_artifacts()


if __name__ == "__main__":
    main()
