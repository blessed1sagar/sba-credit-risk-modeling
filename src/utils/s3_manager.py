"""
AWS S3 artifact management module for SBA Loan ML pipeline.

Handles uploading and downloading of model artifacts and encoders to/from S3.
Enables cloud portability and multi-environment deployment.
"""
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
import logging
from typing import Optional

from src import config

logger = logging.getLogger(__name__)


class S3ArtifactManager:
    """
    AWS S3 artifact synchronization manager with graceful degradation.

    Handles smart sync of ML artifacts (models, encoders) between local storage
    and S3. Downloads artifacts only if missing locally. Gracefully handles
    missing AWS credentials for local development.

    Attributes:
        bucket_name: S3 bucket name.
        region: AWS region.
        s3_client: boto3 S3 client (None if credentials unavailable).
        is_active: Boolean flag indicating if S3 is available.

    Example:
        >>> manager = S3ArtifactManager('sba-credit-risk-artifacts-sagar')
        >>> if manager.is_active:
        ...     manager.sync_inference_artifacts(Path('./models'))
    """

    def __init__(self, bucket_name: str, region: str = 'ap-south-2'):
        """
        Initialize S3ArtifactManager with graceful fallback.

        Args:
            bucket_name: S3 bucket name.
            region: AWS region (default: ap-south-2).

        Note:
            Does NOT raise exception if credentials missing. Sets is_active=False instead.
        """
        self.bucket_name = bucket_name
        self.region = region
        self.is_active = False
        self.s3_client = None

        try:
            self.s3_client = boto3.client('s3', region_name=region)
            # Test credentials by checking bucket access
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.is_active = True
            logger.info(f"✓ S3 client active (bucket: {bucket_name}, region: {region})")
        except NoCredentialsError:
            logger.warning("⚠️  AWS credentials not found. S3 features disabled (local mode).")
        except ClientError as e:
            logger.warning(f"⚠️  S3 bucket access failed: {e}. S3 features disabled.")
        except Exception as e:
            logger.warning(f"⚠️  S3 initialization failed: {e}. Running in local mode.")

    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        """
        Upload file to S3.

        Args:
            local_path: Local file path to upload.
            s3_key: S3 object key (path in bucket).

        Returns:
            True if upload successful or S3 disabled, False on error.

        Example:
            >>> manager.upload_file(Path('models/xgb_tuned.joblib'), 'models/trained/xgb_tuned.joblib')
            True
        """
        if not self.is_active:
            logger.info(f"S3 inactive - skipping upload of {local_path.name}")
            return True  # Not an error - just running locally

        try:
            if not local_path.exists():
                logger.error(f"Local file not found: {local_path}")
                return False

            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key
            )
            logger.info(f"✓ Uploaded {local_path.name} → s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            return False

    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """
        Download file from S3.

        Args:
            s3_key: S3 object key (path in bucket).
            local_path: Local file path to save to.

        Returns:
            True if download successful, False otherwise.

        Example:
            >>> manager.download_file('models/trained/xgb_tuned.joblib', Path('models/trained/xgb_tuned.joblib'))
            True
        """
        if not self.is_active:
            logger.debug(f"S3 inactive - cannot download {s3_key}")
            return False

        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            logger.info(f"✓ Downloaded s3://{self.bucket_name}/{s3_key} → {local_path.name}")
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"S3 object not found: s3://{self.bucket_name}/{s3_key}")
            else:
                logger.error(f"Error downloading from S3: {e}")
            return False

    def file_exists_in_s3(self, s3_key: str) -> bool:
        """
        Check if file exists in S3.

        Args:
            s3_key: S3 object key to check.

        Returns:
            True if file exists, False otherwise.
        """
        if not self.is_active:
            return False

        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def upload_artifact(self, local_path: Path, s3_key: str, artifact_type: str = "artifact") -> bool:
        """
        Upload artifact to S3 with graceful failure handling.

        Args:
            local_path: Local file path to upload.
            s3_key: S3 object key (path in bucket).
            artifact_type: Type description for logging (e.g., "model", "encoder").

        Returns:
            True if upload successful or S3 disabled, False on error.

        Example:
            >>> manager.upload_artifact(Path('models/xgb_tuned.joblib'), 'models/trained/xgb_tuned.joblib', 'model')
            True
        """
        if not self.is_active:
            logger.info(f"S3 inactive - skipping upload of {artifact_type}: {local_path.name}")
            return True  # Not an error - just running locally

        if not local_path.exists():
            logger.error(f"Cannot upload {artifact_type} - file not found: {local_path}")
            return False

        try:
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            logger.info(f"✓ Uploaded {artifact_type}: {local_path.name} → s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading {artifact_type} to S3: {e}")
            return False

    def upload_training_artifacts(self, model_path: Path, encoder_path: Path) -> dict:
        """
        Upload all training artifacts (model + encoder) in one call.

        Args:
            model_path: Local path to trained model.
            encoder_path: Local path to frequency encoder.

        Returns:
            dict with upload status: {'model': bool, 'encoder': bool, 'success': bool}

        Example:
            >>> results = manager.upload_training_artifacts(
            ...     Path('models/trained/xgb_tuned.joblib'),
            ...     Path('models/encoders/frequency_map.pkl')
            ... )
            >>> if results['success']:
            ...     print("All artifacts uploaded")
        """
        results = {
            'model': self.upload_artifact(model_path, config.S3_MODEL_KEY, "model"),
            'encoder': self.upload_artifact(encoder_path, config.S3_ENCODER_KEY, "encoder")
        }
        results['success'] = results['model'] and results['encoder']

        if results['success']:
            logger.info("✓ All training artifacts uploaded successfully")
        elif not self.is_active:
            logger.info("✓ Training artifacts saved locally (S3 disabled)")
        else:
            logger.warning("⚠️  Some artifacts failed to upload to S3")

        return results

    def sync_inference_artifacts(self, local_models_dir: Optional[Path] = None) -> dict:
        """
        Smart sync of inference artifacts from S3.

        Downloads artifacts only if missing locally:
        - models/trained/xgb_tuned.joblib
        - models/encoders/frequency_map.pkl

        If artifacts exist locally, skips download (fast startup).
        If artifacts missing locally, downloads from S3.

        Args:
            local_models_dir: Optional local models directory. If None, uses config.PROJECT_ROOT.

        Returns:
            dict: {'model': 'local'|'s3'|'missing', 'encoder': 'local'|'s3'|'missing'}

        Example:
            >>> results = manager.sync_inference_artifacts()
            >>> if results['model'] == 'missing':
            ...     print("Model not found! Run: python run_pipeline.py")
        """
        if local_models_dir is None:
            local_models_dir = config.PROJECT_ROOT

        logger.info("Syncing inference artifacts from S3...")

        results = {}

        # Define artifact paths
        artifacts = [
            {'name': 'model', 's3_key': config.S3_MODEL_KEY, 'local_path': config.TUNED_MODEL_PATH},
            {'name': 'encoder', 's3_key': config.S3_ENCODER_KEY, 'local_path': config.FREQUENCY_ENCODER_PATH}
        ]

        for artifact in artifacts:
            name = artifact['name']
            s3_key = artifact['s3_key']
            local_path = artifact['local_path']

            if local_path.exists():
                logger.info(f"✓ {name.capitalize()} found locally: {local_path.name}")
                results[name] = 'local'
            else:
                logger.info(f"⚠️  {name.capitalize()} missing locally, attempting S3 download...")
                if self.is_active and self.download_file(s3_key, local_path):
                    logger.info(f"✓ {name.capitalize()} downloaded from S3")
                    results[name] = 's3'
                else:
                    logger.error(
                        f"❌ {name.capitalize()} not found locally or in S3!\n"
                        f"   To generate artifacts, run: python run_pipeline.py"
                    )
                    results[name] = 'missing'

        logger.info("✓ Inference artifact sync complete")
        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize manager (gracefully handles missing credentials)
    manager = S3ArtifactManager(
        bucket_name=config.S3_BUCKET_NAME,
        region=config.S3_REGION
    )

    # Check if S3 is available
    if manager.is_active:
        # Sync inference artifacts
        results = manager.sync_inference_artifacts()
        print(f"Sync results: {results}")
    else:
        print("S3 not available - running in local mode")
