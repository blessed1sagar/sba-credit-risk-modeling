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
    AWS S3 artifact synchronization manager.

    Handles smart sync of ML artifacts (models, encoders) between local storage
    and S3. Downloads artifacts only if missing locally.

    Attributes:
        bucket_name: S3 bucket name.
        region: AWS region.
        s3_client: boto3 S3 client.

    Example:
        >>> manager = S3ArtifactManager('sba-loan-ml-artifacts')
        >>> manager.sync_inference_artifacts(Path('./models'))
    """

    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        """
        Initialize S3ArtifactManager.

        Args:
            bucket_name: S3 bucket name.
            region: AWS region (default: us-east-1).

        Raises:
            NoCredentialsError: If AWS credentials not configured.
        """
        self.bucket_name = bucket_name
        self.region = region

        try:
            self.s3_client = boto3.client('s3', region_name=region)
            logger.info(f"✓ S3 client initialized (bucket: {bucket_name}, region: {region})")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Configure with 'aws configure' or environment variables.")
            raise

    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        """
        Upload file to S3.

        Args:
            local_path: Local file path to upload.
            s3_key: S3 object key (path in bucket).

        Returns:
            True if upload successful, False otherwise.

        Example:
            >>> manager.upload_file(Path('models/xgb_tuned.joblib'), 'models/xgb_tuned.joblib')
            True
        """
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
            >>> manager.download_file('models/xgb_tuned.joblib', Path('models/xgb_tuned.joblib'))
            True
        """
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
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def sync_inference_artifacts(self, local_models_dir: Optional[Path] = None) -> None:
        """
        Smart sync of inference artifacts from S3.

        Downloads artifacts only if missing locally:
        - models/xgb_tuned.joblib
        - data/feature/frequency_encoder.pkl

        If artifacts exist locally, skips download (fast startup).
        If artifacts missing locally, downloads from S3.

        Args:
            local_models_dir: Optional local models directory. If None, uses config.PROJECT_ROOT.

        Example:
            >>> manager.sync_inference_artifacts()
            ✓ Model found locally: models/xgb_tuned.joblib (skipping download)
            ✓ Downloaded s3://.../artifacts/frequency_encoder.pkl → frequency_encoder.pkl
        """
        if local_models_dir is None:
            local_models_dir = config.PROJECT_ROOT

        logger.info("Syncing inference artifacts from S3...")

        # Define artifact paths
        artifacts = [
            {
                's3_key': config.S3_MODEL_KEY,
                'local_path': config.TUNED_MODEL_PATH
            },
            {
                's3_key': config.S3_ENCODER_KEY,
                'local_path': config.FREQUENCY_ENCODER_PATH
            }
        ]

        for artifact in artifacts:
            s3_key = artifact['s3_key']
            local_path = artifact['local_path']

            if local_path.exists():
                logger.info(f"✓ Artifact found locally: {local_path.name} (skipping download)")
            else:
                logger.info(f"Artifact missing locally: {local_path.name}, downloading from S3...")
                success = self.download_file(s3_key, local_path)

                if not success:
                    logger.warning(
                        f"Failed to download {s3_key}. "
                        f"Ensure artifact exists in S3 or provide local path."
                    )

        logger.info("✓ Inference artifact sync complete")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize manager
    manager = S3ArtifactManager(
        bucket_name='sba-loan-ml-artifacts',
        region='us-east-1'
    )

    # Sync inference artifacts
    manager.sync_inference_artifacts()
