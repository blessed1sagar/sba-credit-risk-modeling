"""
Baseline model training module for SBA Loan default prediction.

Trains a baseline XGBoost classifier with class weight handling for imbalanced data.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from typing import Tuple

from src import config

logger = logging.getLogger(__name__)


def load_processed_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data and split into features (X) and target (y).

    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix DataFrame
        - y: Target series (0 = PIF/Good, 1 = CHGOFF/Default)

    Raises:
        FileNotFoundError: If processed data file doesn't exist.
        KeyError: If LoanStatus column is missing.
    """
    logger.info(f"Loading processed data from: {config.PROCESSED_DATA_PATH}")

    try:
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Processed data file not found: {config.PROCESSED_DATA_PATH}")
        raise

    if config.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{config.TARGET_COLUMN}' not found in data")

    # Separate features and target
    y = df[config.TARGET_COLUMN].map(config.TARGET_MAPPING)
    X = df.drop(columns=[config.TARGET_COLUMN])

    logger.info(f"Data loaded: X shape = {X.shape}, y shape = {y.shape}")
    logger.info(
        f"Class distribution: "
        f"Good (0) = {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%), "
        f"Default (1) = {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)"
    )

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = None,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train/test split.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data for test set. If None, uses config.TEST_SIZE.
        random_state: Random seed. If None, uses config.RANDOM_STATE.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Example:
        >>> X_train, X_test, y_train, y_test = split_train_test(X, y)
        >>> print(X_train.shape)
        (44664, 74)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )

    logger.info(f"Train/test split complete:")
    logger.info(f"  Train: {X_train.shape[0]:,} samples")
    logger.info(f"  Test: {X_test.shape[0]:,} samples")
    logger.info(
        f"  Train class distribution: "
        f"Good = {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%), "
        f"Default = {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)"
    )

    return X_train, X_test, y_train, y_test


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost to handle class imbalance.

    Formula: negative_count / positive_count
    This gives more weight to the minority class (defaults).

    Args:
        y_train: Training target vector.

    Returns:
        Scale weight for positive class.

    Example:
        >>> scale_weight = calculate_scale_pos_weight(y_train)
        >>> print(f"scale_pos_weight: {scale_weight:.2f}")
        scale_pos_weight: 12.41
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    scale_pos_weight = n_negative / n_positive

    logger.info(
        f"Class imbalance - Negative: {n_negative:,}, Positive: {n_positive:,}"
    )
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    return scale_pos_weight


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float
) -> XGBClassifier:
    """
    Train baseline XGBoost classifier.

    Uses default hyperparameters from config with class weight handling.

    Args:
        X_train: Training features.
        y_train: Training target.
        scale_pos_weight: Weight for positive class (from calculate_scale_pos_weight).

    Returns:
        Trained XGBoost classifier.

    Example:
        >>> model = train_baseline_model(X_train, y_train, scale_pos_weight)
        >>> print(type(model))
        <class 'xgboost.sklearn.XGBClassifier'>
    """
    logger.info("Training baseline XGBoost model...")

    model = XGBClassifier(
        **config.BASELINE_XGB_PARAMS,
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)

    logger.info("✓ Baseline model trained successfully")

    return model


def save_model(model: XGBClassifier, model_path: str = None) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained XGBoost model.
        model_path: Path to save model. If None, uses config.BASELINE_MODEL_PATH.

    Example:
        >>> save_model(model)
        ✓ Model saved to: models/xgb_baseline.joblib
    """
    if model_path is None:
        model_path = config.BASELINE_MODEL_PATH

    # Ensure models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    logger.info(f"✓ Model saved to: {model_path}")


def run_baseline_training(save_model_flag: bool = True) -> XGBClassifier:
    """
    Execute complete baseline training pipeline.

    Pipeline:
    1. Load processed data
    2. Split into train/test sets
    3. Calculate scale_pos_weight
    4. Train baseline XGBoost model
    5. Save model to disk

    Args:
        save_model_flag: If True, save trained model to disk.

    Returns:
        Trained XGBoost classifier.

    Example:
        >>> model = run_baseline_training()
        >>> print(f"Model type: {type(model)}")
    """
    logger.info("=" * 80)
    logger.info("BASELINE MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    # Step 1: Load data
    logger.info("\n[1/5] Loading processed data...")
    X, y = load_processed_data()

    # Step 2: Train/test split
    logger.info("\n[2/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step 3: Calculate scale_pos_weight
    logger.info("\n[3/5] Calculating class weights...")
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    # Step 4: Train model
    logger.info("\n[4/5] Training baseline model...")
    model = train_baseline_model(X_train, y_train, scale_pos_weight)

    # Step 5: Save model
    if save_model_flag:
        logger.info("\n[5/5] Saving model...")
        save_model(model)
    else:
        logger.info("\n[5/5] Skipping model save (save_model_flag=False)")

    logger.info("\n" + "=" * 80)
    logger.info("✓ BASELINE TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: XGBoost Classifier")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Training samples: {X_train.shape[0]:,}")
    logger.info(f"Test samples: {X_test.shape[0]:,}")
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    if save_model_flag:
        logger.info(f"Model saved to: {config.BASELINE_MODEL_PATH}")

    return model


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run training
    run_baseline_training()
