"""
Hyperparameter tuning module using Optuna for SBA Loan default prediction.

Uses Bayesian optimization to find optimal XGBoost hyperparameters,
maximizing ROC-AUC score with MLflow tracking.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from typing import Tuple, Optional
import optuna
from optuna.samplers import TPESampler

from src import config
from src.training_pipeline.train_baseline import (
    load_processed_data,
    split_train_test,
    calculate_scale_pos_weight
)

logger = logging.getLogger(__name__)


def create_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    cv_folds: int = None
):
    """
    Create Optuna objective function for hyperparameter optimization.

    The objective function performs cross-validation and returns the mean ROC-AUC.

    Args:
        X_train: Training features.
        y_train: Training target.
        scale_pos_weight: Weight for positive class.
        cv_folds: Number of CV folds. If None, uses config.CV_FOLDS.

    Returns:
        Objective function for Optuna.

    Example:
        >>> objective = create_objective(X_train, y_train, scale_pos_weight)
        >>> trial = study.ask()
        >>> score = objective(trial)
    """
    if cv_folds is None:
        cv_folds = config.CV_FOLDS

    def objective(trial):
        """Optuna objective function."""
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
        }

        # Create model with suggested parameters
        model = XGBClassifier(
            **params,
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

        return scores.mean()

    return objective


def run_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_trials: int = None,
    timeout: Optional[int] = None
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization study.

    Args:
        X_train: Training features.
        y_train: Training target.
        scale_pos_weight: Weight for positive class.
        n_trials: Number of trials. If None, uses config.OPTUNA_N_TRIALS.
        timeout: Time limit in seconds. If None, uses config.OPTUNA_TIMEOUT.

    Returns:
        Completed Optuna study object.

    Example:
        >>> study = run_optuna_study(X_train, y_train, scale_pos_weight)
        >>> print(f"Best ROC-AUC: {study.best_value:.4f}")
        Best ROC-AUC: 0.8207
    """
    if n_trials is None:
        n_trials = config.OPTUNA_N_TRIALS

    logger.info("=" * 80)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Objective: Maximize ROC-AUC")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"CV folds: {config.CV_FOLDS}")

    # Create objective function
    objective = create_objective(X_train, y_train, scale_pos_weight)

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config.RANDOM_STATE)
    )

    # Run optimization
    logger.info("\nRunning optimization...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    logger.info("\n✓ Optimization complete")
    logger.info(f"Best ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters:")
    for param, value in study.best_params.items():
        logger.info(f"  {param:20s}: {value}")

    return study


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    scale_pos_weight: float
) -> XGBClassifier:
    """
    Train final model with best hyperparameters from Optuna.

    Args:
        X_train: Training features.
        y_train: Training target.
        best_params: Best parameters from Optuna study.
        scale_pos_weight: Weight for positive class.

    Returns:
        Trained XGBoost classifier with best parameters.

    Example:
        >>> model = train_best_model(X_train, y_train, study.best_params, scale_pos_weight)
    """
    logger.info("Training final model with best parameters...")

    model = XGBClassifier(
        **best_params,
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train)

    logger.info("✓ Best model trained successfully")

    return model


def save_model(model: XGBClassifier, model_path: str = None) -> None:
    """
    Save tuned model to disk.

    Args:
        model: Trained XGBoost model.
        model_path: Path to save model. If None, uses config.TUNED_MODEL_PATH.
    """
    if model_path is None:
        model_path = config.TUNED_MODEL_PATH

    # Ensure models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    logger.info(f"✓ Tuned model saved to: {model_path}")


def log_to_mlflow(
    study: optuna.Study,
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Log Optuna trials and best model to MLflow.

    Args:
        study: Completed Optuna study.
        model: Trained best model.
        X_test: Test features.
        y_test: Test target.

    Note:
        Requires mlflow to be installed.
    """
    try:
        import mlflow
        import mlflow.xgboost
        from src.training_pipeline.evaluation import (
            evaluate_model,
            create_decile_table,
            get_feature_importance
        )

        mlflow.set_tracking_uri(str(config.MLFLOW_TRACKING_URI))

        with mlflow.start_run(run_name="optuna_best_model"):
            # Log best parameters
            for param, value in study.best_params.items():
                mlflow.log_param(param, value)

            # Log best trial score
            mlflow.log_metric("cv_roc_auc", study.best_value)

            # Evaluate on test set
            metrics = evaluate_model(model, X_test, y_test)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            # Create and log decile table
            y_prob = model.predict_proba(X_test)[:, 1]
            decile_table = create_decile_table(y_test.values, y_prob)
            decile_table.to_csv("decile_analysis.csv", index=False)
            mlflow.log_artifact("decile_analysis.csv")

            # Log feature importance
            importance_df = get_feature_importance(model, X_test.columns, top_n=20)
            top_5_features = importance_df.head(5)['feature'].tolist()
            mlflow.log_param("top_5_features", top_5_features)

            # Log model
            mlflow.xgboost.log_model(model, "model")

            logger.info("✓ Metrics and model logged to MLflow")

    except ImportError:
        logger.warning("MLflow not available. Skipping MLflow logging.")
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")


def run_hyperparameter_tuning(
    n_trials: int = None,
    save_model_flag: bool = True,
    log_mlflow: bool = True
) -> XGBClassifier:
    """
    Execute complete hyperparameter tuning pipeline.

    Pipeline:
    1. Load processed data
    2. Split into train/test sets
    3. Calculate scale_pos_weight
    4. Run Optuna optimization
    5. Train best model
    6. Log to MLflow
    7. Save model

    Args:
        n_trials: Number of Optuna trials. If None, uses config.OPTUNA_N_TRIALS.
        save_model_flag: If True, save tuned model to disk.
        log_mlflow: If True, log results to MLflow.

    Returns:
        Trained XGBoost classifier with best hyperparameters.

    Example:
        >>> model = run_hyperparameter_tuning(n_trials=50)
    """
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING PIPELINE")
    logger.info("=" * 80)

    # Step 1: Load data
    logger.info("\n[1/7] Loading processed data...")
    X, y = load_processed_data()

    # Step 2: Train/test split
    logger.info("\n[2/7] Splitting train/test...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step 3: Calculate scale_pos_weight
    logger.info("\n[3/7] Calculating class weights...")
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    # Step 4: Run Optuna
    logger.info("\n[4/7] Running Optuna optimization...")
    study = run_optuna_study(X_train, y_train, scale_pos_weight, n_trials=n_trials)

    # Step 5: Train best model
    logger.info("\n[5/7] Training best model...")
    model = train_best_model(X_train, y_train, study.best_params, scale_pos_weight)

    # Step 6: Log to MLflow
    if log_mlflow:
        logger.info("\n[6/7] Logging to MLflow...")
        log_to_mlflow(study, model, X_test, y_test)
    else:
        logger.info("\n[6/7] Skipping MLflow logging")

    # Step 7: Save model
    if save_model_flag:
        logger.info("\n[7/7] Saving model...")
        save_model(model)
    else:
        logger.info("\n[7/7] Skipping model save")

    logger.info("\n" + "=" * 80)
    logger.info("✓ HYPERPARAMETER TUNING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best ROC-AUC (CV): {study.best_value:.4f}")
    logger.info(f"Number of trials completed: {len(study.trials)}")

    if save_model_flag:
        logger.info(f"Model saved to: {config.TUNED_MODEL_PATH}")

    return model


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tuning
    run_hyperparameter_tuning()
