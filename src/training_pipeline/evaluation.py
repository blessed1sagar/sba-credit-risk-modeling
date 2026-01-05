"""
Model evaluation module for SBA Loan default prediction.

Provides comprehensive evaluation metrics including:
- Standard ML metrics (ROC-AUC, Precision, Recall, F1, Average Precision)
- Credit risk metrics (KS Statistic, Decile Analysis)
- Feature importance analysis
- MLflow logging capabilities
"""
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, accuracy_score
)
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model and return comprehensive metrics dictionary.

    Metrics calculated:
    - ROC-AUC: Area Under ROC Curve
    - Precision: True Positives / (True Positives + False Positives)
    - Recall: True Positives / (True Positives + False Negatives)
    - F1 Score: Harmonic mean of Precision and Recall
    - Average Precision: Area under Precision-Recall curve
    - Accuracy: Correct predictions / Total predictions
    - KS Statistic: Kolmogorov-Smirnov statistic (credit risk metric)

    Args:
        model: Trained model with predict() and predict_proba() methods.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Dictionary of metric names and values.

    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        ROC-AUC: 0.8463
    """
    logger.info("Evaluating model...")

    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate standard metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'average_precision': average_precision_score(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred)
    }

    # Calculate KS statistic (credit risk specific)
    ks_stat, _, _ = calculate_ks_statistic(y_test.values, y_prob)
    metrics['ks_statistic'] = ks_stat

    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric:20s}: {value:.4f}")

    return metrics


def calculate_ks_statistic(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[float, int, pd.DataFrame]:
    """
    Calculate Kolmogorov-Smirnov (KS) statistic for credit risk models.

    The KS statistic measures the maximum separation between cumulative
    distributions of good and bad loans. Higher values indicate better
    discrimination between classes.

    Interpretation:
    - KS > 0.4: Excellent discrimination
    - KS > 0.3: Good discrimination
    - KS > 0.2: Acceptable discrimination
    - KS < 0.2: Poor discrimination

    Args:
        y_true: True labels (0 = good, 1 = bad).
        y_prob: Predicted probabilities.

    Returns:
        Tuple of (ks_statistic, ks_index, ks_dataframe):
        - ks_statistic: Maximum KS value
        - ks_index: Index where max KS occurs
        - ks_dataframe: Full KS calculation DataFrame

    Example:
        >>> ks_stat, ks_idx, df_ks = calculate_ks_statistic(y_test, y_prob)
        >>> print(f"KS Statistic: {ks_stat:.4f}")
        KS Statistic: 0.5311
    """
    # Create DataFrame with actuals and probabilities
    df_ks = pd.DataFrame({
        'actual': y_true,
        'prob': y_prob
    }).sort_values('prob', ascending=False).reset_index(drop=True)

    # Calculate good and bad counts
    df_ks['bad'] = df_ks['actual']
    df_ks['good'] = 1 - df_ks['actual']

    # Calculate cumulative percentages
    df_ks['cum_bad'] = df_ks['bad'].cumsum() / df_ks['bad'].sum()
    df_ks['cum_good'] = df_ks['good'].cumsum() / df_ks['good'].sum()

    # KS is the maximum absolute difference
    df_ks['ks'] = np.abs(df_ks['cum_bad'] - df_ks['cum_good'])

    ks_stat = df_ks['ks'].max()
    ks_index = df_ks['ks'].idxmax()

    logger.info(f"KS Statistic: {ks_stat:.4f}")
    logger.info(f"KS occurs at index: {ks_index} (prob = {df_ks.loc[ks_index, 'prob']:.4f})")

    return ks_stat, ks_index, df_ks


def create_decile_table(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> pd.DataFrame:
    """
    Create decile analysis table for credit risk assessment.

    Divides predictions into 10 equal-sized groups (deciles) and calculates
    performance metrics for each group. This helps understand model performance
    across different risk segments.

    Args:
        y_true: True labels (0 = good, 1 = bad).
        y_prob: Predicted probabilities.

    Returns:
        DataFrame with decile analysis metrics.

    Columns:
        - decile: Decile number (10 = highest risk, 1 = lowest risk)
        - count: Number of loans in decile
        - n_bad: Number of bad loans (defaults)
        - n_good: Number of good loans
        - bad_rate: Default rate (%)
        - min_score, max_score, avg_score: Score range in decile
        - cum_bad, cum_good: Cumulative counts
        - cum_bad_pct, cum_good_pct: Cumulative percentages
        - ks: KS statistic at this decile

    Example:
        >>> decile_df = create_decile_table(y_test, y_prob)
        >>> print(decile_df[['decile', 'count', 'bad_rate', 'ks']].head())
    """
    df_decile = pd.DataFrame({
        'actual': y_true,
        'prob': y_prob
    })

    # Create deciles (10 equal-sized groups)
    df_decile['decile'] = pd.qcut(
        df_decile['prob'],
        q=10,
        labels=False,
        duplicates='drop'
    ) + 1

    # Calculate metrics per decile
    decile_stats = df_decile.groupby('decile').agg(
        count=('actual', 'count'),
        n_bad=('actual', 'sum'),
        min_score=('prob', 'min'),
        max_score=('prob', 'max'),
        avg_score=('prob', 'mean')
    ).reset_index()

    # Calculate additional metrics
    decile_stats['n_good'] = decile_stats['count'] - decile_stats['n_bad']
    decile_stats['bad_rate'] = (
        decile_stats['n_bad'] / decile_stats['count'] * 100
    ).round(2)

    # Cumulative metrics
    total_bad = decile_stats['n_bad'].sum()
    total_good = decile_stats['n_good'].sum()

    decile_stats['cum_bad'] = decile_stats['n_bad'].cumsum()
    decile_stats['cum_good'] = decile_stats['n_good'].cumsum()
    decile_stats['cum_bad_pct'] = (decile_stats['cum_bad'] / total_bad * 100).round(2)
    decile_stats['cum_good_pct'] = (decile_stats['cum_good'] / total_good * 100).round(2)

    # KS per decile
    decile_stats['ks'] = (
        decile_stats['cum_bad_pct'] - decile_stats['cum_good_pct']
    ).abs().round(2)

    # Sort by decile descending (highest risk first)
    decile_stats = decile_stats.sort_values('decile', ascending=False).reset_index(drop=True)

    logger.info(f"Decile table created with {len(decile_stats)} deciles")

    return decile_stats


def get_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank feature importances from model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with features and their importance scores, sorted descending.

    Example:
        >>> importance_df = get_feature_importance(model, X_train.columns, top_n=20)
        >>> print(importance_df.head())
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n).reset_index(drop=True)

    logger.info(f"Top {top_n} feature importances extracted")

    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Top 20 Feature Importances",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        title: Plot title.
        figsize: Figure size (width, height).
        save_path: If provided, save plot to this path.

    Example:
        >>> plot_feature_importance(importance_df, save_path='feature_importance.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(importance_df)))

    ax.barh(
        range(len(importance_df)),
        importance_df['importance'].values,
        color=colors
    )
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {save_path}")

    plt.close()


def log_metrics_to_mlflow(
    metrics: Dict[str, float],
    decile_table: pd.DataFrame,
    importance_df: pd.DataFrame,
    run_name: Optional[str] = None
) -> None:
    """
    Log evaluation metrics and artifacts to MLflow.

    Args:
        metrics: Dictionary of metric names and values.
        decile_table: Decile analysis DataFrame.
        importance_df: Feature importance DataFrame.
        run_name: Optional name for MLflow run.

    Note:
        Requires mlflow to be installed and configured.

    Example:
        >>> log_metrics_to_mlflow(metrics, decile_df, importance_df, "baseline_run")
    """
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name):
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log decile table as artifact
            decile_csv_path = "decile_analysis.csv"
            decile_table.to_csv(decile_csv_path, index=False)
            mlflow.log_artifact(decile_csv_path)

            # Log feature importance plot
            importance_plot_path = "feature_importance.png"
            plot_feature_importance(importance_df, save_path=importance_plot_path)
            mlflow.log_artifact(importance_plot_path)

            # Log top features as parameters
            top_5_features = importance_df.head(5)['feature'].tolist()
            mlflow.log_param("top_5_features", top_5_features)

            logger.info("✓ Metrics and artifacts logged to MLflow")

    except ImportError:
        logger.warning("MLflow not available. Skipping MLflow logging.")
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Evaluation module loaded. Use functions to evaluate models.")
