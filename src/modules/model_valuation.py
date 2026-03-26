"""
Valuation model module — LightGBM regressor for next-period market value.

Public API
----------
prepare_features(dataset)   → X_train, X_test, y_train_log, y_test_log, y_test_raw, feature_cols
train(X_train, y_train_log) → fitted LGBMRegressor
evaluate(model, ...)        → dict of metrics
save(model, feature_cols)   → writes model + column list to models/
load()                      → (LGBMRegressor, feature_cols)
"""

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    CAT_COLS,
    FEATURE_COLUMNS_PATH,
    LEAKAGE_COLS,
    LGB_PARAMS,
    META_COLS,
    MODEL_VALUATION_PATH,
    MODELS_DIR,
    TARGET_VALUATION,
)
from .data_processing import season_period_to_rank

logger = logging.getLogger(__name__)


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_features(
    dataset: pd.DataFrame,
    test_n_periods: int = 2,
) -> Tuple:
    """
    Build the feature matrix and targets from the processed dataset, applying
    a temporal train/test split (hold out the last ``test_n_periods`` periods).

    Returns
    -------
    X_train, X_test, y_train_log, y_test_log, y_test_raw, feature_cols
    """
    df = dataset.copy()

    # Ensure temporal ordering
    if "period_rank" not in df.columns:
        df["period_rank"] = df["season_period"].apply(season_period_to_rank)

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS + LEAKAGE_COLS + [TARGET_VALUATION]
    ]

    X = df[feature_cols].copy()
    X = pd.get_dummies(X, columns=[c for c in CAT_COLS if c in X.columns], drop_first=False)
    feature_cols_encoded = X.columns.tolist()

    y_raw = df[TARGET_VALUATION].values

    # Temporal split
    all_ranks = sorted(df["period_rank"].unique())
    cutoff     = all_ranks[-test_n_periods]

    train_mask = df["period_rank"] < cutoff
    test_mask  = df["period_rank"] >= cutoff

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_log = np.log1p(y_raw[train_mask])
    y_test_log  = np.log1p(y_raw[test_mask])
    y_test_raw  = y_raw[test_mask]

    logger.info(
        "Train: %d rows (%s → %s) | Test: %d rows",
        train_mask.sum(),
        df[train_mask]["season_period"].min(),
        df[train_mask]["season_period"].max(),
        test_mask.sum(),
    )
    return X_train, X_test, y_train_log, y_test_log, y_test_raw, feature_cols_encoded


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    params: dict = None,
) -> LGBMRegressor:
    """
    Train a LightGBM regressor on log-transformed targets.

    Parameters
    ----------
    params : dict, optional
        Override default hyperparameters from ``config.LGB_PARAMS``.
    """
    hp = {**LGB_PARAMS, **(params or {})}
    model = LGBMRegressor(**hp)
    model.fit(X_train, y_train_log)
    logger.info("LightGBM trained on %d samples, %d features", *X_train.shape)
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    model: LGBMRegressor,
    X_test: pd.DataFrame,
    y_test_log: np.ndarray,
    y_test_raw: np.ndarray,
) -> dict:
    """
    Evaluate a trained valuation model and return a metrics dictionary.

    Metrics are reported on both log scale and original EUR scale.
    """
    y_pred_log = model.predict(X_test)
    y_pred_raw = np.maximum(np.expm1(y_pred_log), 0)

    mask = y_test_raw != 0
    mape = float(np.mean(np.abs((y_test_raw[mask] - y_pred_raw[mask]) / y_test_raw[mask])) * 100)

    metrics = {
        # Log scale
        "rmse_log": float(np.sqrt(mean_squared_error(y_test_log, y_pred_log))),
        "mae_log":  float(mean_absolute_error(y_test_log, y_pred_log)),
        "r2_log":   float(r2_score(y_test_log, y_pred_log)),
        # EUR scale
        "rmse_eur":    float(np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))),
        "mae_eur":     float(mean_absolute_error(y_test_raw, y_pred_raw)),
        "median_ae":   float(np.median(np.abs(y_test_raw - y_pred_raw))),
        "r2_eur":      float(r2_score(y_test_raw, y_pred_raw)),
        "mape_pct":    mape,
    }
    logger.info(
        "Valuation model — R²(log)=%.4f  R²(EUR)=%.4f  MAPE=%.2f%%",
        metrics["r2_log"], metrics["r2_eur"], metrics["mape_pct"],
    )
    return metrics


# ── Persistence ────────────────────────────────────────────────────────────────

def save(
    model: LGBMRegressor,
    feature_cols: List[str],
    model_path: Path = MODEL_VALUATION_PATH,
    columns_path: Path = FEATURE_COLUMNS_PATH,
) -> None:
    """Persist the trained model and feature column list to disk."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, columns_path)
    logger.info("Saved valuation model → %s", model_path)
    logger.info("Saved feature columns → %s", columns_path)


def load(
    model_path: Path = MODEL_VALUATION_PATH,
    columns_path: Path = FEATURE_COLUMNS_PATH,
) -> Tuple[LGBMRegressor, List[str]]:
    """Load a persisted valuation model and its feature column list."""
    model       = joblib.load(model_path)
    feature_cols = joblib.load(columns_path)
    logger.info("Loaded valuation model from %s", model_path)
    return model, feature_cols
