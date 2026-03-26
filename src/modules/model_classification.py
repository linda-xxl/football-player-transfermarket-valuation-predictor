"""
Classification model module — XGBoost classifiers for burst and rapid decline.

Public API
----------
prepare_features(dataset, target)   → X_train, X_test, y_train, y_test, feature_cols
train(X_train, y_train, label)      → fitted XGBClassifier
precision_at_k(y_true, y_proba, k)  → float
evaluate(model, X_test, y_test)     → dict of metrics
save(burst_model, decline_model)    → writes models to models/
load_burst() / load_decline()       → XGBClassifier
"""

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .config import (
    CAT_COLS,
    LEAKAGE_COLS,
    META_COLS,
    MODEL_BURST_PATH,
    MODEL_DECLINE_PATH,
    TARGET_VALUATION,
    XGB_BASE_PARAMS,
)
from .data_processing import season_period_to_rank

logger = logging.getLogger(__name__)


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_features(
    dataset: pd.DataFrame,
    target: str,
    test_season: str = "2022-2023_H1",
) -> Tuple:
    """
    Build the feature matrix and target vector, applying a temporal split.

    ``test_season`` is held out; everything before it is training data.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_cols
    """
    df = dataset.copy()

    # Other targets are also excluded (we're training on one at a time)
    other_targets = {"label_burst", "label_rapid_decline", TARGET_VALUATION} - {target}

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS + LEAKAGE_COLS + list(other_targets) + [target]
    ]

    X = df[feature_cols].copy()
    X = pd.get_dummies(X, columns=[c for c in CAT_COLS if c in X.columns], drop_first=False)
    feature_cols_encoded = X.columns.tolist()

    y = df[target].values

    train_mask = df["season_period"] < test_season
    test_mask  = df["season_period"] == test_season

    # Fall back to period_rank-based split when test season has no rows
    if test_mask.sum() == 0:
        logger.warning(
            "Test season '%s' not found; falling back to last period.", test_season
        )
        df["period_rank"] = df["season_period"].apply(season_period_to_rank)
        cutoff     = sorted(df["period_rank"].unique())[-1]
        train_mask = df["period_rank"] < cutoff
        test_mask  = df["period_rank"] >= cutoff

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask],  y[test_mask]

    logger.info(
        "%s — Train: %d rows, %d pos (%.1f%%) | Test: %d rows, %d pos (%.1f%%)",
        target,
        len(y_train), y_train.sum(), y_train.mean() * 100,
        len(y_test),  y_test.sum(),  y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, feature_cols_encoded


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    label: str = "burst",
    params: dict = None,
) -> XGBClassifier:
    """
    Train an XGBoost binary classifier with automatic class-weight balancing.

    Parameters
    ----------
    label : str
        ``'burst'`` or ``'decline'`` — used only for logging.
    params : dict, optional
        Override default hyperparameters from ``config.XGB_BASE_PARAMS``.
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    hp = {**XGB_BASE_PARAMS, "scale_pos_weight": scale_pos_weight, **(params or {})}
    model = XGBClassifier(**hp)
    model.fit(X_train, y_train)

    logger.info(
        "%s classifier trained — %d samples, scale_pos_weight=%.1f",
        label, len(y_train), scale_pos_weight,
    )
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Fraction of true positives among the top-k ranked predictions."""
    order  = np.argsort(-y_proba)
    top_k  = y_true[order[:k]]
    return float(top_k.mean()) if len(top_k) > 0 else 0.0


def evaluate(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label: str = "classifier",
    threshold: float = 0.2,
    k_list: List[int] = None,
) -> dict:
    """
    Evaluate a classification model.

    Returns a dict with ROC-AUC, PR-AUC, Precision@K for several K values,
    and a full classification report string.
    """
    if k_list is None:
        k_list = [10, 50, 100, 200]

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc":  float(roc_auc_score(y_test, y_proba)),
        "pr_auc":   float(average_precision_score(y_test, y_proba)),
        "threshold": threshold,
        "precision_at_k": {
            k: precision_at_k(y_test, y_proba, k) for k in k_list
        },
        "classification_report": classification_report(
            y_test, y_pred, digits=4, zero_division=0
        ),
    }
    logger.info(
        "%s — ROC-AUC=%.4f  PR-AUC=%.4f  P@10=%.4f  P@50=%.4f",
        label,
        metrics["roc_auc"], metrics["pr_auc"],
        metrics["precision_at_k"].get(10, float("nan")),
        metrics["precision_at_k"].get(50, float("nan")),
    )
    return metrics


# ── Persistence ────────────────────────────────────────────────────────────────

def save_burst(model: XGBClassifier, path: Path = MODEL_BURST_PATH) -> None:
    """Persist the burst classifier to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved burst classifier → %s", path)


def save_decline(model: XGBClassifier, path: Path = MODEL_DECLINE_PATH) -> None:
    """Persist the decline classifier to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved decline classifier → %s", path)


def load_burst(path: Path = MODEL_BURST_PATH) -> XGBClassifier:
    """Load the persisted burst classifier."""
    model = joblib.load(path)
    logger.info("Loaded burst classifier from %s", path)
    return model


def load_decline(path: Path = MODEL_DECLINE_PATH) -> XGBClassifier:
    """Load the persisted decline classifier."""
    model = joblib.load(path)
    logger.info("Loaded decline classifier from %s", path)
    return model
