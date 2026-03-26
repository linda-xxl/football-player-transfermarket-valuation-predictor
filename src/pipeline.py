"""
End-to-end training pipeline.

Usage
-----
# Full pipeline (download → process → train → save)
python src/pipeline.py

# Skip download (data already present)
python src/pipeline.py --skip-download

# Skip both download and processing (dataset.csv already built)
python src/pipeline.py --skip-download --skip-processing

The pipeline saves three model artifacts to ``models/``:
    lgb_valuation.pkl   — LightGBM next-period valuation regressor
    xgb_burst.pkl       — XGBoost burst (hidden gems) classifier
    xgb_decline.pkl     — XGBoost rapid decline classifier
    feature_columns.pkl — Encoded feature column list for inference alignment
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src importable when called as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.modules import data_download, data_processing
from src.modules import model_valuation as mv
from src.modules import model_classification as mc
from src.modules.config import (
    DATA_DIR,
    DATASET_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_VALUATION_PATH,
    MODELS_DIR,
    PREDICTED_VALUATIONS_PATH,
    PROCESSED_DIR,
    TARGET_BURST,
    TARGET_DECLINE,
    TARGET_VALUATION,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Step helpers ───────────────────────────────────────────────────────────────

def step_download() -> None:
    logger.info("── Step 1: Download data from Kaggle ──────────────────────────")
    data_download.download(target_dir=DATA_DIR)


def step_process() -> pd.DataFrame:
    logger.info("── Step 2: Data processing pipeline ───────────────────────────")
    dataset = data_processing.run_pipeline(
        data_dir=DATA_DIR,
        processed_dir=PROCESSED_DIR,
    )
    logger.info("Processing complete: %d rows, %d columns", *dataset.shape)
    return dataset


def step_train_valuation(dataset: pd.DataFrame) -> None:
    logger.info("── Step 3: Train LightGBM valuation model ─────────────────────")
    X_train, X_test, y_train_log, y_test_log, y_test_raw, feature_cols = (
        mv.prepare_features(dataset)
    )
    model = mv.train(X_train, y_train_log)
    metrics = mv.evaluate(model, X_test, y_test_log, y_test_raw)

    logger.info(
        "Valuation model results:\n"
        "  R² (log)  = %.4f\n"
        "  R² (EUR)  = %.4f\n"
        "  MAE (EUR) = €%.0f\n"
        "  MAPE      = %.2f%%",
        metrics["r2_log"], metrics["r2_eur"],
        metrics["mae_eur"], metrics["mape_pct"],
    )

    # Retrain on ALL data for production (maximise coverage for demo predictions)
    logger.info("Retraining on full dataset for deployment …")
    X_all = pd.get_dummies(
        dataset[[c for c in dataset.columns
                 if c not in mv.META_COLS + mv.LEAKAGE_COLS + [TARGET_VALUATION]]],
        columns=[c for c in mv.CAT_COLS if c in dataset.columns],
        drop_first=False,
    )
    # Re-align to the same columns used in evaluation
    X_all = X_all.reindex(columns=feature_cols, fill_value=0)
    y_all_log = np.log1p(dataset[TARGET_VALUATION].values)

    production_model = mv.train(X_all, y_all_log)
    mv.save(production_model, feature_cols)
    logger.info("Valuation model saved.")


def step_train_classifiers(dataset: pd.DataFrame) -> None:
    logger.info("── Step 4: Train XGBoost burst classifier ──────────────────────")
    X_train_b, X_test_b, y_train_b, y_test_b, feat_cols_b = mc.prepare_features(
        dataset, target=TARGET_BURST
    )
    burst_model = mc.train(X_train_b, y_train_b, label="burst")
    burst_metrics = mc.evaluate(burst_model, X_test_b, y_test_b, label="burst")
    logger.info(
        "Burst model — ROC-AUC=%.4f  PR-AUC=%.4f  P@10=%.3f",
        burst_metrics["roc_auc"], burst_metrics["pr_auc"],
        burst_metrics["precision_at_k"].get(10, float("nan")),
    )

    logger.info("── Step 5: Train XGBoost decline classifier ────────────────────")
    X_train_d, X_test_d, y_train_d, y_test_d, feat_cols_d = mc.prepare_features(
        dataset, target=TARGET_DECLINE
    )
    decline_model = mc.train(X_train_d, y_train_d, label="decline")
    decline_metrics = mc.evaluate(decline_model, X_test_d, y_test_d, label="decline")
    logger.info(
        "Decline model — ROC-AUC=%.4f  PR-AUC=%.4f  P@10=%.3f",
        decline_metrics["roc_auc"], decline_metrics["pr_auc"],
        decline_metrics["precision_at_k"].get(10, float("nan")),
    )

    mc.save_burst(burst_model)
    mc.save_decline(decline_model)
    logger.info("Classifier models saved.")


def step_export_predictions(dataset: pd.DataFrame) -> None:
    logger.info("── Step 6: Export batch predictions ───────────────────────────")
    model, feature_cols = mv.load(MODEL_VALUATION_PATH, FEATURE_COLUMNS_PATH)

    X_all = pd.get_dummies(
        dataset[[c for c in dataset.columns
                 if c not in mv.META_COLS + mv.LEAKAGE_COLS + [TARGET_VALUATION]]],
        columns=[c for c in mv.CAT_COLS if c in dataset.columns],
        drop_first=False,
    )
    X_all = X_all.reindex(columns=feature_cols, fill_value=0)

    pred_log = model.predict(X_all)
    pred_eur = np.expm1(pred_log)

    out = dataset.copy()
    out["predicted_valuation"] = pred_eur
    out["pred_valuation_change"] = out["predicted_valuation"] - out["period_end_valuation"]
    out["actual_valuation_change"] = out[TARGET_VALUATION] - out["period_end_valuation"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTED_VALUATIONS_PATH, index=False)
    logger.info("Predictions saved: %d rows → %s", len(out), PREDICTED_VALUATIONS_PATH)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(skip_download: bool = False, skip_processing: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download
    if not skip_download:
        step_download()
    else:
        logger.info("Skipping download (--skip-download).")

    # 2. Process
    if not skip_processing:
        dataset = step_process()
    else:
        logger.info("Skipping processing (--skip-processing); loading %s …", DATASET_PATH)
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"{DATASET_PATH} not found. Remove --skip-processing to regenerate it."
            )
        dataset = pd.read_csv(DATASET_PATH)
        logger.info("Loaded dataset: %d rows, %d columns", *dataset.shape)

    # 3. Train valuation model
    step_train_valuation(dataset)

    # 4 & 5. Train classifiers
    step_train_classifiers(dataset)

    # 6. Export batch predictions
    step_export_predictions(dataset)

    logger.info("=== Pipeline complete. Models saved to %s ===", MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Football Player Valuation — full training pipeline"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download (use existing data/ files).",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip data processing (use existing data/dataset.csv).",
    )
    args = parser.parse_args()
    run(skip_download=args.skip_download, skip_processing=args.skip_processing)
