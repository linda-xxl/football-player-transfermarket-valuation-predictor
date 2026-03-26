"""
Central configuration: all paths, constants, feature definitions, and
model hyperparameters. Import from here rather than hardcoding values elsewhere.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
# Resolves to …/Football-Player-Valuation-Prediction/
ROOT = Path(__file__).resolve().parents[2]

# ── Directory paths ────────────────────────────────────────────────────────────
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR    = ROOT / "models"

# ── Processed / model artifact paths ──────────────────────────────────────────
DATASET_PATH         = DATA_DIR / "dataset.csv"
X_FEATURES_PATH      = PROCESSED_DIR / "X_features.csv"
Y_TARGET_PATH        = PROCESSED_DIR / "y_target.csv"

PREDICTED_VALUATIONS_PATH = PROCESSED_DIR / "predicted_valuations.csv"

MODEL_VALUATION_PATH  = MODELS_DIR / "lgb_valuation.pkl"
MODEL_BURST_PATH      = MODELS_DIR / "xgb_burst.pkl"
MODEL_DECLINE_PATH    = MODELS_DIR / "xgb_decline.pkl"
FEATURE_COLUMNS_PATH  = MODELS_DIR / "feature_columns.pkl"

# ── Kaggle dataset identifier ──────────────────────────────────────────────────
KAGGLE_DATASET = "davidcariboo/player-scores"

# ── Column groups shared across training and inference ─────────────────────────
META_COLS = [
    "player_id",
    "season_period",
    "domestic_competition_id",
    "player_club_id",
    "period_rank",
]

# Columns that leak future information — always excluded from model input
LEAKAGE_COLS = [
    "valuation_2y_ahead",
    "valuation_log_change_2y",
    "label_burst",
    "label_rapid_decline",
    "date_of_birth",
    "points_earned",   # redundant with win_rate; kept in dataset for reference
]

CAT_COLS = ["position", "foot", "career_stage"]

TARGET_VALUATION = "next_valuation"
TARGET_BURST     = "label_burst"
TARGET_DECLINE   = "label_rapid_decline"

# ── Top-5 European leagues ─────────────────────────────────────────────────────
TOP_5_LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1"]

# ── Feature engineering constants ─────────────────────────────────────────────
PEAK_AGE    = 27       # consensus age of peak valuation
MIN_MINUTES = 90       # minimum minutes to compute per-90 stats

# ── Target label thresholds ────────────────────────────────────────────────────
PERIODS_AHEAD              = 4     # 4 half-seasons ≈ 2 years
BURST_PERCENTILE           = 0.95  # top 5% valuation growth
BURST_VALUE_CAP_PERCENTILE = 0.50  # only flag players currently below median value
DECLINE_PERCENTILE         = 0.05  # bottom 5% valuation change
DECLINE_MIN_AGE            = 30    # only flag veterans

# ── LightGBM hyperparameters (next-period valuation) ──────────────────────────
LGB_PARAMS = dict(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# ── XGBoost hyperparameters (burst / decline classifiers) ─────────────────────
XGB_BASE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=0,
    n_jobs=-1,
)
