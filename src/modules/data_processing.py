"""
Data processing module.

Reproduces the full pipeline from Notebook 1 (EDA) as clean, importable
functions:

    load_raw_data       → dict of raw DataFrames
    merge_and_clean     → single merged DataFrame at match-level
    aggregate_by_period → one row per (player, season_period)
    attach_valuations   → merge in market valuations and next-period target
    create_targets      → add label_burst and label_rapid_decline
    engineer_features   → derived stats, interaction terms, availability metrics
    add_lag_features    → lag-1/lag-2/rolling-2 history for key columns
    run_pipeline        → orchestrates all steps; saves dataset.csv + processed CSVs
"""

import logging
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    DATASET_PATH,
    DECLINE_MIN_AGE,
    DECLINE_PERCENTILE,
    BURST_PERCENTILE,
    BURST_VALUE_CAP_PERCENTILE,
    MIN_MINUTES,
    PEAK_AGE,
    PERIODS_AHEAD,
    PROCESSED_DIR,
    TOP_5_LEAGUES,
    X_FEATURES_PATH,
    Y_TARGET_PATH,
)

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _assign_season_half(date: pd.Timestamp) -> str:
    """Map a date to its season-half label (e.g. ``'2021-2022_H1'``)."""
    year, month = date.year, date.month
    if 7 <= month <= 12:
        return f"{year}-{year + 1}_H1"
    else:
        return f"{year - 1}-{year}_H2"


def season_period_to_rank(sp: str) -> int:
    """
    Convert a season-period string to a sortable integer.

    H1 (Jul–Dec) comes first within a season, H2 (Jan–Jun) comes second.
    E.g. ``'2021-2022_H1'`` < ``'2021-2022_H2'`` < ``'2022-2023_H1'``.
    """
    season, half = sp.rsplit("_", 1)
    start_year = int(season.split("-")[0])
    half_rank = 0 if half == "H1" else 1
    return start_year * 2 + half_rank


def _parse_transfer_record(value) -> float:
    """Convert strings like ``'+€3.05m'`` or ``'€-25.26m'`` to signed integers."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if "+-0" in value or value == "0":
        return 0.0
    is_negative = "-" in value
    numbers = re.findall(r"[\d.]+", value)
    if not numbers:
        return np.nan
    amount = float(numbers[0])
    multiplier = 1_000_000 if "m" in value.lower() else (1_000 if "k" in value.lower() else 1)
    result = float(amount * multiplier)
    return -result if is_negative else result


# ── Step 1: Load raw CSVs ──────────────────────────────────────────────────────

def load_raw_data(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from ``data_dir`` into a dictionary keyed by filename stem.

    Returns
    -------
    dict
        Keys: ``'appearances'``, ``'games'``, ``'players'``, ``'clubs'``,
        ``'player_valuations'``, ``'game_lineups'``, ``'game_events'``,
        ``'transfers'``, ``'club_games'``, ``'competitions'``
    """
    data_dir = Path(data_dir)
    raw: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        key = csv_path.stem
        raw[key] = pd.read_csv(csv_path, encoding="UTF-8")
        logger.info("Loaded %-30s %s", key, raw[key].shape)
    return raw


# ── Step 2–6: Merge and clean ──────────────────────────────────────────────────

def merge_and_clean(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Select relevant columns, merge all tables at match level, handle missing
    data, parse aggregate scores, and compute per-game match results.

    Parameters
    ----------
    raw : dict
        Output of :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Match-level DataFrame with one row per (player, game).
    """
    # ── Select columns ─────────────────────────────────────────────────────────
    appearances = raw["appearances"].drop(
        columns=["appearance_id", "player_current_club_id", "player_name"],
        errors="ignore",
    )
    games = raw["games"].drop(
        columns=["url", "referee", "stadium", "home_club_name", "away_club_name"],
        errors="ignore",
    )
    players = raw["players"].drop(
        columns=[
            "first_name", "last_name", "country_of_birth", "city_of_birth",
            "country_of_citizenship", "contract_expiration_date",
            "image_url", "url", "current_club_name",
            "current_club_domestic_competition_id", "last_season",
        ],
        errors="ignore",
    )
    clubs = raw["clubs"].drop(
        columns=["coach_name", "filename", "url", "stadium_name", "last_season",
                 "total_market_value"],
        errors="ignore",
    )

    # ── Merge 1: appearances × games ──────────────────────────────────────────
    player_games = pd.merge(appearances, games, on="game_id", how="inner",
                            suffixes=("", "_game"))
    dup_cols = [c for c in player_games.columns if c.endswith("_game")]
    player_games = player_games.drop(columns=dup_cols)

    # ── Merge 2: × players ────────────────────────────────────────────────────
    player_data = pd.merge(player_games, players, on="player_id", how="inner")

    # ── Merge 3: × clubs (for squad context) ──────────────────────────────────
    player_club_data = pd.merge(
        player_data, clubs,
        left_on="player_club_id", right_on="club_id", how="left",
    )
    player_club_data = player_club_data.rename(columns={
        "name_x": "player_name",
        "name_y": "club_name",
    })

    # ── Flag and handle missing data ───────────────────────────────────────────
    player_club_data["date_of_birth_missing"] = (
        player_club_data["date_of_birth"].isna().astype(int)
    )

    # Drop structurally missing / leaking columns
    drop_cols = [
        "home_club_position", "away_club_position",
        "agent_name",
        "market_value_in_eur", "highest_market_value_in_eur",
        "player_code",
    ]
    player_club_data = player_club_data.drop(columns=drop_cols, errors="ignore")

    # Drop rows missing critical identifiers or match scores
    player_club_data = player_club_data.dropna(
        subset=["home_club_id", "away_club_id", "home_club_goals", "away_club_goals"],
    )
    player_club_data = player_club_data.dropna(
        subset=["home_club_manager_name", "away_club_manager_name"],
    )

    # Attendance: cascading group median fill then global median
    if "attendance" in player_club_data.columns:
        for group_cols in [["player_club_id", "season"], ["player_club_id"], ["season"]]:
            valid_groups = [c for c in group_cols if c in player_club_data.columns]
            if valid_groups:
                player_club_data["attendance"] = player_club_data["attendance"].fillna(
                    player_club_data.groupby(valid_groups)["attendance"].transform("median")
                )
        player_club_data["attendance"] = player_club_data["attendance"].fillna(
            player_club_data["attendance"].median()
        )

    # Mode / median imputation for remaining columns
    impute_rules = {
        "home_club_formation":   "mode",
        "away_club_formation":   "mode",
        "date_of_birth":         "mode",
        "sub_position":          "mode",
        "club_average_age":      "median",
        "foreigners_percentage": "median",
    }
    for col, strategy in impute_rules.items():
        if col not in player_club_data.columns:
            continue
        if strategy == "mode":
            fill_val = player_club_data[col].mode()
            if not fill_val.empty:
                player_club_data[col] = player_club_data[col].fillna(fill_val[0])
        else:
            player_club_data[col] = player_club_data[col].fillna(
                player_club_data[col].median()
            )

    # ── Parse aggregate score ──────────────────────────────────────────────────
    if "aggregate" in player_club_data.columns:
        try:
            scores = player_club_data["aggregate"].str.split(":", expand=True).astype(float)
            player_club_data[["home_score", "away_score"]] = scores.values
        except Exception:
            # Fall back to goals columns when aggregate is unparseable
            player_club_data["home_score"] = player_club_data.get("home_club_goals", 0)
            player_club_data["away_score"] = player_club_data.get("away_club_goals", 0)
        player_club_data = player_club_data.drop(columns=["aggregate"], errors="ignore")

    # ── Per-game match result from player's perspective ────────────────────────
    # Encoding: win=1, draw=2, loss=0  (sum across period captures result quality)
    if "home_score" in player_club_data.columns:
        is_home  = player_club_data["player_club_id"] == player_club_data["home_club_id"]
        is_away  = player_club_data["player_club_id"] == player_club_data["away_club_id"]
        home_win = player_club_data["home_score"] > player_club_data["away_score"]
        away_win = player_club_data["away_score"] > player_club_data["home_score"]
        draw     = player_club_data["home_score"] == player_club_data["away_score"]

        player_club_data["result"] = np.select(
            [draw, (is_home & home_win) | (is_away & away_win)],
            [2, 1],
            default=0,
        )

    # ── Compute age from date_of_birth and game date ───────────────────────────
    if "date_of_birth" in player_club_data.columns and "date" in player_club_data.columns:
        player_club_data["date"]         = pd.to_datetime(player_club_data["date"],         errors="coerce")
        player_club_data["date_of_birth"] = pd.to_datetime(player_club_data["date_of_birth"], errors="coerce")
        player_club_data["age"] = (
            (player_club_data["date"] - player_club_data["date_of_birth"])
            .dt.days / 365.25
        )

    # ── Assign season-half labels ──────────────────────────────────────────────
    player_club_data["date"] = pd.to_datetime(player_club_data["date"], errors="coerce")
    player_club_data = player_club_data.dropna(subset=["date"])
    player_club_data["season_period"] = player_club_data["date"].apply(_assign_season_half)

    logger.info("merge_and_clean: %s rows", len(player_club_data))
    return player_club_data


# ── Step 8: Aggregate to season-period level ───────────────────────────────────

def aggregate_by_period(player_club_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match-level data to one row per (player_id, season_period).

    Numeric stats are summed; player and club attributes take the last
    observed value within the period.
    """
    agg_dict = {
        # Performance totals
        "goals":              ("goals",              "sum"),
        "assists":            ("assists",             "sum"),
        "yellow_cards":       ("yellow_cards",        "sum"),
        "red_cards":          ("red_cards",           "sum"),
        "minutes_played":     ("minutes_played",      "sum"),
        "games_played":       ("game_id",             "count"),
        "result":             ("result",              "sum"),
        # Player attributes (last value in period)
        "position":           ("position",            "last"),
        "foot":               ("foot",                "last"),
        "height_in_cm":       ("height_in_cm",        "last"),
        "date_of_birth":      ("date_of_birth",       "last"),
        "age":                ("age",                 "last"),
        # Club / competition context
        "player_club_id":          ("player_club_id",          "last"),
        "domestic_competition_id": ("domestic_competition_id", "last"),
        "home_club_goals":         ("home_club_goals",          "mean"),
        "away_club_goals":         ("away_club_goals",          "mean"),
    }

    # Optional club squad context (column names vary across dataset versions)
    optional = {
        "squad_size":        ("squad_size",        "last"),
        "average_age":       ("average_age",        "last"),
        "foreigners_number": ("foreigners_number",  "last"),
        # Some versions use club_ prefix
        "club_squad_size":        ("club_squad_size",        "last"),
        "club_average_age":       ("club_average_age",       "last"),
        "club_foreigners_number": ("club_foreigners_number", "last"),
    }
    for col, spec in optional.items():
        if spec[0] in player_club_data.columns:
            agg_dict[col] = spec

    perf = (
        player_club_data
        .groupby(["player_id", "season_period"])
        .agg(**agg_dict)
        .reset_index()
    )

    # Normalise squad-context column names (drop prefixed duplicates)
    rename_map = {
        "club_squad_size":        "squad_size",
        "club_average_age":       "average_age",
        "club_foreigners_number": "foreigners_number",
    }
    perf = perf.rename(columns={k: v for k, v in rename_map.items() if k in perf.columns})
    # If both exist after rename, keep only the non-prefixed version
    for col in ["squad_size", "average_age", "foreigners_number"]:
        dupes = [c for c in perf.columns if c == col]
        if len(dupes) > 1:
            perf = perf.loc[:, ~perf.columns.duplicated()]

    logger.info("aggregate_by_period: %s rows", len(perf))
    return perf


# ── Step 10–11: Attach market valuations ──────────────────────────────────────

def attach_valuations(perf: pd.DataFrame, player_valuations: pd.DataFrame) -> pd.DataFrame:
    """
    Merge period-end market valuations into the aggregated performance table
    and compute the next-period valuation target.
    """
    pv = player_valuations.copy()
    pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
    pv = pv.dropna(subset=["date"])
    pv["season_period"] = pv["date"].apply(_assign_season_half)

    valuations_by_period = (
        pv.sort_values(["player_id", "season_period", "date"])
        .groupby(["player_id", "season_period"])
        .agg(period_end_valuation=("market_value_in_eur", "last"))
        .reset_index()
    )

    # Next-period valuation (shifted within each player's timeline)
    valuations_by_period = valuations_by_period.sort_values(["player_id", "season_period"])
    valuations_by_period["next_valuation"] = (
        valuations_by_period.groupby("player_id")["period_end_valuation"].shift(-1)
    )

    dataset = pd.merge(
        perf,
        valuations_by_period[["player_id", "season_period", "period_end_valuation", "next_valuation"]],
        on=["player_id", "season_period"],
        how="inner",
    )
    dataset = dataset.dropna(subset=["next_valuation"])
    dataset = dataset.sort_values(["player_id", "season_period"]).reset_index(drop=True)

    logger.info("attach_valuations: %s rows after inner join", len(dataset))
    return dataset


# ── Step 12: Create binary target labels ──────────────────────────────────────

def create_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Append ``label_burst`` and ``label_rapid_decline`` binary columns.

    Burst   — top-5% 2-year valuation growth AND current valuation <= median
    Decline — bottom-5% 2-year valuation change AND player age >= 30
    """
    df = dataset.copy()
    df = df.sort_values(["player_id", "season_period"]).reset_index(drop=True)

    # 2-year-ahead valuation
    df["valuation_2y_ahead"] = (
        df.groupby("player_id")["period_end_valuation"].shift(-PERIODS_AHEAD)
    )

    # Log-change (only valid when both values are positive)
    valid = (df["period_end_valuation"] > 0) & (df["valuation_2y_ahead"] > 0)
    df["valuation_log_change_2y"] = np.where(
        valid,
        np.log(df["valuation_2y_ahead"]) - np.log(df["period_end_valuation"]),
        np.nan,
    )
    df = df.dropna(subset=["valuation_log_change_2y"]).reset_index(drop=True)

    burst_threshold   = df["valuation_log_change_2y"].quantile(BURST_PERCENTILE)
    decline_threshold = df["valuation_log_change_2y"].quantile(DECLINE_PERCENTILE)
    value_cap         = df["period_end_valuation"].quantile(BURST_VALUE_CAP_PERCENTILE)

    df["label_burst"] = (
        (df["valuation_log_change_2y"] >= burst_threshold) &
        (df["period_end_valuation"] <= value_cap)
    ).astype(int)

    df["label_rapid_decline"] = (
        (df["valuation_log_change_2y"] <= decline_threshold) &
        (df["age"] >= DECLINE_MIN_AGE)
    ).astype(int)

    logger.info(
        "create_targets: burst=%d (%.1f%%), decline=%d (%.1f%%)",
        df["label_burst"].sum(), df["label_burst"].mean() * 100,
        df["label_rapid_decline"].sum(), df["label_rapid_decline"].mean() * 100,
    )
    return df


# ── Step 9 + 13–18: Derived / engineered features ─────────────────────────────

def engineer_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features used by the models:
    per-game rates, per-90 stats, momentum, availability, team-relative
    contribution, career stage, top-5 league flag, and win rate.
    """
    df = dataset.copy()
    gp = df["games_played"].replace(0, np.nan)   # safe denominator

    # ── Per-game ──────────────────────────────────────────────────────────────
    df["goals_per_game"]              = df["goals"]      / gp
    df["assists_per_game"]            = df["assists"]     / gp
    df["minutes_per_game"]            = df["minutes_played"] / gp
    df["goal_contributions"]          = df["goals"] + df["assists"]
    df["goal_contributions_per_game"] = df["goal_contributions"] / gp
    df["cards_per_game"]              = (df["yellow_cards"] + df["red_cards"]) / gp

    # ── Per-90 (zero-fill when < MIN_MINUTES played) ─────────────────────────
    safe_min = df["minutes_played"].where(df["minutes_played"] >= MIN_MINUTES, other=np.nan)
    df["goals_per_90"]   = (df["goals"]   / safe_min * 90).fillna(0)
    df["assists_per_90"] = (df["assists"] / safe_min * 90).fillna(0)

    # ── Momentum (change vs previous period) ─────────────────────────────────
    for col in ["goals", "assists", "minutes_played", "games_played"]:
        prev = df.groupby("player_id")[col].shift(1)
        df[f"{col}_momentum"] = df[col] - prev.fillna(0)

    # ── Age / peak interactions ────────────────────────────────────────────────
    df["years_to_peak"]      = PEAK_AGE - df["age"]
    df["age_x_goals_per_90"] = df["age"] * df["goals_per_90"]

    df["career_stage"] = pd.cut(
        df["age"],
        bins=[0, 23, 29, 100],
        labels=["emerging", "peak", "veteran"],
        right=False,
    ).astype(str)

    # ── Availability & starter status ─────────────────────────────────────────
    df["availability_rate"] = (
        df["minutes_played"] / (df["games_played"] * 90)
    ).clip(0, 1)
    df["is_regular_starter"] = (df["minutes_per_game"] >= 70).astype(int)

    # ── Team context ───────────────────────────────────────────────────────────
    df["team_avg_goals"] = (df["home_club_goals"] + df["away_club_goals"]) / 2

    team_output = (df["team_avg_goals"] * df["games_played"]).replace(0, np.nan)
    df["goal_share"]         = (df["goals"]             / team_output).fillna(0)
    df["contribution_share"] = (df["goal_contributions"] / team_output).fillna(0)

    # ── League tier ───────────────────────────────────────────────────────────
    df["is_top5_league"] = df["domestic_competition_id"].isin(TOP_5_LEAGUES).astype(int)

    # ── Win rate ──────────────────────────────────────────────────────────────
    # result is sum of per-game codes (win=1, draw=2, loss=0)
    df["win_rate"] = (df["result"] / (df["games_played"] * 2)).clip(0, 1)

    # ── Points earned (football: win=3, draw=1; included for dataset compat) ──
    # Derived from win_rate and games: not recoverable exactly, so we proxy
    df["points_earned"] = df["win_rate"] * df["games_played"] * 2

    logger.info("engineer_features: done (%d columns)", len(df.columns))
    return df


# ── Step 19: Lag and rolling features ─────────────────────────────────────────

def add_lag_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag-1, lag-2, and rolling-2-period mean features for key performance
    and valuation columns.  All NaN values (first periods) are filled with 0.
    """
    df = dataset.copy()
    df["period_rank"] = df["season_period"].apply(season_period_to_rank)
    df = df.sort_values(["player_id", "period_rank"]).reset_index(drop=True)

    lag_cols = [
        "period_end_valuation",
        "goals", "assists", "minutes_played", "games_played",
        "goals_per_90", "assists_per_90",
        "goal_contributions_per_game", "win_rate",
    ]

    for col in lag_cols:
        if col not in df.columns:
            continue
        grp = df.groupby("player_id")[col]
        df[f"{col}_lag1"] = grp.shift(1)
        df[f"{col}_lag2"] = grp.shift(2)
        df[f"{col}_rolling2_mean"] = grp.transform(
            lambda x: x.shift(1).rolling(2, min_periods=1).mean()
        )

    # Valuation-specific trend features
    df["valuation_pct_change"] = (
        df.groupby("player_id")["period_end_valuation"].pct_change()
    )
    df["valuation_pct_change_lag1"] = (
        df.groupby("player_id")["valuation_pct_change"].shift(1)
    )
    df["valuation_trend"] = df["period_end_valuation_lag1"] - df["period_end_valuation_lag2"]
    df["goals_trend"]     = df.get("goals_lag1", 0) - df.get("goals_lag2", 0)

    # Fill NaN from first-period lags
    lag_like_cols = [
        c for c in df.columns
        if any(s in c for s in ["_lag1", "_lag2", "_rolling2", "_trend", "_pct_change"])
    ]
    df[lag_like_cols] = df[lag_like_cols].fillna(0)

    logger.info("add_lag_features: added %d lag columns", len(lag_like_cols))
    return df


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: Path = DATA_DIR,
    output_dir: Path = None,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Run the full data processing pipeline end-to-end.

    Steps
    -----
    1. Load raw CSVs
    2. Merge and clean at match level
    3. Aggregate to (player, season_period) rows
    4. Attach market valuations and next-period target
    5. Create binary target labels
    6. Engineer derived features
    7. Add lag / rolling features
    8. Save ``dataset.csv`` and processed CSVs

    Returns
    -------
    pd.DataFrame
        The final feature-engineered dataset.
    """
    data_dir      = Path(data_dir)
    processed_dir = Path(processed_dir)
    output_path   = Path(output_dir) / "dataset.csv" if output_dir else DATASET_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Data processing pipeline START ===")

    # 1. Load
    raw = load_raw_data(data_dir)

    # 2. Merge & clean
    player_club_data = merge_and_clean(raw)

    # 3. Aggregate
    perf = aggregate_by_period(player_club_data)

    # 4. Attach valuations
    dataset = attach_valuations(perf, raw["player_valuations"])

    # 5. Create targets
    dataset = create_targets(dataset)

    # 6. Engineer features
    dataset = engineer_features(dataset)

    # 7. Lag features
    dataset = add_lag_features(dataset)

    # 8. Save outputs
    dataset.to_csv(output_path, index=False)
    logger.info("Saved dataset → %s  (%s rows, %s cols)", output_path, *dataset.shape)

    # Save X / y splits for classification models (backward compat with Notebook 3)
    _LEAKAGE = [
        "valuation_2y_ahead", "valuation_log_change_2y",
        "label_burst", "label_rapid_decline",
        "date_of_birth", "points_earned",
    ]
    _META = ["player_id", "season_period", "domestic_competition_id",
             "player_club_id", "period_rank"]
    _TARGET = "next_valuation"

    feature_cols = [c for c in dataset.columns if c not in _META + _LEAKAGE + [_TARGET]]
    X = dataset[["player_id", "season_period"] + feature_cols]
    y = dataset[["label_burst", "label_rapid_decline", _TARGET]]

    X.to_csv(X_FEATURES_PATH, index=False)
    y.to_csv(Y_TARGET_PATH, index=False)
    logger.info("Saved X_features → %s", X_FEATURES_PATH)
    logger.info("Saved y_target   → %s", Y_TARGET_PATH)
    logger.info("=== Data processing pipeline END ===")

    return dataset
