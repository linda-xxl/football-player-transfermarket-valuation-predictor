"""
Predictor module — inference utilities consumed by the Streamlit app.

The ``FootballPredictor`` class loads pre-trained models and the processed
dataset once, then exposes three clean prediction methods:

    find_players(query)                   → fuzzy player search results
    predict_valuation(player_id)          → next-period market value
    get_hidden_gems(season, positions)    → burst-probability ranked table
    get_rapid_declines(season, positions) → decline-probability ranked table
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as fuzz_process

from .config import (
    CAT_COLS,
    DATA_DIR,
    DATASET_PATH,
    FEATURE_COLUMNS_PATH,
    LEAKAGE_COLS,
    META_COLS,
    MODEL_BURST_PATH,
    MODEL_DECLINE_PATH,
    MODEL_VALUATION_PATH,
    TARGET_VALUATION,
)
from . import model_valuation as mv
from . import model_classification as mc

logger = logging.getLogger(__name__)

_EUR_CAP = 2_500_000   # burst label definition: current value ≤ median (~€2.5M)
_VETERAN_AGE = 30       # decline label definition: age ≥ 30


class FootballPredictor:
    """
    Wraps all inference operations for the Streamlit app.

    Load once on startup; all heavy computation (model loading, data joining)
    is done in ``__init__`` and cached in instance attributes.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``dataset.csv`` and ``players.csv``.
    models_dir : Path
        Directory containing saved ``.pkl`` model artifacts.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        models_dir: Path = None,
    ) -> None:
        self._data_dir   = Path(data_dir)
        self._models_dir = Path(models_dir) if models_dir else MODEL_VALUATION_PATH.parent

        self.lgb_model:    object = None
        self.burst_clf:    object = None
        self.decline_clf:  object = None
        self.feature_cols: List[str] = []

        self.dataset:        pd.DataFrame = None
        self.players:        pd.DataFrame = None
        self.latest_records: pd.DataFrame = None

        self._load_data()
        self._load_models()
        self._build_latest_records()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load the processed dataset and player name table."""
        dataset_path = DATASET_PATH
        players_path = self._data_dir / "players.csv"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"dataset.csv not found at {dataset_path}. "
                "Run `python src/pipeline.py` first to process the data."
            )
        self.dataset = pd.read_csv(dataset_path)
        logger.info("Loaded dataset: %s rows", len(self.dataset))

        if players_path.exists():
            # Load only name/club — avoid column conflicts (players.csv also has 'position')
            self.players = pd.read_csv(players_path)[
                ["player_id", "name", "current_club_name", "country_of_birth"]
            ]
            logger.info("Loaded players: %s rows", len(self.players))
        else:
            logger.warning("players.csv not found; player names will be unavailable.")
            self.players = pd.DataFrame(columns=["player_id", "name", "current_club_name"])

    def _load_models(self) -> None:
        """Load all three saved model artifacts from disk."""
        missing = [
            p for p in [MODEL_VALUATION_PATH, MODEL_BURST_PATH, MODEL_DECLINE_PATH, FEATURE_COLUMNS_PATH]
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Model artifacts not found: {missing}. "
                "Run `python src/pipeline.py` first to train and save the models."
            )
        self.lgb_model, self.feature_cols = mv.load(MODEL_VALUATION_PATH, FEATURE_COLUMNS_PATH)
        self.burst_clf   = mc.load_burst(MODEL_BURST_PATH)
        self.decline_clf = mc.load_decline(MODEL_DECLINE_PATH)
        logger.info("All models loaded successfully.")

    def _build_latest_records(self) -> None:
        """
        Build a table of each player's most recent season record, enriched
        with name and club from ``players.csv``.
        """
        if "period_rank" not in self.dataset.columns:
            from .data_processing import season_period_to_rank
            self.dataset["period_rank"] = self.dataset["season_period"].apply(season_period_to_rank)

        latest = (
            self.dataset.sort_values("period_rank")
            .groupby("player_id")
            .last()
            .reset_index()
        )
        self.latest_records = latest.merge(self.players, on="player_id", how="left")
        logger.info("Latest records built: %d players", len(self.latest_records))

    def _build_X(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Build the one-hot encoded feature matrix from a subset of records,
        aligning columns to the training schema.
        """
        feature_raw_cols = [
            c for c in self.dataset.columns
            if c not in META_COLS + LEAKAGE_COLS + [TARGET_VALUATION]
        ]
        X = rows[feature_raw_cols].copy()
        X = pd.get_dummies(X, columns=[c for c in CAT_COLS if c in X.columns], drop_first=False)
        X = X.reindex(columns=self.feature_cols, fill_value=0)
        return X

    # ── Public API ─────────────────────────────────────────────────────────────

    def find_players(self, query: str, top_n: int = 8) -> pd.DataFrame:
        """
        Fuzzy-match ``query`` against all player names and return the top
        matches as a DataFrame.

        Columns: player_id, name, club, position, latest_season,
                 current_value_eur, match_score
        """
        all_names = self.latest_records["name"].dropna().unique().tolist()
        matches   = fuzz_process.extract(query, all_names, scorer=fuzz.WRatio, limit=top_n)

        rows = []
        for name, score, _ in matches:
            rec = self.latest_records[self.latest_records["name"] == name].iloc[0]
            rows.append({
                "player_id":         rec["player_id"],
                "name":              rec["name"],
                "club":              rec.get("current_club_name", ""),
                "position":          rec.get("position", ""),
                "latest_season":     rec["season_period"],
                "current_value_eur": rec["period_end_valuation"],
                "match_score":       score,
            })
        return pd.DataFrame(rows)

    def predict_valuation(self, player_id: int) -> dict:
        """
        Predict the next-period market value for a player using their most
        recent available record.

        Returns
        -------
        dict with keys:
            name, club, position, latest_season,
            current_value_eur, predicted_value_eur, pct_change
        """
        row = self.latest_records[self.latest_records["player_id"] == player_id]
        if row.empty:
            raise ValueError(f"player_id {player_id} not found in dataset.")

        X_input  = self._build_X(row)
        pred_log = float(self.lgb_model.predict(X_input)[0])
        pred_eur = float(np.expm1(pred_log))
        curr_eur = float(row["period_end_valuation"].values[0])
        pct      = (pred_eur - curr_eur) / curr_eur * 100 if curr_eur > 0 else float("nan")

        return {
            "name":                row["name"].values[0],
            "club":                row.get("current_club_name", pd.Series([""]) ).values[0],
            "position":            row["position"].values[0] if "position" in row.columns else "",
            "latest_season":       row["season_period"].values[0],
            "current_value_eur":   curr_eur,
            "predicted_value_eur": pred_eur,
            "pct_change":          pct,
        }

    def get_hidden_gems(
        self,
        season_period: str,
        positions: Optional[List[str]] = None,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """
        Return the top-``top_k`` players most likely to burst in value,
        scored using data from ``season_period``.

        Parameters
        ----------
        season_period : str
            Season half to use as model input (e.g. ``'2021-2022_H1'``).
        positions : list of str, optional
            If provided, filter results to these positions.
        top_k : int
            Maximum number of rows to return.
        """
        rows = self.dataset[self.dataset["season_period"] == season_period].copy()
        if rows.empty:
            return pd.DataFrame()

        X          = self._build_X(rows)
        burst_prob = self.burst_clf.predict_proba(X)[:, 1]

        out = rows[["player_id", "season_period", "period_end_valuation", "age",
                    "position", "domestic_competition_id"]].copy()
        out["burst_probability"] = burst_prob
        out = out.merge(self.players, on="player_id", how="left")

        # Filter: undervalued players only (consistent with label definition)
        out = out[out["period_end_valuation"] <= _EUR_CAP]

        if positions:
            out = out[out["position"].isin(positions)]

        out = out.sort_values("burst_probability", ascending=False).head(top_k)
        out = out.reset_index(drop=True)
        out.index += 1
        return out

    def get_rapid_declines(
        self,
        season_period: str,
        positions: Optional[List[str]] = None,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """
        Return the top-``top_k`` players most at risk of rapid valuation
        decline, scored using data from ``season_period``.

        Parameters
        ----------
        season_period : str
            Season half to use as model input (e.g. ``'2021-2022_H1'``).
        positions : list of str, optional
            If provided, filter results to these positions.
        top_k : int
            Maximum number of rows to return.
        """
        rows = self.dataset[self.dataset["season_period"] == season_period].copy()
        if rows.empty:
            return pd.DataFrame()

        X            = self._build_X(rows)
        decline_prob = self.decline_clf.predict_proba(X)[:, 1]

        out = rows[["player_id", "season_period", "period_end_valuation", "age",
                    "position", "domestic_competition_id"]].copy()
        out["decline_probability"] = decline_prob
        out = out.merge(self.players, on="player_id", how="left")

        # Filter: veterans only (consistent with label definition)
        out = out[out["age"] >= _VETERAN_AGE]

        if positions:
            out = out[out["position"].isin(positions)]

        out = out.sort_values("decline_probability", ascending=False).head(top_k)
        out = out.reset_index(drop=True)
        out.index += 1
        return out

    def predict_classification(self, player_id: int) -> dict:
        """
        Return burst and decline probabilities for a player's latest record.

        Returns
        -------
        dict with keys: burst_probability, decline_probability
        """
        row = self.latest_records[self.latest_records["player_id"] == player_id]
        if row.empty:
            raise ValueError(f"player_id {player_id} not found in dataset.")

        X_input = self._build_X(row)
        burst_prob   = float(self.burst_clf.predict_proba(X_input)[:, 1][0])
        decline_prob = float(self.decline_clf.predict_proba(X_input)[:, 1][0])
        return {"burst_probability": burst_prob, "decline_probability": decline_prob}

    def get_player_prediction_history(self, player_id: int) -> pd.DataFrame:
        """
        Return all season records for a player with actual and model-predicted
        next-period valuations.

        Columns: season_period, period_end_valuation, [next_valuation], predicted_valuation
        """
        rows = self.dataset[self.dataset["player_id"] == player_id].copy()
        if rows.empty:
            return pd.DataFrame()

        rows = rows.sort_values("period_rank")
        X_all     = self._build_X(rows)
        pred_log  = self.lgb_model.predict(X_all)
        pred_eur  = np.expm1(pred_log)
        rows["predicted_valuation"] = pred_eur

        cols = ["season_period", "period_end_valuation"]
        if TARGET_VALUATION in rows.columns:
            cols.append(TARGET_VALUATION)
        cols.append("predicted_valuation")
        return rows[cols].reset_index(drop=True)

    def get_latest_season(self) -> str:
        """Return the most recent season period in the dataset."""
        seasons = self.get_available_seasons()
        return seasons[-1] if seasons else ""

    def get_all_players_list(self) -> pd.DataFrame:
        """
        Return a DataFrame of all players with their latest-season info,
        suitable for populating a searchable dropdown.

        Columns: player_id, name, current_club_name, position,
                 season_period, period_end_valuation
        """
        cols = ["player_id", "name", "current_club_name", "position",
                "season_period", "period_end_valuation"]
        available = [c for c in cols if c in self.latest_records.columns]
        df = self.latest_records[available].copy()
        df = df.dropna(subset=["name"])
        return df.sort_values("name").reset_index(drop=True)

    def get_available_seasons(self) -> List[str]:
        """Return all season periods present in the dataset, sorted chronologically."""
        from .data_processing import season_period_to_rank
        seasons = self.dataset["season_period"].unique().tolist()
        return sorted(seasons, key=season_period_to_rank)

    def get_positions(self) -> List[str]:
        """Return all player position values present in the dataset."""
        return sorted(self.dataset["position"].dropna().unique().tolist())
