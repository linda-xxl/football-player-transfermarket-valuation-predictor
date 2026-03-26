"""
player_profile.py — helper functions for the root app.py dashboard.

Provides data loading and player-specific query utilities used by the
CFFC 'Momentum' Engine Streamlit app.
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_minimal_data(data_dir: str = "data") -> dict:
    """Load the CSVs needed for the player profile section."""
    return {
        "players_df":           pd.read_csv(f"{data_dir}/players.csv"),
        "appearances_df":       pd.read_csv(f"{data_dir}/appearances.csv"),
        "games_df":             pd.read_csv(f"{data_dir}/games.csv"),
        "player_valuations_df": pd.read_csv(f"{data_dir}/player_valuations.csv"),
        "transfers_df":         pd.read_csv(f"{data_dir}/transfers.csv"),
    }


def get_player_profile(players_df: pd.DataFrame, player_id: int) -> dict:
    """Return a dict of profile fields for a given player_id, or None."""
    row = players_df[players_df["player_id"] == player_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_transfer_history(transfers_df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Return transfer rows for a player, sorted by date."""
    df = transfers_df[transfers_df["player_id"] == player_id].copy()
    if df.empty:
        return df
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    return df.sort_values("transfer_date").reset_index(drop=True)


def get_valuation_history(player_valuations_df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Return valuation rows for a player, sorted by date."""
    df = player_valuations_df[player_valuations_df["player_id"] == player_id].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def get_game_history(
    appearances_df: pd.DataFrame,
    games_df: pd.DataFrame,
    player_id: int,
) -> pd.DataFrame:
    """
    Return a per-game performance table for a player.

    Columns: date, opponent_club_name, is_home, team_goals, opp_goals,
             result, goals, assists, minutes_played
    """
    apps = appearances_df[appearances_df["player_id"] == player_id].copy()
    if apps.empty:
        return apps

    games = games_df.copy()

    # Merge appearances with game details
    merged = apps.merge(games, on="game_id", how="left")

    # Determine home/away and derive opponent / scores
    merged["is_home"] = merged["player_club_id"] == merged["home_club_id"]

    merged["team_goals"] = merged.apply(
        lambda r: r["home_club_goals"] if r["is_home"] else r["away_club_goals"], axis=1
    )
    merged["opp_goals"] = merged.apply(
        lambda r: r["away_club_goals"] if r["is_home"] else r["home_club_goals"], axis=1
    )
    merged["opponent_club_name"] = merged.apply(
        lambda r: r["away_club_name"] if r["is_home"] else r["home_club_name"], axis=1
    )

    merged["result"] = merged.apply(
        lambda r: "W" if r["team_goals"] > r["opp_goals"]
        else ("L" if r["team_goals"] < r["opp_goals"] else "D"),
        axis=1,
    )

    merged["date"] = pd.to_datetime(merged["date_x"] if "date_x" in merged.columns else merged["date"], errors="coerce")

    cols = ["date", "opponent_club_name", "is_home", "team_goals", "opp_goals",
            "result", "goals", "assists", "minutes_played"]
    return merged[cols].sort_values("date").reset_index(drop=True)


def plot_appearances_scatter(games: pd.DataFrame, ax=None):
    """
    Scatter plot of goals+assists per appearance, coloured by result.
    Green = Win, Red = Loss, Gray = Draw.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    color_map = {"W": "green", "L": "red", "D": "gray"}
    colors = games["result"].map(color_map).fillna("gray")

    ax.scatter(
        range(len(games)),
        games["goals"] + games["assists"],
        c=colors,
        alpha=0.7,
        edgecolors="none",
        s=40,
    )
    ax.set_xlabel("Appearance #")
    ax.set_ylabel("Goals + Assists")
    ax.set_title("Goals + Assists per Appearance")
    ax.grid(True, alpha=0.2)
