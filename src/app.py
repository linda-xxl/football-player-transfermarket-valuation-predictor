"""
Streamlit app — Football Player Valuation & Scouting Tool.

Run with:
    streamlit run src/app.py

Tabs
----
1. Player Prediction   — search / select a player, predict next-season value
                         plus hidden-gem and rapid-decline probabilities.
2. Scouting Hub        — shows latest available season, top-K hidden gems
                         and decline watch list with optional position filter.
3. Player Profile      — detailed profile, actual-vs-predicted valuation
                         history, market value timeline, game stats, transfers.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.modules.predictor import FootballPredictor
from src.player_profile import (
    get_game_history,
    get_player_profile,
    get_transfer_history,
    get_valuation_history,
    plot_appearances_scatter,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Valuation & Scouting",
    page_icon="⚽",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_eur(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"€{value / 1_000:.0f}K"
    return f"€{value:.0f}"


def _pct_badge(pct: float) -> str:
    if pd.isna(pct):
        return "—"
    arrow = "▲" if pct >= 0 else "▼"
    return f"{arrow} {abs(pct):.1f}%"


# ── Load ML predictor (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and data …")
def load_predictor() -> FootballPredictor:
    return FootballPredictor()


# ── Load raw CSV data for profile tab (cached) ─────────────────────────────────
@st.cache_data(show_spinner="Loading player data …")
def load_raw_data() -> dict:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    required = {
        "players_df":           data_dir / "players.csv",
        "appearances_df":       data_dir / "appearances.csv",
        "club_games_df":        data_dir / "club_games.csv",
        "player_valuations_df": data_dir / "player_valuations.csv",
        "games_df":             data_dir / "games.csv",
        "transfers_df":         data_dir / "transfers.csv",
    }
    return {key: pd.read_csv(path) for key, path in required.items()}


def _calculate_pmi(appearances_df: pd.DataFrame, club_games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute single-game and 10-game rolling PMI score."""
    merged = appearances_df.merge(
        club_games_df,
        left_on=["game_id", "player_club_id"],
        right_on=["game_id", "club_id"],
        how="left",
    ).fillna(0)

    merged["PMI_score"] = (
        (merged["minutes_played"] / 90 * 10)
        + (merged["goals"] * 15)
        + (merged["assists"] * 10)
        - (merged["yellow_cards"] * 5)
        - (merged["red_cards"] * 15)
        + (merged["is_win"] * 5)
    )
    merged = merged.sort_values(["player_id", "date"])
    merged["PMI_10_game_rolling_avg"] = (
        merged.groupby("player_id")["PMI_score"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    cols = ["player_id", "player_name", "date", "PMI_score", "PMI_10_game_rolling_avg"]
    return merged[cols].dropna(subset=["player_name"])


# ── Page header ────────────────────────────────────────────────────────────────
st.title("⚽ Football Player Valuation & Scouting Tool")
st.caption(
    "LightGBM (next-period valuation) · XGBoost (hidden gems / decline detection) · "
    "Data: Transfermarkt via Kaggle"
)

# ── Load predictor — show friendly error if models are missing ─────────────────
try:
    predictor = load_predictor()
except FileNotFoundError as exc:
    st.error(
        "**Models not found.** Run the training pipeline first:\n\n"
        "```\npython src/pipeline.py --skip-download --skip-processing\n```\n\n"
        f"Detail: `{exc}`"
    )
    st.stop()

# ── Pre-compute common values ──────────────────────────────────────────────────
available_seasons  = predictor.get_available_seasons()
available_positions = predictor.get_positions()
latest_season      = predictor.get_latest_season()
all_players_df     = predictor.get_all_players_list()

# ── Load raw profile data (optional — tab 3 degrades gracefully if absent) ────
try:
    raw_data = load_raw_data()
    raw_data_ok = True
except Exception:
    raw_data = {}
    raw_data_ok = False

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_scout, tab_profile = st.tabs([
    "📈 Player Prediction",
    "💎 Scouting Hub",
    "👤 Player Profile & Analytics",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Player Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Player Valuation & Label Prediction")
    st.markdown(
        """
        Search for a player, select them from the list, then click **Predict**.
        The models will return:
        - **Next-season market value** (LightGBM regressor, based on most recent season record)
        - **Hidden Gem probability** — likelihood of a rapid value surge
        - **Rapid Decline probability** — likelihood of a steep value drop
        """
    )

    search_pred = st.text_input(
        "🔍 Search player name",
        placeholder="Type at least 2 characters …",
        key="pred_search",
    )

    if len(search_pred.strip()) < 2:
        st.caption("Type at least 2 characters to filter the player list.")
    else:
        mask_pred = all_players_df["name"].str.contains(
            search_pred.strip(), case=False, na=False
        )
        filtered_pred = all_players_df[mask_pred]

        if filtered_pred.empty:
            st.warning("No players match your search. Try a different spelling.")
        else:
            # Pre-build label dict for O(1) format_func lookup
            label_dict_pred = {
                row["player_id"]: (
                    f"{row['name']}  —  {row.get('current_club_name') or ''}  "
                    f"[{row['season_period']}]"
                )
                for _, row in filtered_pred.iterrows()
            }

            st.caption(f"{len(filtered_pred):,} player(s) found.")
            selected_pred_id = st.selectbox(
                "Select a player",
                options=filtered_pred["player_id"].tolist(),
                format_func=lambda pid: label_dict_pred.get(pid, str(pid)),
                key="pred_player",
            )

            if st.button("🔮 Predict", key="btn_predict"):
                val_result = predictor.predict_valuation(int(selected_pred_id))
                clf_result = predictor.predict_classification(int(selected_pred_id))

                st.divider()
                st.subheader(f"Prediction: {val_result['name']}")

                col_info, col_cls, col_chart = st.columns([1.2, 1.2, 1.0])

                with col_info:
                    pct = val_result["pct_change"]
                    pct_color = "green" if pct >= 0 else "red"
                    st.markdown(
                        f"""
                        | | |
                        |---|---|
                        | **Club** | {val_result['club']} |
                        | **Position** | {val_result['position']} |
                        | **Latest data from** | {val_result['latest_season']} |
                        | **Current valuation** | {_fmt_eur(val_result['current_value_eur'])} |
                        | **Predicted next period** | {_fmt_eur(val_result['predicted_value_eur'])} |
                        | **Implied change** | :{pct_color}[**{_pct_badge(pct)}**] |
                        """
                    )
                    st.caption(
                        "⚠️ Predictions are based on historical patterns up to the "
                        "latest available data. Intra-season events are not reflected."
                    )

                with col_cls:
                    bp = clf_result["burst_probability"]
                    dp = clf_result["decline_probability"]

                    st.markdown("**💎 Hidden Gem Probability**")
                    if bp >= 0.5:
                        st.success(f"**{bp:.1%}** — Likely Hidden Gem")
                    else:
                        st.info(f"**{bp:.1%}**")
                    st.progress(float(bp))

                    st.markdown("**📉 Rapid Decline Probability**")
                    if dp >= 0.5:
                        st.error(f"**{dp:.1%}** — Decline Risk")
                    else:
                        st.info(f"**{dp:.1%}**")
                    st.progress(float(dp))

                    st.caption(
                        "Probabilities are model estimates, not guarantees. "
                        "Hidden gem threshold: player currently ≤ €2.5M. "
                        "Decline threshold: age ≥ 30."
                    )

                with col_chart:
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    labels = ["Current\nValuation", "Predicted\nNext Period"]
                    vals   = [val_result["current_value_eur"], val_result["predicted_value_eur"]]
                    colors = ["#aec6cf", "#77dd77" if pct >= 0 else "#ff6961"]
                    bars = ax.bar(labels, [v / 1e6 for v in vals], color=colors,
                                  width=0.4, edgecolor="white")
                    for bar, v in zip(bars, vals):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            _fmt_eur(v),
                            ha="center", va="bottom", fontsize=9,
                        )
                    ax.set_ylabel("Market Value (€M)")
                    ax.set_ylim(0, max(vals) / 1e6 * 1.35)
                    ax.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Scouting Hub
# ══════════════════════════════════════════════════════════════════════════════
with tab_scout:
    st.header("Scouting Hub — Hidden Gems & Decline Watch List")

    st.success(f"📅 **Latest available season in dataset: {latest_season}**")

    st.markdown(
        """
        Select a **season period** (defaults to the latest) and optionally filter
        by **position**. The models rank all eligible players by probability.

        - **Hidden gems** — undervalued players (≤ €2.5M) most likely to surge in
          value over the next 2 years.
        - **Decline watch list** — veterans (age ≥ 30) most at risk of a steep
          valuation drop over the next 2 years.
        """
    )

    col_season, col_pos, col_k = st.columns([2, 2, 1])

    with col_season:
        # Default index = 0 because available_seasons[::-1] puts latest first
        selected_season = st.selectbox(
            "📅 Season period",
            options=available_seasons[::-1],
            index=0,
            key="scout_season",
        )

    with col_pos:
        selected_positions = st.multiselect(
            "🎽 Filter by position (blank = all)",
            options=available_positions,
            key="scout_positions",
        )

    with col_k:
        top_k = st.number_input(
            "Top K", min_value=5, max_value=200, value=20, step=5,
            key="scout_top_k",
        )

    if st.button("🔍 Find Players", key="btn_scout"):
        pos_filter = selected_positions if selected_positions else None

        col_gems, col_decline = st.columns(2)

        # ── Hidden Gems ───────────────────────────────────────────────────
        with col_gems:
            st.subheader("💎 Hidden Gems")
            st.caption(
                f"Top-{top_k} undervalued players (≤ €2.5M) by burst probability "
                f"in **{selected_season}**"
            )

            gems = predictor.get_hidden_gems(
                season_period=selected_season,
                positions=pos_filter,
                top_k=top_k,
            )

            if gems.empty:
                st.info("No undervalued players found for this season / position filter.")
            else:
                dg = gems[["name", "current_club_name", "position", "age",
                            "period_end_valuation", "burst_probability"]].copy()
                dg["period_end_valuation"] = dg["period_end_valuation"].apply(_fmt_eur)
                dg["burst_probability"]    = dg["burst_probability"].apply(lambda x: f"{x:.1%}")
                dg["age"]                  = dg["age"].apply(lambda x: f"{x:.0f}")
                dg = dg.rename(columns={
                    "name": "Player", "current_club_name": "Club",
                    "position": "Position", "age": "Age",
                    "period_end_valuation": "Current Value",
                    "burst_probability": "Burst Prob.",
                })
                st.dataframe(dg, use_container_width=True, hide_index=False)

                fig, ax = plt.subplots(figsize=(5, max(3, len(gems) * 0.35)))
                ax.barh(gems["name"].fillna("Unknown").values[::-1],
                        gems["burst_probability"].values[::-1],
                        color="#77dd77", edgecolor="white")
                ax.set_xlabel("Burst Probability")
                ax.set_title("Burst Probability Ranking", fontweight="bold", fontsize=10)
                ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ── Decline Watch List ─────────────────────────────────────────────
        with col_decline:
            st.subheader("📉 Decline Watch List")
            st.caption(
                f"Top-{top_k} veterans (age ≥ 30) by decline probability "
                f"in **{selected_season}**"
            )

            declines = predictor.get_rapid_declines(
                season_period=selected_season,
                positions=pos_filter,
                top_k=top_k,
            )

            if declines.empty:
                st.info("No veteran players found for this season / position filter.")
            else:
                dd = declines[["name", "current_club_name", "position", "age",
                               "period_end_valuation", "decline_probability"]].copy()
                dd["period_end_valuation"]  = dd["period_end_valuation"].apply(_fmt_eur)
                dd["decline_probability"]   = dd["decline_probability"].apply(lambda x: f"{x:.1%}")
                dd["age"]                   = dd["age"].apply(lambda x: f"{x:.0f}")
                dd = dd.rename(columns={
                    "name": "Player", "current_club_name": "Club",
                    "position": "Position", "age": "Age",
                    "period_end_valuation": "Current Value",
                    "decline_probability": "Decline Prob.",
                })
                st.dataframe(dd, use_container_width=True, hide_index=False)

                fig, ax = plt.subplots(figsize=(5, max(3, len(declines) * 0.35)))
                ax.barh(declines["name"].fillna("Unknown").values[::-1],
                        declines["decline_probability"].values[::-1],
                        color="#ff6961", edgecolor="white")
                ax.set_xlabel("Decline Probability")
                ax.set_title("Decline Probability Ranking", fontweight="bold", fontsize=10)
                ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with st.expander("ℹ️ How are these predictions made?"):
        st.markdown(
            """
            **Hidden Gems** are identified by an XGBoost binary classifier trained
            to detect players whose valuation will grow into the **top 5th percentile**
            over the next 2 years (≈ 6.7× increase), filtered to players currently
            valued at or below the market median (≤ €2.5M).

            **Decline Watch List** uses a separate XGBoost classifier trained on
            players whose valuation falls into the **bottom 5th percentile** over
            2 years (≈ 78% drop), restricted to players aged 30 or over.

            Both models handle severe class imbalance via `scale_pos_weight` and are
            evaluated using **Precision@K** and **PR-AUC**. See Notebook 3 for full
            evaluation details.
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Player Profile & Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.header("Player Profile & Analytics")

    if not raw_data_ok:
        st.error(
            "Raw data files could not be loaded. "
            "Ensure the `data/` directory is populated by running the pipeline."
        )
        st.stop()

    # ── Player search ──────────────────────────────────────────────────────────
    st.subheader("Select a Player")

    search_prof = st.text_input(
        "🔍 Search player name",
        placeholder="Type at least 2 characters …",
        key="prof_search",
    )

    if len(search_prof.strip()) < 2:
        st.caption("Type at least 2 characters to filter the player list.")
        st.stop()

    mask_prof = all_players_df["name"].str.contains(
        search_prof.strip(), case=False, na=False
    )
    filtered_prof = all_players_df[mask_prof]

    if filtered_prof.empty:
        st.warning("No players match your search.")
        st.stop()

    label_dict_prof = {
        row["player_id"]: f"{row['name']}  —  {row.get('current_club_name') or ''}"
        for _, row in filtered_prof.iterrows()
    }
    st.caption(f"{len(filtered_prof):,} player(s) found.")
    prof_player_id = int(st.selectbox(
        "Select player",
        options=filtered_prof["player_id"].tolist(),
        format_func=lambda pid: label_dict_prof.get(pid, str(pid)),
        key="prof_player",
    ))

    st.divider()

    # ── Profile info ───────────────────────────────────────────────────────────
    profile = get_player_profile(raw_data["players_df"], prof_player_id)

    if profile:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Player Information")
            st.write(f"**Name:** {profile.get('name', 'N/A')}")
            dob = profile.get("date_of_birth")
            if dob and pd.notna(dob):
                dob_dt = pd.to_datetime(dob, errors="coerce")
                if pd.notna(dob_dt):
                    age = (datetime.now() - dob_dt).days // 365
                    st.write(f"**Age:** {age}")
            st.write(f"**Country:** {profile.get('country_of_citizenship', 'N/A')}")
            st.write(f"**Position:** {profile.get('position', 'N/A')}")
            st.write(f"**Sub-Position:** {profile.get('sub_position', 'N/A')}")

        with col2:
            st.subheader("Physical Attributes")
            st.write(f"**Height:** {profile.get('height_in_cm', 'N/A')} cm")
            st.write(f"**Foot:** {profile.get('foot', 'N/A')}")
            st.write(f"**Current Club:** {profile.get('current_club_name', 'N/A')}")

        with col3:
            st.subheader("Market Value")
            mv_cur = profile.get("market_value_in_eur")
            mv_top = profile.get("highest_market_value_in_eur")
            st.write(f"**Current:** {_fmt_eur(mv_cur) if mv_cur else 'N/A'}")
            st.write(f"**Highest:** {_fmt_eur(mv_top) if mv_top else 'N/A'}")

            transfers_d = get_transfer_history(raw_data["transfers_df"], prof_player_id)
            if transfers_d is not None and not transfers_d.empty:
                last_t = transfers_d.iloc[-1]
                tf = last_t.get("transfer_fee")
                if pd.notna(tf) and tf > 0:
                    st.write(f"**Latest Transfer Fee:** {_fmt_eur(tf)}")
                else:
                    st.write("**Latest Transfer Fee:** N/A")
            else:
                st.write("**Latest Transfer Fee:** N/A")

    # ── Actual vs Predicted Valuations ─────────────────────────────────────────
    pred_hist = predictor.get_player_prediction_history(prof_player_id)
    if not pred_hist.empty:
        st.subheader("Actual vs Predicted Valuations")
        fig_pred = go.Figure()

        if "next_valuation" in pred_hist.columns:
            fig_pred.add_trace(go.Scatter(
                x=pred_hist["season_period"],
                y=pred_hist["next_valuation"],
                mode="lines+markers",
                name="Actual Next-Period Value",
                line=dict(dash="solid", color="forestgreen"),
            ))

        fig_pred.add_trace(go.Scatter(
            x=pred_hist["season_period"],
            y=pred_hist["predicted_valuation"],
            mode="lines+markers",
            name="Model Predicted Value",
            line=dict(dash="dash", color="orange"),
        ))

        fig_pred.update_layout(
            title="Actual vs Predicted Valuations (all seasons)",
            xaxis_title="Season Period",
            yaxis_title="Value (EUR)",
            hovermode="x unified",
            template="plotly_dark",
            height=450,
            xaxis=dict(type="category"),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # ── Market Value Timeline ──────────────────────────────────────────────────
    valuations   = get_valuation_history(raw_data["player_valuations_df"], prof_player_id)
    transfers_d  = get_transfer_history(raw_data["transfers_df"], prof_player_id)

    if valuations is not None and not valuations.empty:
        st.subheader("Market Value Timeline")

        fig_val = px.line(
            valuations, x="date", y="market_value_in_eur",
            title="Market Value Over Time with Transfer Events",
            labels={"market_value_in_eur": "Market Value (EUR)", "date": "Date"},
            markers=True,
        )

        if transfers_d is not None and not transfers_d.empty:
            tp = transfers_d.copy()
            tp["transfer_date"] = pd.to_datetime(tp["transfer_date"], errors="coerce")
            tp = tp.dropna(subset=["transfer_date"])

            if not tp.empty:
                t_vals = []
                for t_date in tp["transfer_date"]:
                    idx = (valuations["date"] - t_date).abs().argmin()
                    t_vals.append(valuations.iloc[idx]["market_value_in_eur"])
                tp["market_value_in_eur"] = t_vals

                scatter_trace = px.scatter(
                    tp, x="transfer_date", y="market_value_in_eur",
                    hover_data={"from_club_name": True, "to_club_name": True},
                ).data[0]
                scatter_trace.marker.color = "orange"
                scatter_trace.marker.size  = 12
                scatter_trace.name         = "Transfer Events"
                fig_val.add_trace(scatter_trace)

        fig_val.update_layout(hovermode="x unified", template="plotly_dark", height=450)
        st.plotly_chart(fig_val, use_container_width=True)

    # ── PMI — Player Momentum Index ────────────────────────────────────────────
    st.subheader("Player Momentum Index (PMI)")
    try:
        pmi_df = _calculate_pmi(raw_data["appearances_df"], raw_data["club_games_df"])
        pmi_player = pmi_df[pmi_df["player_id"] == prof_player_id].copy()
        pmi_player["date"] = pd.to_datetime(pmi_player["date"], errors="coerce")
        pmi_player = pmi_player.sort_values("date")

        if not pmi_player.empty:
            fig_pmi = go.Figure()
            fig_pmi.add_trace(go.Scatter(
                x=pmi_player["date"],
                y=pmi_player["PMI_10_game_rolling_avg"],
                mode="lines",
                name="10-game Rolling PMI",
                line=dict(color="royalblue"),
            ))
            fig_pmi.add_trace(go.Scatter(
                x=pmi_player["date"],
                y=pmi_player["PMI_score"],
                mode="markers",
                name="Single-Game PMI",
                marker=dict(size=4, color="lightblue", opacity=0.6),
            ))
            fig_pmi.update_layout(
                title="Player Momentum Index Over Time",
                xaxis_title="Date",
                yaxis_title="PMI Score",
                hovermode="x unified",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig_pmi, use_container_width=True)
        else:
            st.info("No PMI data available for this player.")
    except Exception:
        st.info("PMI could not be calculated (appearances or club_games data missing).")

    # ── Games Performance ──────────────────────────────────────────────────────
    games = get_game_history(raw_data["appearances_df"], raw_data["games_df"], prof_player_id)

    if games is not None and not games.empty:
        st.subheader("Career Performance Stats")

        total_goals   = int(games["goals"].sum())
        total_assists = int(games["assists"].sum())
        total_minutes = int(games["minutes_played"].sum())
        total_games   = len(games)
        total_wins    = int((games["result"] == "W").sum())

        goals_per_90   = total_goals   / total_minutes * 90 if total_minutes > 0 else 0
        assists_per_90 = total_assists / total_minutes * 90 if total_minutes > 0 else 0
        win_pct        = total_wins    / total_games   * 100 if total_games > 0 else 0
        mins_per_game  = total_minutes / total_games         if total_games > 0 else 0

        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Total Goals",   total_goals)
        tc2.metric("Total Assists", total_assists)
        tc3.metric("Total Games",   total_games)
        tc4.metric("Total Wins",    total_wins)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Goals / 90",     f"{goals_per_90:.2f}")
        c2.metric("Assists / 90",   f"{assists_per_90:.2f}")
        c3.metric("Mins / Game",    f"{mins_per_game:.1f}")
        c4.metric("Win %",          f"{win_pct:.1f}%")

        st.write("**Recent Match History (last 10):**")
        recent = games.tail(10)[
            ["date", "opponent_club_name", "is_home", "team_goals",
             "opp_goals", "result", "goals", "assists", "minutes_played"]
        ].copy()
        recent.columns = ["Date", "Opponent", "Home", "Team Goals",
                          "Opp Goals", "Result", "Goals", "Assists", "Minutes"]
        recent["Home"] = recent["Home"].apply(lambda x: "Home" if x else "Away")
        st.dataframe(recent, use_container_width=True, hide_index=True)

        # ── Visualizations ─────────────────────────────────────────────────────
        st.subheader("Performance Visualizations")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.write("**Appearances Scatter (Goals + Assists)**")
            st.caption("🟢 Win  🔴 Loss  ⚫ Draw")
            fig_sc, ax_sc = plt.subplots(figsize=(10, 4))
            plot_appearances_scatter(games, ax=ax_sc)
            ax_sc.set_ylim(bottom=0)
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        with vcol2:
            st.write("**Minutes per Appearance**")
            fig_m, ax_m = plt.subplots(figsize=(10, 4))
            ax_m.hist(games["minutes_played"], bins=15,
                      color="steelblue", edgecolor="black", alpha=0.7)
            ax_m.set_xlabel("Minutes Played")
            ax_m.set_ylabel("Frequency")
            ax_m.set_title("Distribution of Minutes Played")
            ax_m.grid(True, alpha=0.2)
            st.pyplot(fig_m)
            plt.close(fig_m)

    # ── Transfer History ───────────────────────────────────────────────────────
    if transfers_d is not None and not transfers_d.empty:
        st.subheader("Transfer History")
        td = transfers_d.copy()
        td["transfer_date"] = pd.to_datetime(
            td["transfer_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        st.dataframe(td, use_container_width=True, hide_index=True)
