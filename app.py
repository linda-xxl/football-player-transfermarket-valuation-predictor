import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from src import player_profile

# Page configuration
st.set_page_config(layout="wide", page_title="CFFC 'Momentum' Engine")

# Title
st.title("CFFC Player Momentum Index (PMI) Engine 🚀")


@st.cache_data
def load_data():
    """Load the required CSV files."""
    players_df = pd.read_csv("data/players.csv")
    appearances_df = pd.read_csv("data/appearances.csv")
    club_games_df = pd.read_csv("data/club_games.csv")
    player_valuations_df = pd.read_csv("data/player_valuations.csv")
    games_df = pd.read_csv("data/games.csv")

    return players_df, appearances_df, club_games_df, player_valuations_df, games_df


def calculate_pmi(appearances_df, club_games_df, players_df):
    """Calculate the Player Momentum Index (PMI) with 10-game rolling average."""

    # Merge appearances with club_games on game_id and player_club_id/club_id
    merged_df = appearances_df.merge(
        club_games_df,
        left_on=["game_id", "player_club_id"],
        right_on=["game_id", "club_id"],
        how="left"
    )    # Fill NaN values with 0
    merged_df = merged_df.fillna(0)

    # Calculate single-game PMI score
    merged_df["PMI_score"] = (
        (merged_df["minutes_played"] / 90 * 10) +
        (merged_df["goals"] * 15) +
        (merged_df["assists"] * 10) -
        (merged_df["yellow_cards"] * 5) -
        (merged_df["red_cards"] * 15) +
        (merged_df["is_win"] * 5)
    )

    # Sort by player_id and date
    merged_df = merged_df.sort_values(by=["player_id", "date"])

    # Calculate 10-game rolling average for each player
    merged_df["PMI_10_game_rolling_avg"] = (
        merged_df.groupby("player_id")["PMI_score"].rolling(
            window=10, min_periods=1).mean().reset_index(drop=True)
    )

    # The player_name is already in appearances_df, so we don't need to merge again
    # Just return the relevant columns including raw PMI_score
    final_df = merged_df[["player_id", "player_name",
                          "date", "PMI_score", "PMI_10_game_rolling_avg", "is_win"]].copy()

    # Return only the columns needed for visualization
    return final_df.dropna(subset=["player_name"])


# Load data
players_df, appearances_df, club_games_df, player_valuations_df, games_df = load_data()

# Calculate PMI
pmi_df = calculate_pmi(appearances_df, club_games_df, players_df)

# Convert player_name to string to handle mixed types
pmi_df["player_name"] = pmi_df["player_name"].astype(str)

# Create list of unique player names
player_names = sorted(pmi_df["player_name"].dropna().unique().tolist())

# Sidebar for player selection with advanced filters
st.sidebar.header("Select Player ⚽")

# A1. Add Player Name Search
player_name_search = st.sidebar.text_input(
    "Search player by name:",
    placeholder="Type name to filter..."
)

# A2. Add Position Filter
available_positions = sorted(
    [p for p in players_df["position"].dropna().unique() if isinstance(p, str)]
)
selected_positions = st.sidebar.multiselect(
    "Filter by position:",
    options=available_positions,
    default=None
)

# (PMI filter removed - slider deleted per request)

# A4. Add Market Value Filter
min_market_value = float(player_valuations_df["market_value_in_eur"].min())
max_market_value = float(player_valuations_df["market_value_in_eur"].max())

market_value_range = st.sidebar.slider(
    "Filter by current market value (EUR):",
    min_value=min_market_value,
    max_value=max_market_value,
    value=(min_market_value, max_market_value),
    step=100000.0
)

# (Total Points Added filter removed - slider deleted per request)

# Apply filters to create filtered player list
filtered_players = pmi_df.copy()

# Filter by name search
if player_name_search:
    filtered_players = filtered_players[
        filtered_players["player_name"].str.contains(
            player_name_search, case=False, na=False
        )
    ]

# Filter by position
if selected_positions:
    # Get player IDs with selected positions
    players_with_positions = players_df[
        players_df["position"].isin(selected_positions)
    ]["player_id"].tolist()
    filtered_players = filtered_players[
        filtered_players["player_id"].isin(players_with_positions)
    ]

# (Average PMI filter removed)

# Filter by market value
# Get latest market value for each player
latest_valuations = player_valuations_df.sort_values("date").drop_duplicates(
    "player_id", keep="last"
)[["player_id", "market_value_in_eur"]].rename(columns={"market_value_in_eur": "current_market_value"})
filtered_players = filtered_players.merge(
    latest_valuations, on="player_id", how="left"
)
filtered_players = filtered_players[
    (filtered_players["current_market_value"] >= market_value_range[0]) &
    (filtered_players["current_market_value"] <= market_value_range[1])
]

# (Total Points Added filter removed)
# Get unique player names from filtered data
filtered_player_names = sorted(
    filtered_players["player_name"].dropna().unique().tolist()
)

# Display filter result counts
st.sidebar.caption(f"{len(filtered_player_names)} players found")

# Show filtered count
if len(filtered_player_names) == 0:
    st.sidebar.warning("No players match the selected filters.")
    selected_player = None
else:
    selected_player = st.sidebar.selectbox(
        "Choose a player:",
        options=filtered_player_names
    )

# Filter data for selected player only if player is selected
if selected_player is not None:
    player_data = pmi_df[pmi_df["player_name"] == selected_player].copy()
    player_data = player_data.sort_values(by="date")
else:
    player_data = pd.DataFrame()

# Main page header
if selected_player is not None:
    st.header(f"{selected_player}'s Momentum & Market Value")
else:
    st.header("Please select a player from the filters")

# Get player_id for market value lookup
if selected_player is not None:
    player_id_list = pmi_df[pmi_df["player_name"]
                            == selected_player]["player_id"].unique()
else:
    player_id_list = []

if len(player_id_list) > 0 and selected_player is not None:
    player_id = int(player_id_list[0])

    # Get valuation data for the selected player
    valuations = player_valuations_df[player_valuations_df["player_id"] == player_id].copy(
    )
    if not valuations.empty:
        valuations["date"] = pd.to_datetime(
            valuations["date"], errors="coerce")
        valuations = valuations.sort_values("date")

    # Merge PMI data with valuations on date using merge_asof for proper handling of mismatched dates
    combined_data = player_data.copy()
    combined_data["date"] = pd.to_datetime(
        combined_data["date"], errors="coerce")
    combined_data = combined_data.sort_values("date")

    # Prepare valuations DataFrame
    valuations_clean = valuations[["date", "market_value_in_eur"]].copy()
    valuations_clean["date"] = pd.to_datetime(
        valuations_clean["date"], errors="coerce")
    valuations_clean = valuations_clean.sort_values(
        "date").drop_duplicates(subset=["date"], keep="last")

    # Use merge_asof to merge on date with direction='backward'
    # This finds the most recent market value for each PMI date
    combined_data = pd.merge_asof(
        combined_data,
        valuations_clean,
        on="date",
        direction="backward"
    )

    # ---------------------------------------------
    # Actual vs Predicted Valuations — loaded from ML predictor
    # ---------------------------------------------
    import sys as _sys
    _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    try:
        from src.modules.predictor import FootballPredictor as _FP
        import plotly.graph_objects as go

        @st.cache_resource(show_spinner=False)
        def _load_pred():
            return _FP()

        _predictor = _load_pred()
        pred_hist  = _predictor.get_player_prediction_history(player_id)

        if not pred_hist.empty:
            fig = go.Figure()

            if "next_valuation" in pred_hist.columns:
                fig.add_trace(go.Scatter(
                    x=pred_hist["season_period"],
                    y=pred_hist["next_valuation"],
                    mode="lines+markers",
                    name="Actual Next-Period Value",
                    line=dict(dash="solid", color="forestgreen"),
                ))

            fig.add_trace(go.Scatter(
                x=pred_hist["season_period"],
                y=pred_hist["predicted_valuation"],
                mode="lines+markers",
                name="Model Predicted Value",
                line=dict(dash="dash", color="orange"),
            ))

            fig.update_layout(
                title=f"{selected_player} — Actual vs Predicted Valuations",
                xaxis_title="Season Period",
                yaxis_title="Value (EUR)",
                hovermode="x unified",
                template="plotly_dark",
                height=500,
                xaxis=dict(type="category"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction history available for this player.")
    except FileNotFoundError:
        st.info(
            "Actual vs Predicted chart not available — models not trained yet. "
            "Run `python src/pipeline.py --skip-download --skip-processing` first."
        )
    except Exception:
        st.info("Actual vs Predicted chart could not be loaded.")

# ============================================================================
# DETAILED PLAYER PROFILE & ANALYTICS SECTION
# ============================================================================

st.header("Detailed Player Profile & Analytics")

# Load additional data using player_profile module
data = player_profile.load_minimal_data(data_dir="data")

# Find the player_id for the selected player
# Get the player_id from the filtered pmi_df
player_id_list = pmi_df[pmi_df["player_name"]
                        == selected_player]["player_id"].unique()

if len(player_id_list) > 0:
    player_id = int(player_id_list[0])

    # Get player profile information
    profile = player_profile.get_player_profile(data["players_df"], player_id)

    if profile:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Player Information")
            st.write(f"**Name:** {profile.get('name', 'N/A')}")
            # B6: Add age calculation and display
            dob = profile.get('date_of_birth')
            if dob and pd.notna(dob):
                dob = pd.to_datetime(dob, errors='coerce')
                if pd.notna(dob):
                    from datetime import datetime
                    age = (datetime.now() - dob).days // 365
                    st.write(f"**Age:** {age}")
                else:
                    st.write(f"**Age:** N/A")
            st.write(
                f"**Country:** {profile.get('country_of_citizenship', 'N/A')}")
            st.write(f"**Position:** {profile.get('position', 'N/A')}")
            st.write(f"**Sub-Position:** {profile.get('sub_position', 'N/A')}")

        with col2:
            st.subheader("Physical Attributes")
            st.write(f"**Height:** {profile.get('height_in_cm', 'N/A')} cm")
            st.write(f"**Foot:** {profile.get('foot', 'N/A')}")
            st.write(
                f"**Current Club:** {profile.get('current_club_name', 'N/A')}")

        with col3:
            st.subheader("Market Value")
            market_value = profile.get('market_value_in_eur')
            highest_value = profile.get('highest_market_value_in_eur')
            if market_value:
                st.write(f"**Current:** €{market_value:,.0f}")
            else:
                st.write("**Current:** N/A")
            if highest_value:
                st.write(f"**Highest:** €{highest_value:,.0f}")
            else:
                st.write("**Highest:** N/A")

            # 3. Show latest transfer fee
            transfers = player_profile.get_transfer_history(
                data["transfers_df"], player_id
            )
            if transfers is not None and not transfers.empty:
                latest_transfer = transfers.iloc[-1]
                transfer_fee = latest_transfer.get('transfer_fee')
                if pd.notna(transfer_fee) and transfer_fee > 0:
                    st.write(f"**Latest Transfer Fee:** €{transfer_fee:,.0f}")
                else:
                    st.write("**Latest Transfer Fee:** N/A")
            else:
                st.write("**Latest Transfer Fee:** N/A")

    # Get game history
    games = player_profile.get_game_history(
        data["appearances_df"], data["games_df"], player_id
    )

    # --- Market Value Timeline (moved above Games Performance) ---
    # Compute transfer history (used for markers) and valuation history
    transfers = player_profile.get_transfer_history(
        data["transfers_df"], player_id
    )

    valuations = player_profile.get_valuation_history(
        data["player_valuations_df"], player_id
    )

    if valuations is not None and not valuations.empty:
        st.subheader("Market Value Timeline")

        fig_valuation = px.line(
            valuations,
            x="date",
            y="market_value_in_eur",
            title="Market Value Over Time with Transfer Events",
            labels={
                "market_value_in_eur": "Market Value (EUR)", "date": "Date"},
            markers=True
        )

        # Add transfer event markers if transfers exist
        if transfers is not None and not transfers.empty:
            transfers_plot = transfers.copy()
            transfers_plot["transfer_date"] = pd.to_datetime(
                transfers_plot["transfer_date"], errors="coerce")
            transfers_plot = transfers_plot.dropna(subset=["transfer_date"])

            if not transfers_plot.empty:
                transfer_values = []
                for t_date in transfers_plot["transfer_date"]:
                    closest_idx = (valuations["date"] - t_date).abs().argmin()
                    transfer_values.append(
                        valuations.iloc[closest_idx]["market_value_in_eur"])

                transfers_plot["market_value_in_eur"] = transfer_values

                fig_valuation.add_trace(
                    px.scatter(
                        transfers_plot,
                        x="transfer_date",
                        y="market_value_in_eur",
                        hover_data={"from_club_name": True,
                                    "to_club_name": True}
                    ).data[0]
                )
                fig_valuation.data[-1].marker.color = "orange"
                fig_valuation.data[-1].marker.size = 12
                fig_valuation.data[-1].name = "Transfer Events"

        fig_valuation.update_layout(
            hovermode="x unified",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig_valuation, use_container_width=True)

    if games is not None and not games.empty:
        st.subheader("Games Performance")

        # Display key stats from entire career
        total_goals = int(games["goals"].sum())
        total_assists = int(games["assists"].sum())
        total_minutes = int(games["minutes_played"].sum())
        total_games = len(games)
        total_wins = (games["result"] == "W").sum()

        goals_per_90 = (total_goals / total_minutes) * \
            90 if total_minutes > 0 else 0
        assists_per_90 = (total_assists / total_minutes) * \
            90 if total_minutes > 0 else 0
        win_pct = (total_wins / total_games) * 100 if total_games > 0 else 0

        # Minutes per 90 (as requested: Total Minutes / Total Games)
        minutes_per_90 = (
            total_minutes / total_games) if total_games > 0 else 0

        # Totals row: Total Goals, Total Assists, Total Games, Total Wins
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        with tcol1:
            st.metric("Total Goals", int(total_goals))
        with tcol2:
            st.metric("Total Assists", int(total_assists))
        with tcol3:
            st.metric("Total Games", int(total_games))
        with tcol4:
            st.metric("Total Wins", int(total_wins))

        # Simple gray captions below top row of metrics
        tcap1, tcap2, tcap3, tcap4 = st.columns(4)
        with tcap1:
            st.caption("Career total goals")
        with tcap2:
            st.caption("Career total assists")
        with tcap3:
            st.caption("Career total games")
        with tcap4:
            st.caption("Career total wins")

        # Per-90 / derived metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Goals Per 90", f"{goals_per_90:.2f}")
        with col2:
            st.metric("Assists Per 90", f"{assists_per_90:.2f}")
        with col3:
            st.metric("Minutes Per 90", f"{minutes_per_90:.1f}")
        with col4:
            st.metric("Win %", f"{win_pct:.1f}%")

        # Captions / formula explanations under the derived metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("(Total Goals / Total Minutes) * 90")
        with c2:
            st.caption("(Total Assists / Total Minutes) * 90")
        with c3:
            st.caption("(Total Minutes / Total Games)")
        with c4:
            st.caption("(Total Wins / Total Games) * 100")

        # Display recent games table
        recent_games = games.tail(10)
        st.write("**Recent Match History:**")
        games_display = recent_games[[
            "date", "opponent_club_name", "is_home", "team_goals", "opp_goals",
            "result", "goals", "assists", "minutes_played"
        ]].copy()
        games_display.columns = [
            "Date", "Opponent", "Home", "Team Goals", "Opp Goals",
            "Result", "Goals", "Assists", "Minutes"
        ]
        games_display["Home"] = games_display["Home"].apply(
            lambda x: "Home" if x else "Away")
        st.dataframe(games_display, use_container_width=True)

        # Add matplotlib charts
        st.subheader("Performance Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Appearances Scatter Plot**")
            st.caption("🟢 Green: Wins | 🔴 Red: Losses | ⚫ Gray: Draws")
            import matplotlib.pyplot as plt
            fig_appearances, ax = plt.subplots(figsize=(10, 4))
            player_profile.plot_appearances_scatter(games, ax=ax)
            # B5: Ensure Y-axis always starts from 0
            ax.set_ylim(bottom=0)
            st.pyplot(fig_appearances)

        with col2:
            st.write("**Minutes per Appearance**")
            st.caption(
                "Distribution of minutes played across all appearances. Blue bars show frequency.")
            fig_minutes, ax = plt.subplots(figsize=(10, 4))
            games_plot = games.copy()
            ax.hist(games_plot["minutes_played"], bins=15,
                    color="steelblue", edgecolor="black", alpha=0.7)
            ax.set_xlabel("Minutes Played")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Minutes Played")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig_minutes)

        # Transfer History (displayed after performance visuals)
        if transfers is not None and not transfers.empty:
            st.subheader("Transfer History")
            transfers_display = transfers.copy()
            transfers_display["transfer_date"] = pd.to_datetime(
                transfers_display["transfer_date"]
            ).dt.strftime("%Y-%m-%d")
            st.dataframe(transfers_display, use_container_width=True)


else:
    st.warning(f"Could not find detailed profile data for {selected_player}")
