"""
CBB Betting Model Dashboard
Streamlit app for live game predictions and odds comparison.

Usage:
    streamlit run dashboard.py
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)
ESPN_ODDS = (
    "https://sports.core.api.espn.com/v2/sports/basketball/"
    "leagues/mens-college-basketball/events/{gid}/competitions/{gid}/odds"
)

EXCLUDE_PROVIDER_IDS = {59}  # live odds


# -------------------------------------------------------------------
# Load model and data (cached)
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained LR model."""
    path = MODELS_DIR / "logistic_regression.joblib"
    model_dict = joblib.load(path)
    return model_dict


@st.cache_data(ttl=300)
def load_features():
    """Load the features file with all rolling stats."""
    path = PROCESSED_DIR / "features_v2.csv"
    df = pd.read_csv(path)
    df["game_id"] = df["game_id"].astype(str)
    return df


@st.cache_data(ttl=300)
def load_elo_ratings():
    """Load current ELO ratings."""
    path = MODELS_DIR / "elo_ratings.json"
    with open(path) as f:
        data = json.load(f)
    return data


@st.cache_data
def load_feature_cols():
    """Load the list of feature columns the model expects."""
    path = PROCESSED_DIR / "feature_cols.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# -------------------------------------------------------------------
# ESPN API helpers
# -------------------------------------------------------------------
@st.cache_data(ttl=120)
def fetch_scoreboard(date_str):
    """Fetch ESPN scoreboard for a given date."""
    try:
        resp = requests.get(ESPN_SCOREBOARD, params={
            "dates": date_str,
            "groups": "50",
            "limit": "400",
        }, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error("Failed to fetch scoreboard: {}".format(e))
        return None


@st.cache_data(ttl=120)
def fetch_odds(game_id):
    """Fetch odds for a specific game."""
    try:
        url = ESPN_ODDS.format(gid=game_id)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        # Filter out live odds
        return [i for i in items if int(i.get("provider", {}).get("id", 0)) not in EXCLUDE_PROVIDER_IDS]
    except Exception:
        return []


def moneyline_to_implied(ml):
    """Convert American moneyline to implied probability."""
    if ml is None or pd.isna(ml):
        return None
    ml = float(ml)
    if ml == 0:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)

def kelly_sizing(model_prob, market_prob, ml, fraction=0.25):
    """
    Compute fractional Kelly bet size.
    Returns (kelly_pct, bet_amount) or (0, 0) if no edge.
    """
    if model_prob is None or market_prob is None or ml is None:
        return 0, 0
    ml = float(ml)
    if ml < 0:
        decimal_odds = 1 + (100 / abs(ml))
    else:
        decimal_odds = 1 + (ml / 100)
    edge = model_prob - market_prob
    if edge <= 0:
        return 0, 0
    kelly_pct = edge / (decimal_odds - 1)
    fractional = kelly_pct * fraction
    return round(fractional, 4), decimal_odds

BETS_FILE = Path("data/bets_log.csv")


def load_bets():
    """Load existing bets log."""
    if BETS_FILE.exists():
        try:
            df = pd.read_csv(BETS_FILE)
            if df.empty:
                return pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_bet(game_id, date, home_team, away_team, bet_side, bet_amount,
             model_prob, market_prob, decimal_odds):
    """Save a new bet to the log."""
    row = {
        "game_id": str(game_id),
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "bet_side": bet_side,
        "bet_amount": round(bet_amount, 2),
        "model_prob": round(model_prob, 4),
        "market_prob": round(market_prob, 4),
        "decimal_odds": round(decimal_odds, 4),
        "result": None,
        "profit": None,
    }
    df = load_bets()
    if not df.empty:
        existing = df[(df["game_id"].astype(str) == str(game_id)) & (df["bet_side"] == bet_side)]
        if not existing.empty:
            return False
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    BETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BETS_FILE, index=False)
    return True


def remove_bet(game_id, bet_side):
    """Remove a bet from the log."""
    df = load_bets()
    if df.empty:
        return
    mask = (df["game_id"].astype(str) == str(game_id)) & (df["bet_side"] == bet_side)
    df = df[~mask]
    df.to_csv(BETS_FILE, index=False)


def settle_bets():
    """Check completed games and calculate P&L using ESPN API."""
    df = load_bets()
    if df.empty:
        return df

    for idx, row in df.iterrows():
        if pd.notna(row.get("profit")):
            continue  # already settled

        gid = str(row["game_id"])

        # Hit ESPN summary for this game
        try:
            resp = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/"
                "mens-college-basketball/summary?event={}".format(gid),
                timeout=10,
            )
            data = resp.json()
            header = data.get("header", {})
            competitions = header.get("competitions", [{}])
            if not competitions:
                continue
            comp = competitions[0]

            # Check if game is complete
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue

            # Get scores
            competitors = comp.get("competitors", [])
            home_score = away_score = 0
            for c in competitors:
                if c.get("homeAway") == "home":
                    home_score = int(c.get("score", 0))
                else:
                    away_score = int(c.get("score", 0))

            home_won = home_score > away_score
            bet_won = (row["bet_side"] == "home" and home_won) or \
                      (row["bet_side"] == "away" and not home_won)

            if bet_won:
                profit = row["bet_amount"] * (row["decimal_odds"] - 1)
            else:
                profit = -row["bet_amount"]

            df.at[idx, "result"] = "W" if bet_won else "L"
            df.at[idx, "profit"] = round(profit, 2)

        except Exception:
            continue

    df.to_csv(BETS_FILE, index=False)
    return df


def parse_scoreboard(data):
    """Parse ESPN scoreboard into a clean list of games."""
    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})

        home = away = None
        for t in comp.get("competitors", []):
            if t.get("homeAway") == "home":
                home = t
            else:
                away = t

        if not home or not away:
            continue

        home_team = home.get("team", {})
        away_team = away.get("team", {})

        game = {
            "game_id": event.get("id", ""),
            "date": event.get("date", ""),
            "name": event.get("shortName", ""),
            "status": status.get("description", ""),
            "status_name": status.get("name", ""),
            "completed": status.get("completed", False),
            "home_team": home_team.get("displayName", ""),
            "away_team": away_team.get("displayName", ""),
            "home_abbr": home_team.get("abbreviation", ""),
            "away_abbr": away_team.get("abbreviation", ""),
            "home_id": home_team.get("id", ""),
            "away_id": away_team.get("id", ""),
            "home_score": home.get("score", ""),
            "away_score": away.get("score", ""),
            "home_logo": home_team.get("logo", ""),
            "away_logo": away_team.get("logo", ""),
            "neutral_site": comp.get("neutralSite", False),
        }
        games.append(game)

    return games


# -------------------------------------------------------------------
# Prediction engine
# -------------------------------------------------------------------
def get_team_latest_features(features_df, team_name, team_id):
    """
    Get the most recent rolling features for a team.
    Looks for the team's latest game in features_v2.csv
    and returns their rolling stats.
    """
    # Try matching by team name (as home or away)
    home_games = features_df[features_df["home_team"] == team_name].sort_values("date", ascending=False)
    away_games = features_df[features_df["away_team"] == team_name].sort_values("date", ascending=False)

    team_features = {}

    # Get rolling stats from most recent game (prefer most recent regardless of home/away)
    latest_home_date = home_games["date"].iloc[0] if len(home_games) > 0 else ""
    latest_away_date = away_games["date"].iloc[0] if len(away_games) > 0 else ""

    if latest_home_date >= latest_away_date and len(home_games) > 0:
        row = home_games.iloc[0]
        # When team was home, their stats are in home_ columns
        for col in features_df.columns:
            if col.startswith("home_roll_") or col.startswith("home_season_"):
                key = col.replace("home_", "")
                team_features[key] = row[col]
        team_features["elo"] = row.get("home_elo_pre", 1500)
        team_features["rest_days"] = row.get("home_rest_days", 3)
    elif len(away_games) > 0:
        row = away_games.iloc[0]
        # When team was away, their stats are in away_ columns
        for col in features_df.columns:
            if col.startswith("away_roll_") or col.startswith("away_season_"):
                key = col.replace("away_", "")
                team_features[key] = row[col]
        team_features["elo"] = row.get("away_elo_pre", 1500)
        team_features["rest_days"] = row.get("away_rest_days", 3)

    return team_features


def predict_matchup(model_dict, features_df, feature_cols, home_team, away_team,
                    home_id="", away_id="", neutral=False):
    """
    Predict P(home_win) for a matchup using current team features.
    """
    home_feats = get_team_latest_features(features_df, home_team, home_id)
    away_feats = get_team_latest_features(features_df, away_team, away_id)

    if not home_feats or not away_feats:
        return None, "Missing features"

    home_elo = home_feats.get("elo", 1500)
    away_elo = away_feats.get("elo", 1500)

    # Build feature dict matching the model's expected columns
    game_features = {
        "elo_diff": home_elo - away_elo,
        "elo_sum": home_elo + away_elo,
        "is_neutral": 1 if neutral else 0,
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "rest_diff": home_feats.get("rest_days", 3) - away_feats.get("rest_days", 3),
        "home_rest_days": home_feats.get("rest_days", 3),
        "away_rest_days": away_feats.get("rest_days", 3),
        "season_progress": 0.85,  # late season
    }

    # Rolling diffs (short window)
    rolling_metrics = [
        "oe", "de", "net_eff", "efg", "opp_efg", "to_rate",
        "forced_to_rate", "orb_pct", "drb_pct", "ft_rate", "tpr",
        "ast_rate", "tempo",
    ]

    for metric in rolling_metrics:
        home_val = home_feats.get("roll_{}".format(metric), np.nan)
        away_val = away_feats.get("roll_{}".format(metric), np.nan)
        game_features["roll_{}_diff".format(metric)] = home_val - away_val if pd.notna(home_val) and pd.notna(away_val) else 0

    # Season diffs (full season window)
    for metric in rolling_metrics:
        home_val = home_feats.get("season_{}".format(metric), np.nan)
        away_val = away_feats.get("season_{}".format(metric), np.nan)
        game_features["season_{}_diff".format(metric)] = home_val - away_val if pd.notna(home_val) and pd.notna(away_val) else 0

    # Predict
    try:
        model = model_dict["model"]
        scaler = model_dict.get("scaler")
        X = pd.DataFrame([game_features])[feature_cols]

        if scaler:
            X = scaler.transform(X)

        prob = model.predict_proba(X)[0, 1]
        return round(float(prob), 4), None
    except Exception as e:
        return None, str(e)


# -------------------------------------------------------------------
# Display helpers
# -------------------------------------------------------------------
def format_prob(prob):
    """Format probability as percentage."""
    if prob is None:
        return "N/A"
    return "{:.1f}%".format(prob * 100)


def format_ml(ml):
    """Format moneyline."""
    if ml is None:
        return "N/A"
    ml = float(ml)
    if ml > 0:
        return "+{}".format(int(ml))
    return str(int(ml))


def edge_color(edge):
    """Return color based on edge magnitude."""
    if edge is None:
        return "gray"
    if edge > 0.05:
        return "green"
    elif edge > 0.02:
        return "orange"
    else:
        return "red"


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CBB Betting Model",
        page_icon="🏀",
        layout="wide",
    )

    st.title("🏀 CBB Betting Model")
    st.caption("Logistic Regression + Rolling Efficiency Metrics | Live Predictions")

    # Load model and data
    try:
        model_dict = load_model()
        features_df = load_features()
        feature_cols = load_feature_cols()
        elo_ratings = load_elo_ratings()
    except Exception as e:
        st.error("Error loading model/data: {}. Run the pipeline first.".format(e))
        return

    # Sidebar
    st.sidebar.header("Settings")

    # Date picker
    today = datetime.now()
    selected_date = st.sidebar.date_input(
        "Game Date",
        value=today,
        min_value=today - timedelta(days=30),
        max_value=today + timedelta(days=7),
    )
    date_str = selected_date.strftime("%Y%m%d")

    # Edge threshold
    min_edge = st.sidebar.slider("Minimum Edge (%)", 0, 15, 3, 1) / 100
    # Bankroll management (persistent)
    CONFIG_FILE = Path("data/config.json")
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    else:
        config = {"total_deposited": 0}
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)

    # Calculate current bankroll from deposits + P&L - pending bets
    bets_for_bankroll = load_bets()
    settled_profit = 0
    pending_count = 0
    pending_amount = 0
    if not bets_for_bankroll.empty and "profit" in bets_for_bankroll.columns:
        settled_profit = bets_for_bankroll["profit"].dropna().sum()
        pending_count = bets_for_bankroll["profit"].isna().sum()
        pending_bets = bets_for_bankroll[bets_for_bankroll["profit"].isna()]
        pending_amount = pending_bets["bet_amount"].sum() if not pending_bets.empty else 0
    bankroll = config["total_deposited"] + settled_profit - pending_amount

    # Display
    st.sidebar.header("Bankroll")
    st.sidebar.metric("Current Bankroll", "${:,.2f}".format(bankroll),
                       delta="${:+,.2f}".format(settled_profit) if settled_profit != 0 else None)
    st.sidebar.caption("Deposited: ${:,.0f} | P&L: ${:+,.2f}".format(
        config["total_deposited"], settled_profit))
    if pending_count > 0:
        st.sidebar.caption("{} bets pending".format(pending_count))

    # Deposit funds
    add_funds = st.sidebar.number_input("Add Funds ($)", value=0, step=50, min_value=0)
    if add_funds > 0 and st.sidebar.button("Deposit"):
        config["total_deposited"] += add_funds
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
        st.rerun()

    kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Tabs
    tab_today, tab_lookup, tab_rankings, tab_pnl, tab_backtest = st.tabs([
        "📅 Today's Games", "🔍 Matchup Lookup", "📊 ELO Rankings", "💰 P&L Tracker", "📈 Backtest"
    ])

    # =================================================================
    # TAB 1: Today's Games
    # =================================================================
    with tab_today:
        st.header("Games for {}".format(selected_date.strftime("%B %d, %Y")))

        scoreboard = fetch_scoreboard(date_str)
        if not scoreboard:
            st.warning("Could not fetch scoreboard.")
            return

        games = parse_scoreboard(scoreboard)

        if not games:
            st.info("No games scheduled for this date.")
            return

        st.write("**{} games found**".format(len(games)))

        for game in games:
            with st.container():
                st.divider()

                # Predict
                prob, error = predict_matchup(
                    model_dict, features_df, feature_cols,
                    game["home_team"], game["away_team"],
                    game["home_id"], game["away_id"],
                    game["neutral_site"],
                )

                # Fetch odds
                odds_items = fetch_odds(game["game_id"])

                # Parse market odds
                market_home_prob = None
                market_ml_home = None
                market_ml_away = None
                market_spread = None
                provider_name = None

                if odds_items:
                    item = odds_items[0]
                    provider_name = item.get("provider", {}).get("name", "")
                    market_ml_home = item.get("homeTeamOdds", {}).get("moneyLine")
                    market_ml_away = item.get("awayTeamOdds", {}).get("moneyLine")
                    market_spread = item.get("spread")

                    home_imp = moneyline_to_implied(market_ml_home)
                    away_imp = moneyline_to_implied(market_ml_away)
                    if home_imp and away_imp:
                        total = home_imp + away_imp
                        market_home_prob = home_imp / total

                # Compute edge
                edge_home = None
                edge_away = None
                if prob is not None and market_home_prob is not None:
                    edge_home = prob - market_home_prob
                    edge_away = (1 - prob) - (1 - market_home_prob)

                # Layout
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                with col1:
                    if game["completed"]:
                        st.write("**{} {} - {} {}**  ✅".format(
                            game["away_abbr"], game["away_score"],
                            game["home_score"], game["home_abbr"],
                        ))
                    elif game["status_name"] == "STATUS_SCHEDULED":
                        st.write("**{} @ {}**  ⏰ {}".format(
                            game["away_abbr"], game["home_abbr"],
                            game["status"],
                        ))
                    else:
                        st.write("**{} {} - {} {}**  🔴 LIVE".format(
                            game["away_abbr"], game["away_score"],
                            game["home_score"], game["home_abbr"],
                        ))

                    if game["neutral_site"]:
                        st.caption("📍 Neutral site")

                with col2:
                    st.write("**Model**")
                    if prob is not None:
                        st.write("{}: {}".format(game["home_abbr"], format_prob(prob)))
                        st.write("{}: {}".format(game["away_abbr"], format_prob(1 - prob)))
                    elif error:
                        st.write("⚠️ {}".format(error))

                with col3:
                    st.write("**Market**")
                    if market_home_prob is not None:
                        st.write("{}: {} ({})".format(
                            game["home_abbr"],
                            format_prob(market_home_prob),
                            format_ml(market_ml_home),
                        ))
                        st.write("{}: {} ({})".format(
                            game["away_abbr"],
                            format_prob(1 - market_home_prob),
                            format_ml(market_ml_away),
                        ))
                        if market_spread is not None:
                            st.caption("Spread: {}".format(market_spread))
                        if provider_name:
                            st.caption("via {}".format(provider_name))
                    else:
                        st.write("No odds available")

                with col4:
                    st.write("**Edge / Bet**")
                    if edge_home is not None:
                        has_home_edge = edge_home > min_edge
                        has_away_edge = edge_away > min_edge

                        if has_home_edge:
                            k_pct, dec_odds = kelly_sizing(prob, market_home_prob, market_ml_home, kelly_fraction)
                            bet_amt = round(bankroll * k_pct, 2)
                            st.write("Edge: {:+.1f}% {}".format(edge_home * 100, game["home_abbr"]))
                            if bet_amt > 0:
                                st.write("${:.2f} ({:.1f}%)".format(bet_amt, k_pct * 100))

                                existing = load_bets()
                                already_bet = False
                                if not existing.empty:
                                    already_bet = ((existing["game_id"].astype(str) == str(game["game_id"])) &
                                                   (existing["bet_side"] == "home")).any()

                                if already_bet:
                                    st.write("✅ Bet placed")
                                    if st.button("Remove", key="rm_home_{}".format(game["game_id"])):
                                        remove_bet(game["game_id"], "home")
                                        st.rerun()
                                elif not game["completed"]:
                                    if st.button("Place Bet", key="bet_home_{}".format(game["game_id"]), type="primary"):
                                        save_bet(game["game_id"], game["date"],
                                                 game["home_team"], game["away_team"],
                                                 "home", bet_amt, prob,
                                                 market_home_prob, dec_odds)
                                        st.rerun()

                        elif has_away_edge:
                            k_pct, dec_odds = kelly_sizing(1 - prob, 1 - market_home_prob, market_ml_away, kelly_fraction)
                            bet_amt = round(bankroll * k_pct, 2)
                            st.write("Edge: {:+.1f}% {}".format(edge_away * 100, game["away_abbr"]))
                            if bet_amt > 0:
                                st.write("${:.2f} ({:.1f}%)".format(bet_amt, k_pct * 100))

                                existing = load_bets()
                                already_bet = False
                                if not existing.empty:
                                    already_bet = ((existing["game_id"].astype(str) == str(game["game_id"])) &
                                                   (existing["bet_side"] == "away")).any()

                                if already_bet:
                                    st.write("✅ Bet placed")
                                    if st.button("Remove", key="rm_away_{}".format(game["game_id"])):
                                        remove_bet(game["game_id"], "away")
                                        st.rerun()
                                elif not game["completed"]:
                                    if st.button("Place Bet", key="bet_away_{}".format(game["game_id"]), type="primary"):
                                        save_bet(game["game_id"], game["date"],
                                                 game["home_team"], game["away_team"],
                                                 "away", bet_amt, 1 - prob,
                                                 1 - market_home_prob, dec_odds)
                                        st.rerun()
                        else:
                            st.write("No edge")
                    else:
                        st.write("—")

    # =================================================================
    # TAB 2: Matchup Lookup
    # =================================================================
    with tab_lookup:
        st.header("Custom Matchup Prediction")

        # Get all team names
        all_teams = sorted(set(
            features_df["home_team"].unique().tolist() +
            features_df["away_team"].unique().tolist()
        ))

        # Filter to 2026 season teams
        season_2026 = features_df[features_df["season"] == 2026]
        current_teams = sorted(set(
            season_2026["home_team"].unique().tolist() +
            season_2026["away_team"].unique().tolist()
        ))

        if not current_teams:
            current_teams = all_teams

        col1, col2 = st.columns(2)
        with col1:
            away_team = st.selectbox("Away Team", current_teams, index=0)
        with col2:
            home_team = st.selectbox("Home Team", current_teams, index=min(1, len(current_teams)-1))

        neutral = st.checkbox("Neutral Site")

        if st.button("Predict", type="primary"):
            prob, error = predict_matchup(
                model_dict, features_df, feature_cols,
                home_team, away_team,
                neutral=neutral,
            )

            if prob is not None:
                st.write("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(home_team, format_prob(prob))
                with col2:
                    st.metric(away_team, format_prob(1 - prob))

                # Show feature breakdown
                home_feats = get_team_latest_features(features_df, home_team, "")
                away_feats = get_team_latest_features(features_df, away_team, "")

                st.write("---")
                st.subheader("Team Comparison")

                comparison = {
                    "Metric": [],
                    home_team: [],
                    away_team: [],
                    "Advantage": [],
                }

                metrics = [
                    ("ELO Rating", "elo", True),
                    ("Off. Efficiency", "roll_oe", True),
                    ("Def. Efficiency", "roll_de", False),
                    ("eFG%", "roll_efg", True),
                    ("Turnover Rate", "roll_to_rate", False),
                    ("Off. Reb %", "roll_orb_pct", True),
                    ("Def. Reb %", "roll_drb_pct", True),
                    ("FT Rate", "roll_ft_rate", True),
                    ("Tempo", "roll_tempo", None),
                ]

                for label, key, higher_better in metrics:
                    h_val = home_feats.get(key, np.nan)
                    a_val = away_feats.get(key, np.nan)

                    comparison["Metric"].append(label)
                    comparison[home_team].append(round(h_val, 2) if pd.notna(h_val) else "N/A")
                    comparison[away_team].append(round(a_val, 2) if pd.notna(a_val) else "N/A")

                    if pd.notna(h_val) and pd.notna(a_val) and higher_better is not None:
                        if higher_better:
                            comparison["Advantage"].append(home_team if h_val > a_val else away_team)
                        else:
                            comparison["Advantage"].append(home_team if h_val < a_val else away_team)
                    else:
                        comparison["Advantage"].append("—")

                st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

            elif error:
                st.error("Prediction failed: {}".format(error))

    # =================================================================
    # TAB 3: ELO Rankings
    # =================================================================
    with tab_rankings:
        st.header("Current ELO Rankings")

        elo_data = elo_ratings.get("ratings", elo_ratings)
        if isinstance(elo_data, dict):
            # Sort by rating
            sorted_teams = sorted(elo_data.items(), key=lambda x: x[1], reverse=True)

            n_show = st.slider("Show top N teams", 10, 100, 25)

            rankings = []
            for i, (team, rating) in enumerate(sorted_teams[:n_show]):
                rankings.append({
                    "Rank": i + 1,
                    "Team": team,
                    "ELO": round(rating, 1),
                    "vs Average": round(rating - 1500, 1),
                })

            st.dataframe(pd.DataFrame(rankings), use_container_width=True, hide_index=True)

    # =================================================================
    # TAB 4: P&L Tracker
    # =================================================================
    with tab_pnl:
        st.header("P&L Tracker")

        # Manual bet entry
        with st.expander("Add Bet Manually"):
            st.caption("Log a bet you placed but didn't track in the dashboard.")
            
            manual_col1, manual_col2 = st.columns(2)
            with manual_col1:
                manual_date = st.date_input("Game Date", key="manual_date")
                manual_date_str = manual_date.strftime("%Y%m%d")
            with manual_col2:
                manual_side = st.selectbox("Bet Side", ["home", "away"], key="manual_side")

            # Fetch games for that date
            manual_scoreboard = fetch_scoreboard(manual_date_str)
            if manual_scoreboard:
                manual_games = parse_scoreboard(manual_scoreboard)
                game_options = {
                    "{} @ {} ({})".format(g["away_abbr"], g["home_abbr"], g["status"]): g
                    for g in manual_games
                }
                if game_options:
                    selected_game_label = st.selectbox("Game", list(game_options.keys()), key="manual_game")
                    selected_game = game_options[selected_game_label]

                    manual_col3, manual_col4, manual_col5 = st.columns(3)
                    with manual_col3:
                        manual_amount = st.number_input("Bet Amount ($)", value=100, min_value=1, key="manual_amt")
                    with manual_col4:
                        manual_ml = st.number_input("Moneyline", value=-110, key="manual_ml")
                    with manual_col5:
                        ml_val = float(manual_ml)
                        if ml_val < 0:
                            manual_decimal = 1 + (100 / abs(ml_val))
                        else:
                            manual_decimal = 1 + (ml_val / 100)
                        st.metric("Decimal Odds", "{:.3f}".format(manual_decimal))

                    if st.button("Log Bet", key="manual_log", type="primary"):
                        # Compute market prob from moneyline
                        if ml_val < 0:
                            market_prob = abs(ml_val) / (abs(ml_val) + 100)
                        else:
                            market_prob = 100 / (ml_val + 100)

                        saved = save_bet(
                            selected_game["game_id"],
                            selected_game["date"],
                            selected_game["home_team"],
                            selected_game["away_team"],
                            manual_side,
                            manual_amount,
                            0.0,  # no model prob for manual bets
                            market_prob,
                            manual_decimal,
                        )
                        if saved:
                            st.success("Bet logged!")
                            st.rerun()
                        else:
                            st.warning("Bet already exists for this game/side.")
                else:
                    st.warning("No games found for this date.")

        bets = settle_bets()

        if bets.empty:
            st.info("No bets placed yet. Browse today's games and click 'Place Bet' on games with edges.")
        else:
            settled = bets[bets["profit"].notna()].copy()
            pending = bets[bets["profit"].isna()].copy()

            # Summary metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                total_profit = settled["profit"].sum() if not settled.empty else 0
                current = bankroll
                st.metric("Bankroll", "${:,.0f}".format(current),
                          delta="${:+,.0f}".format(total_profit))

            with col2:
                wins = (settled["result"] == "W").sum() if not settled.empty else 0
                losses = (settled["result"] == "L").sum() if not settled.empty else 0
                st.metric("Record", "{}-{}".format(wins, losses))

            with col3:
                win_rate = wins / len(settled) if not settled.empty and len(settled) > 0 else 0
                st.metric("Win Rate", "{:.1%}".format(win_rate))

            with col4:
                total_wagered = settled["bet_amount"].sum() if not settled.empty else 0
                roi = total_profit / total_wagered if total_wagered > 0 else 0
                st.metric("ROI", "{:.1%}".format(roi))

            with col5:
                st.metric("Pending", "{}".format(len(pending)))

            with col6:
                st.metric("P&L", "{:+,.2f}".format(total_profit))

            # Cumulative P&L chart
            if not settled.empty:
                settled = settled.sort_values("date").reset_index(drop=True)
                settled["cumulative_pnl"] = settled["profit"].cumsum()
                st.subheader("Cumulative P&L")
                st.line_chart(settled["cumulative_pnl"])

            # Pending bets
            if not pending.empty:
                total_locked = pending["bet_amount"].sum()
                st.subheader("Pending Bets — ${:,.2f} locked".format(total_locked))
                display_pending = pending[["date", "home_team", "away_team",
                                           "bet_side", "bet_amount", "model_prob",
                                           "market_prob", "decimal_odds"]].copy()
                display_pending["potential_profit"] = display_pending.apply(
                    lambda r: r["bet_amount"] * (r["decimal_odds"] - 1), axis=1)
                display_pending["bet_amount"] = display_pending["bet_amount"].apply(
                    lambda x: "${:.2f}".format(x))
                display_pending["potential_profit"] = display_pending["potential_profit"].apply(
                    lambda x: "+${:.2f}".format(x))
                display_pending["model_prob"] = display_pending["model_prob"].apply(
                    lambda x: "{:.1%}".format(x))
                display_pending["market_prob"] = display_pending["market_prob"].apply(
                    lambda x: "{:.1%}".format(x))
                display_pending = display_pending.drop(columns=["decimal_odds"])
                display_pending = display_pending.sort_values("date", ascending=False)
                st.dataframe(display_pending, use_container_width=True, hide_index=True)

            # Settled bets
            if not settled.empty:
                st.subheader("Settled Bets")
                display_settled = settled[["date", "home_team", "away_team",
                                           "bet_side", "bet_amount", "result",
                                           "profit", "cumulative_pnl"]].copy()
                display_settled = display_settled.sort_values("date", ascending=False)
                display_settled["bet_amount"] = display_settled["bet_amount"].apply(
                    lambda x: "${:.0f}".format(x))
                display_settled["profit"] = display_settled["profit"].apply(
                    lambda x: "${:+,.0f}".format(x))
                display_settled["cumulative_pnl"] = display_settled["cumulative_pnl"].apply(
                    lambda x: "${:+,.0f}".format(x))
                st.dataframe(display_settled, use_container_width=True, hide_index=True)

            # Clear all bets button
            st.divider()
            if st.button("🗑️ Clear All Bets", type="secondary"):
                if BETS_FILE.exists():
                    BETS_FILE.unlink()
                st.rerun()

    # =================================================================
    # TAB 5: Backtest (placeholder)
    # =================================================================
    with tab_backtest:
        st.header("Backtest Results")
        st.info("Run `python pull_odds.py --season 2025` then `python main.py --step backtest` to populate this tab.")

        # Check if backtest results exist
        backtest_path = PROCESSED_DIR / "backtest_results.csv"
        if backtest_path.exists():
            results = pd.read_csv(backtest_path)
            st.write("**{} bets tracked**".format(len(results)))

            col1, col2, col3 = st.columns(3)
            with col1:
                wins = results["won"].sum() if "won" in results.columns else 0
                total = len(results)
                st.metric("Win Rate", "{:.1%}".format(wins / total if total > 0 else 0))
            with col2:
                profit = results["profit"].sum() if "profit" in results.columns else 0
                st.metric("Total Profit", "${:.2f}".format(profit))
            with col3:
                roi = profit / results["bet_size"].sum() if "bet_size" in results.columns and results["bet_size"].sum() > 0 else 0
                st.metric("ROI", "{:.1%}".format(roi))

            st.dataframe(results.tail(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
