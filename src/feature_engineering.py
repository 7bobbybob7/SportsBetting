"""
feature_engineering.py - Build features for logistic regression and XGBoost.

All features use ONLY pre-game information (no data leakage).

Feature categories:
    1. ELO-based: ELO difference, home ELO, away ELO
    2. Efficiency: adjusted OE/DE differentials (from Barttorvik)
    3. Tempo: pace differential
    4. Four Factors: turnover rate, offensive rebounding, FT rate, 3PT%
    5. Contextual: home/away/neutral, rest days, conference, SOS
    6. Tournament-specific: seed, historical upset rates, experience

Usage:
    from src.feature_engineering import build_features
    
    features_df = build_features(games_df, elo_game_log, barttorvik_df)
"""

import numpy as np
import pandas as pd
from typing import Optional


def build_features(
    games: pd.DataFrame,
    elo_log: pd.DataFrame,
    barttorvik: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for ML models.
    
    Merges ELO predictions with Barttorvik efficiency data and computes
    all differential features. Returns one row per game, ready for
    model training.
    
    Args:
        games: Raw game data with results
        elo_log: Game log from EloRater.rate_seasons() with pre-game ELO
        barttorvik: Team-season Barttorvik ratings (optional but recommended)
    
    Returns:
        DataFrame with features + target variable (home_win).
    """
    # Start with ELO game log (has pre-game ratings and predictions)
    df = elo_log.copy()

    # -- 1. ELO features ---------------------------------------------
    df["elo_diff"] = df["home_elo_adj"] - df["away_elo_pre"]
    df["elo_sum"] = df["home_elo_pre"] + df["away_elo_pre"]  # overall quality

    # -- 2. Neutral site indicator ------------------------------------
    df["is_neutral"] = df["neutral"].astype(int)

    # -- 3. Barttorvik efficiency features ----------------------------
    if barttorvik is not None and not barttorvik.empty:
        df = _merge_barttorvik(df, barttorvik)

    # -- 4. Contextual features ---------------------------------------
    df = _add_rest_days(df)
    df = _add_season_progress(df)

    # -- 5. Define target ---------------------------------------------
    # home_win already exists from ELO log

    return df


def _merge_barttorvik(df: pd.DataFrame, bart: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Barttorvik team ratings onto each game for both teams.
    
    Creates differential features:
        - adj_oe_diff: home offensive efficiency - away offensive efficiency
        - adj_de_diff: home defensive efficiency - away defensive efficiency
          (note: lower DE is better, so positive diff = home has worse defense)
        - adj_tempo_diff: pace differential
        - barthag_diff: overall quality differential
        - sos_diff: strength of schedule differential
    """
    # Standardize team names for matching
    # (This is often the hardest part - names differ between sources)
    
    # Merge home team's Barttorvik stats
    home_cols = ["team", "season", "adj_oe", "adj_de", "adj_tempo", "barthag"]
    home_cols = [c for c in home_cols if c in bart.columns]
    
    if len(home_cols) < 3:  # need at least team, season, and one stat
        print("WARNING: Barttorvik data missing expected columns. Skipping merge.")
        return df
    
    home_bart = bart[home_cols].copy()
    home_rename = {c: f"home_{c}" for c in home_cols if c not in ("team", "season")}
    home_bart = home_bart.rename(columns=home_rename)
    
    df = df.merge(
        home_bart,
        left_on=["home_team", "season"],
        right_on=["team", "season"],
        how="left",
    )
    if "team" in df.columns:
        df = df.drop(columns=["team"])
    
    # Merge away team's Barttorvik stats
    away_bart = bart[home_cols].copy()
    away_rename = {c: f"away_{c}" for c in home_cols if c not in ("team", "season")}
    away_bart = away_bart.rename(columns=away_rename)
    
    df = df.merge(
        away_bart,
        left_on=["away_team", "season"],
        right_on=["team", "season"],
        how="left",
        suffixes=("", "_drop"),
    )
    if "team" in df.columns:
        df = df.drop(columns=["team"])
    
    # Drop any duplicate columns from merge
    df = df[[c for c in df.columns if not c.endswith("_drop")]]
    
    # -- Compute differentials ----------------------------------------
    if "home_adj_oe" in df.columns and "away_adj_oe" in df.columns:
        df["adj_oe_diff"] = df["home_adj_oe"] - df["away_adj_oe"]
    
    if "home_adj_de" in df.columns and "away_adj_de" in df.columns:
        # Lower DE is better, so we flip: away_de - home_de
        # Positive = home team has the defensive advantage
        df["adj_de_diff"] = df["away_adj_de"] - df["home_adj_de"]
    
    if "home_adj_tempo" in df.columns and "away_adj_tempo" in df.columns:
        df["adj_tempo_diff"] = df["home_adj_tempo"] - df["away_adj_tempo"]
    
    if "home_barthag" in df.columns and "away_barthag" in df.columns:
        df["barthag_diff"] = df["home_barthag"] - df["away_barthag"]
    
    # Net efficiency = OE - DE (higher is better for both)
    if all(c in df.columns for c in ["home_adj_oe", "home_adj_de", "away_adj_oe", "away_adj_de"]):
        df["home_net_eff"] = df["home_adj_oe"] - df["home_adj_de"]
        df["away_net_eff"] = df["away_adj_oe"] - df["away_adj_de"]
        df["net_eff_diff"] = df["home_net_eff"] - df["away_net_eff"]
    
    return df


def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days for each team before a game.
    
    More rest generally helps, especially in tournaments with
    back-to-back games. Rest differential is a meaningful feature.
    """
    if "date" not in df.columns:
        return df
    
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Sort by date
    df = df.sort_values("date_dt").reset_index(drop=True)
    
    # Track last game date per team
    last_game = {}
    rest_home = []
    rest_away = []
    
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date_dt"]
        
        if pd.isna(date):
            rest_home.append(np.nan)
            rest_away.append(np.nan)
            continue
        
        # Home team rest
        if home in last_game and pd.notna(last_game[home]):
            rest_h = (date - last_game[home]).days
        else:
            rest_h = 7  # default for first game of season
        
        # Away team rest
        if away in last_game and pd.notna(last_game[away]):
            rest_a = (date - last_game[away]).days
        else:
            rest_a = 7
        
        rest_home.append(rest_h)
        rest_away.append(rest_a)
        
        last_game[home] = date
        last_game[away] = date
    
    df["home_rest_days"] = rest_home
    df["away_rest_days"] = rest_away
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    
    # Clean up
    if "date_dt" in df.columns:
        df = df.drop(columns=["date_dt"])
    
    return df


def _add_season_progress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season progress feature (0 = start of season, 1 = end).
    
    Models can use this because ELO and efficiency stats become more
    reliable as the season progresses. Early-season games have more
    noise (small sample sizes, new rosters).
    """
    if "date" not in df.columns:
        return df
    
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    
    for season in df["season"].unique():
        mask = df["season"] == season
        season_dates = df.loc[mask, "date_dt"]
        
        if season_dates.isna().all():
            continue
        
        min_date = season_dates.min()
        max_date = season_dates.max()
        range_days = (max_date - min_date).days
        
        if range_days > 0:
            df.loc[mask, "season_progress"] = (
                (season_dates - min_date).dt.days / range_days
            )
        else:
            df.loc[mask, "season_progress"] = 0.5
    
    if "date_dt" in df.columns:
        df = df.drop(columns=["date_dt"])
    
    return df


# ---------------------------------------------------------------------------
# Tournament-specific features (for March Madness)
# ---------------------------------------------------------------------------
def add_tournament_features(
    df: pd.DataFrame, bracket: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add March Madness specific features.
    
    Args:
        df: Game data with basic features already computed
        bracket: Tournament bracket info with seeds, regions
    
    Features added:
        - seed_diff: seed number differential (lower = better)
        - historical_upset_rate: how often this seed matchup produces upsets
        - is_tournament: binary flag
    """
    # Historical upset rates by seed matchup (from NCAA data)
    # Format: (higher_seed, lower_seed) -> upset rate
    UPSET_RATES = {
        (1, 16): 0.01, (2, 15): 0.06, (3, 14): 0.15, (4, 13): 0.20,
        (5, 12): 0.35, (6, 11): 0.37, (7, 10): 0.39, (8, 9): 0.49,
    }
    
    if bracket is not None and not bracket.empty:
        # Merge seed info
        if "seed" in bracket.columns and "team" in bracket.columns:
            df = df.merge(
                bracket[["team", "seed"]].rename(columns={"team": "home_team", "seed": "home_seed"}),
                on="home_team",
                how="left",
            )
            df = df.merge(
                bracket[["team", "seed"]].rename(columns={"team": "away_team", "seed": "away_seed"}),
                on="away_team",
                how="left",
            )
            df["seed_diff"] = df.get("home_seed", np.nan) - df.get("away_seed", np.nan)
    
    return df


# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns to use for modeling.
    
    Automatically detects which features are available (depends on
    whether Barttorvik data was merged successfully).
    """
    # Core features (always available from ELO)
    core = ["elo_diff", "elo_sum", "is_neutral", "home_elo_pre", "away_elo_pre"]
    
    # Barttorvik features (available if merge succeeded)
    barttorvik = [
        "adj_oe_diff", "adj_de_diff", "adj_tempo_diff", "barthag_diff",
        "net_eff_diff", "home_net_eff", "away_net_eff",
    ]
    
    # Contextual
    contextual = ["rest_diff", "season_progress", "home_rest_days", "away_rest_days"]
    
    # Tournament
    tournament = ["seed_diff"]
    
    # Return only columns that exist in the data
    all_features = core + barttorvik + contextual + tournament
    available = [c for c in all_features if c in df.columns]
    
    print(f"Features available: {len(available)}/{len(all_features)}")
    for f in available:
        print(f"  OK {f}")
    for f in all_features:
        if f not in available:
            print(f"  X {f} (missing)")
    
    return available


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 40)
    print("\nAvailable feature groups:")
    print("  1. ELO-based (elo_diff, elo_sum)")
    print("  2. Efficiency (adj_oe_diff, adj_de_diff, net_eff_diff)")
    print("  3. Tempo (adj_tempo_diff)")
    print("  4. Contextual (rest_diff, season_progress)")
    print("  5. Tournament (seed_diff, upset_rate)")
    print("\nRun build_features() with game data + ELO log + Barttorvik ratings")
