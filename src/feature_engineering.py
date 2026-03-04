"""
feature_engineering.py - Build features for logistic regression and XGBoost.

All features use ONLY pre-game information (no data leakage).

Feature categories:
    1. ELO-based: ELO difference, home ELO, away ELO
    2. Efficiency: adjusted OE/DE differentials (from Barttorvik)
    3. Tempo: pace differential
    4. Contextual: home/away/neutral, rest days, season progress
    5. Tournament-specific: seed, historical upset rates

Usage:
    from src.feature_engineering import build_features

    features_df = build_features(games_df, elo_game_log, barttorvik_df)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# ESPN -> Barttorvik team name mapping
# ---------------------------------------------------------------------------
# ESPN uses full names like "Connecticut Huskies", Barttorvik uses "Connecticut"
# This builds a fuzzy matcher plus a manual override table for tricky cases.

MANUAL_NAME_MAP = {
    # ESPN name -> Barttorvik name
    "UConn Huskies": "Connecticut",
    "Connecticut Huskies": "Connecticut",
    "LSU Tigers": "LSU",
    "UNLV Rebels": "UNLV",
    "VCU Rams": "VCU",
    "UCF Knights": "UCF",
    "USC Trojans": "USC",
    "SMU Mustangs": "SMU",
    "TCU Horned Frogs": "TCU",
    "BYU Cougars": "BYU",
    "UNC Greensboro Spartans": "UNC Greensboro",
    "UNC Asheville Bulldogs": "UNC Asheville",
    "UNC Wilmington Seahawks": "UNC Wilmington",
    "UTEP Miners": "UTEP",
    "UTSA Roadrunners": "UTSA",
    "UT Arlington Mavericks": "UT Arlington",
    "UT Martin Skyhawks": "UT Martin",
    "Miami Hurricanes": "Miami FL",
    "Miami (OH) RedHawks": "Miami OH",
    "Saint Mary's Gaels": "Saint Mary's",
    "St. John's Red Storm": "St. John's",
    "Saint Joseph's Hawks": "Saint Joseph's",
    "Saint Louis Billikens": "Saint Louis",
    "Saint Peter's Peacocks": "Saint Peter's",
    "Ole Miss Rebels": "Mississippi",
    "Pitt Panthers": "Pittsburgh",
    "UMass Minutemen": "Massachusetts",
    "UMass Lowell River Hawks": "UMass Lowell",
    "USC Upstate Spartans": "USC Upstate",
    "Detroit Mercy Titans": "Detroit Mercy",
    "Loyola Chicago Ramblers": "Loyola Chicago",
    "Loyola Marymount Lions": "Loyola Marymount",
    "Loyola (MD) Greyhounds": "Loyola MD",
    "LIU Sharks": "LIU",
    "FDU Knights": "Fairleigh Dickinson",
    "SIU Edwardsville Cougars": "SIU Edwardsville",
    "SIUE Cougars": "SIU Edwardsville",
    "Southern Illinois Salukis": "Southern Illinois",
    "Green Bay Phoenix": "Green Bay",
    "Little Rock Trojans": "Little Rock",
    "IUPUI Jaguars": "IUPUI",
    "Purdue Fort Wayne Mastodons": "Purdue Fort Wayne",
    "Texas A&M-CC Islanders": "Texas A&M Corpus Chris",
    "Texas A&M-Corpus Christi Islanders": "Texas A&M Corpus Chris",
    "Hawai'i Rainbow Warriors": "Hawaii",
    "NC State Wolfpack": "N.C. State",
    "Penn Quakers": "Pennsylvania",
    "Army Black Knights": "Army",
    "Navy Midshipmen": "Navy",
    "NIU Huskies": "Northern Illinois",
    "SFA Lumberjacks": "Stephen F. Austin",
    "Stephen F. Austin Lumberjacks": "Stephen F. Austin",
    "UIC Flames": "Illinois Chicago",
    "NJIT Highlanders": "NJIT",
}


def _normalize_name(name):
    """Strip common mascot suffixes to get base team name."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    # Common mascot words to strip
    mascots = [
        "Wildcats", "Tigers", "Bears", "Bulldogs", "Eagles", "Hawks",
        "Panthers", "Lions", "Warriors", "Knights", "Cougars", "Mustangs",
        "Huskies", "Rams", "Rebels", "Aggies", "Longhorns", "Wolverines",
        "Spartans", "Buckeyes", "Badgers", "Hoosiers", "Boilermakers",
        "Hawkeyes", "Cyclones", "Jayhawks", "Sooners", "Cowboys", "Miners",
        "Hokies", "Cavaliers", "Terrapins", "Nittany Lions", "Scarlet Knights",
        "Volunteers", "Commodores", "Crimson Tide", "Fighting Irish",
        "Blue Devils", "Tar Heels", "Wolfpack", "Demon Deacons",
        "Orange", "Cardinal", "Bruins", "Trojans", "Ducks", "Beavers",
        "Sun Devils", "Buffaloes", "Utes", "Coyotes", "Aztecs", "Falcons",
        "Broncos", "Lobos", "Mountaineers", "Red Raiders", "Horned Frogs",
        "Razorbacks", "Gamecocks", "Gators", "Seminoles", "Hurricanes",
        "Yellow Jackets", "Owls", "Thundering Herd", "Bobcats", "Rockets",
        "Zips", "Golden Flashes", "RedHawks", "Bearcats", "Musketeers",
        "Billikens", "Flyers", "Explorers", "Gaels", "Pilots", "Toreros",
        "Waves", "Dons", "Broncs", "Peacocks", "Red Storm", "Friars",
        "Bluejays", "Hoyas", "Pirates", "Shockers", "Salukis", "Braves",
        "Lumberjacks", "Phoenix", "Leathernecks", "Penguins", "Mastodons",
        "Kangaroos", "Flames", "Norse", "Grizzlies", "Skyhawks", "Governors",
        "Racers", "Colonels", "Hilltoppers", "Jaguars", "Blazers", "49ers",
        "Chanticleers", "Monarchs", "Dukes", "Tribe", "Spiders", "Keydets",
        "Paladins", "Catamounts", "Retrievers", "Seawolves", "Terriers",
        "Great Danes", "River Hawks", "Minutemen", "Bonnies", "Red Foxes",
        "Jaspers", "Stags", "Greyhounds", "Leopards", "Mountain Hawks",
        "Raiders", "Bison", "Ramblers", "Sharks", "Highlanders", "Anteaters",
        "Matadors", "Gauchos", "Highlanders", "Titans", "Roadrunners",
        "Islanders", "Seahawks", "Mavericks", "Mean Green", "Thunderbirds",
        "Rainbow Warriors", "Black Knights", "Midshipmen",
    ]
    # Sort longest first so "Nittany Lions" matches before "Lions"
    mascots.sort(key=len, reverse=True)
    for mascot in mascots:
        if name.endswith(" " + mascot):
            return name[: -(len(mascot) + 1)].strip()
    return name


def build_team_name_map(espn_names, bart_names):
    """
    Build a mapping from ESPN team names to Barttorvik team names.

    Strategy:
        1. Check manual override table
        2. Try stripping mascot and matching
        3. Try substring matching
    """
    mapping = {}
    bart_set = set(bart_names)
    bart_lower = {b.lower(): b for b in bart_names}

    for espn_name in espn_names:
        # 1. Manual override
        if espn_name in MANUAL_NAME_MAP:
            mapping[espn_name] = MANUAL_NAME_MAP[espn_name]
            continue

        # 2. Strip mascot
        base = _normalize_name(espn_name)
        if base in bart_set:
            mapping[espn_name] = base
            continue

        # 3. Case-insensitive match
        if base.lower() in bart_lower:
            mapping[espn_name] = bart_lower[base.lower()]
            continue

        # 3b. Try State <-> St. conversion
        if " State" in base:
            alt = base.replace(" State", " St.")
            if alt in bart_set:
                mapping[espn_name] = alt
                continue
            if alt.lower() in bart_lower:
                mapping[espn_name] = bart_lower[alt.lower()]
                continue
        if " St." in base:
            alt = base.replace(" St.", " State")
            if alt in bart_set:
                mapping[espn_name] = alt
                continue

        # 4. Check if Barttorvik name is substring of ESPN name
        matched = False
        for bart_name in bart_names:
            if bart_name.lower() in espn_name.lower() and len(bart_name) > 3:
                mapping[espn_name] = bart_name
                matched = True
                break

        if not matched:
            # Leave unmapped - will result in NaN features (handled gracefully)
            mapping[espn_name] = None

    return mapping


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_features(
    games,
    elo_log,
    barttorvik=None,
):
    """
    Build the full feature matrix for ML models.

    Merges ELO predictions with Barttorvik efficiency data and computes
    all differential features. Returns one row per game, ready for
    model training.
    """
    # Start with ELO game log (has pre-game ratings and predictions)
    df = elo_log.copy()

    # -- 1. ELO features --
    df["elo_diff"] = df["home_elo_adj"] - df["away_elo_pre"]
    df["elo_sum"] = df["home_elo_pre"] + df["away_elo_pre"]

    # -- 2. Neutral site indicator --
    df["is_neutral"] = df["neutral"].astype(int)

    # -- 3. Barttorvik efficiency features --
    if barttorvik is not None and not barttorvik.empty:
        df = _merge_barttorvik(df, barttorvik)

    # -- 4. Contextual features --
    df = _add_rest_days(df)
    df = _add_season_progress(df)

    return df


def _merge_barttorvik(df, bart):
    """
    Merge Barttorvik team ratings onto each game for both teams.

    Handles the column name mapping (Barttorvik uses AdjOE, AdjDE, etc.)
    and the team name mapping (ESPN full names vs Barttorvik short names).
    """
    # Standardize Barttorvik column names
    bart = bart.copy()
    col_map = {}
    for col in bart.columns:
        cl = col.strip().lower().replace(" ", "_").replace(".", "")
        col_map[col] = cl
    bart = bart.rename(columns=col_map)

    # Map known column names to our internal names
    rename = {
        "adjoe": "adj_oe",
        "adjde": "adj_de",
        "adj_t": "adj_tempo",
        "adjt": "adj_tempo",
        "barthag": "barthag",
        "team": "team",
        "season": "season",
    }
    for old, new in rename.items():
        if old in bart.columns and old != new:
            bart = bart.rename(columns={old: new})

    # Force numeric types (Google Sheets copy-paste puts rank on second line)
    # e.g. "112.1\n56" -> 112.1
    for col in ["adj_oe", "adj_de", "adj_tempo", "barthag"]:
        if col in bart.columns:
            bart[col] = bart[col].astype(str).str.split("\n").str[0].str.strip()
            bart[col] = pd.to_numeric(bart[col], errors="coerce")

    # Clean team names: Google Sheets copy-paste includes seed info on second line
    # e.g. "Connecticut\n4 seed, CHAMP" -> "Connecticut"
    if "team" in bart.columns:
        bart["team"] = bart["team"].astype(str).str.split("\n").str[0].str.strip()
        # Also remove any trailing seed/record info like "4 seed, CHA"
        bart["team"] = bart["team"].str.replace(r"\d+ seed.*$", "", regex=True).str.strip()
        # Remove trailing commas or whitespace
        bart["team"] = bart["team"].str.rstrip(", ")

    # Check we have the columns we need
    needed = ["team", "season", "adj_oe", "adj_de", "barthag"]
    available = [c for c in needed if c in bart.columns]
    if len(available) < 3:
        print("WARNING: Barttorvik missing expected columns.")
        print("  Have: {}".format(list(bart.columns[:10])))
        print("  Need: {}".format(needed))
        return df

    # Build team name mapping
    espn_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    bart_teams = bart["team"].unique()
    name_map = build_team_name_map(espn_teams, bart_teams)

    # Add mapped name columns
    df["home_bart_name"] = df["home_team"].map(name_map)
    df["away_bart_name"] = df["away_team"].map(name_map)

    # Report match rate
    total = len(espn_teams)
    matched = sum(1 for v in name_map.values() if v is not None)
    print("  Team name matching: {}/{} ESPN teams matched to Barttorvik ({:.0f}%)".format(
        matched, total, 100 * matched / total if total > 0 else 0
    ))
    if matched < total:
        unmatched = [k for k, v in name_map.items() if v is None]
        if len(unmatched) <= 20:
            print("  Unmatched: {}".format(unmatched))
        else:
            print("  Unmatched (first 20): {}".format(unmatched[:20]))

    # Select Barttorvik stats to merge
    stat_cols = [c for c in ["adj_oe", "adj_de", "adj_tempo", "barthag"] if c in bart.columns]
    merge_cols = ["team", "season"] + stat_cols

    # Merge home team stats
    home_bart = bart[merge_cols].copy()
    home_rename = {c: "home_{}".format(c) for c in stat_cols}
    home_bart = home_bart.rename(columns=home_rename)

    df = df.merge(
        home_bart,
        left_on=["home_bart_name", "season"],
        right_on=["team", "season"],
        how="left",
    )
    if "team" in df.columns:
        df = df.drop(columns=["team"])

    # Merge away team stats
    away_bart = bart[merge_cols].copy()
    away_rename = {c: "away_{}".format(c) for c in stat_cols}
    away_bart = away_bart.rename(columns=away_rename)

    df = df.merge(
        away_bart,
        left_on=["away_bart_name", "season"],
        right_on=["team", "season"],
        how="left",
        suffixes=("", "_drop"),
    )
    if "team" in df.columns:
        df = df.drop(columns=["team"])
    df = df[[c for c in df.columns if not c.endswith("_drop")]]

    # Clean up temp columns
    df = df.drop(columns=["home_bart_name", "away_bart_name"], errors="ignore")

    # -- Force numeric types (Google Sheets export can produce strings) --
    for col in df.columns:
        if col.startswith(("home_adj_", "away_adj_", "home_barthag", "away_barthag",
                           "home_net_", "away_net_")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -- Compute differentials --
    if "home_adj_oe" in df.columns and "away_adj_oe" in df.columns:
        df["adj_oe_diff"] = df["home_adj_oe"] - df["away_adj_oe"]

    if "home_adj_de" in df.columns and "away_adj_de" in df.columns:
        # Lower DE is better, so flip: away_de - home_de
        # Positive = home team has the defensive advantage
        df["adj_de_diff"] = df["away_adj_de"] - df["home_adj_de"]

    if "home_adj_tempo" in df.columns and "away_adj_tempo" in df.columns:
        df["adj_tempo_diff"] = df["home_adj_tempo"] - df["away_adj_tempo"]

    if "home_barthag" in df.columns and "away_barthag" in df.columns:
        df["barthag_diff"] = df["home_barthag"] - df["away_barthag"]

    # Net efficiency = OE - DE (higher is better)
    if all(c in df.columns for c in ["home_adj_oe", "home_adj_de", "away_adj_oe", "away_adj_de"]):
        df["home_net_eff"] = df["home_adj_oe"] - df["home_adj_de"]
        df["away_net_eff"] = df["away_adj_oe"] - df["away_adj_de"]
        df["net_eff_diff"] = df["home_net_eff"] - df["away_net_eff"]

    # Report merge success
    if "home_adj_oe" in df.columns:
        merge_rate = df["home_adj_oe"].notna().mean()
        print("  Barttorvik merge rate: {:.1f}% of games have efficiency data".format(100 * merge_rate))

    return df


def _add_rest_days(df):
    """Compute rest days for each team before a game."""
    if "date" not in df.columns:
        return df

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date_dt").reset_index(drop=True)

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

        if home in last_game and pd.notna(last_game[home]):
            rest_h = (date - last_game[home]).days
        else:
            rest_h = 7

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

    if "date_dt" in df.columns:
        df = df.drop(columns=["date_dt"])

    return df


def _add_season_progress(df):
    """Add season progress feature (0 = start, 1 = end)."""
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
def add_tournament_features(df, bracket=None):
    """Add March Madness specific features."""
    UPSET_RATES = {
        (1, 16): 0.01, (2, 15): 0.06, (3, 14): 0.15, (4, 13): 0.20,
        (5, 12): 0.35, (6, 11): 0.37, (7, 10): 0.39, (8, 9): 0.49,
    }

    if bracket is not None and not bracket.empty:
        if "seed" in bracket.columns and "team" in bracket.columns:
            df = df.merge(
                bracket[["team", "seed"]].rename(
                    columns={"team": "home_team", "seed": "home_seed"}
                ),
                on="home_team",
                how="left",
            )
            df = df.merge(
                bracket[["team", "seed"]].rename(
                    columns={"team": "away_team", "seed": "away_seed"}
                ),
                on="away_team",
                how="left",
            )
            df["seed_diff"] = df.get("home_seed", np.nan) - df.get(
                "away_seed", np.nan
            )

    return df


# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------
def get_feature_columns(df):
    """Return the list of feature columns available for modeling."""
    core = ["elo_diff", "elo_sum", "is_neutral", "home_elo_pre", "away_elo_pre"]

    barttorvik = [
        "adj_oe_diff", "adj_de_diff", "adj_tempo_diff", "barthag_diff",
        "net_eff_diff", "home_net_eff", "away_net_eff",
    ]

    contextual = ["rest_diff", "season_progress", "home_rest_days", "away_rest_days"]

    tournament = ["seed_diff"]

    all_features = core + barttorvik + contextual + tournament
    available = [c for c in all_features if c in df.columns]

    print("Features available: {}/{}".format(len(available), len(all_features)))
    for f in available:
        print("  + {}".format(f))
    for f in all_features:
        if f not in available:
            print("  - {} (missing)".format(f))

    return available
