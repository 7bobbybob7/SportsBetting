"""
simulate_tournament.py - Monte Carlo bracket simulation for 2026 NCAA Tournament.

Uses the trained LR model to predict game-by-game probabilities,
then simulates the full bracket N times to compute each team's
probability of reaching each round.

Usage:
    python simulate_tournament.py                    # 10,000 simulations
    python simulate_tournament.py --n 100000         # 100,000 simulations
    python simulate_tournament.py --n 10000 --csv    # Export results to CSV
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

ROUNDS = [
    "R64",        # Round of 64 (First Round)
    "R32",        # Round of 32 (Second Round)
    "S16",        # Sweet 16
    "E8",         # Elite 8
    "F4",         # Final Four
    "Championship",
    "Champion",
]


# -------------------------------------------------------------------
# 2026 NCAA Tournament Bracket
# -------------------------------------------------------------------
# First Four games (played before R64)
FIRST_FOUR = [
    {"team_a": "UMBC Retrievers", "team_b": "Howard Bison", "slot": "MIDWEST_16a"},
    {"team_a": "Prairie View A&M Panthers", "team_b": "Lehigh Mountain Hawks", "slot": "SOUTH_16a"},
    {"team_a": "Texas Longhorns", "team_b": "NC State Wolfpack", "slot": "WEST_11"},
    {"team_a": "Miami (OH) RedHawks", "team_b": "SMU Mustangs", "slot": "MIDWEST_11"},
]

# Each region is a list of 8 first-round matchups in bracket order:
# (1v16), (8v9), (5v12), (4v13), (6v11), (3v14), (7v10), (2v15)
# Winners of adjacent pairs play each other in the next round.
BRACKET = {
    "East": [
        {"seed": 1, "team": "Duke Blue Devils"},
        {"seed": 16, "team": "Siena Saints"},
        {"seed": 8, "team": "Ohio State Buckeyes"},
        {"seed": 9, "team": "TCU Horned Frogs"},
        {"seed": 5, "team": "St. John's Red Storm"},
        {"seed": 12, "team": "Northern Iowa Panthers"},
        {"seed": 4, "team": "Kansas Jayhawks"},
        {"seed": 13, "team": "California Baptist Lancers"},
        {"seed": 6, "team": "Louisville Cardinals"},
        {"seed": 11, "team": "South Florida Bulls"},
        {"seed": 3, "team": "Michigan State Spartans"},
        {"seed": 14, "team": "North Dakota State Bison"},
        {"seed": 7, "team": "UCLA Bruins"},
        {"seed": 10, "team": "UCF Knights"},
        {"seed": 2, "team": "UConn Huskies"},
        {"seed": 15, "team": "Furman Paladins"},
    ],
    "South": [
        {"seed": 1, "team": "Florida Gators"},
        {"seed": 16, "team": "FIRST_FOUR_SOUTH_16a"},  # Prairie View/Lehigh winner
        {"seed": 8, "team": "Clemson Tigers"},
        {"seed": 9, "team": "Iowa Hawkeyes"},
        {"seed": 5, "team": "Vanderbilt Commodores"},
        {"seed": 12, "team": "McNeese Cowboys"},
        {"seed": 4, "team": "Nebraska Cornhuskers"},
        {"seed": 13, "team": "Troy Trojans"},
        {"seed": 6, "team": "North Carolina Tar Heels"},
        {"seed": 11, "team": "VCU Rams"},
        {"seed": 3, "team": "Illinois Fighting Illini"},
        {"seed": 14, "team": "Pennsylvania Quakers"},
        {"seed": 7, "team": "Saint Mary's Gaels"},
        {"seed": 10, "team": "Texas A&M Aggies"},
        {"seed": 2, "team": "Houston Cougars"},
        {"seed": 15, "team": "Idaho Vandals"},
    ],
    "West": [
        {"seed": 1, "team": "Arizona Wildcats"},
        {"seed": 16, "team": "Long Island University Sharks"},
        {"seed": 8, "team": "Villanova Wildcats"},
        {"seed": 9, "team": "Utah State Aggies"},
        {"seed": 5, "team": "Wisconsin Badgers"},
        {"seed": 12, "team": "High Point Panthers"},
        {"seed": 4, "team": "Arkansas Razorbacks"},
        {"seed": 13, "team": "Hawai'i Rainbow Warriors"},
        {"seed": 6, "team": "BYU Cougars"},
        {"seed": 11, "team": "FIRST_FOUR_WEST_11"},  # Texas/NC State winner
        {"seed": 3, "team": "Gonzaga Bulldogs"},
        {"seed": 14, "team": "Kennesaw State Owls"},
        {"seed": 7, "team": "Miami Hurricanes"},
        {"seed": 10, "team": "Missouri Tigers"},
        {"seed": 2, "team": "Purdue Boilermakers"},
        {"seed": 15, "team": "Queens University Royals"},
    ],
    "Midwest": [
        {"seed": 1, "team": "Michigan Wolverines"},
        {"seed": 16, "team": "FIRST_FOUR_MIDWEST_16a"},  # UMBC/Howard winner
        {"seed": 8, "team": "Georgia Bulldogs"},
        {"seed": 9, "team": "Saint Louis Billikens"},
        {"seed": 5, "team": "Texas Tech Red Raiders"},
        {"seed": 12, "team": "Akron Zips"},
        {"seed": 4, "team": "Alabama Crimson Tide"},
        {"seed": 13, "team": "Hofstra Pride"},
        {"seed": 6, "team": "Tennessee Volunteers"},
        {"seed": 11, "team": "FIRST_FOUR_MIDWEST_11"},  # SMU/Miami OH winner
        {"seed": 3, "team": "Virginia Cavaliers"},
        {"seed": 14, "team": "Wright State Raiders"},
        {"seed": 7, "team": "Kentucky Wildcats"},
        {"seed": 10, "team": "Santa Clara Broncos"},
        {"seed": 2, "team": "Iowa State Cyclones"},
        {"seed": 15, "team": "Tennessee State Tigers"},
    ],
}

# Final Four bracket: East vs South, West vs Midwest
FINAL_FOUR_MATCHUPS = [("East", "South"), ("West", "Midwest")]


# -------------------------------------------------------------------
# Model + features loading
# -------------------------------------------------------------------
def load_model():
    path = MODELS_DIR / "logistic_regression.joblib"
    return joblib.load(path)


def load_features():
    path = PROCESSED_DIR / "features_v2.csv"
    df = pd.read_csv(path)
    df["game_id"] = df["game_id"].astype(str)
    return df


def load_feature_cols():
    path = PROCESSED_DIR / "feature_cols.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# -------------------------------------------------------------------
# Team feature lookup (same as dashboard)
# -------------------------------------------------------------------
def get_team_features(features_df, team_name):
    """Get most recent rolling features for a team."""
    home_games = features_df[features_df["home_team"] == team_name].sort_values("date", ascending=False)
    away_games = features_df[features_df["away_team"] == team_name].sort_values("date", ascending=False)

    team_features = {}

    latest_home_date = home_games["date"].iloc[0] if len(home_games) > 0 else ""
    latest_away_date = away_games["date"].iloc[0] if len(away_games) > 0 else ""

    if latest_home_date >= latest_away_date and len(home_games) > 0:
        row = home_games.iloc[0]
        for col in features_df.columns:
            if col.startswith("home_roll_") or col.startswith("home_season_"):
                key = col.replace("home_", "")
                team_features[key] = row[col]
        team_features["elo"] = row.get("home_elo_pre", 1500)
        team_features["rest_days"] = row.get("home_rest_days", 3)
    elif len(away_games) > 0:
        row = away_games.iloc[0]
        for col in features_df.columns:
            if col.startswith("away_roll_") or col.startswith("away_season_"):
                key = col.replace("away_", "")
                team_features[key] = row[col]
        team_features["elo"] = row.get("away_elo_pre", 1500)
        team_features["rest_days"] = row.get("away_rest_days", 3)

    return team_features


def predict_matchup(model_dict, feature_cols, features_df, team_a, team_b):
    """
    Predict P(team_a wins) for a neutral site matchup.
    Team A is treated as 'home' in the model (arbitrary for neutral).
    """
    feats_a = get_team_features(features_df, team_a)
    feats_b = get_team_features(features_df, team_b)

    if not feats_a or not feats_b:
        # Fallback: use ELO only
        elo_a = feats_a.get("elo", 1500) if feats_a else 1500
        elo_b = feats_b.get("elo", 1500) if feats_b else 1500
        elo_diff = elo_a - elo_b
        return 1 / (1 + 10 ** (-elo_diff / 400))

    elo_a = feats_a.get("elo", 1500)
    elo_b = feats_b.get("elo", 1500)

    game_features = {
        "elo_diff": elo_a - elo_b,
        "elo_sum": elo_a + elo_b,
        "is_neutral": 1,  # all tournament games are neutral
        "home_elo_pre": elo_a,
        "away_elo_pre": elo_b,
        "rest_diff": 0,
        "home_rest_days": 3,
        "away_rest_days": 3,
        "season_progress": 0.95,
    }

    rolling_metrics = [
        "oe", "de", "net_eff", "efg", "opp_efg", "to_rate",
        "forced_to_rate", "orb_pct", "drb_pct", "ft_rate", "tpr",
        "ast_rate", "tempo",
    ]

    for metric in rolling_metrics:
        a_val = feats_a.get("roll_{}".format(metric), np.nan)
        b_val = feats_b.get("roll_{}".format(metric), np.nan)
        game_features["roll_{}_diff".format(metric)] = (
            a_val - b_val if pd.notna(a_val) and pd.notna(b_val) else 0
        )

    for metric in rolling_metrics:
        a_val = feats_a.get("season_{}".format(metric), np.nan)
        b_val = feats_b.get("season_{}".format(metric), np.nan)
        game_features["season_{}_diff".format(metric)] = (
            a_val - b_val if pd.notna(a_val) and pd.notna(b_val) else 0
        )

    try:
        model = model_dict["model"]
        scaler = model_dict.get("scaler")
        X = pd.DataFrame([game_features])[feature_cols]
        if scaler:
            X = scaler.transform(X)
        prob = model.predict_proba(X)[0, 1]
        return float(prob)
    except Exception:
        # Fallback to ELO
        elo_diff = elo_a - elo_b
        return 1 / (1 + 10 ** (-elo_diff / 400))


# -------------------------------------------------------------------
# Pre-compute all pairwise probabilities
# -------------------------------------------------------------------
def compute_all_probs(teams, model_dict, feature_cols, features_df):
    """Pre-compute P(A beats B) for all pairs to avoid repeated model calls."""
    print("Pre-computing pairwise probabilities for {} teams...".format(len(teams)))
    probs = {}
    for i, a in enumerate(teams):
        for b in teams:
            if a == b:
                continue
            if (a, b) in probs:
                continue
            p = predict_matchup(model_dict, feature_cols, features_df, a, b)
            probs[(a, b)] = p
            probs[(b, a)] = 1 - p

        if (i + 1) % 10 == 0:
            print("  {}/{} teams computed".format(i + 1, len(teams)))

    print("Done: {} pairwise probabilities".format(len(probs)))
    return probs


# -------------------------------------------------------------------
# Simulate one bracket
# -------------------------------------------------------------------
def simulate_region(region_teams, probs, rng):
    """
    Simulate a region from R64 through Elite 8.
    region_teams: list of 16 team names in bracket order.
    Returns the region champion.
    """
    current = list(region_teams)

    # R64: 16 -> 8
    winners = []
    for i in range(0, 16, 2):
        a, b = current[i], current[i + 1]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        winners.append(winner)
    current = winners

    # R32: 8 -> 4
    winners = []
    for i in range(0, 8, 2):
        a, b = current[i], current[i + 1]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        winners.append(winner)
    current = winners

    # S16: 4 -> 2
    winners = []
    for i in range(0, 4, 2):
        a, b = current[i], current[i + 1]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        winners.append(winner)
    current = winners

    # E8: 2 -> 1
    a, b = current[0], current[1]
    p = probs.get((a, b), 0.5)
    champion = a if rng.random() < p else b

    return champion


def simulate_region_with_tracking(region_teams, probs, rng, round_counts):
    """
    Simulate a region and track which round each team reaches.
    Returns region champion.
    """
    current = list(region_teams)

    # All 16 teams make R64
    for t in current:
        round_counts[t]["R64"] += 1

    # R64 -> R32
    winners = []
    for i in range(0, 16, 2):
        a, b = current[i], current[i + 1]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        winners.append(winner)
        round_counts[winner]["R32"] += 1
    current = winners

    # R32 -> S16
    winners = []
    for i in range(0, 8, 2):
        a, b = current[i], current[i + 1]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        winners.append(winner)
        round_counts[winner]["S16"] += 1
    current = winners

    # S16 -> E8
    winners = []
    for i in range(0, 4, 2):
        a, b = current[i], current[i + 1]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        winners.append(winner)
        round_counts[winner]["E8"] += 1
    current = winners

    # E8 -> F4
    a, b = current[0], current[1]
    p = probs.get((a, b), 0.5)
    champion = a if rng.random() < p else b
    round_counts[champion]["F4"] += 1

    return champion


def simulate_tournament(bracket, probs, rng, round_counts):
    """Simulate full tournament. Returns champion."""
    region_champs = {}
    for region_name, teams_data in bracket.items():
        team_names = [t["team"] for t in teams_data]
        champ = simulate_region_with_tracking(team_names, probs, rng, round_counts)
        region_champs[region_name] = champ

    # Final Four
    for region_a, region_b in FINAL_FOUR_MATCHUPS:
        a = region_champs[region_a]
        b = region_champs[region_b]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        round_counts[winner]["Championship"] += 1

        if (region_a, region_b) == FINAL_FOUR_MATCHUPS[0]:
            finalist_1 = winner
        else:
            finalist_2 = winner

    # Championship
    p = probs.get((finalist_1, finalist_2), 0.5)
    champion = finalist_1 if rng.random() < p else finalist_2
    round_counts[champion]["Champion"] += 1

    return champion


# -------------------------------------------------------------------
# Resolve First Four
# -------------------------------------------------------------------
def resolve_first_four(bracket, probs, rng):
    """Simulate First Four games and update bracket with winners."""
    bracket = {k: list(v) for k, v in bracket.items()}  # deep copy

    for ff in FIRST_FOUR:
        a, b = ff["team_a"], ff["team_b"]
        p = probs.get((a, b), 0.5)
        winner = a if rng.random() < p else b
        slot = ff["slot"]

        # Find and replace placeholder in bracket
        for region_name, teams in bracket.items():
            for i, t in enumerate(teams):
                placeholder = "FIRST_FOUR_{}".format(slot)
                if t["team"] == placeholder:
                    bracket[region_name][i] = {"seed": t["seed"], "team": winner}

    return bracket


# -------------------------------------------------------------------
# Main simulation
# -------------------------------------------------------------------
def run_simulation(n_sims=10000, seed=42):
    print("=" * 60)
    print("2026 NCAA TOURNAMENT MONTE CARLO SIMULATION")
    print("=" * 60)

    # Load model and data
    print("\nLoading model and features...")
    model_dict = load_model()
    features_df = load_features()
    feature_cols = load_feature_cols()

    # Collect all team names
    all_teams = set()
    for region_teams in BRACKET.values():
        for t in region_teams:
            if not t["team"].startswith("FIRST_FOUR"):
                all_teams.add(t["team"])
    for ff in FIRST_FOUR:
        all_teams.add(ff["team_a"])
        all_teams.add(ff["team_b"])

    # Check which teams exist in our features
    known_teams = set(features_df["home_team"].unique()) | set(features_df["away_team"].unique())
    missing = all_teams - known_teams
    if missing:
        print("\nWARNING: {} teams not found in features:".format(len(missing)))
        for t in sorted(missing):
            print("  - {}".format(t))
        print("These teams will use ELO-only fallback predictions.\n")

    # Pre-compute probabilities
    all_teams_list = sorted(all_teams)
    probs = compute_all_probs(all_teams_list, model_dict, feature_cols, features_df)

    # Initialize round counters
    round_counts = defaultdict(lambda: defaultdict(int))

    rng = np.random.default_rng(seed)

    print("\nRunning {} simulations...".format(n_sims))
    start = time.time()
    champion_counts = defaultdict(int)

    for sim in range(n_sims):
        # Resolve First Four each simulation
        bracket = resolve_first_four(BRACKET, probs, rng)

        # Simulate full tournament
        champion = simulate_tournament(bracket, probs, rng, round_counts)
        champion_counts[champion] += 1

        if (sim + 1) % (n_sims // 10) == 0:
            elapsed = time.time() - start
            print("  {}/{} ({:.0f}%) | {:.1f}s elapsed".format(
                sim + 1, n_sims, 100 * (sim + 1) / n_sims, elapsed
            ))

    elapsed = time.time() - start
    print("\nDone! {} simulations in {:.1f}s".format(n_sims, elapsed))

    # Build results table
    results = []
    for team in all_teams_list:
        row = {"Team": team}
        for round_name in ROUNDS:
            count = round_counts[team][round_name]
            row[round_name] = round(count / n_sims * 100, 1)
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("Champion", ascending=False).reset_index(drop=True)

    # Print results
    print("\n" + "=" * 100)
    print("CHAMPIONSHIP PROBABILITIES")
    print("=" * 100)
    print("{:<35} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}".format(
        "Team", "R64", "R32", "S16", "E8", "F4", "Title G", "Champ"
    ))
    print("-" * 100)

    for _, row in df.head(30).iterrows():
        print("{:<35} {:>5.1f}% {:>5.1f}% {:>5.1f}% {:>5.1f}% {:>5.1f}% {:>7.1f}% {:>7.1f}%".format(
            row["Team"],
            row["R64"], row["R32"], row["S16"], row["E8"],
            row["F4"], row["Championship"], row["Champion"],
        ))

    # Print Final Four most likely teams
    print("\n" + "=" * 60)
    print("MOST LIKELY FINAL FOUR")
    print("=" * 60)
    f4 = df.nlargest(4, "F4")
    for _, row in f4.iterrows():
        print("  {}: {:.1f}%".format(row["Team"], row["F4"]))

    print("\n" + "=" * 60)
    print("MOST LIKELY CHAMPION")
    print("=" * 60)
    champ = df.iloc[0]
    print("  {}: {:.1f}%".format(champ["Team"], champ["Champion"]))

    return df


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo tournament simulation")
    parser.add_argument("--n", type=int, default=10000, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--csv", action="store_true", help="Export results to CSV")
    args = parser.parse_args()

    df = run_simulation(n_sims=args.n, seed=args.seed)

    if args.csv:
        out_path = PROCESSED_DIR / "tournament_simulation.csv"
        df.to_csv(out_path, index=False)
        print("\nSaved to {}".format(out_path))
