"""
pull_odds.py - Pull historical odds from ESPN's public API and compute
de-vigged consensus market probabilities.

ESPN odds endpoint:
    sports.core.api.espn.com/v2/sports/basketball/leagues/
    mens-college-basketball/events/{id}/competitions/{id}/odds

Features:
    - Resume capability: skips games already pulled
    - De-vigs moneylines using multiplicative method
    - Computes consensus probability across multiple books
    - Filters out live/in-game odds (provider id 59)
    - Rate limiting (0.3s between requests)

Usage:
    python pull_odds.py                     # Pull odds for all games in all_games.csv
    python pull_odds.py --season 2025       # Pull single season
    python pull_odds.py --season 2025 --devig-only   # Just recompute de-vig on existing data
"""

import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

ESPN_ODDS_BASE = (
    "https://sports.core.api.espn.com/v2/sports/basketball/"
    "leagues/mens-college-basketball/events/{game_id}/"
    "competitions/{game_id}/odds"
)

# Provider IDs to EXCLUDE (live/in-game odds, not closing lines)
EXCLUDE_PROVIDERS = {59}  # "ESPN Bet - Live Odds"

RAW_ODDS_FILE = ODDS_DIR / "raw_odds.csv"
DEVIGGED_FILE = ODDS_DIR / "odds_devigged.csv"
CONSENSUS_FILE = ODDS_DIR / "consensus_odds.csv"


# -------------------------------------------------------------------
# Moneyline conversion & de-vigging
# -------------------------------------------------------------------
def moneyline_to_implied(ml):
    """
    Convert American moneyline to implied probability.
    e.g., -200 -> 0.6667, +150 -> 0.4000
    """
    if ml is None or pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml == 0:
        return np.nan
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def devig_multiplicative(home_implied, away_implied):
    """
    Remove vig using the multiplicative (normalization) method.
    The simplest and most common approach:
        true_prob = implied_prob / sum(implied_probs)

    Returns (home_fair, away_fair) as true probabilities summing to 1.
    """
    if pd.isna(home_implied) or pd.isna(away_implied):
        return np.nan, np.nan
    total = home_implied + away_implied
    if total <= 0:
        return np.nan, np.nan
    return home_implied / total, away_implied / total


# -------------------------------------------------------------------
# Pull odds from ESPN
# -------------------------------------------------------------------
def pull_game_odds(game_id):
    """
    Pull odds for a single game from ESPN's core API.
    Returns a list of dicts, one per provider.
    """
    url = ESPN_ODDS_BASE.format(game_id=game_id)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    items = data.get("items", [])
    if not items:
        return []

    results = []
    for item in items:
        provider = item.get("provider", {})
        provider_id = int(provider.get("id", 0))
        provider_name = provider.get("name", "Unknown")

        # Skip live/in-game odds
        if provider_id in EXCLUDE_PROVIDERS:
            continue

        home_odds = item.get("homeTeamOdds", {})
        away_odds = item.get("awayTeamOdds", {})

        home_ml = home_odds.get("moneyLine", None)
        away_ml = away_odds.get("moneyLine", None)
        spread = item.get("spread", None)
        over_under = item.get("overUnder", None)

        # Get team info
        home_favorite = home_odds.get("favorite", False)

        results.append({
            "game_id": str(game_id),
            "provider_id": provider_id,
            "provider_name": provider_name,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread": spread,
            "over_under": over_under,
            "home_favorite": home_favorite,
        })

    return results


def pull_odds_bulk(game_ids, test_limit=None):
    """
    Pull odds for all game IDs with resume capability.
    """
    # Load existing for resume
    existing_ids = set()
    existing_rows = []
    if RAW_ODDS_FILE.exists():
        existing = pd.read_csv(RAW_ODDS_FILE)
        existing_ids = set(existing["game_id"].astype(str).unique())
        existing_rows = existing.to_dict("records")
        print("Resuming: {} games already have odds".format(len(existing_ids)))

    to_pull = [gid for gid in game_ids if str(gid) not in existing_ids]
    if test_limit:
        to_pull = to_pull[:test_limit]

    total = len(to_pull)
    if total == 0:
        print("All odds already pulled!")
        return

    print("Pulling odds for {} games...".format(total))

    new_rows = []
    success = 0
    no_odds = 0
    failures = 0
    start_time = time.time()

    for i, game_id in enumerate(to_pull):
        results = pull_game_odds(game_id)

        if results:
            new_rows.extend(results)
            success += 1
        else:
            no_odds += 1

        # Progress every 100 games
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print("  [{}/{}] {:.0f}% | {} with odds, {} without | ETA: {:.0f} min".format(
                i + 1, total, 100 * (i + 1) / total,
                success, no_odds, remaining / 60
            ))

            # Checkpoint every 1000 games
            if (i + 1) % 1000 == 0:
                _save_raw(existing_rows + new_rows)

        time.sleep(0.3)

    # Final save
    all_rows = existing_rows + new_rows
    _save_raw(all_rows)

    print("\nDone! {} games with odds, {} without".format(success, no_odds))
    print("Total odds rows: {}".format(len(all_rows)))


def _save_raw(rows):
    """Save raw odds to CSV."""
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RAW_ODDS_FILE, index=False)


# -------------------------------------------------------------------
# De-vig and compute consensus
# -------------------------------------------------------------------
def compute_devigged_odds():
    """
    Take raw odds, convert moneylines to implied probabilities,
    remove vig, and compute per-game consensus market probability.
    """
    if not RAW_ODDS_FILE.exists():
        print("No raw odds found. Run pull first.")
        return

    df = pd.read_csv(RAW_ODDS_FILE)
    print("Loaded {} raw odds rows ({} games)".format(
        len(df), df["game_id"].nunique()
    ))

    # Convert moneylines to implied probabilities
    df["home_implied"] = df["home_ml"].apply(moneyline_to_implied)
    df["away_implied"] = df["away_ml"].apply(moneyline_to_implied)

    # De-vig each provider's line
    devigged = df.apply(
        lambda row: devig_multiplicative(row["home_implied"], row["away_implied"]),
        axis=1,
        result_type="expand",
    )
    df["home_fair"] = devigged[0]
    df["away_fair"] = devigged[1]

    # Check vig levels
    df["vig"] = df["home_implied"] + df["away_implied"] - 1
    avg_vig = df["vig"].mean()
    print("Average vig across all books: {:.2%}".format(avg_vig))

    # Save full de-vigged data
    df.to_csv(DEVIGGED_FILE, index=False)
    print("Saved de-vigged odds -> {}".format(DEVIGGED_FILE))

    # Compute consensus: average de-vigged probability across all providers
    consensus = df.groupby("game_id").agg(
        home_fair_mean=("home_fair", "mean"),
        away_fair_mean=("away_fair", "mean"),
        home_fair_median=("away_fair", lambda x: 1 - x.median()),  # just use mean
        n_providers=("provider_id", "nunique"),
        providers=("provider_name", lambda x: ", ".join(sorted(x.unique()))),
        avg_vig=("vig", "mean"),
        spread_mean=("spread", "mean"),
        over_under_mean=("over_under", "mean"),
    ).reset_index()

    # Rename for merging with features
    consensus = consensus.rename(columns={
        "home_fair_mean": "market_prob_home",
        "away_fair_mean": "market_prob_away",
    })

    consensus["game_id"] = consensus["game_id"].astype(str)

    consensus.to_csv(CONSENSUS_FILE, index=False)
    print("Saved consensus odds -> {}".format(CONSENSUS_FILE))
    print("Games with consensus odds: {}".format(len(consensus)))

    # Summary stats
    print("\nProvider coverage:")
    provider_counts = df["provider_name"].value_counts()
    for name, count in provider_counts.head(10).items():
        games = df[df["provider_name"] == name]["game_id"].nunique()
        print("  {:<35} {} games".format(name, games))

    print("\nAverage providers per game: {:.1f}".format(consensus["n_providers"].mean()))

    return consensus


# -------------------------------------------------------------------
# Merge odds into features file
# -------------------------------------------------------------------
def merge_odds_with_features():
    """
    Attach consensus market probabilities to the features file
    so the backtester can compare model probs vs market probs.
    """
    features_path = PROCESSED_DIR / "features.csv"
    if not features_path.exists():
        print("No features.csv found. Run the pipeline first.")
        return

    if not CONSENSUS_FILE.exists():
        print("No consensus odds found. Run pull + devig first.")
        return

    features = pd.read_csv(features_path)
    consensus = pd.read_csv(CONSENSUS_FILE)

    features["game_id"] = features["game_id"].astype(str)
    consensus["game_id"] = consensus["game_id"].astype(str)

    # Merge
    before = len(features)
    features = features.merge(
        consensus[["game_id", "market_prob_home", "market_prob_away",
                    "n_providers", "avg_vig", "spread_mean", "over_under_mean"]],
        on="game_id",
        how="left",
    )

    # Report coverage
    has_odds = features["market_prob_home"].notna().sum()
    total = len(features)
    print("Features: {} total games".format(total))
    print("With market odds: {} ({:.1%})".format(has_odds, has_odds / total))

    # Per-season breakdown
    print("\nOdds coverage by season:")
    for season in sorted(features["season"].unique()):
        season_data = features[features["season"] == season]
        season_odds = season_data["market_prob_home"].notna().sum()
        print("  {}: {}/{} games ({:.1%})".format(
            season, season_odds, len(season_data),
            season_odds / len(season_data) if len(season_data) > 0 else 0
        ))

    features.to_csv(features_path, index=False)
    print("\nSaved updated features with market odds -> {}".format(features_path))


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull and process ESPN odds data")
    parser.add_argument("--season", type=int, default=None,
                        help="Pull single season only")
    parser.add_argument("--test", type=int, default=None,
                        help="Test with N games only")
    parser.add_argument("--devig-only", action="store_true",
                        help="Skip pulling, just recompute de-vig on existing data")
    parser.add_argument("--merge", action="store_true",
                        help="Merge consensus odds into features.csv")
    args = parser.parse_args()

    print("=" * 50)
    print("ESPN ODDS PULLER")
    print("=" * 50 + "\n")

    if args.devig_only:
        compute_devigged_odds()
    elif args.merge:
        merge_odds_with_features()
    else:
        # Load game IDs
        games = pd.read_csv(PROCESSED_DIR / "all_games.csv")
        print("Total games available: {}".format(len(games)))

        if args.season:
            games = games[games["season"] == args.season]
            print("Filtered to season {}: {} games".format(args.season, len(games)))

        game_ids = games["game_id"].astype(str).tolist()

        # Pull odds
        pull_odds_bulk(game_ids, test_limit=args.test)

        # De-vig
        print("\n" + "-" * 50)
        print("DE-VIGGING ODDS")
        print("-" * 50 + "\n")
        compute_devigged_odds()

        # Merge into features
        print("\n" + "-" * 50)
        print("MERGING INTO FEATURES")
        print("-" * 50 + "\n")
        merge_odds_with_features()

    print("\nDone!")
