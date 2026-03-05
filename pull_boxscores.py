"""
pull_boxscores.py - Pull team-level box scores from ESPN summary API.

Endpoint:
    https://site.api.espn.com/apis/site/v2/sports/basketball/
    mens-college-basketball/summary?event={game_id}

Extracts: FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, AST, TO, STL, BLK, PTS, PF
for both teams in each game.

Features:
    - Resume capability: skips games already pulled
    - Progress tracking with ETA
    - Rate limiting (0.3s between requests)
    - Error logging

Usage:
    python pull_boxscores.py                  # Pull all games
    python pull_boxscores.py --test 5         # Test with 5 games
    python pull_boxscores.py --season 2025    # Pull single season
"""

import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"

# Output file
BOXSCORE_FILE = PROCESSED_DIR / "boxscores.csv"


def parse_team_stats(team_data):
    """Parse ESPN team statistics array into a flat dict."""
    stats = {}
    team_info = team_data.get("team", {})
    stats["team_name"] = team_info.get("displayName", "")
    stats["team_id"] = team_info.get("id", "")
    stats["team_abbr"] = team_info.get("abbreviation", "")
    stats["team_homeaway"] = team_data.get("homeAway", "")

    for stat in team_data.get("statistics", []):
        name = stat.get("name", "")
        display = stat.get("displayValue", "")

        # Stats that come as "made-attempted" format
        if "-" in name and "-" in display and "Pct" not in name:
            parts = display.split("-")
            if len(parts) == 2:
                try:
                    made_key = name.split("-")[0]
                    att_key = name.split("-")[1]
                    stats[made_key] = int(parts[0])
                    stats[att_key] = int(parts[1])
                except (ValueError, IndexError):
                    pass
            continue

        # Percentage stats
        if "Pct" in name:
            try:
                stats[name] = float(display)
            except ValueError:
                pass
            continue

        # Count stats (rebounds, assists, etc.)
        try:
            stats[name] = int(display)
        except ValueError:
            try:
                stats[name] = float(display)
            except ValueError:
                stats[name] = display

    return stats


def pull_boxscore(game_id):
    """Pull box score for a single game. Returns list of 2 team stat dicts."""
    try:
        resp = requests.get(
            ESPN_SUMMARY,
            params={"event": game_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        boxscore = data.get("boxscore", {})
        teams = boxscore.get("teams", [])

        if len(teams) != 2:
            return None

        results = []
        for team_data in teams:
            stats = parse_team_stats(team_data)
            stats["game_id"] = str(game_id)
            results.append(stats)

        return results

    except Exception as e:
        return None


def pull_all_boxscores(game_ids, test_limit=None):
    """
    Pull box scores for all game IDs with resume capability.

    Args:
        game_ids: List of ESPN game IDs
        test_limit: If set, only pull this many games (for testing)
    """
    # Load existing data for resume
    existing_ids = set()
    existing_rows = []
    if BOXSCORE_FILE.exists():
        existing = pd.read_csv(BOXSCORE_FILE)
        existing_ids = set(existing["game_id"].astype(str).unique())
        existing_rows = existing.to_dict("records")
        print("Resuming: {} games already pulled".format(len(existing_ids)))

    # Filter to unpulled games
    to_pull = [gid for gid in game_ids if str(gid) not in existing_ids]

    if test_limit:
        to_pull = to_pull[:test_limit]

    total = len(to_pull)
    if total == 0:
        print("All games already pulled!")
        return

    print("Pulling box scores for {} games...".format(total))
    start_time = time.time()

    new_rows = []
    success = 0
    failures = 0

    for i, game_id in enumerate(to_pull):
        result = pull_boxscore(game_id)

        if result:
            new_rows.extend(result)
            success += 1
        else:
            failures += 1

        # Progress update every 50 games
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            eta_min = remaining / 60

            print("  [{}/{}] {:.0f}% | {} ok, {} failed | ETA: {:.0f} min".format(
                i + 1, total, 100 * (i + 1) / total,
                success, failures, eta_min
            ))

            # Save checkpoint every 500 games
            if (i + 1) % 500 == 0:
                _save_checkpoint(existing_rows + new_rows)

        time.sleep(0.3)

    # Final save
    all_rows = existing_rows + new_rows
    _save_checkpoint(all_rows)

    print("\nDone! {} games pulled ({} failed)".format(success, failures))
    print("Total box score rows: {}".format(len(all_rows)))
    print("Saved to: {}".format(BOXSCORE_FILE))


def _save_checkpoint(rows):
    """Save current progress to CSV."""
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(BOXSCORE_FILE, index=False)


def flatten_boxscores():
    """
    Convert the raw box score data into a clean per-game format.

    Input: boxscores.csv (2 rows per game, one per team)
    Output: boxscores_flat.csv (1 row per game with home_ and away_ prefixes)
    """
    if not BOXSCORE_FILE.exists():
        print("No boxscores.csv found. Run pull first.")
        return

    df = pd.read_csv(BOXSCORE_FILE)
    print("Loaded {} box score rows".format(len(df)))

    # Split into home and away
    home = df[df["team_homeaway"] == "home"].copy()
    away = df[df["team_homeaway"] == "away"].copy()

    # Prefix columns
    skip_cols = ["game_id", "team_homeaway"]
    home_renamed = {"game_id": "game_id"}
    for col in home.columns:
        if col not in skip_cols:
            home_renamed[col] = "home_" + col
    home = home.rename(columns=home_renamed)

    away_renamed = {"game_id": "game_id"}
    for col in away.columns:
        if col not in skip_cols:
            away_renamed[col] = "away_" + col
    away = away.rename(columns=away_renamed)

    # Merge on game_id
    flat = home.merge(away, on="game_id", how="inner")

    out_path = PROCESSED_DIR / "boxscores_flat.csv"
    flat.to_csv(out_path, index=False)
    print("Saved {} games to {}".format(len(flat), out_path))

    # Print sample columns
    print("\nColumns: {}".format(list(flat.columns[:20])))

    return flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None,
                        help="Test with N games only")
    parser.add_argument("--season", type=int, default=None,
                        help="Pull single season only")
    parser.add_argument("--flatten", action="store_true",
                        help="Just flatten existing data")
    args = parser.parse_args()

    if args.flatten:
        flatten_boxscores()
    else:
        # Load game IDs from existing data
        games = pd.read_csv(PROCESSED_DIR / "all_games.csv")
        print("Total games available: {}".format(len(games)))

        if args.season:
            games = games[games["season"] == args.season]
            print("Filtered to season {}: {} games".format(args.season, len(games)))

        game_ids = games["game_id"].astype(str).tolist()
        pull_all_boxscores(game_ids, test_limit=args.test)

        # Auto-flatten after pulling
        print("\nFlattening box scores...")
        flatten_boxscores()
