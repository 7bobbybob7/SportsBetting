"""
update_games.py - Incrementally pull new games and append to all_games.csv.

Instead of re-pulling all 8 seasons, this:
  1. Reads existing all_games.csv
  2. Finds the latest date already pulled
  3. Pulls only new games from that date to today
  4. Appends to all_games.csv (deduped by game_id)

Also pulls box scores for any new games automatically.

Usage:
    python update_games.py                    # Pull from last date to today
    python update_games.py --from 20251101    # Pull from specific date
    python update_games.py --season 2026      # Pull full season (Nov-today)
"""

import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary"
)

ALL_GAMES_PATH = PROCESSED_DIR / "all_games.csv"
BOXSCORE_PATH = PROCESSED_DIR / "boxscores.csv"
FLAT_BOXSCORE_PATH = PROCESSED_DIR / "boxscores_flat.csv"


# -------------------------------------------------------------------
# ESPN game parser (same logic as pull_data_v2.py)
# -------------------------------------------------------------------
def parse_espn_event(event, season):
    """Parse a single ESPN scoreboard event into a flat dict."""
    try:
        comps = event.get("competitions", [])
        if not comps:
            return None
        comp = comps[0]

        status = comp.get("status", {}).get("type", {})
        if not status.get("completed", False):
            return None

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            return None

        home = away = None
        for t in competitors:
            if t.get("homeAway") == "home":
                home = t
            else:
                away = t

        if not home or not away:
            return None

        home_score = int(home.get("score", 0))
        away_score = int(away.get("score", 0))

        return {
            "game_id": event.get("id", ""),
            "date": event.get("date", ""),
            "season": season,
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "home_id": home.get("team", {}).get("id", ""),
            "away_id": away.get("team", {}).get("id", ""),
            "home_score": home_score,
            "away_score": away_score,
            "home_win": 1 if home_score > away_score else 0,
            "margin": home_score - away_score,
            "neutral_site": comp.get("neutralSite", False),
            "conference_game": comp.get("conferenceCompetition", False),
        }
    except Exception:
        return None


def date_to_season(date_str):
    """Convert a date string to NCAA season year.
    Games Nov-Dec are season = next year. Games Jan-Apr are season = that year.
    E.g., Nov 2025 -> season 2026, Mar 2026 -> season 2026.
    """
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", ""))
        except Exception:
            return 2026  # fallback
    if dt.month >= 11:
        return dt.year + 1
    else:
        return dt.year


# -------------------------------------------------------------------
# Pull new games
# -------------------------------------------------------------------
def pull_new_games(start_date, end_date=None):
    """
    Pull games from ESPN day-by-day from start_date to end_date.

    Args:
        start_date: datetime object or string YYYYMMDD
        end_date: datetime object or string YYYYMMDD (default: today)

    Returns:
        DataFrame of new games
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y%m%d")
    if end_date is None:
        end_date = pd.Timestamp.now().tz_localize(None)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y%m%d")

    if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)
    total_days = (end_date - start_date).days + 1
    print("Pulling games from {} to {} ({} days)".format(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        total_days,
    ))

    new_games = []
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")

        try:
            resp = requests.get(ESPN_SCOREBOARD, params={
                "dates": date_str,
                "groups": "50",
                "limit": "400",
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for event in data.get("events", []):
                season = date_to_season(date_str)
                game = parse_espn_event(event, season)
                if game:
                    new_games.append(game)

        except Exception as e:
            print("  Error on {}: {}".format(date_str, e))

        current += timedelta(days=1)

        # Progress every 20 days
        days_done = (current - start_date).days
        if days_done % 20 == 0 and days_done > 0:
            pct = int(100 * days_done / total_days)
            print("  {}% done ({} games found)".format(pct, len(new_games)))

        time.sleep(0.3)

    if new_games:
        print("Found {} new games".format(len(new_games)))
        return pd.DataFrame(new_games)

    print("No new games found")
    return pd.DataFrame()


# -------------------------------------------------------------------
# Box score pulling (same as pull_boxscores.py)
# -------------------------------------------------------------------
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

        if "-" in name and "-" in display and "Pct" not in name:
            parts = display.split("-")
            if len(parts) == 2:
                try:
                    stats[name.split("-")[0]] = int(parts[0])
                    stats[name.split("-")[1]] = int(parts[1])
                except (ValueError, IndexError):
                    pass
            continue

        if "Pct" in name:
            try:
                stats[name] = float(display)
            except ValueError:
                pass
            continue

        try:
            stats[name] = int(display)
        except ValueError:
            try:
                stats[name] = float(display)
            except ValueError:
                stats[name] = display

    return stats


def pull_boxscores_for_games(game_ids):
    """Pull box scores for a list of game IDs. Returns list of row dicts."""
    # Check which ones we already have
    existing_ids = set()
    if BOXSCORE_PATH.exists():
        existing = pd.read_csv(BOXSCORE_PATH)
        existing_ids = set(existing["game_id"].astype(str).unique())

    to_pull = [gid for gid in game_ids if str(gid) not in existing_ids]

    if not to_pull:
        print("All box scores already exist")
        return []

    print("Pulling box scores for {} new games...".format(len(to_pull)))

    new_rows = []
    success = 0
    failures = 0
    start_time = time.time()

    for i, game_id in enumerate(to_pull):
        try:
            resp = requests.get(ESPN_SUMMARY, params={"event": game_id}, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            teams = data.get("boxscore", {}).get("teams", [])
            if len(teams) == 2:
                for team_data in teams:
                    stats = parse_team_stats(team_data)
                    stats["game_id"] = str(game_id)
                    new_rows.append(stats)
                success += 1
            else:
                failures += 1
        except Exception:
            failures += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(to_pull):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(to_pull) - i - 1) / rate if rate > 0 else 0
            print("  [{}/{}] {} ok, {} failed | ETA: {:.0f} min".format(
                i + 1, len(to_pull), success, failures, remaining / 60
            ))

        time.sleep(0.3)

    print("Done: {} box scores pulled ({} failed)".format(success, failures))
    return new_rows


def append_boxscores(new_rows):
    """Append new box score rows to boxscores.csv and rebuild flat file."""
    if not new_rows:
        return

    new_df = pd.DataFrame(new_rows)

    if BOXSCORE_PATH.exists():
        existing = pd.read_csv(BOXSCORE_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(BOXSCORE_PATH, index=False)
    print("Saved {} total box score rows".format(len(combined)))

    # Rebuild flat file
    home = combined[combined["team_homeaway"] == "home"].copy()
    away = combined[combined["team_homeaway"] == "away"].copy()

    skip_cols = ["game_id", "team_homeaway"]
    home = home.rename(columns={c: "home_" + c for c in home.columns if c not in skip_cols})
    away = away.rename(columns={c: "away_" + c for c in away.columns if c not in skip_cols})

    flat = home.merge(away, on="game_id", how="inner")
    flat.to_csv(FLAT_BOXSCORE_PATH, index=False)
    print("Saved {} flat box scores".format(len(flat)))


# -------------------------------------------------------------------
# Main update logic
# -------------------------------------------------------------------
def update(start_date=None, season=None):
    """
    Main update function.

    If no args: detect last date in all_games.csv, pull from there to today.
    If --from: pull from that date.
    If --season: pull the full season (Nov start to today).
    """

    # Determine start date
    if season:
        start_date = datetime(season - 1, 11, 1)
        print("Pulling full season {} (from {})".format(
            season, start_date.strftime("%Y-%m-%d")
        ))
    elif start_date:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y%m%d")
    else:
        # Auto-detect from existing data
        if ALL_GAMES_PATH.exists():
            existing = pd.read_csv(ALL_GAMES_PATH)
            last_date = pd.to_datetime(existing["date"]).max()
            start_date = last_date - timedelta(days=1)  # overlap by 1 day for safety
            print("Existing data through {}".format(last_date.strftime("%Y-%m-%d")))
            print("Pulling from {} to catch any missed games".format(
                start_date.strftime("%Y-%m-%d")
            ))
        else:
            print("No existing all_games.csv found.")
            print("Run pull_data_v2.py first for historical data, or use --season 2026")
            return

    # Pull new games
    new_games = pull_new_games(start_date)

    if new_games.empty:
        print("No new games to add")
        return

    # Merge with existing
    if ALL_GAMES_PATH.exists():
        existing = pd.read_csv(ALL_GAMES_PATH)
        print("Existing games: {}".format(len(existing)))

        combined = pd.concat([existing, new_games], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")
        combined = combined.sort_values(["season", "date"]).reset_index(drop=True)
    else:
        combined = new_games

    combined.to_csv(ALL_GAMES_PATH, index=False)
    print("Updated all_games.csv: {} total games".format(len(combined)))

    # Show games per season
    print("\nGames per season:")
    print(combined.groupby("season").size().to_string())

    # Pull box scores for new games
    new_ids = new_games["game_id"].astype(str).tolist()
    new_rows = pull_boxscores_for_games(new_ids)
    append_boxscores(new_rows)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally update CBB game data")
    parser.add_argument("--from", dest="from_date", type=str, default=None,
                        help="Start date (YYYYMMDD)")
    parser.add_argument("--season", type=int, default=None,
                        help="Pull full season (e.g., 2026 for 2025-26)")
    args = parser.parse_args()

    print("=" * 50)
    print("CBB DATA UPDATE")
    print("=" * 50 + "\n")

    update(start_date=args.from_date, season=args.season)
