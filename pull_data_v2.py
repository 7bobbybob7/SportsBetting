"""
pull_data_v2.py - Pull all CBB data.
Hits ESPN's public API directly - no broken libraries needed.

Usage:
    python pull_data_v2.py
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
from io import StringIO

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


# ---------------------------------------------------------------
# PART 1: Game data from ESPN API directly
# ---------------------------------------------------------------
def pull_games_espn(start_year=2018, end_year=2025):
    """
    Pull game results directly from ESPN's public scoreboard API.
    Iterates day-by-day through each season (Nov - Apr).
    """
    all_games = []

    for season in range(start_year, end_year + 1):
        print("[ESPN API] Pulling season {}...".format(season))

        # CBB season: Nov of (season-1) to Apr of (season)
        start_date = datetime(season - 1, 11, 1)
        end_date = datetime(season, 4, 15)

        season_games = []
        current = start_date
        days_done = 0
        total_days = (end_date - start_date).days

        while current <= end_date:
            date_str = current.strftime("%Y%m%d")

            try:
                resp = requests.get(ESPN_BASE, params={
                    "dates": date_str,
                    "groups": "50",
                    "limit": "400",
                }, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                for event in data.get("events", []):
                    game = parse_espn_event(event, season)
                    if game:
                        season_games.append(game)

            except Exception:
                pass

            current += timedelta(days=1)
            days_done += 1

            # Progress update every 20 days
            if days_done % 20 == 0:
                pct = int(100 * days_done / total_days)
                print("  {}% ({} games so far)".format(pct, len(season_games)))

            time.sleep(0.3)

        if season_games:
            df = pd.DataFrame(season_games)
            path = RAW_DIR / "games_{}.csv".format(season)
            df.to_csv(path, index=False)
            all_games.append(df)
            print("  DONE: {} games for {}".format(len(df), season))
        else:
            print("  WARNING: No games for {}".format(season))

    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        path = PROCESSED_DIR / "all_games.csv"
        combined.to_csv(path, index=False)
        print("\nTotal games: {}".format(len(combined)))
        return combined
    return pd.DataFrame()


def parse_espn_event(event, season):
    """Parse a single ESPN scoreboard event into a flat dict."""
    try:
        comps = event.get("competitions", [])
        if not comps:
            return None

        comp = comps[0]

        # Only completed games
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


# ---------------------------------------------------------------
# PART 2: Barttorvik ratings
# ---------------------------------------------------------------
def pull_barttorvik(start_year=2018, end_year=2025):
    """Pull team ratings from Barttorvik."""
    all_ratings = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for season in range(start_year, end_year + 1):
        print("[Barttorvik] Pulling ratings for {}...".format(season))

        # Try the getteam endpoint
        url = "https://barttorvik.com/team-tables_getter.php"
        params = {"year": season, "type": "pointed"}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            text = resp.text.strip()

            # Try parsing as HTML table
            try:
                tables = pd.read_html(StringIO(text))
                if tables and len(tables[0]) > 50:
                    df = tables[0]
                    df["season"] = season
                    path = RAW_DIR / "barttorvik_{}.csv".format(season)
                    df.to_csv(path, index=False)
                    all_ratings.append(df)
                    print("  OK {} teams".format(len(df)))
                    time.sleep(2)
                    continue
            except Exception:
                pass

            # Try as tab-separated
            try:
                df = pd.read_csv(StringIO(text), sep="\t")
                if len(df) > 50:
                    df["season"] = season
                    path = RAW_DIR / "barttorvik_{}.csv".format(season)
                    df.to_csv(path, index=False)
                    all_ratings.append(df)
                    print("  OK {} teams (TSV)".format(len(df)))
                    time.sleep(2)
                    continue
            except Exception:
                pass

            # Try as comma-separated
            try:
                df = pd.read_csv(StringIO(text))
                if len(df) > 50:
                    df["season"] = season
                    path = RAW_DIR / "barttorvik_{}.csv".format(season)
                    df.to_csv(path, index=False)
                    all_ratings.append(df)
                    print("  OK {} teams (CSV)".format(len(df)))
                    time.sleep(2)
                    continue
            except Exception:
                pass

            # Save raw response for debugging
            debug_path = RAW_DIR / "barttorvik_{}_raw.txt".format(season)
            with open(debug_path, "w") as f:
                f.write(text[:2000])
            print("  FAILED: saved raw response to {} for debugging".format(debug_path))

        except requests.RequestException as e:
            print("  ERROR: {}".format(e))

        time.sleep(2)

    if all_ratings:
        combined = pd.concat(all_ratings, ignore_index=True)
        path = PROCESSED_DIR / "barttorvik_all.csv"
        combined.to_csv(path, index=False)
        print("\nTotal team-seasons: {}".format(len(combined)))
        return combined
    else:
        print("\nBarttorvik auto-pull failed.")
        print("Manual download option:")
        print("  1. Go to https://barttorvik.com/#")
        print("  2. Select year in dropdown")
        print("  3. Copy the table -> paste into Google Sheets -> File -> Download as CSV")
        print("  4. Save as data/raw/barttorvik_YEAR.csv")
        return pd.DataFrame()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("CBB DATA PULL v2 - ESPN API Direct")
    print("=" * 50)
    print("This will take about 20-30 minutes (hitting ESPN day-by-day)")
    print("Let it run - you'll see progress updates.\n")

    # Game data
    print("--- GAME DATA ---\n")
    games = pull_games_espn(2018, 2025)

    # Barttorvik
    print("\n--- BARTTORVIK RATINGS ---\n")
    ratings = pull_barttorvik(2018, 2025)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Games: {} rows".format(len(games)))
    print("Barttorvik: {} rows".format(len(ratings)))

    if len(games) > 0:
        print("\nGames per season:")
        print(games.groupby("season").size().to_string())

    print("\nFiles saved to:")
    print("  data/raw/       (per-season files)")
    print("  data/processed/ (combined files)")
