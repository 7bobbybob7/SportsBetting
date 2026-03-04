"""
pull_data.py - Pull all CBB data using reliable methods.

Usage:
    python pull_data.py
"""

import time
from pathlib import Path
import pandas as pd
import requests
from io import StringIO

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# PART 1: Game data via CBBpy
# ---------------------------------------------------------------
def pull_games_cbbpy(start_year=2018, end_year=2025):
    """
    Pull game results using CBBpy.
    Season convention: 2024 = the 2023-24 season.
    """
    from cbbpy.mens_scraper import get_games_season

    all_games = []

    for season in range(start_year, end_year + 1):
        print("[CBBpy] Pulling games for {}...".format(season))
        try:
            game_info, box_scores, pbp = get_games_season(season)

            if game_info is not None and len(game_info) > 0:
                game_info["season"] = season
                path = RAW_DIR / "games_cbbpy_{}.csv".format(season)
                game_info.to_csv(path, index=False)
                all_games.append(game_info)
                print("  OK {} games saved".format(len(game_info)))
            else:
                print("  WARNING: No data for {}".format(season))
        except Exception as e:
            print("  ERROR: {}".format(e))

        time.sleep(2)  # be polite to ESPN

    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        path = PROCESSED_DIR / "all_games.csv"
        combined.to_csv(path, index=False)
        print("\nTotal games: {}".format(len(combined)))
        return combined
    else:
        print("WARNING: No games pulled")
        return pd.DataFrame()


# ---------------------------------------------------------------
# PART 1B: Backup - hoopR parquet files (if CBBpy fails)
# ---------------------------------------------------------------
def pull_games_hoopr(start_year=2018, end_year=2025):
    """
    Download pre-built parquet files from hoopR GitHub releases.
    No scraping needed - just downloading static files.
    """
    base_url = "https://github.com/sportsdataverse/hoopR-mbb-data/releases/download/mbb_schedule"
    all_games = []

    for season in range(start_year, end_year + 1):
        url = "{}/mbb_schedule_{}.parquet".format(base_url, season)
        print("[hoopR] Downloading {}...".format(season))
        try:
            df = pd.read_parquet(url)
            path = RAW_DIR / "games_hoopr_{}.parquet".format(season)
            df.to_parquet(path, index=False)
            df["season"] = season
            all_games.append(df)
            print("  OK {} games".format(len(df)))
        except Exception as e:
            print("  ERROR: {}".format(e))

    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        path = PROCESSED_DIR / "all_games.csv"
        combined.to_csv(path, index=False)
        print("\nTotal games: {}".format(len(combined)))
        return combined
    else:
        return pd.DataFrame()


# ---------------------------------------------------------------
# PART 2: Barttorvik ratings
# ---------------------------------------------------------------
def pull_barttorvik(start_year=2018, end_year=2025):
    """
    Pull team ratings from Barttorvik using their data export URL.
    """
    all_ratings = []

    for season in range(start_year, end_year + 1):
        print("[Barttorvik] Pulling ratings for {}...".format(season))

        # Try the main team stats page
        url = "https://barttorvik.com/trank.php"
        params = {
            "year": season,
            "sort": "",
            "lastx": "0",
            "hession": "All",
            "begin": "",
            "end": "",
            "top": "0",
            "revession": "0",
            "cession": "All",
            "start": "",
            "csv": "1",  # request CSV format
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            # Try parsing as CSV
            try:
                df = pd.read_csv(StringIO(resp.text))
                if len(df) > 0:
                    df["season"] = season
                    path = RAW_DIR / "barttorvik_{}.csv".format(season)
                    df.to_csv(path, index=False)
                    all_ratings.append(df)
                    print("  OK {} teams".format(len(df)))
                    time.sleep(2)
                    continue
            except Exception:
                pass

            # Fallback: try parsing as HTML table
            try:
                tables = pd.read_html(StringIO(resp.text))
                if tables and len(tables[0]) > 0:
                    df = tables[0]
                    df["season"] = season
                    path = RAW_DIR / "barttorvik_{}.csv".format(season)
                    df.to_csv(path, index=False)
                    all_ratings.append(df)
                    print("  OK {} teams (from HTML)".format(len(df)))
                    time.sleep(2)
                    continue
            except Exception:
                pass

            print("  WARNING: Could not parse response for {}".format(season))

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
        print("WARNING: No Barttorvik data pulled")
        print("You can manually download from https://barttorvik.com")
        print("Click any season -> Export to CSV -> save to data/raw/")
        return pd.DataFrame()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("CBB DATA PULL")
    print("=" * 50)

    # Pull game data
    print("\n--- GAME DATA ---\n")
    games = pull_games_hoopr(2018, 2025)

    # If CBBpy fails or returns too few games, try hoopR
    #if len(games) < 10000:
    #    print("\nCBBpy returned fewer games than expected.")
    #    print("Trying hoopR parquet files as backup...\n")
    #    games = pull_games_hoopr(2018, 2025)

    # Pull Barttorvik
    print("\n--- BARTTORVIK RATINGS ---\n")
    ratings = pull_barttorvik(2018, 2025)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Games: {} rows".format(len(games)))
    print("Barttorvik: {} rows".format(len(ratings)))
    print("\nFiles saved to:")
    print("  data/raw/       (per-season files)")
    print("  data/processed/ (combined files)")
