"""
data_loader.py - Pull college basketball data from multiple sources.

Sources:
    1. sportsdataverse (ESPN API wrapper) - game results, box scores, schedules
    2. CBBpy - lightweight NCAA game scraper (backup/supplement)
    3. Barttorvik - advanced efficiency metrics, T-Rank ratings
    4. hoopR GitHub releases - pre-built parquet files for bulk historical data

Usage:
    from src.data_loader import load_all_games, load_barttorvik_ratings
    
    games = load_all_games(start_year=2018, end_year=2025)
    ratings = load_barttorvik_ratings(start_year=2018, end_year=2025)
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Barttorvik base URL for scraping team-level season stats
BARTTORVIK_URL = "https://barttorvik.com/team-tables_getter.php"

# hoopR GitHub releases for pre-built parquet (bulk download fallback)
HOOPR_BASE = (
    "https://github.com/sportsdataverse/hoopR-mbb-data/releases/download"
)


# ---------------------------------------------------------------------------
# 1. sportsdataverse - ESPN API wrapper
# ---------------------------------------------------------------------------
def load_games_sportsdataverse(season: int, save: bool = True) -> pd.DataFrame:
    """
    Pull all D1 men's basketball games for a given season using sportsdataverse.
    
    The 'season' parameter uses the NCAA convention: season=2024 means the
    2023-24 academic year (games from Nov 2023 - Apr 2024).
    
    Returns a DataFrame with columns:
        game_id, date, season, home_team, away_team, home_score, away_score,
        home_win, neutral_site, conference_game, ...
    """
    try:
        from sportsdataverse.mbb import espn_mbb_schedule
    except ImportError:
        print("sportsdataverse not installed. Run: pip install sportsdataverse")
        return pd.DataFrame()

    print(f"[sportsdataverse] Pulling schedule for {season}...")
    
    try:
        schedule = espn_mbb_schedule(dates=season)
    except Exception as e:
        print(f"  WARNING: Error pulling {season}: {e}")
        return pd.DataFrame()
    
    if schedule is None or schedule.empty:
        print(f"  WARNING: No data returned for {season}")
        return pd.DataFrame()

    # Standardize column names - sportsdataverse schema can vary
    df = schedule.copy()
    
    # Map to our standard schema
    rename_map = {
        "game_id": "game_id",
        "game_date": "date",
        "season": "season",
        "home_display_name": "home_team",
        "away_display_name": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "neutral_site": "neutral_site",
        "conference_competition": "conference_game",
    }
    
    # Only rename columns that exist
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)
    
    # Ensure required columns exist
    if "home_score" in df.columns and "away_score" in df.columns:
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["margin"] = df["home_score"] - df["away_score"]
    
    df["season"] = season
    
    # Filter to completed games only (both scores present)
    df = df.dropna(subset=["home_score", "away_score"])
    
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"games_sdv_{season}.csv"
        df.to_csv(path, index=False)
        print(f"  OK Saved {len(df)} games -> {path}")
    
    return df


def load_games_sportsdataverse_bulk(
    start_year: int = 2018, end_year: int = 2025, save: bool = True
) -> pd.DataFrame:
    """Pull games for multiple seasons and concatenate."""
    frames = []
    for season in range(start_year, end_year + 1):
        df = load_games_sportsdataverse(season, save=save)
        if not df.empty:
            frames.append(df)
        time.sleep(1)  # be polite to ESPN API
    
    if not frames:
        return pd.DataFrame()
    
    all_games = pd.concat(frames, ignore_index=True)
    
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / "all_games_sdv.csv"
        all_games.to_csv(path, index=False)
        print(f"\nOK Total: {len(all_games)} games across {start_year}-{end_year}")
    
    return all_games


# ---------------------------------------------------------------------------
# 2. CBBpy - lightweight NCAA scraper (backup / supplement)
# ---------------------------------------------------------------------------
def load_games_cbbpy(season: int, save: bool = True) -> pd.DataFrame:
    """
    Pull game results using CBBpy. Uses the same season convention.
    
    CBBpy returns game-level data with team names, scores, and basic info.
    Good backup if sportsdataverse has gaps.
    """
    try:
        from cbbpy.mens_scraper import get_games_season
    except ImportError:
        print("cbbpy not installed. Run: pip install cbbpy")
        return pd.DataFrame()

    print(f"[CBBpy] Pulling games for {season}...")
    
    try:
        # CBBpy returns (game_info_df, box_scores_df, pbp_df)
        game_info, _, _ = get_games_season(season)
    except Exception as e:
        print(f"  WARNING: Error pulling {season}: {e}")
        return pd.DataFrame()
    
    if game_info is None or game_info.empty:
        print(f"  WARNING: No data returned for {season}")
        return pd.DataFrame()
    
    df = game_info.copy()
    df["season"] = season
    
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"games_cbbpy_{season}.csv"
        df.to_csv(path, index=False)
        print(f"  OK Saved {len(df)} games -> {path}")
    
    return df


# ---------------------------------------------------------------------------
# 3. hoopR GitHub releases - bulk parquet download
# ---------------------------------------------------------------------------
def load_games_hoopr(season: int, save: bool = True) -> pd.DataFrame:
    """
    Download pre-built game parquet files from hoopR-mbb-data GitHub releases.
    
    This is the fastest way to get bulk historical data. Files are ~2-5MB each.
    URL pattern: .../mbb_schedule_{season}.parquet
    """
    url = f"{HOOPR_BASE}/mbb_schedule/mbb_schedule_{season}.parquet"
    
    print(f"[hoopR] Downloading schedule for {season}...")
    
    try:
        df = pd.read_parquet(url)
    except Exception as e:
        print(f"  WARNING: Error downloading {season}: {e}")
        return pd.DataFrame()
    
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"games_hoopr_{season}.parquet"
        df.to_parquet(path, index=False)
        print(f"  OK Saved {len(df)} games -> {path}")
    
    return df


# ---------------------------------------------------------------------------
# 4. Barttorvik - advanced efficiency metrics
# ---------------------------------------------------------------------------
def load_barttorvik_ratings(season: int, save: bool = True) -> pd.DataFrame:
    """
    Scrape team-level season ratings from Barttorvik.
    
    Returns per-team stats for the given season:
        team, conference, record, adj_oe (adjusted offensive efficiency),
        adj_de (adjusted defensive efficiency), adj_tempo, barthag (win prob 
        vs avg D1 team), t_rank, sos (strength of schedule), ...
    
    Season convention: season=2024 -> 2023-24 season.
    """
    print(f"[Barttorvik] Scraping team ratings for {season}...")
    
    params = {
        "year": season,
        "type": "pointed",  # full season stats
    }
    
    try:
        resp = requests.get(BARTTORVIK_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: Request failed: {e}")
        return pd.DataFrame()
    
    # Barttorvik returns tab-separated or HTML table data
    # The exact format depends on the endpoint; try parsing as HTML table first
    try:
        tables = pd.read_html(resp.text)
        if tables:
            df = tables[0]
        else:
            # Try CSV/TSV parsing
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), sep="\t")
    except Exception:
        # Fallback: try direct CSV parsing
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(resp.text), sep=",")
        except Exception as e:
            print(f"  WARNING: Could not parse response: {e}")
            return pd.DataFrame()
    
    df["season"] = season
    
    # Standardize column names (Barttorvik uses various naming conventions)
    col_map = {
        "Team": "team",
        "Conf": "conference",
        "Conf.": "conference",
        "AdjOE": "adj_oe",
        "AdjDE": "adj_de",
        "AdjT": "adj_tempo",
        "Barthag": "barthag",
        "Rec": "record",
        "Rk": "t_rank",
    }
    existing = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing)
    
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"barttorvik_{season}.csv"
        df.to_csv(path, index=False)
        print(f"  OK Saved {len(df)} teams -> {path}")
    
    return df


def load_barttorvik_bulk(
    start_year: int = 2018, end_year: int = 2025, save: bool = True
) -> pd.DataFrame:
    """Pull Barttorvik ratings for multiple seasons."""
    frames = []
    for season in range(start_year, end_year + 1):
        df = load_barttorvik_ratings(season, save=save)
        if not df.empty:
            frames.append(df)
        time.sleep(2)  # be polite
    
    if not frames:
        return pd.DataFrame()
    
    all_ratings = pd.concat(frames, ignore_index=True)
    
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / "barttorvik_all.csv"
        all_ratings.to_csv(path, index=False)
        print(f"\nOK Total: {len(all_ratings)} team-seasons")
    
    return all_ratings


# ---------------------------------------------------------------------------
# 5. Barttorvik game-level predictions (for benchmarking)
# ---------------------------------------------------------------------------
def load_barttorvik_game_predictions(season: int) -> pd.DataFrame:
    """
    Scrape Barttorvik's game-level win probability predictions.
    
    These serve as our benchmark - we want to match or beat T-Rank's
    prediction accuracy.
    
    URL: https://barttorvik.com/schedule.php?year={season}
    """
    url = f"https://barttorvik.com/schedule.php?year={season}"
    
    print(f"[Barttorvik] Scraping game predictions for {season}...")
    
    try:
        tables = pd.read_html(url)
        if tables:
            df = tables[0]
            df["season"] = season
            return df
    except Exception as e:
        print(f"  WARNING: Could not scrape game predictions: {e}")
    
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. Master data pipeline
# ---------------------------------------------------------------------------
def load_all_games(
    start_year: int = 2018,
    end_year: int = 2025,
    source: str = "sportsdataverse",
    save: bool = True,
) -> pd.DataFrame:
    """
    Master function: load game data from the specified source.
    
    Args:
        start_year: First season (NCAA convention, e.g. 2018 = 2017-18)
        end_year: Last season
        source: "sportsdataverse", "cbbpy", or "hoopr"
        save: Whether to save intermediate and final CSVs
    
    Returns:
        DataFrame with all games, standardized columns.
    """
    print(f"\n{'='*60}")
    print(f"Loading CBB games: {start_year}-{end_year} via {source}")
    print(f"{'='*60}\n")
    
    if source == "sportsdataverse":
        return load_games_sportsdataverse_bulk(start_year, end_year, save)
    elif source == "cbbpy":
        frames = []
        for season in range(start_year, end_year + 1):
            df = load_games_cbbpy(season, save)
            if not df.empty:
                frames.append(df)
            time.sleep(1)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elif source == "hoopr":
        frames = []
        for season in range(start_year, end_year + 1):
            df = load_games_hoopr(season, save)
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        raise ValueError(f"Unknown source: {source}")


def build_master_dataset(
    start_year: int = 2018, end_year: int = 2025
) -> pd.DataFrame:
    """
    Build the complete dataset by merging game results with Barttorvik ratings.
    
    This is the main entry point for the data pipeline:
        1. Load all game results
        2. Load Barttorvik team ratings for each season
        3. Merge team ratings onto each game (for both home and away teams)
        4. Save the master dataset
    
    The resulting DataFrame has one row per game with:
        - Game info (date, teams, scores, margin, neutral site)
        - Home team's Barttorvik ratings (adj_oe, adj_de, adj_tempo, etc.)
        - Away team's Barttorvik ratings
    """
    # Step 1: Game results
    games = load_all_games(start_year, end_year, source="sportsdataverse")
    if games.empty:
        print("WARNING: No game data loaded. Try a different source.")
        return pd.DataFrame()
    
    # Step 2: Barttorvik ratings
    ratings = load_barttorvik_bulk(start_year, end_year)
    if ratings.empty:
        print("WARNING: No Barttorvik data loaded. Returning games only.")
        return games
    
    # Step 3: Merge - home team ratings
    home_cols = {col: f"home_{col}" for col in ratings.columns 
                 if col not in ("team", "season")}
    home_ratings = ratings.rename(columns=home_cols)
    
    games = games.merge(
        home_ratings,
        left_on=["home_team", "season"],
        right_on=["team", "season"],
        how="left",
    )
    if "team" in games.columns:
        games = games.drop(columns=["team"])
    
    # Step 4: Merge - away team ratings
    away_cols = {col: f"away_{col}" for col in ratings.columns 
                 if col not in ("team", "season")}
    away_ratings = ratings.rename(columns=away_cols)
    
    games = games.merge(
        away_ratings,
        left_on=["away_team", "season"],
        right_on=["team", "season"],
        how="left",
        suffixes=("", "_away"),
    )
    if "team" in games.columns:
        games = games.drop(columns=["team"])
    
    # Step 5: Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "master_dataset.csv"
    games.to_csv(path, index=False)
    print(f"\nOK Master dataset: {len(games)} games with team ratings -> {path}")
    
    return games


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pull CBB data")
    parser.add_argument("--start", type=int, default=2018, help="Start season")
    parser.add_argument("--end", type=int, default=2025, help="End season")
    parser.add_argument(
        "--source",
        choices=["sportsdataverse", "cbbpy", "hoopr"],
        default="sportsdataverse",
    )
    parser.add_argument("--full", action="store_true", help="Build full master dataset")
    args = parser.parse_args()
    
    if args.full:
        build_master_dataset(args.start, args.end)
    else:
        load_all_games(args.start, args.end, source=args.source)
