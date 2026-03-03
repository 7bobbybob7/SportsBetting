#!/usr/bin/env python3
"""
Quick Start Script — Run this first!

This script:
1. Verifies your environment is set up correctly
2. Downloads game data from hoopR GitHub releases
3. Runs the ELO system on historical data
4. Prints current top-25 rankings and season accuracy

Usage:
    python quickstart.py

Before running:
    pip install -r requirements.txt
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Verify required packages are installed."""
    print("Checking dependencies...")
    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "plotly": "plotly",
        "requests": "requests",
    }
    
    missing = []
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ✗ {pip_name} — MISSING")
            missing.append(pip_name)
    
    # Optional but recommended
    optional = {"sportsdataverse": "sportsdataverse", "cbbpy": "cbbpy"}
    for import_name, pip_name in optional.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pip_name} (optional)")
        except ImportError:
            print(f"  ⚠ {pip_name} (optional, not installed)")
    
    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required dependencies installed!\n")
    return True


def download_sample_data():
    """
    Download a sample season of data from hoopR GitHub releases.
    
    This gives you team box scores from ESPN for the 2024 season as a 
    quick way to verify the pipeline works before downloading everything.
    """
    import pandas as pd
    from pathlib import Path
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cache = data_dir / "sample_games.csv"
    if cache.exists():
        print(f"Sample data already cached at {cache}")
        return pd.read_csv(cache)
    
    print("Downloading sample game data from hoopR...")
    
    try:
        # Try hoopR GitHub releases first
        url = (
            "https://github.com/sportsdataverse/hoopR-mbb-data/releases/download/"
            "espn_mens_college_basketball_team_boxscores/"
            "espn_mens_college_basketball_team_boxscores_2024.parquet"
        )
        df = pd.read_parquet(url)
        print(f"  ✓ Downloaded {len(df)} team box score records for 2024")
        
        # Transform to game-level data
        # hoopR data has one row per team per game — pivot to game level
        games = transform_team_box_to_games(df)
        games.to_csv(cache, index=False)
        print(f"  ✓ Transformed to {len(games)} games")
        
        return games
        
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print("  Creating synthetic sample data for testing...")
        games = create_synthetic_data()
        games.to_csv(cache, index=False)
        return games


def transform_team_box_to_games(team_box_df):
    """
    Transform hoopR team box data (one row per team per game)
    into game-level data (one row per game).
    """
    import pandas as pd
    
    df = team_box_df.copy()
    
    # hoopR columns vary — adapt based on what's available
    # Common columns: game_id, team_short_display_name, home_away, 
    # team_score, opponent_team_score
    
    if "game_id" not in df.columns:
        print("  Warning: Expected column 'game_id' not found. Available columns:")
        print(f"  {list(df.columns[:20])}...")
        return pd.DataFrame()
    
    # Split into home and away
    home = df[df["home_away"] == "home"].copy() if "home_away" in df.columns else pd.DataFrame()
    away = df[df["home_away"] == "away"].copy() if "home_away" in df.columns else pd.DataFrame()
    
    if len(home) == 0 or len(away) == 0:
        print("  Warning: Could not split into home/away. Check data format.")
        return pd.DataFrame()
    
    # Merge on game_id
    name_col = "team_short_display_name" if "team_short_display_name" in df.columns else "team_display_name"
    score_col = "team_score" if "team_score" in df.columns else "points"
    
    games = pd.merge(
        home[["game_id", name_col, score_col, "game_date"]].rename(columns={
            name_col: "home_team",
            score_col: "home_score",
        }),
        away[["game_id", name_col, score_col]].rename(columns={
            name_col: "away_team",
            score_col: "away_score",
        }),
        on="game_id",
        how="inner",
    )
    
    games["season"] = 2024
    games = games.sort_values("game_date").reset_index(drop=True)
    
    return games


def create_synthetic_data():
    """Create synthetic game data for testing the pipeline."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    teams = [
        "Connecticut", "Houston", "Purdue", "North Carolina", "Tennessee",
        "Auburn", "Iowa State", "Duke", "Marquette", "Arizona",
        "Creighton", "Illinois", "Gonzaga", "Baylor", "Kentucky",
        "Kansas", "Michigan State", "Alabama", "BYU", "Clemson",
        "San Diego State", "Texas Tech", "Wisconsin", "Colorado State",
        "New Mexico", "Drake", "Nevada", "Dayton", "South Carolina",
        "Florida Atlantic",
    ]
    
    games = []
    for game_num in range(500):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        
        # Simulate scores
        home_score = int(np.random.normal(72, 12))
        away_score = int(np.random.normal(69, 12))
        
        games.append({
            "game_id": f"game_{game_num:04d}",
            "game_date": f"2024-{(game_num // 50 + 11) % 12 + 1:02d}-{game_num % 28 + 1:02d}",
            "season": 2024,
            "home_team": home,
            "away_team": away,
            "home_score": max(home_score, 40),
            "away_score": max(away_score, 40),
            "neutral_site": np.random.random() < 0.15,
        })
    
    return pd.DataFrame(games)


def run_elo_demo(games_df):
    """Run the ELO system on the sample data and show results."""
    from src.elo import EloRatingSystem
    
    print("\n" + "=" * 50)
    print("Running ELO Rating System")
    print("=" * 50)
    
    elo = EloRatingSystem(
        k_factor=25,
        home_advantage=100,
        season_regression=0.50,
        mov_cap=25,
    )
    
    # Process the games
    results = elo.process_season(
        games_df,
        season=2024,
        home_col="home_team",
        away_col="away_team",
        home_score_col="home_score",
        away_score_col="away_score",
    )
    
    accuracy = results["correct_prediction"].mean()
    
    print(f"\nGames processed: {len(results)}")
    print(f"Prediction accuracy: {accuracy:.1%}")
    
    # Show top 25
    print("\nTop 25 Rankings:")
    print("-" * 35)
    rankings = elo.get_rankings(25)
    for rank, row in rankings.iterrows():
        print(f"  {rank:>3}. {row['team']:<25} {row['elo_rating']:.0f}")
    
    # Show some predictions
    print("\nSample Predictions vs Actuals:")
    print("-" * 70)
    sample = results.tail(10)
    for _, game in sample.iterrows():
        winner = "✓" if game["correct_prediction"] else "✗"
        print(
            f"  {winner} {game['home_team']:<20} ({game['home_win_prob']:.0%}) "
            f"vs {game['away_team']:<20} | "
            f"{game['home_score']}-{game['away_score']}"
        )
    
    return elo, results


def demo_odds_conversion():
    """Quick demo of odds conversion utilities."""
    from src.utils import (
        american_to_implied_prob,
        remove_vig_from_american,
        kelly_from_american,
    )
    
    print("\n" + "=" * 50)
    print("Odds Conversion Demo")
    print("=" * 50)
    
    # Example: Duke -200 vs. UNC +170
    print("\nExample: Duke -200 vs UNC +170")
    
    duke_implied = american_to_implied_prob(-200)
    unc_implied = american_to_implied_prob(170)
    print(f"  Raw implied: Duke {duke_implied:.1%}, UNC {unc_implied:.1%}")
    print(f"  Total: {duke_implied + unc_implied:.1%} (vig = {(duke_implied + unc_implied - 1) * 100:.1f}%)")
    
    duke_fair, unc_fair = remove_vig_from_american(-200, 170)
    print(f"  Fair (vig-free): Duke {duke_fair:.1%}, UNC {unc_fair:.1%}")
    
    # If our model says Duke has 72% chance
    model_prob = 0.72
    kelly = kelly_from_american(model_prob, -200, fraction=0.25)
    print(f"\n  Our model says Duke: {model_prob:.0%}")
    print(f"  Edge: {(model_prob - duke_fair) * 100:.1f}%")
    print(f"  Kelly bet size (0.25x): {kelly:.1%} of bankroll")


def main():
    print("=" * 50)
    print("CBB Betting Model — Quick Start")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and re-run.")
        sys.exit(1)
    
    # Step 2: Download sample data
    games = download_sample_data()
    
    if len(games) == 0:
        print("Failed to load any data. Check your internet connection.")
        sys.exit(1)
    
    print(f"\nLoaded {len(games)} games")
    print(f"Columns: {list(games.columns)}")
    print(f"Teams: {games['home_team'].nunique()} unique")
    
    # Step 3: Run ELO demo
    elo, results = run_elo_demo(games)
    
    # Step 4: Odds conversion demo
    demo_odds_conversion()
    
    # Step 5: Next steps
    print("\n" + "=" * 50)
    print("✓ Setup complete! Next steps:")
    print("=" * 50)
    print("1. Get an Odds API key: https://the-odds-api.com")
    print("   Set it: export ODDS_API_KEY='your_key_here'")
    print("2. Download full historical data (2018-2025):")
    print("   Edit seasons list in data_loader.py")
    print("3. Open notebooks/01_data_exploration.ipynb")
    print("4. Tune ELO parameters: see src/elo.py tune_elo_parameters()")
    print("5. Train ML models: src/feature_engineering.py + scikit-learn")
    print("=" * 50)


if __name__ == "__main__":
    main()
