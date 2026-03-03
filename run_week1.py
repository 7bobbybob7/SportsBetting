"""
Week 1 Getting Started Script
==============================
Run this to set up your data pipeline and build the initial ELO system.

Steps:
    1. Load game data from ESPN (2019-2025 seasons)
    2. Build and validate the custom ELO system
    3. Tune ELO hyperparameters
    4. Generate current team ratings
    5. Quick evaluation

Usage:
    cd cbb-betting-model
    pip install -r requirements.txt
    cp .env.example .env     # Add your Odds API key
    python run_week1.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.data_loader import DataLoader
from src.elo import EloRatingSystem, EloConfig, tune_elo
from src.utils import log_loss_score, brier_score


def main():
    print("=" * 60)
    print("CBB BETTING MODEL — WEEK 1: DATA + ELO")
    print("=" * 60)

    # ──────────────────────────────────────────
    # Step 1: Load game data
    # ──────────────────────────────────────────
    print("\n📊 STEP 1: Loading game data from ESPN")
    print("-" * 40)

    loader = DataLoader()

    # Start with recent seasons. This will take a while on first run
    # since it hits the ESPN API day by day. Subsequent runs use cache.
    seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    games = loader.build_game_dataset(seasons)

    if games.empty:
        print("\n⚠️  Could not load game data automatically.")
        print("This might be due to network issues or API rate limits.")
        print("Try running again, or load one season at a time:")
        print("  loader.load_season_games_espn(2025)")
        return

    print(f"\n✅ Loaded {len(games)} games across {len(seasons)} seasons")
    print(f"   Seasons: {games['season'].unique()}")
    print(f"   Date range: {games['date'].min()} to {games['date'].max()}")

    # ──────────────────────────────────────────
    # Step 2: Build ELO system with default params
    # ──────────────────────────────────────────
    print("\n🏀 STEP 2: Running ELO system")
    print("-" * 40)

    config = EloConfig(
        k_factor=25,
        home_advantage=100,
        season_regression=0.45,
        mov_multiplier=True,
        mov_cap=25,
    )
    elo = EloRatingSystem(config)
    history = elo.run_seasons(games)

    # ──────────────────────────────────────────
    # Step 3: Evaluate
    # ──────────────────────────────────────────
    print("\n📈 STEP 3: Evaluation")
    print("-" * 40)

    probs = history["home_win_prob"].values
    outcomes = history["home_won"].astype(float).values

    accuracy = ((probs > 0.5) == outcomes.astype(bool)).mean()
    ll = log_loss_score(probs, outcomes)
    bs = brier_score(probs, outcomes)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Log Loss:  {ll:.4f}")
    print(f"  Brier:     {bs:.4f}")

    # Evaluate by season
    print("\n  Per-season accuracy:")
    for season in sorted(games["season"].unique()):
        mask = history.index.isin(games[games["season"] == season].index)
        if mask.sum() > 0:
            season_acc = ((probs[mask] > 0.5) == outcomes[mask].astype(bool)).mean()
            print(f"    {season}: {season_acc:.3f} ({mask.sum()} games)")

    # ──────────────────────────────────────────
    # Step 4: Current ratings (Top 25)
    # ──────────────────────────────────────────
    print("\n🏆 STEP 4: Current ELO Rankings (Top 25)")
    print("-" * 40)

    top25 = elo.get_ratings(top_n=25)
    for _, row in top25.iterrows():
        print(f"  {row['rank']:3d}. {row['team']:<30s} {row['elo_rating']:.0f}")

    # ──────────────────────────────────────────
    # Step 5: Tune parameters (optional — takes a few minutes)
    # ──────────────────────────────────────────
    print("\n🔧 STEP 5: Hyperparameter tuning")
    print("-" * 40)
    print("  Skipping full grid search (uncomment below to run)")
    print("  This tests 60 parameter combinations — takes ~5-10 min")

    # Uncomment to run tuning:
    # best_config, results = tune_elo(
    #     games,
    #     param_grid={
    #         "k_factor": [15, 20, 25, 30],
    #         "home_advantage": [60, 80, 100, 120],
    #         "season_regression": [0.3, 0.4, 0.5],
    #     },
    #     metric="log_loss",
    # )
    # print(results.head(10))

    # ──────────────────────────────────────────
    # Step 6: Test a prediction
    # ──────────────────────────────────────────
    print("\n🎯 STEP 6: Sample predictions")
    print("-" * 40)

    # Example: predict a neutral-site tournament game
    sample_teams = top25["team"].head(4).tolist()
    if len(sample_teams) >= 2:
        pred = elo.predict(sample_teams[0], sample_teams[1], neutral=True)
        t1, t2 = sample_teams[0], sample_teams[1]
        print(f"  {t1} vs {t2} (neutral site):")
        print(f"    {t1}: {pred[f'{t1}_win_prob']:.1%} (ELO: {pred[f'{t1}_rating']:.0f})")
        print(f"    {t2}: {pred[f'{t2}_win_prob']:.1%} (ELO: {pred[f'{t2}_rating']:.0f})")

    # ──────────────────────────────────────────
    # Step 7: Save outputs
    # ──────────────────────────────────────────
    print("\n💾 STEP 7: Saving outputs")
    print("-" * 40)

    history.to_parquet("data/processed/elo_history.parquet")
    elo.get_ratings().to_csv("data/processed/current_ratings.csv", index=False)

    print("  Saved: data/processed/elo_history.parquet")
    print("  Saved: data/processed/current_ratings.csv")

    print("\n" + "=" * 60)
    print("✅ Week 1 complete! Next steps:")
    print("  1. Review current_ratings.csv — do the rankings look right?")
    print("  2. Uncomment Step 5 to tune hyperparameters")
    print("  3. Load Barttorvik data and compare your ELO vs T-Rank")
    print("  4. Start feature engineering (Week 2)")
    print("=" * 60)


if __name__ == "__main__":
    main()
