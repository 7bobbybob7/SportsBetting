"""
main.py - End-to-end pipeline for the CBB betting model.

Run this script to execute the full pipeline:
    1. Pull game data (sportsdataverse / CBBpy)
    2. Pull Barttorvik advanced metrics
    3. Build ELO ratings (with optional hyperparameter tuning)
    4. Engineer features
    5. Train logistic regression + XGBoost
    6. Evaluate against validation set
    7. Backtest with Kelly criterion
    8. Save models and results

Usage:
    python main.py                    # Full pipeline
    python main.py --step data        # Just pull data
    python main.py --step elo         # Just run ELO (assumes data exists)
    python main.py --step train       # Just train models (assumes features exist)
    python main.py --step backtest    # Just run backtest (assumes models exist)
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_all_games, load_barttorvik_bulk, build_master_dataset
from src.elo import EloRater, tune_elo
from src.feature_engineering import build_features, get_feature_columns
from src.models import train_all_models, save_model
from src.backtester import Backtester
from src.utils import normalize_team_column, data_quality_report


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_SEASONS = list(range(2018, 2024))   # 2017-18 through 2022-23
VAL_SEASONS = [2024]                       # 2023-24
TEST_SEASONS = [2025]                      # 2024-25 regular season
LIVE_SEASON = [2026]                       # 2025-26 March Madness

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def step_data():
    """Step 1: Pull all raw data."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)
    
    # Game results
    games = load_all_games(
        start_year=2018, end_year=2025, source="sportsdataverse"
    )
    if games.empty:
        print("WARNING: sportsdataverse failed. Trying hoopR...")
        games = load_all_games(
            start_year=2018, end_year=2025, source="hoopr"
        )
    
    data_quality_report(games, "Game Data")
    
    # Barttorvik ratings
    ratings = load_barttorvik_bulk(start_year=2018, end_year=2025)
    data_quality_report(ratings, "Barttorvik Ratings")
    
    return games, ratings


def step_elo(games: pd.DataFrame, tune: bool = False):
    """Step 2: Build ELO ratings."""
    print("\n" + "=" * 60)
    print("STEP 2: ELO RATINGS")
    print("=" * 60)
    
    if tune:
        print("Tuning ELO hyperparameters...")
        result = tune_elo(
            games,
            train_seasons=TRAIN_SEASONS,
            val_seasons=VAL_SEASONS,
        )
        best = result["best_params"]
        print(f"Best params: {best}")
        elo = EloRater(**best)
    else:
        elo = EloRater(
            base_k=20,
            home_advantage=100,
            season_regression=0.45,
            mov_multiplier=True,
            mov_cap=25,
        )
    
    # Rate all seasons
    game_log = elo.rate_seasons(games, start_year=2018, end_year=2025)
    
    # Evaluate
    print("\nELO Evaluation (all seasons):")
    elo.evaluate()
    
    print("\nELO Evaluation (by season):")
    by_season = elo.evaluate_by_season()
    print(by_season.to_string())
    
    # Top 25
    print("\nCurrent Top 25:")
    print(elo.get_ratings(top_n=25).to_string())
    
    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    elo.save(str(MODELS_DIR / "elo_ratings.json"))
    game_log.to_csv(DATA_DIR / "processed" / "elo_game_log.csv", index=False)
    
    return elo, game_log


def step_features(game_log: pd.DataFrame, barttorvik: pd.DataFrame):
    """Step 3: Engineer features."""
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    features = build_features(game_log, game_log, barttorvik)
    feature_cols = get_feature_columns(features)
    
    # Save
    features.to_csv(DATA_DIR / "processed" / "features.csv", index=False)
    
    data_quality_report(features, "Feature Matrix")
    
    return features, feature_cols


def step_train(features: pd.DataFrame, feature_cols: list[str]):
    """Step 4: Train models."""
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)
    
    results = train_all_models(
        features,
        train_seasons=TRAIN_SEASONS,
        val_seasons=VAL_SEASONS,
        feature_cols=feature_cols,
        tune=True,
    )
    
    # Save models
    if "logistic_regression" in results:
        save_model(results["logistic_regression"]["model"], "logistic_regression")
    if "xgboost" in results:
        save_model(results["xgboost"]["model"], "xgboost")
    
    return results


def step_backtest(features: pd.DataFrame, model_results: dict):
    """Step 5: Backtest with Kelly criterion."""
    print("\n" + "=" * 60)
    print("STEP 5: BACKTESTING")
    print("=" * 60)
    
    # Use test seasons for backtesting
    test_data = features[features["season"].isin(TEST_SEASONS)].copy()
    
    if test_data.empty:
        print("WARNING: No test data available for backtesting")
        return
    
    # Need market probabilities for backtesting
    # If we don't have odds data yet, use ELO as a proxy benchmark
    if "market_prob" not in test_data.columns:
        print("WARNING: No market probabilities. Using ELO win prob as proxy.")
        test_data["market_prob"] = test_data["home_win_prob"]
        # Shift slightly to simulate model having an edge
        # In production, this will be replaced with actual odds
    
    # For each model, get predictions on test set
    for model_name in ["logistic_regression", "xgboost"]:
        if model_name not in model_results:
            continue
        
        print(f"\n--- Backtesting {model_name} ---")
        
        model_dict = model_results[model_name]["model"]
        model = model_dict["model"]
        scaler = model_dict.get("scaler")
        feature_cols = model_dict["feature_cols"]
        
        X_test = test_data[feature_cols].copy()
        mask = X_test.notna().all(axis=1)
        X_clean = X_test[mask]
        
        if scaler:
            X_input = scaler.transform(X_clean)
        else:
            X_input = X_clean
        
        model_probs = model.predict_proba(X_input)[:, 1]
        
        test_clean = test_data[mask].copy()
        test_clean["model_prob"] = model_probs
        
        # Run backtest
        bt = Backtester(
            bankroll=10_000,
            kelly_fraction=0.25,
            min_edge=0.03,
        )
        bt.run(test_clean)
        bt.summary()


def main():
    """Run the full pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["data", "elo", "features", "train", "backtest", "all"],
        default="all",
    )
    parser.add_argument("--tune-elo", action="store_true")
    args = parser.parse_args()
    
    print("+==========================================+")
    print("|    CBB BETTING MODEL PIPELINE            |")
    print("|    March Madness 2026 Edition            |")
    print("+==========================================+")
    
    if args.step in ("data", "all"):
        games, barttorvik = step_data()
    
    if args.step in ("elo", "all"):
        if args.step == "elo":
            games = pd.read_csv(DATA_DIR / "processed" / "all_games_sdv.csv")
            barttorvik = pd.read_csv(DATA_DIR / "processed" / "barttorvik_all.csv")
        elo, game_log = step_elo(games, tune=args.tune_elo)
    
    if args.step in ("features", "all"):
        if args.step == "features":
            game_log = pd.read_csv(DATA_DIR / "processed" / "elo_game_log.csv")
            barttorvik = pd.read_csv(DATA_DIR / "processed" / "barttorvik_all.csv")
        features, feature_cols = step_features(game_log, barttorvik)
    
    if args.step in ("train", "all"):
        if args.step == "train":
            features = pd.read_csv(DATA_DIR / "processed" / "features.csv")
            feature_cols = get_feature_columns(features)
        model_results = step_train(features, feature_cols)
    
    if args.step in ("backtest", "all"):
        if args.step == "backtest":
            features = pd.read_csv(DATA_DIR / "processed" / "features.csv")
            # Load models...
        step_backtest(features, model_results)
    
    print("\nOK Pipeline complete!")


if __name__ == "__main__":
    main()
