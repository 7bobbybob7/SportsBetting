"""
main.py - End-to-end pipeline for the CBB betting model.

Usage:
    python main.py                    # Full pipeline
    python main.py --step data        # Verify data exists
    python main.py --step elo         # Just run ELO
    python main.py --step features    # Just build features
    python main.py --step train       # Just train models
    python main.py --step backtest    # Just run backtest
"""

import argparse
from pathlib import Path

import pandas as pd

from src.elo import EloRater, tune_elo
from src.feature_engineering import build_features, get_feature_columns
from src.models import train_all_models, save_model
from src.backtester import Backtester
from src.utils import data_quality_report


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_SEASONS = list(range(2018, 2024))   # 2017-18 through 2022-23
VAL_SEASONS = [2024]                       # 2023-24
TEST_SEASONS = [2025]                      # 2024-25 regular season

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def step_data():
    """Step 1: Load pre-pulled data and verify it."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)

    # Game results (from ESPN API via pull_data_v2.py)
    games_path = DATA_DIR / "processed" / "all_games.csv"
    if not games_path.exists():
        print("ERROR: {} not found.".format(games_path))
        print("Run pull_data_v2.py first to pull game data.")
        return None, None

    games = pd.read_csv(games_path)
    print("Loaded {} games from {}".format(len(games), games_path))
    data_quality_report(games, "Game Data")

    # Barttorvik ratings (manually downloaded CSVs)
    bart_path = DATA_DIR / "processed" / "barttorvik_all.csv"
    if not bart_path.exists():
        print("WARNING: {} not found.".format(bart_path))
        print("Barttorvik data not available - will use ELO-only features.")
        barttorvik = None
    else:
        barttorvik = pd.read_csv(bart_path)
        print("Loaded {} team-seasons from {}".format(len(barttorvik), bart_path))
        data_quality_report(barttorvik, "Barttorvik Ratings")

    return games, barttorvik


def step_elo(games, tune=False):
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
        print("Best params: {}".format(best))
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
    metrics = elo.evaluate()
    for k, v in metrics.items():
        if k != "calibration":
            print("  {}: {}".format(k, v))

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


def step_features(game_log, barttorvik):
    """Step 3: Engineer features."""
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)

    features = build_features(game_log, game_log, barttorvik)
    feature_cols = get_feature_columns(features)

    features.to_csv(DATA_DIR / "processed" / "features.csv", index=False)
    data_quality_report(features, "Feature Matrix")

    return features, feature_cols


def step_train(features, feature_cols):
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

    if "logistic_regression" in results:
        save_model(results["logistic_regression"]["model"], "logistic_regression")
    if "xgboost" in results:
        save_model(results["xgboost"]["model"], "xgboost")

    return results


def step_backtest(features, model_results):
    """Step 5: Backtest with Kelly criterion."""
    print("\n" + "=" * 60)
    print("STEP 5: BACKTESTING")
    print("=" * 60)

    test_data = features[features["season"].isin(TEST_SEASONS)].copy()

    if test_data.empty:
        print("WARNING: No test data available for backtesting")
        return

    if "market_prob" not in test_data.columns:
        print("WARNING: No market probabilities. Using ELO win prob as proxy.")
        test_data["market_prob"] = test_data["home_win_prob"]

    for model_name in ["logistic_regression", "xgboost"]:
        if model_name not in model_results:
            continue

        print("\n--- Backtesting {} ---".format(model_name))

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

    print("=" * 50)
    print("  CBB BETTING MODEL PIPELINE")
    print("  March Madness 2026 Edition")
    print("=" * 50)

    if args.step in ("data", "all"):
        games, barttorvik = step_data()
        if games is None:
            return

    if args.step in ("elo", "all"):
        if args.step == "elo":
            games = pd.read_csv(DATA_DIR / "processed" / "all_games.csv")
            barttorvik_path = DATA_DIR / "processed" / "barttorvik_all.csv"
            barttorvik = pd.read_csv(barttorvik_path) if barttorvik_path.exists() else None
        elo, game_log = step_elo(games, tune=args.tune_elo)

    if args.step in ("features", "all"):
        if args.step == "features":
            game_log = pd.read_csv(DATA_DIR / "processed" / "elo_game_log.csv")
            barttorvik_path = DATA_DIR / "processed" / "barttorvik_all.csv"
            barttorvik = pd.read_csv(barttorvik_path) if barttorvik_path.exists() else None
        features, feature_cols = step_features(game_log, barttorvik)

    if args.step in ("train", "all"):
        if args.step == "train":
            features = pd.read_csv(DATA_DIR / "processed" / "features.csv")
            feature_cols = get_feature_columns(features)
        model_results = step_train(features, feature_cols)

    if args.step in ("backtest", "all"):
        if args.step == "backtest":
            features = pd.read_csv(DATA_DIR / "processed" / "features.csv")
            feature_cols = get_feature_columns(features)
            model_results = {}
        step_backtest(features, model_results)

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
