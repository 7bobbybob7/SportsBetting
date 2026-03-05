"""
main.py - End-to-end pipeline for the CBB betting model.

Pipeline:
    1. Load game data + box scores (already pulled)
    2. Compute rolling efficiency metrics from box scores (no leakage)
    3. Build ELO ratings
    4. Merge ELO + rolling stats into feature matrix
    5. Train logistic regression + XGBoost
    6. Backtest with Kelly criterion
    7. Save models and results

Usage:
    python main.py                    # Full pipeline
    python main.py --step data        # Verify data exists
    python main.py --step rolling     # Compute rolling stats
    python main.py --step elo         # Build ELO ratings
    python main.py --step features    # Merge ELO + rolling into features
    python main.py --step train       # Train models
    python main.py --step backtest    # Run backtest
"""

import argparse
from pathlib import Path

import pandas as pd

from src.elo import EloRater, tune_elo
from src.rolling_stats import compute_game_stats, compute_rolling_features
from src.models import train_all_models, save_model
from src.backtester import Backtester
from src.utils import data_quality_report


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_SEASONS = list(range(2018, 2024))   # 2017-18 through 2022-23
VAL_SEASONS = [2024]                       # 2023-24
TEST_SEASONS = [2025]                      # 2024-25 regular season

ROLLING_WINDOW = 10   # recent form window
SEASON_WINDOW = 100   # effectively full season

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PROCESSED_DIR = DATA_DIR / "processed"


def step_data():
    """Step 1: Verify all required data exists."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA VERIFICATION")
    print("=" * 60)

    games_path = PROCESSED_DIR / "all_games.csv"
    if not games_path.exists():
        print("ERROR: {} not found. Run pull_data_v2.py first.".format(games_path))
        return None

    games = pd.read_csv(games_path)
    print("Games: {} rows".format(len(games)))

    boxscores_path = PROCESSED_DIR / "boxscores_flat.csv"
    if not boxscores_path.exists():
        print("ERROR: {} not found. Run pull_boxscores.py first.".format(boxscores_path))
        return None

    boxscores = pd.read_csv(boxscores_path)
    print("Box scores: {} rows".format(len(boxscores)))

    return games


def step_rolling():
    """Step 2: Compute rolling efficiency metrics from box scores."""
    print("\n" + "=" * 60)
    print("STEP 2: ROLLING STATS (short={}, season={})".format(ROLLING_WINDOW, SEASON_WINDOW))
    print("=" * 60)

    games = pd.read_csv(PROCESSED_DIR / "all_games.csv")
    boxscores = pd.read_csv(PROCESSED_DIR / "boxscores_flat.csv")

    # Per-game stats
    print("Computing per-game stats...")
    game_stats = compute_game_stats(boxscores)
    game_stats.to_csv(PROCESSED_DIR / "game_stats.csv", index=False)

    # Short window (recent form)
    print("\nComputing short rolling features (window={})...".format(ROLLING_WINDOW))
    short = compute_rolling_features(games, game_stats, window=ROLLING_WINDOW)
    short.to_csv(PROCESSED_DIR / "rolling_features.csv", index=False)

    # Full season window
    print("\nComputing season rolling features (window={})...".format(SEASON_WINDOW))
    full = compute_rolling_features(games, game_stats, window=SEASON_WINDOW)

    # Rename full season columns: roll_ -> season_
    full_cols = [c for c in full.columns if c.startswith(("home_roll_", "away_roll_", "roll_"))]
    full_rename = {c: c.replace("roll_", "season_") for c in full_cols}
    full = full.rename(columns=full_rename)

    # Merge season stats onto short window
    season_cols = [c for c in full.columns if "season_" in c] + ["game_id"]
    short["game_id"] = short["game_id"].astype(str)
    full["game_id"] = full["game_id"].astype(str)
    combined = short.merge(full[season_cols], on="game_id", how="left")

    combined.to_csv(PROCESSED_DIR / "rolling_features_combined.csv", index=False)

    return combined


def step_elo(games, tune=False):
    """Step 3: Build ELO ratings."""
    print("\n" + "=" * 60)
    print("STEP 3: ELO RATINGS")
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

    game_log = elo.rate_seasons(games, start_year=2018, end_year=2025)

    # Evaluate
    print("\nELO Evaluation (all seasons):")
    metrics = elo.evaluate()
    for k, v in metrics.items():
        if k != "calibration":
            print("  {}: {}".format(k, v))

    print("\nCurrent Top 25:")
    print(elo.get_ratings(top_n=25).to_string())

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    elo.save(str(MODELS_DIR / "elo_ratings.json"))
    game_log.to_csv(PROCESSED_DIR / "elo_game_log.csv", index=False)

    return elo, game_log


def step_features():
    """Step 4: Merge ELO + rolling stats into final feature matrix."""
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 60)

    # Load components
    elo_log = pd.read_csv(PROCESSED_DIR / "elo_game_log.csv")

    # Use combined rolling if available, else short-only
    combined_path = PROCESSED_DIR / "rolling_features_combined.csv"
    short_path = PROCESSED_DIR / "rolling_features.csv"
    if combined_path.exists():
        rolling = pd.read_csv(combined_path)
        print("Using combined rolling features (short + season)")
    else:
        rolling = pd.read_csv(short_path)
        print("Using short-only rolling features")

    # Merge ELO onto rolling features
    elo_cols = [
        "game_id", "home_elo_pre", "away_elo_pre", "home_elo_adj",
        "home_win_prob", "home_win", "neutral", "season",
    ]
    elo_log["game_id"] = elo_log["game_id"].astype(str)
    rolling["game_id"] = rolling["game_id"].astype(str)

    features = rolling.merge(elo_log[elo_cols], on="game_id", how="left", suffixes=("", "_elo"))

    if "season_elo" in features.columns:
        features = features.drop(columns=["season_elo"])

    # ELO features
    features["elo_diff"] = features["home_elo_adj"] - features["away_elo_pre"]
    features["elo_sum"] = features["home_elo_pre"] + features["away_elo_pre"]
    features["is_neutral"] = features["neutral"].astype(int)

    # Rest days and season progress
    features = _add_rest_and_progress(features)

    # Define feature columns
    elo_features = ["elo_diff", "elo_sum", "is_neutral", "home_elo_pre", "away_elo_pre"]
    roll_diffs = [c for c in features.columns if c.startswith("roll_") and c.endswith("_diff")]
    season_diffs = [c for c in features.columns if c.startswith("season_") and c.endswith("_diff")]
    context_feats = ["rest_diff", "home_rest_days", "away_rest_days", "season_progress"]
    context_avail = [c for c in context_feats if c in features.columns]

    feature_cols = elo_features + roll_diffs + season_diffs + context_avail

    print("Features: {}".format(len(feature_cols)))
    print("  ELO: {}".format(len(elo_features)))
    print("  Rolling diffs: {}".format(len(roll_diffs)))
    print("  Season diffs: {}".format(len(season_diffs)))
    print("  Context: {}".format(len(context_avail)))

    # Save
    features.to_csv(PROCESSED_DIR / "features_v2.csv", index=False)

    with open(PROCESSED_DIR / "feature_cols.txt", "w") as fh:
        for c in feature_cols:
            fh.write(c + "\n")

    data_quality_report(features, "Feature Matrix v2")

    return features, feature_cols


def _add_rest_and_progress(features):
    """Add rest days and season progress features."""
    import numpy as np

    features = features.copy()
    features["date_dt"] = pd.to_datetime(features["date"], errors="coerce")
    features = features.sort_values("date_dt").reset_index(drop=True)

    last_game = {}
    rest_home = []
    rest_away = []

    for _, row in features.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date_dt"]

        if pd.isna(date):
            rest_home.append(np.nan)
            rest_away.append(np.nan)
            continue

        if home in last_game and pd.notna(last_game[home]):
            rest_h = (date - last_game[home]).days
        else:
            rest_h = 7

        if away in last_game and pd.notna(last_game[away]):
            rest_a = (date - last_game[away]).days
        else:
            rest_a = 7

        rest_home.append(rest_h)
        rest_away.append(rest_a)
        last_game[home] = date
        last_game[away] = date

    features["home_rest_days"] = rest_home
    features["away_rest_days"] = rest_away
    features["rest_diff"] = features["home_rest_days"] - features["away_rest_days"]

    # Season progress
    for season in features["season"].unique():
        mask = features["season"] == season
        dates = features.loc[mask, "date_dt"]
        if dates.isna().all():
            continue
        min_d = dates.min()
        max_d = dates.max()
        rng = (max_d - min_d).days
        if rng > 0:
            features.loc[mask, "season_progress"] = (dates - min_d).dt.days / rng
        else:
            features.loc[mask, "season_progress"] = 0.5

    features = features.drop(columns=["date_dt"], errors="ignore")
    return features


def step_train(features=None, feature_cols=None):
    """Step 5: Train models."""
    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING")
    print("=" * 60)

    if features is None:
        features = pd.read_csv(PROCESSED_DIR / "features_v2.csv")
    if feature_cols is None:
        with open(PROCESSED_DIR / "feature_cols.txt") as fh:
            feature_cols = [line.strip() for line in fh if line.strip()]

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


def step_backtest(features=None, model_results=None, feature_cols=None):
    """Step 6: Backtest with Kelly criterion."""
    print("\n" + "=" * 60)
    print("STEP 6: BACKTESTING")
    print("=" * 60)

    if features is None:
        features = pd.read_csv(PROCESSED_DIR / "features_v2.csv")

    test_data = features[features["season"].isin(TEST_SEASONS)].copy()

    if test_data.empty:
        print("WARNING: No test data available for backtesting")
        return

    if "market_prob" not in test_data.columns:
        print("WARNING: No market probabilities. Using ELO win prob as proxy.")
        test_data["market_prob"] = test_data["home_win_prob"]

    if model_results is None:
        print("WARNING: No model results passed. Skipping.")
        return

    for model_name in ["logistic_regression", "xgboost"]:
        if model_name not in model_results:
            continue

        print("\n--- Backtesting {} ---".format(model_name))

        model_dict = model_results[model_name]["model"]
        model = model_dict["model"]
        scaler = model_dict.get("scaler")
        feat_cols = model_dict["feature_cols"]

        X_test = test_data[feat_cols].copy()
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
        choices=["data", "rolling", "elo", "features", "train", "backtest", "all"],
        default="all",
    )
    parser.add_argument("--tune-elo", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("  CBB BETTING MODEL PIPELINE v2")
    print("  March Madness 2026 Edition")
    print("  Rolling Stats (no leakage)")
    print("=" * 50)

    if args.step in ("data", "all"):
        games = step_data()
        if games is None:
            return

    if args.step in ("rolling", "all"):
        rolling = step_rolling()

    if args.step in ("elo", "all"):
        games = pd.read_csv(PROCESSED_DIR / "all_games.csv")
        elo, game_log = step_elo(games, tune=args.tune_elo)

    if args.step in ("features", "all"):
        features, feature_cols = step_features()

    if args.step in ("train", "all"):
        if args.step == "train":
            features = None
            feature_cols = None
        model_results = step_train(features, feature_cols)

    if args.step in ("backtest", "all"):
        if args.step == "backtest":
            features = pd.read_csv(PROCESSED_DIR / "features_v2.csv")
            model_results = None
            feature_cols = None
        step_backtest(features, model_results, feature_cols)

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
