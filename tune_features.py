"""
tune_features.py - Tune rolling window size and test feature combinations.

Tests:
    1. Rolling window sizes: 8, 10, 12, 15, 20
    2. Feature sets: diffs only vs diffs + individual stats
    3. With/without rest days and season progress

Usage:
    python tune_features.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier

from src.rolling_stats import compute_game_stats, compute_rolling_features

PROCESSED_DIR = Path("data/processed")

TRAIN_SEASONS = list(range(2018, 2024))
VAL_SEASONS = [2024]


def add_elo_features(rolling, elo_log):
    """Merge ELO features onto rolling features."""
    elo_cols = [
        "game_id", "home_elo_pre", "away_elo_pre", "home_elo_adj",
        "home_win_prob", "home_win", "neutral", "season",
    ]
    elo_log["game_id"] = elo_log["game_id"].astype(str)
    rolling["game_id"] = rolling["game_id"].astype(str)

    features = rolling.merge(elo_log[elo_cols], on="game_id", how="left", suffixes=("", "_elo"))
    if "season_elo" in features.columns:
        features = features.drop(columns=["season_elo"])

    features["elo_diff"] = features["home_elo_adj"] - features["away_elo_pre"]
    features["elo_sum"] = features["home_elo_pre"] + features["away_elo_pre"]
    features["is_neutral"] = features["neutral"].astype(int)

    return features


def add_rest_days(features):
    """Compute rest days for each team."""
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
    features["date_dt"] = pd.to_datetime(features["date"], errors="coerce")
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


def evaluate_config(features, feature_cols, label=""):
    """Train XGBoost and LR, return validation metrics."""
    train = features[features["season"].isin(TRAIN_SEASONS)]
    val = features[features["season"].isin(VAL_SEASONS)]

    X_train = train[feature_cols].copy()
    y_train = train["home_win"].astype(int)
    X_val = val[feature_cols].copy()
    y_val = val["home_win"].astype(int)

    # Drop NaN rows
    train_mask = X_train.notna().all(axis=1)
    val_mask = X_val.notna().all(axis=1)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]

    if len(X_train) < 100 or len(X_val) < 100:
        return None

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_val)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    # LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    lr = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    return {
        "config": label,
        "n_features": len(feature_cols),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "xgb_accuracy": round(accuracy_score(y_val, xgb_preds), 4),
        "xgb_log_loss": round(log_loss(y_val, xgb_probs), 4),
        "lr_accuracy": round(accuracy_score(y_val, lr_preds), 4),
        "lr_log_loss": round(log_loss(y_val, lr_probs), 4),
    }


def main():
    print("=" * 60)
    print("  FEATURE TUNING")
    print("=" * 60)

    # Load base data
    games = pd.read_csv(PROCESSED_DIR / "all_games.csv")
    boxscores = pd.read_csv(PROCESSED_DIR / "boxscores_flat.csv")
    elo_log = pd.read_csv(PROCESSED_DIR / "elo_game_log.csv")

    print("Computing per-game stats...")
    game_stats = compute_game_stats(boxscores)

    results = []

    # -- Test 1: Window sizes --
    print("\n--- Testing rolling window sizes ---")
    for window in [8, 10, 12, 15, 20]:
        print("  Window={}...".format(window))
        start = time.time()

        rolling = compute_rolling_features(games, game_stats, window=window, min_games=3)
        features = add_elo_features(rolling, elo_log.copy())

        # Diffs only
        elo_feats = ["elo_diff", "elo_sum", "is_neutral", "home_elo_pre", "away_elo_pre"]
        roll_diffs = [c for c in features.columns if c.startswith("roll_") and c.endswith("_diff")]
        feat_cols = elo_feats + roll_diffs

        r = evaluate_config(features, feat_cols, "window={} diffs".format(window))
        if r:
            results.append(r)
            print("    XGB: {:.4f} acc, {:.4f} ll | LR: {:.4f} acc, {:.4f} ll ({:.0f}s)".format(
                r["xgb_accuracy"], r["xgb_log_loss"],
                r["lr_accuracy"], r["lr_log_loss"],
                time.time() - start))

    # Find best window
    best_window_result = min(results, key=lambda x: x["xgb_log_loss"])
    best_window = int(best_window_result["config"].split("=")[1].split()[0])
    print("\n  Best window: {} (XGB log loss: {})".format(best_window, best_window_result["xgb_log_loss"]))

    # -- Test 2: Feature sets with best window --
    print("\n--- Testing feature combinations (window={}) ---".format(best_window))

    rolling = compute_rolling_features(games, game_stats, window=best_window, min_games=3)
    features = add_elo_features(rolling, elo_log.copy())
    features = add_rest_days(features)

    elo_feats = ["elo_diff", "elo_sum", "is_neutral", "home_elo_pre", "away_elo_pre"]
    roll_diffs = [c for c in features.columns if c.startswith("roll_") and c.endswith("_diff")]
    roll_individual = [c for c in features.columns if c.startswith(("home_roll_", "away_roll_"))]
    context_feats = ["rest_diff", "home_rest_days", "away_rest_days", "season_progress"]
    context_avail = [c for c in context_feats if c in features.columns]

    configs = [
        ("ELO only", elo_feats),
        ("ELO + diffs", elo_feats + roll_diffs),
        ("ELO + diffs + context", elo_feats + roll_diffs + context_avail),
        ("ELO + diffs + individual", elo_feats + roll_diffs + roll_individual),
        ("ELO + diffs + individual + context", elo_feats + roll_diffs + roll_individual + context_avail),
    ]

    for name, feat_cols in configs:
        print("  {}...".format(name))
        r = evaluate_config(features, feat_cols, name)
        if r:
            results.append(r)
            print("    XGB: {:.4f} acc, {:.4f} ll | LR: {:.4f} acc, {:.4f} ll".format(
                r["xgb_accuracy"], r["xgb_log_loss"],
                r["lr_accuracy"], r["lr_log_loss"]))

    # -- Summary --
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    df = pd.DataFrame(results)
    df = df.sort_values("xgb_log_loss")
    print(df[["config", "n_features", "xgb_accuracy", "xgb_log_loss", "lr_accuracy", "lr_log_loss"]].to_string(index=False))

    # Save best config info
    best = df.iloc[0]
    print("\nBest config: {} ({} features)".format(best["config"], best["n_features"]))
    print("  XGB: {:.4f} accuracy, {:.4f} log loss".format(best["xgb_accuracy"], best["xgb_log_loss"]))
    print("  LR:  {:.4f} accuracy, {:.4f} log loss".format(best["lr_accuracy"], best["lr_log_loss"]))


if __name__ == "__main__":
    main()
