"""
models.py - Train and evaluate prediction models.

Models:
    1. Logistic Regression - directly outputs calibrated probabilities
    2. XGBoost - captures nonlinear feature interactions

Both models predict P(home_win) given pre-game features.

Usage:
    from src.models import train_all_models, predict_game
    
    results = train_all_models(features_df, train_seasons, val_seasons)
    prob = predict_game(model, features_dict)
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    tune: bool = True,
) -> dict:
    """
    Train logistic regression model.
    
    LR is our baseline ML model because:
        - Directly outputs calibrated probabilities
        - Coefficients are interpretable (good for interviews)
        - Fast to train, easy to debug
        - Often competitive with complex models for binary prediction
    
    Returns:
        Dict with model, scaler, feature_cols, and training info.
    """
    X = X_train[feature_cols].copy()
    y = y_train.copy()

    # Drop rows with NaN features
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if tune:
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        }
        grid = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=5,
            scoring="neg_log_loss",
            n_jobs=-1,
        )
        grid.fit(X_scaled, y)
        model = grid.best_estimator_
        print(f"  LR best params: {grid.best_params_}")
        print(f"  LR CV log loss: {-grid.best_score_:.4f}")
    else:
        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        model.fit(X_scaled, y)

    # Feature importance (coefficients)
    coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values(
        key=abs, ascending=False
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "coefficients": coefs,
        "n_train": len(y),
    }


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    tune: bool = True,
) -> dict:
    """
    Train XGBoost gradient boosted tree model.
    
    XGBoost typically performs best because it captures:
        - Nonlinear relationships (e.g., high OE + fast tempo interaction)
        - Feature interactions the linear model misses
        - Handles missing values natively
    
    Returns:
        Dict with model, feature_cols, and training info.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("WARNING: xgboost not installed. Run: pip install xgboost")
        return {}

    X = X_train[feature_cols].copy()
    y = y_train.copy()

    # Drop NaN rows
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if tune:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 3],
        }
        
        # Use random search for faster tuning (full grid too large)
        from sklearn.model_selection import RandomizedSearchCV
        
        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        
        search = RandomizedSearchCV(
            base,
            param_grid,
            n_iter=50,
            cv=5,
            scoring="neg_log_loss",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)
        model = search.best_estimator_
        print(f"  XGB best params: {search.best_params_}")
        print(f"  XGB CV log loss: {-search.best_score_:.4f}")
    else:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(X, y)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return {
        "model": model,
        "scaler": None,  # XGBoost doesn't need scaling
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "n_train": len(y),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model_dict: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a trained model on test data.
    
    Returns comprehensive metrics:
        - accuracy, log_loss, brier_score, auc_roc
        - calibration breakdown by probability bin
        - performance by neutral/home games
    """
    feature_cols = model_dict["feature_cols"]
    model = model_dict["model"]
    scaler = model_dict.get("scaler")

    X = X_test[feature_cols].copy()
    y = y_test.copy()

    # Drop NaN
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    # Scale if needed
    if scaler:
        X_input = scaler.transform(X)
    else:
        X_input = X

    # Predictions
    probs = model.predict_proba(X_input)[:, 1]  # P(home_win)
    preds = (probs >= 0.5).astype(int)

    # Metrics
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y, preds), 4),
        "log_loss": round(log_loss(y, probs), 4),
        "brier_score": round(brier_score_loss(y, probs), 4),
        "auc_roc": round(roc_auc_score(y, probs), 4),
        "n_test": len(y),
    }

    # Calibration
    metrics["calibration"] = _calibration_bins(probs, y.values)

    print(f"\n{model_name} Results:")
    for k, v in metrics.items():
        if k not in ("calibration", "model"):
            print(f"  {k}: {v}")

    return metrics


def _calibration_bins(probs: np.ndarray, actuals: np.ndarray) -> list[dict]:
    """Compute calibration: predicted probability vs actual win rate."""
    bins = np.arange(0, 1.05, 0.1)
    result = []
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            result.append({
                "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "predicted": round(probs[mask].mean(), 3),
                "actual": round(actuals[mask].mean(), 3),
                "count": int(mask.sum()),
            })
    return result


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def train_all_models(
    features: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    feature_cols: list[str],
    target: str = "home_win",
    tune: bool = True,
) -> dict:
    """
    Train and evaluate all models, returning a comparison.
    
    Args:
        features: Full feature DataFrame
        train_seasons: Seasons for training
        val_seasons: Seasons for validation
        feature_cols: Which features to use
        target: Target column name
        tune: Whether to tune hyperparameters
    
    Returns:
        Dict with trained models and comparison DataFrame.
    """
    # Split
    train = features[features["season"].isin(train_seasons)]
    val = features[features["season"].isin(val_seasons)]

    X_train = train[feature_cols]
    y_train = train[target].astype(int)
    X_val = val[feature_cols]
    y_val = val[target].astype(int)

    print(f"Training: {len(y_train)} games ({train_seasons})")
    print(f"Validation: {len(y_val)} games ({val_seasons})")
    print(f"Features: {len(feature_cols)}")
    print()

    results = {}

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train, feature_cols, tune)
    lr_metrics = evaluate_model(lr, X_val, y_val, "Logistic Regression")
    results["logistic_regression"] = {"model": lr, "metrics": lr_metrics}

    # 2. XGBoost
    print("\nTraining XGBoost...")
    xgb = train_xgboost(X_train, y_train, feature_cols, tune)
    if xgb:
        xgb_metrics = evaluate_model(xgb, X_val, y_val, "XGBoost")
        results["xgboost"] = {"model": xgb, "metrics": xgb_metrics}

    # Comparison table
    comparison = pd.DataFrame(
        [r["metrics"] for r in results.values()]
    ).set_index("model")
    
    if "calibration" in comparison.columns:
        comparison = comparison.drop(columns=["calibration"])
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string())

    results["comparison"] = comparison
    return results


# ---------------------------------------------------------------------------
# Prediction for new games
# ---------------------------------------------------------------------------
def predict_game(
    model_dict: dict,
    features: dict,
) -> float:
    """
    Predict P(home_win) for a single game using a trained model.
    
    Args:
        model_dict: Output from train_logistic_regression or train_xgboost
        features: Dict of feature_name -> value for this game
    
    Returns:
        Float probability of home team winning.
    """
    feature_cols = model_dict["feature_cols"]
    model = model_dict["model"]
    scaler = model_dict.get("scaler")

    X = pd.DataFrame([features])[feature_cols]

    if scaler:
        X = scaler.transform(X)

    prob = model.predict_proba(X)[0, 1]
    return round(float(prob), 4)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_model(model_dict: dict, name: str):
    """Save trained model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model_dict, path)
    print(f"OK Saved model -> {path}")


def load_model(name: str) -> dict:
    """Load trained model from disk."""
    path = MODELS_DIR / f"{name}.joblib"
    model_dict = joblib.load(path)
    print(f"OK Loaded model from {path}")
    return model_dict
