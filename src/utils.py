"""
utils.py - Shared utility functions.

Key utilities:
    - Team name normalization (the hardest part of CBB data work)
    - Calibration curve plotting
    - General helpers
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------
# Different sources use different names. This is the #1 data pain point.
# Example: "UConn" vs "Connecticut" vs "Connecticut Huskies" vs "CONN"

TEAM_NAME_MAP = {
    # Common aliases -> canonical name
    "UConn": "Connecticut",
    "UCONN": "Connecticut",
    "Connecticut Huskies": "Connecticut",
    "UNC": "North Carolina",
    "North Carolina Tar Heels": "North Carolina",
    "UK": "Kentucky",
    "Kentucky Wildcats": "Kentucky",
    "KU": "Kansas",
    "Kansas Jayhawks": "Kansas",
    "Duke Blue Devils": "Duke",
    "Gonzaga Bulldogs": "Gonzaga",
    "UCLA Bruins": "UCLA",
    "Michigan St.": "Michigan State",
    "Michigan St": "Michigan State",
    "Ohio St.": "Ohio State",
    "Ohio St": "Ohio State",
    "Penn St.": "Penn State",
    "Penn St": "Penn State",
    "Mich. State": "Michigan State",
    "N. Carolina": "North Carolina",
    "N Carolina": "North Carolina",
    "St. John's": "St. John's (NY)",
    "Saint John's": "St. John's (NY)",
    "St John's": "St. John's (NY)",
    "Saint Mary's": "Saint Mary's (CA)",
    "St. Mary's": "Saint Mary's (CA)",
    "USC": "Southern California",
    "Southern Cal": "Southern California",
    "Ole Miss": "Mississippi",
    "UNLV": "UNLV",
    "VCU": "VCU",
    "TCU": "TCU",
    "SMU": "SMU",
    "BYU": "BYU",
    "LSU": "LSU",
    "UCF": "UCF",
    "FAU": "Florida Atlantic",
    "Fla. Atlantic": "Florida Atlantic",
    "FDU": "Fairleigh Dickinson",
    "FGCU": "Florida Gulf Coast",
    "UNC Asheville": "UNC Asheville",
    "UNCG": "UNC Greensboro",
    # Add more as needed when merging data sources
}


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name to a canonical form.
    
    Handles:
        - Known aliases (UConn -> Connecticut)
        - Removing mascot suffixes (Duke Blue Devils -> Duke)
        - Standardizing abbreviations (St. -> Saint)
    """
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    
    # Check direct mapping first
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    
    # Try without common suffixes (mascots)
    # This is a rough heuristic - expand as needed
    for suffix in [
        " Wildcats", " Bulldogs", " Bears", " Tigers", " Eagles",
        " Hawks", " Lions", " Panthers", " Cougars", " Knights",
        " Mustangs", " Cardinals", " Rams", " Hoyas", " Bruins",
        " Trojans", " Crimson Tide", " Fighting Irish", " Tar Heels",
        " Blue Devils", " Jayhawks", " Huskies", " Spartans",
        " Wolverines", " Buckeyes", " Badgers", " Gators", " Sooners",
        " Longhorns", " Volunteers", " Seminoles", " Terrapins",
    ]:
        if name.endswith(suffix):
            stripped = name[: -len(suffix)]
            if stripped in TEAM_NAME_MAP:
                return TEAM_NAME_MAP[stripped]
            return stripped
    
    return name


def normalize_team_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Apply team name normalization to a DataFrame column."""
    df = df.copy()
    df[col] = df[col].apply(normalize_team_name)
    return df


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def calibration_table(
    probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Create a calibration table: predicted probability vs actual win rate.
    
    A well-calibrated model has predicted ~ actual in every bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            rows.append({
                "bin_low": bins[i],
                "bin_high": bins[i + 1],
                "bin_label": f"{bins[i]:.0%}-{bins[i+1]:.0%}",
                "predicted_mean": probs[mask].mean(),
                "actual_mean": actuals[mask].mean(),
                "count": mask.sum(),
                "pct_of_total": mask.sum() / len(probs),
            })

    return pd.DataFrame(rows)


def plot_calibration(
    probs: np.ndarray,
    actuals: np.ndarray,
    model_name: str = "Model",
    save_path: str = None,
):
    """Plot calibration curve (predicted vs actual probability)."""
    import matplotlib.pyplot as plt

    cal = calibration_table(probs, actuals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.scatter(
        cal["predicted_mean"],
        cal["actual_mean"],
        s=cal["count"] * 2,  # size by sample count
        alpha=0.7,
        zorder=5,
    )
    ax1.plot(cal["predicted_mean"], cal["actual_mean"], "o-", alpha=0.7, label=model_name)
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Actual Win Rate")
    ax1.set_title(f"Calibration Curve: {model_name}")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Prediction distribution
    ax2.hist(probs, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Predicted Probability (Home Win)")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.axvline(0.5, color="red", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"OK Saved calibration plot -> {save_path}")
    
    plt.show()


def plot_bankroll(
    bankroll_history: list[float],
    title: str = "Bankroll Over Time",
    save_path: str = None,
):
    """Plot bankroll growth chart from backtest."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(bankroll_history, color="steelblue", linewidth=1.5)
    ax.axhline(bankroll_history[0], color="red", linestyle="--", alpha=0.5, label="Starting bankroll")
    ax.fill_between(
        range(len(bankroll_history)),
        bankroll_history[0],
        bankroll_history,
        alpha=0.1,
        color="steelblue",
    )
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """Quick data quality check."""
    report = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "null_pct": {col: f"{df[col].isna().mean():.1%}" for col in df.columns if df[col].isna().any()},
        "duplicates": 0,
    }

    print(f"\n{'='*40}")
    print(f"Data Quality: {name}")
    print(f"{'='*40}")
    print(f"  Rows: {report['rows']:,}")
    print(f"  Columns: {report['columns']}")
    print(f"  Duplicates: {report['duplicates']}")
    if report["null_pct"]:
        print(f"  Null columns:")
        for col, pct in report["null_pct"].items():
            print(f"    {col}: {pct}")
    else:
        print("  No null values OK")

    return report
