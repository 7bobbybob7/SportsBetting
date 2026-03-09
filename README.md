# 🏀 College Basketball Betting Model

An end-to-end college basketball prediction system built for live testing during the **2026 NCAA Tournament**.

Three models — custom ELO ratings, logistic regression, and XGBoost — predict game outcomes and identify value bets against sportsbook lines. All features use only pre-game information (zero data leakage).

## Architecture

```
ESPN Public API (47K games, 2018-2025)
    ↓
pull_data_v2.py — game results, scores, schedules
pull_boxscores.py — team box scores (FG, 3PT, FT, REB, AST, TO, STL, BLK)
    ↓
elo.py — custom ELO system (CBB-adapted: 45% regression, MOV cap, HCA)
rolling_stats.py — rolling efficiency metrics (10-game + full season windows)
    ↓
main.py --step features — 35 features: ELO, efficiency, tempo, rest, context
    ↓
models.py — logistic regression + XGBoost → P(home_win)
    ↓
backtester.py — Kelly criterion (0.25x), P&L tracking, drawdown analysis
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/7bobbybob7/SportsBetting.git
cd cbb-betting-model
pip install -r requirements.txt

# Pull data (one-time)
python pull_data_v2.py          # Game results (~30 min)
python pull_boxscores.py        # Box scores (~8 hours, resume-capable)

# Run the full pipeline
python main.py

# Or run individual steps
python main.py --step data        # Verify data exists
python main.py --step rolling     # Compute rolling stats from box scores
python main.py --step elo         # Build ELO ratings
python main.py --step features    # Merge ELO + rolling + context
python main.py --step train       # Train LR + XGBoost
python main.py --step backtest    # Simulate betting performance
```

## Data Sources (all free, no API keys needed)

| Source | Data | Coverage |
|--------|------|----------|
| ESPN Scoreboard API | Game results, scores, schedules | 2018–2025 (46,843 games) |
| ESPN Summary API | Team box scores per game | 2018–2025 (46,832 games) |

## Models

**1. Custom ELO** — Built from scratch with CBB-specific adaptations:
- 45% season regression (transfer portal, freshman impact)
- 100-point home court advantage (~3.5 points)
- Margin-of-victory multiplier with diminishing returns, capped at 25
- Tunable via grid search over K-factor, regression, HCA, MOV cap

**2. Logistic Regression** — Interpretable baseline with calibrated probabilities.

**3. XGBoost** — Captures nonlinear feature interactions between efficiency metrics.

## Features (35 total)

Rolling efficiency metrics computed from ESPN box scores, using only pre-game data:

- **ELO (5):** elo_diff, elo_sum, is_neutral, home/away ELO
- **Recent form (13):** 10-game rolling averages — offensive/defensive efficiency, eFG%, opponent eFG%, turnover rate, forced TO rate, ORB%, DRB%, FT rate, 3PT rate, assist rate, tempo
- **Season identity (13):** full-season rolling averages — same metrics
- **Context (4):** rest days differential, home/away rest days, season progress

## Evaluation

All models evaluated on:
- **Log loss** (primary metric — measures probability calibration)
- **Brier score**, accuracy, AUC-ROC
- **Calibration curves** — "when model says 70%, do teams win ~70%?"

| Model | Accuracy | Log Loss | Brier Score | AUC-ROC |
|-------|----------|----------|-------------|---------|
| ELO baseline | 72.4% | 0.5606 | 0.1887 | — |
| Logistic Regression | 72.7% | 0.5268 | 0.1782 | 0.773 |
| XGBoost | 72.8% | 0.5277 | 0.1786 | 0.772 |

## Backtesting

- **Kelly Criterion** (0.25x fractional) bankroll management
- Minimum 3% edge threshold to place a bet
- 5% max bet cap per game
- Tracks: ROI, max drawdown, Sharpe ratio, win rate, yield

## Project Structure

```
cbb-betting-model/
├── main.py                  # End-to-end pipeline (v2, rolling stats)
├── pull_data_v2.py          # ESPN API game data scraper
├── pull_boxscores.py        # ESPN box score scraper (resume-capable)
├── tune_features.py         # Feature/window tuning experiments
├── src/
│   ├── elo.py               # Custom ELO rating system
│   ├── rolling_stats.py     # Rolling efficiency from box scores
│   ├── models.py            # Logistic regression + XGBoost
│   ├── backtester.py        # Kelly criterion + P&L
│   ├── odds_fetcher.py      # Odds API integration
│   ├── utils.py             # Team name normalization, viz helpers
│   └── tournament.py        # Tournament bracket predictions
├── data/
│   ├── raw/                 # Per-season CSVs from ESPN
│   └── processed/           # Combined datasets, features, rolling stats
├── models/                  # Saved .joblib model files
├── notebooks/               # EDA and analysis
└── app/                     # Streamlit dashboard
```

## Training / Test Split

| Split | Seasons | Purpose |
|-------|---------|---------|
| Train | 2018–2023 (~34K games) | Build ratings, train models |
| Validation | 2023–24 (~6K games) | Tune hyperparameters |
| Test | 2024–25 (~6K games) | Backtest performance |
| **Live** | **March 2026** | **Real-time tournament predictions** |

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Requests
