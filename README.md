# 🏀 College Basketball Betting Model

An end-to-end college basketball prediction system built for live testing during the **2026 NCAA Tournament**.

Three models — custom ELO ratings, logistic regression, and XGBoost — predict game outcomes and identify value bets against the market using The Odds API.

## Architecture

```
Raw Data (ESPN, Barttorvik, Odds API)
    ↓
data_loader.py — pull & clean game results + efficiency metrics
    ↓
elo.py — custom ELO system (CBB-adapted: 45% regression, MOV cap, HCA)
    ↓
feature_engineering.py — 15+ features: ELO diff, efficiency, tempo, rest, SOS
    ↓
models.py — logistic regression + XGBoost → P(home_win)
    ↓
odds_fetcher.py — live/historical odds, consensus line, de-vig
    ↓
backtester.py — Kelly criterion (0.25x), P&L tracking, drawdown analysis
    ↓
Streamlit dashboard — predictions, calibration, live March Madness tracker
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/cbb-betting-model.git
cd cbb-betting-model
pip install -r requirements.txt

# Set up Odds API key (free: https://the-odds-api.com)
cp .env.example .env
# Edit .env with your API key

# Run the full pipeline
python main.py

# Or run individual steps
python main.py --step data        # Pull game data + Barttorvik
python main.py --step elo         # Build ELO ratings
python main.py --step features    # Engineer features
python main.py --step train       # Train LR + XGBoost
python main.py --step backtest    # Simulate betting performance
```

## Data Sources (all free)

| Source | Data | Coverage |
|--------|------|----------|
| sportsdataverse | Game results, scores, schedules | 2003–present |
| Barttorvik | Adjusted efficiency, tempo, T-Rank | 2008–present |
| The Odds API | Moneylines, spreads, totals | 2020–present |
| hoopR-mbb-data | Bulk parquet files (backup) | 2003–present |

## Models

**1. Custom ELO** — Built from scratch with CBB-specific adaptations:
- 45% season regression (transfer portal, freshman impact)
- 100-point home court advantage (~3.5 points)
- Margin-of-victory multiplier with diminishing returns, capped at 25
- Tunable via grid search over K-factor, regression, HCA, MOV cap

**2. Logistic Regression** — Interpretable baseline with calibrated probabilities.

**3. XGBoost** — Captures nonlinear feature interactions for best accuracy.

## Evaluation

All models evaluated on:
- **Log loss** (primary metric — measures probability calibration)
- **Brier score**, accuracy, AUC-ROC
- **Calibration curves** — "when model says 70%, do teams win ~70%?"
- **Market comparison** — model probabilities vs de-vigged consensus odds
- **Benchmark** — compared against Barttorvik T-Rank predictions

## Backtesting

- **Kelly Criterion** (0.25x fractional) bankroll management
- Minimum 3% edge threshold to place a bet
- Separate regular season vs March Madness P&L
- Tracks: ROI, max drawdown, Sharpe ratio, closing line value

## Project Structure

```
cbb-betting-model/
├── main.py                  # End-to-end pipeline
├── src/
│   ├── data_loader.py       # Pull data from ESPN, Barttorvik, hoopR
│   ├── elo.py               # Custom ELO rating system
│   ├── feature_engineering.py
│   ├── models.py            # Logistic regression + XGBoost
│   ├── backtester.py        # Kelly criterion + P&L
│   ├── odds_fetcher.py      # The Odds API integration
│   └── utils.py             # Team name normalization, viz helpers
├── data/
│   ├── raw/                 # Raw downloads
│   ├── processed/           # Cleaned + merged datasets
│   └── odds/                # Historical odds snapshots
├── models/                  # Saved model files
├── notebooks/               # EDA and analysis
└── app/                     # Streamlit dashboard
```

## Training / Test Split

| Split | Seasons | Purpose |
|-------|---------|---------|
| Train | 2018–2023 (~33K games) | Build ratings, train models |
| Validation | 2023–24 | Tune hyperparameters |
| Test | 2024–25 | Backtest performance |
| **Live** | **March 2026** | **Real-time tournament predictions** |

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Plotly, Streamlit
