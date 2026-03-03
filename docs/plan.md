# College Basketball Betting Model — Phase 1 Project Plan

## Why CBB > NFL (for now)
- March Madness starts March 17 — you can test LIVE in 2 weeks
- ~5,500+ D1 games/season = massive dataset vs NFL's 272
- Tournament is the ultimate test: 67 games, single elimination, massive public betting
- "I built this model and ran it live during March Madness" >>> any backtest story
- NFL can be added later as a second sport extension

---

## Data Sources (All Free)

### 1. Game Data — `sportsdataverse` (Python) + `CBBpy`
- **sportsdataverse:** `pip install sportsdataverse` — Python companion to hoopR
  - Play-by-play data, team box scores, player box scores, schedules
  - Coverage: 2003–present for men's college basketball via ESPN API
  - Provides: `espn_mbb_pbp()`, `espn_mbb_team_box()`, `espn_mbb_player_box()`
- **CBBpy:** `pip install cbbpy` — lightweight ESPN scraper for NCAA basketball
  - Game info, box scores, play-by-play for any D1 game
  - Simple API: `get_game_boxscore()`, `get_game_pbp()`, `get_game_info()`
- **hoopR-mbb-data (GitHub releases):** Pre-built parquet files for team/player box scores
  - Direct download: `https://github.com/sportsdataverse/hoopR-mbb-data`
  - Fastest way to get bulk historical data without API rate limits

### 2. Advanced Metrics — Barttorvik (Free) + KenPom ($25/yr, optional)
- **Barttorvik (barttorvik.com):**
  - Free, no subscription needed
  - Adjusted offensive/defensive efficiency, tempo, Barthag (win probability)
  - T-Rank power rankings back to 2008
  - Has a built-in game prediction model to benchmark against
  - Adds recency bias (down-weights games >40 days old) — interesting modeling choice
  - Access via: scraping or the cbbdata R API (free key)
- **KenPom (kenpom.com):**
  - $25/year subscription — worth it but optional
  - Gold standard for CBB analytics: adjusted efficiency margin (AdjEM)
  - Python scraper: `pip install kenpompy`
  - If you get it, use as feature input AND as benchmark to beat

### 3. Historical Odds — The Odds API + Kaggle
- **The Odds API:** `https://the-odds-api.com`
  - Free tier: 500 requests/month
  - Historical NCAA basketball odds from late 2020 (moneylines, spreads, totals)
  - Multiple bookmakers (DraftKings, FanDuel, etc.)
  - Can also get LIVE odds for March Madness testing
- **Kaggle datasets:**
  - "College Basketball Dataset" — team stats + tournament results
  - Various March Madness datasets (refreshed annually for Kaggle competitions)
- **OddsShark NCAAB Database:** Free tool for building custom betting reports

### 4. Supplementary
- **ESPN API (via sportsdataverse):** Live scores, schedules, box scores
- **NCAA official data:** Tournament seeding, RPI/NET rankings
- **Recruiting data:** 247Sports via `recruitR-py` (future Phase 3 feature)

### Data Strategy
- **Training set:** 2018-2023 seasons (~33,000 games)
- **Validation set:** 2023-24 season (~5,500 games)
- **Test set:** 2024-25 regular season (~5,500 games)
- **LIVE test:** 2026 March Madness tournament (67 games, real-time)
- **Why 2018+:** Modern CBB (shot clock changes, 3-point revolution, transfer portal era)

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Language | Python 3.10+ | You know it, industry standard |
| Data | Pandas, NumPy | Data wrangling |
| ML | Scikit-learn, XGBoost | Models + evaluation |
| Visualization | Plotly, Matplotlib | Interactive + static charts |
| Dashboard | Streamlit | Fast, Python-native, professional |
| Version Control | Git + GitHub | Portfolio visibility |
| Live Odds | The Odds API | Real-time March Madness testing |

---

## Models to Build (3 total)

### Model 1: Custom ELO Rating System (build from scratch)
**Why:** Same reasons as NFL plan — shows you understand the math. But CBB ELO has a unique wrinkle: 363 D1 teams with wildly varying strengths (Duke vs. a mid-major), and massive roster turnover every year.

**CBB-specific ELO design:**
- All teams start at 1500
- After each game, ratings update based on result + margin of victory
- **Key CBB adaptations:**
  - **Season regression:** Heavier than NFL (~40-50% toward mean) because of roster turnover, transfers, freshmen
  - **Home court advantage:** Bigger in CBB than any other sport (~3.5 pts). Vary by venue (Cameron Indoor ≠ generic arena)
  - **Margin of victory cap:** Blowouts happen more in CBB; cap the MOV multiplier to avoid overweighting 30+ point wins over weak teams
  - **Strength of schedule weighting:** Beating a top-25 team should move your rating more than beating a sub-200 team
- Convert ELO to win probability: `P(win) = 1 / (1 + 10^(-elo_diff/400))`
- **Benchmark:** Compare your ELO predictions against Barttorvik's T-Rank predictions

**Interview talking point:** "College basketball ELO is harder than NFL ELO because you have 363 teams with massive strength variance and 30-40% roster turnover annually. I had to tune the season regression much more aggressively and cap margin-of-victory to avoid overweighting blowouts against weak opponents."

### Model 2: Logistic Regression
**Why:** Calibrated probabilities, interpretable, strong baseline

**Features to engineer (CBB-specific):**
- ELO rating difference (from your Model 1)
- Home/away/neutral indicator (neutral site is common in CBB, especially tournaments)
- Adjusted offensive efficiency differential (points per 100 possessions)
- Adjusted defensive efficiency differential
- Tempo differential (possessions per game — style matchup matters)
- Turnover rate differential
- Offensive rebounding rate differential
- Free throw rate differential
- 3-point shooting % differential (rolling average)
- Strength of schedule (average opponent ELO)
- Rest days differential
- Conference strength tier (Power 6 vs mid-major vs low-major)
- Days since season start (teams improve over the season)
- Tournament experience indicator (for March Madness predictions)

**Important:** All features use ONLY pre-game data. No leakage.

### Model 3: XGBoost
**Why:** Nonlinear feature interactions are particularly valuable in CBB — a fast-tempo team vs a slow-tempo team creates dynamics that logistic regression can't capture

**Same features + XGBoost finds interactions like:**
- "High offensive efficiency + fast tempo against slow defensive team" → bigger edge than either metric alone
- Conference mismatches (Big 12 defensive team vs. high-scoring mid-major)
- Late-season form vs early-season noise

**Hyperparameter tuning with Optuna or GridSearch**

---

## March Madness Specific Features (Bonus)

The tournament is where this project really shines. Add these features for tournament games:
- **Seed difference** (1-seed vs 16-seed, etc.)
- **Historical upset rate by seed matchup** (e.g., 12 vs 5 upsets happen ~35% of the time)
- **Geographic advantage** (team playing closer to home)
- **Tournament experience** (coach's tournament record, program history)
- **"Bubble team" flag** (teams that barely made it tend to be motivated/undervalued)
- **Public betting % vs line movement** (sharp money indicator — future enhancement)

---

## Evaluation Pipeline

### 1. Accuracy Metrics
- Overall accuracy (% games predicted correctly)
- Log loss (the real metric — penalizes confident wrong predictions)
- Brier score (calibration quality)
- AUC-ROC (how well the model separates winners from losers)

### 2. Calibration Analysis
- Calibration curves for each model
- When model says 70%, do those teams win ~70%?
- **CBB-specific:** Separate calibration for favorites vs underdogs, home vs neutral site

### 3. Comparison Against the Market
- Convert closing moneyline odds to implied probabilities (remove vig)
- Compare your probabilities vs the market's
- Where does your model disagree? Is it profitable?

### 4. Comparison Against Established Models
- Benchmark against Barttorvik's game predictions (free, public)
- Benchmark against KenPom's predictions (if you subscribe)
- Benchmark against FiveThirtyEight-style ELO
- **This is unique to CBB** — you have public benchmarks to measure against, which makes your evaluation much more rigorous

### 5. Edge Analysis
- Calculate `edge = your_prob - market_implied_prob`
- Which game types does your model find edge? (home underdogs? conference mismatches? tournament games?)
- Segment by: conference, seed, home/away/neutral, time of season

---

## Backtesting & Bankroll Management

### Kelly Criterion
- `f* = (bp - q) / b` where b = decimal_odds - 1, p = your win probability
- **Use fractional Kelly (0.25x)** — CBB has higher variance than NFL
- **Minimum edge threshold:** Only bet when edge > 3-5%

### Simulation
- $10,000 starting bankroll
- Backtest across regular season + tournament games
- Track: cumulative P&L, ROI %, max drawdown, Sharpe ratio, CLV
- **Separate tournament P&L** — March Madness should be its own analysis

### LIVE March Madness Test (The Killer Feature)
- Before each tournament game, model generates:
  - Win probability for each team
  - Comparison against current betting line
  - Recommended bet (if edge exists) with Kelly sizing
- Track predictions in real-time on the dashboard
- Post-tournament: full analysis of how the model performed live
- **This is your #1 interview talking point**

---

## Streamlit Dashboard Pages

### Page 1: Model Overview
- Model comparison table (accuracy, log loss, Brier score, AUC)
- Calibration curves for all 3 models
- Feature importance chart (XGBoost)
- Logistic regression coefficient chart

### Page 2: Market Comparison
- Scatter: your probability vs market implied probability
- Edge distribution histogram
- Profitable bets highlighted
- Filters: by conference, home/away/neutral, season phase

### Page 3: Backtest Results
- Bankroll growth chart
- Game-by-game bet log
- P&L by season, month, conference
- Max drawdown + Sharpe ratio
- **Separate March Madness performance section**

### Page 4: March Madness LIVE (the showpiece)
- Current tournament bracket with your predictions overlaid
- Pre-game: win probability, edge vs market, recommended bet
- Post-game: results tracker, running P&L
- Upset alerts: games where model says the underdog has value
- Historical bracket simulation: how would your model have done in past tournaments?

### Page 5: Team Deep Dive
- Select team → ELO progression, efficiency trends, model performance
- Head-to-head comparison tool
- Conference strength rankings

---

## Project Structure

```
cbb-betting-model/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                    # Raw downloaded data (gitignored)
│   ├── processed/              # Cleaned, merged datasets
│   └── odds/                   # Historical + live odds data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_elo_system.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_backtesting.ipynb
│   └── 06_march_madness_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion (sportsdataverse, CBBpy, odds)
│   ├── feature_engineering.py  # All feature computation
│   ├── elo.py                  # Custom ELO system (CBB-adapted)
│   ├── models.py               # Model training & evaluation
│   ├── backtester.py           # Kelly criterion & P&L simulation
│   ├── odds_fetcher.py         # Live odds from The Odds API
│   ├── tournament.py           # March Madness specific logic
│   └── utils.py                # Odds conversion, helpers
├── app/
│   ├── app.py                  # Streamlit main app
│   └── pages/
│       ├── model_overview.py
│       ├── market_comparison.py
│       ├── backtest_results.py
│       ├── march_madness_live.py
│       └── team_deep_dive.py
└── models/                     # Saved model artifacts
    └── .gitkeep
```

---

## Timeline (Adjusted for March Madness Deadline)

### ⚡ SPRINT Phase: Before Tournament (Now → March 15)
**Goal: Have a working model generating tournament predictions by Selection Sunday**

**Week 1 (March 2-8): Data + ELO**
- [ ] Set up repo, environment, project structure
- [ ] Pull game data via sportsdataverse/CBBpy (2018-2025 seasons)
- [ ] Scrape Barttorvik efficiency data for features
- [ ] Pull historical odds from The Odds API / Kaggle
- [ ] Merge and clean all datasets
- [ ] Implement custom ELO system, validate against Barttorvik T-Rank
- [ ] Quick EDA notebook

**Week 2 (March 9-15): Models + Quick Dashboard**
- [ ] Engineer all features (efficiency, tempo, SOS, etc.)
- [ ] Train logistic regression + XGBoost on 2018-2023 data
- [ ] Validate on 2023-24 season
- [ ] Quick calibration check
- [ ] Set up Odds API for live odds
- [ ] Build minimal Streamlit dashboard (model predictions + odds comparison)
- [ ] **Selection Sunday (March 15):** Generate bracket predictions

### 🏀 LIVE Phase: During Tournament (March 17 - April 6)
- [ ] Before each round: generate predictions for all games
- [ ] Compare against live betting lines
- [ ] Track results in real-time on dashboard
- [ ] Log every prediction and its outcome

### 🔧 POLISH Phase: After Tournament (April 7+)
**Week 5-6: Full Evaluation**
- [ ] Complete calibration analysis (all 3 models)
- [ ] Full backtest with Kelly criterion across historical seasons
- [ ] Edge analysis: where does model find value?
- [ ] March Madness post-mortem: how did live predictions perform?

**Week 7-8: Dashboard + Portfolio**
- [ ] Build out all 5 Streamlit pages
- [ ] March Madness live results page (populated with real data)
- [ ] Clean visualizations
- [ ] Comprehensive README
- [ ] Deploy on Streamlit Cloud
- [ ] Code cleanup, docstrings, tests

---

## Resume Bullet Points (draft)

> **College Basketball Betting Model** | Python, Scikit-learn, XGBoost, Streamlit, Git
> - Built a custom ELO rating system adapted for college basketball's 363-team ecosystem, tuning season regression, home-court advantage, and margin-of-victory capping across 33,000+ historical games.
> - Trained logistic regression and XGBoost models using 15+ engineered features (adjusted efficiency, tempo, SOS), comparing predictions against Barttorvik benchmarks and historical closing betting lines.
> - Deployed the model live during the 2026 NCAA Tournament, generating real-time win probabilities and edge-based bet recommendations for all 67 tournament games.
> - Built a Streamlit dashboard featuring calibration analysis, Kelly criterion backtesting (X% ROI over Y seasons), and a live March Madness prediction tracker.

---

## Key Differences vs NFL Plan (What Makes CBB Unique)

| Aspect | NFL | College Basketball |
|--------|-----|-------------------|
| Games/season | 272 | 5,500+ |
| Teams | 32 | 363 |
| Roster stability | High | Low (transfers, freshmen) |
| Home court effect | ~3 pts | ~3.5 pts (louder, more impactful) |
| Neutral site games | Rare | Common (tournaments) |
| Public benchmarks | Limited | Barttorvik, KenPom (great for validation) |
| Live testing opportunity | Sept 2026 | March 2026 (NOW) |
| Variance | Lower | Higher (more upsets) |
| Market efficiency | Very efficient | Less efficient (more games, less coverage) |

That last row is key — CBB betting markets are considered less efficient than NFL, especially for mid-major games and early-season matchups. More inefficiency = more potential edge for your model.

---

## Future Phases

- **Phase 2 — Spreads:** Predict margin of victory, backtest spread betting
- **Phase 3 — Player Props:** Player-level models (points, rebounds, assists o/u)
- **Phase 4 — Market Maker Sim:** Flip to sportsbook perspective
- **Phase 5 — NFL Extension:** Port the framework to NFL
- **Phase 6 — Multi-sport:** Unified framework across CBB + NFL
