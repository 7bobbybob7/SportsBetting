"""
elo.py - Custom ELO rating system for college basketball.

CBB-specific adaptations vs standard ELO:
    1. Heavy season regression (40-50%) due to roster turnover / transfer portal
    2. Large home-court advantage (~3.5 points = ~100 ELO points)
    3. Margin-of-victory multiplier with cap (avoid overweighting blowouts)
    4. Neutral-site game handling (tournaments, March Madness)
    5. 363-team ecosystem with massive strength variance

The ELO system serves two purposes:
    A. Standalone prediction model (ELO difference -> win probability)
    B. Feature generator for logistic regression / XGBoost models

Usage:
    from src.elo import EloRater
    
    elo = EloRater()
    results = elo.rate_seasons(games_df, start_year=2018, end_year=2025)
    
    # Get current ratings
    elo.get_ratings()
    
    # Predict a game
    prob = elo.predict("Duke", "North Carolina", neutral=False)
"""

import numpy as np
import pandas as pd
from typing import Optional


class EloRater:
    """
    College basketball ELO rating system.
    
    Hyperparameters (all tunable):
        base_k:             Base K-factor controlling update magnitude (default: 20)
        home_advantage:     ELO points added to home team (default: 100 ~ 3.5 pts)
        season_regression:  Fraction regressed toward mean each season (default: 0.45)
        mov_multiplier:     Whether to scale updates by margin of victory (default: True)
        mov_cap:            Max margin counted in MOV multiplier (default: 25)
        initial_elo:        Starting ELO for all teams (default: 1500)
    """

    def __init__(
        self,
        base_k: float = 20.0,
        home_advantage: float = 100.0,
        season_regression: float = 0.45,
        mov_multiplier: bool = True,
        mov_cap: int = 25,
        initial_elo: float = 1500.0,
    ):
        self.base_k = base_k
        self.home_advantage = home_advantage
        self.season_regression = season_regression
        self.mov_multiplier = mov_multiplier
        self.mov_cap = mov_cap
        self.initial_elo = initial_elo

        # Current ratings: team_name -> elo
        self.ratings: dict[str, float] = {}

        # Full history: list of dicts for every game processed
        self.game_log: list[dict] = []

        # ELO snapshots: team_name -> list of (date, elo)
        self.history: dict[str, list[tuple]] = {}

    # ------------------------------------------------------------------
    # Core ELO math
    # ------------------------------------------------------------------
    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """
        Standard ELO expected score (win probability for team A).
        
        P(A wins) = 1 / (1 + 10^((B - A) / 400))
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _mov_factor(self, margin: int) -> float:
        """
        Margin-of-victory multiplier.
        
        Scales the K-factor by how much the winner won by, with diminishing
        returns and a hard cap. This rewards convincing wins more than
        squeakers, but prevents 40-point blowouts against cupcakes from
        inflating ratings.
        
        Formula: log(1 + min(|margin|, cap)) / log(1 + cap) * scale
        
        The log ensures diminishing returns:
            - 5-point win  -> ~1.0x
            - 10-point win -> ~1.3x
            - 20-point win -> ~1.7x
            - 25-point win -> ~1.85x (cap)
        """
        if not self.mov_multiplier:
            return 1.0

        capped_margin = min(abs(margin), self.mov_cap)
        # Normalize so a ~5-point win gives ~1.0x multiplier
        factor = np.log1p(capped_margin) / np.log1p(5)
        return max(factor, 0.5)  # floor at 0.5x to never under-weight a game

    def _get_k(self, margin: int) -> float:
        """Effective K-factor for a game, adjusted by margin of victory."""
        return self.base_k * self._mov_factor(margin)

    # ------------------------------------------------------------------
    # Season management
    # ------------------------------------------------------------------
    def regress_to_mean(self):
        """
        Regress all ratings toward 1500 at the start of a new season.
        
        CBB uses heavier regression than NFL (~45% vs ~33%) because:
            - Transfer portal: ~30-40% roster turnover annually
            - Freshmen: top recruits immediately impact rosters
            - Graduation: seniors leave every year
            - Coaching changes: frequent, especially mid-majors
        
        new_elo = mean + (1 - regression) * (old_elo - mean)
        
        With 45% regression:
            - A 1700 team -> 1500 + 0.55 * 200 = 1610
            - A 1300 team -> 1500 + 0.55 * (-200) = 1390
        """
        for team in self.ratings:
            self.ratings[team] = (
                self.initial_elo
                + (1 - self.season_regression) 
                * (self.ratings[team] - self.initial_elo)
            )

    def _ensure_team(self, team: str):
        """Initialize a team at the base ELO if we haven't seen them."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_elo
            self.history[team] = []

    # ------------------------------------------------------------------
    # Process a single game
    # ------------------------------------------------------------------
    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        date: str = "",
        game_id: str = "",
        season: int = 0,
    ) -> dict:
        """
        Process one game result and update ratings.
        
        Args:
            home_team:  Name of the home team
            away_team:  Name of the away team
            home_score: Final score for home team
            away_score: Final score for away team
            neutral:    True if played at a neutral site
            date:       Game date (for logging)
            game_id:    Unique game identifier
            season:     Season year (NCAA convention)
        
        Returns:
            Dict with pre-game ratings, prediction, actual result, and updates.
        """
        self._ensure_team(home_team)
        self._ensure_team(away_team)

        # Pre-game ratings
        home_elo = self.ratings[home_team]
        away_elo = self.ratings[away_team]

        # Apply home-court advantage (skip for neutral sites)
        hca = 0.0 if neutral else self.home_advantage
        home_elo_adj = home_elo + hca

        # Pre-game win probability
        home_win_prob = self.expected_score(home_elo_adj, away_elo)

        # Actual result
        margin = home_score - away_score
        home_win = 1 if margin > 0 else 0
        actual = 1.0 if margin > 0 else 0.0  # for ELO update (0.5 for ties, but CBB doesn't tie)

        # Effective K-factor
        k = self._get_k(margin)

        # ELO update
        home_delta = k * (actual - home_win_prob)
        away_delta = -home_delta  # zero-sum

        # Apply updates
        self.ratings[home_team] += home_delta
        self.ratings[away_team] += away_delta

        # Log everything
        record = {
            "game_id": game_id,
            "date": date,
            "season": season,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "margin": margin,
            "neutral": neutral,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "home_elo_adj": home_elo_adj,
            "home_win_prob": home_win_prob,
            "home_win": home_win,
            "k_factor": k,
            "home_delta": home_delta,
            "home_elo_post": self.ratings[home_team],
            "away_elo_post": self.ratings[away_team],
        }

        self.game_log.append(record)
        self.history[home_team].append((date, self.ratings[home_team]))
        self.history[away_team].append((date, self.ratings[away_team]))

        return record

    # ------------------------------------------------------------------
    # Process full seasons
    # ------------------------------------------------------------------
    def rate_seasons(
        self,
        games: pd.DataFrame,
        start_year: int = 2018,
        end_year: int = 2025,
        home_col: str = "home_team",
        away_col: str = "away_team",
        home_score_col: str = "home_score",
        away_score_col: str = "away_score",
        date_col: str = "date",
        season_col: str = "season",
        neutral_col: str = "neutral_site",
        game_id_col: str = "game_id",
    ) -> pd.DataFrame:
        """
        Process multiple seasons of games chronologically.
        
        Applies season regression between each season, then processes
        all games within each season in date order.
        
        Args:
            games: DataFrame with game results (must have the column names
                   specified by the *_col parameters)
            start_year, end_year: Range of seasons to process
            *_col: Column name mappings
        
        Returns:
            DataFrame of the full game log with ELO predictions.
        """
        self.game_log = []  # reset

        for season in range(start_year, end_year + 1):
            # Regress ratings at the start of each new season
            if season > start_year:
                self.regress_to_mean()
                print(f"  -> Season {season}: regressed to mean (regression={self.season_regression})")

            # Filter games for this season
            season_games = games[games[season_col] == season].copy()
            
            if season_games.empty:
                print(f"  WARNING: No games for season {season}")
                continue

            # Sort by date
            season_games = season_games.sort_values(date_col).reset_index(drop=True)

            print(f"  Processing {len(season_games)} games for {season}...")

            for _, row in season_games.iterrows():
                # Handle neutral site - could be bool, string, NaN
                neutral = False
                if neutral_col in row.index:
                    val = row[neutral_col]
                    neutral = bool(val) if pd.notna(val) else False

                self.update(
                    home_team=str(row[home_col]),
                    away_team=str(row[away_col]),
                    home_score=int(row[home_score_col]),
                    away_score=int(row[away_score_col]),
                    neutral=neutral,
                    date=str(row.get(date_col, "")),
                    game_id=str(row.get(game_id_col, "")),
                    season=season,
                )

        result = pd.DataFrame(self.game_log)
        print(f"\nOK Processed {len(result)} total games")
        print(f"  Teams rated: {len(self.ratings)}")
        
        return result

    # ------------------------------------------------------------------
    # Predictions & queries
    # ------------------------------------------------------------------
    def predict(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> dict:
        """
        Predict outcome of a future game using current ratings.
        
        Returns dict with:
            - home_win_prob: probability home team wins
            - away_win_prob: probability away team wins
            - home_elo, away_elo: current ratings
            - elo_diff: adjusted difference (home - away + HCA)
            - predicted_margin: rough point spread estimate
        """
        self._ensure_team(home_team)
        self._ensure_team(away_team)

        home_elo = self.ratings[home_team]
        away_elo = self.ratings[away_team]

        hca = 0.0 if neutral else self.home_advantage
        elo_diff = (home_elo + hca) - away_elo

        home_prob = self.expected_score(home_elo + hca, away_elo)

        # Rough margin estimate: ~28 ELO points ~ 1 point
        predicted_margin = elo_diff / 28.0

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": elo_diff,
            "home_win_prob": round(home_prob, 4),
            "away_win_prob": round(1 - home_prob, 4),
            "predicted_margin": round(predicted_margin, 1),
            "neutral": neutral,
        }

    def get_ratings(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Return current ratings as a sorted DataFrame."""
        df = pd.DataFrame(
            [{"team": k, "elo": v} for k, v in self.ratings.items()]
        )
        df = df.sort_values("elo", ascending=False).reset_index(drop=True)
        df.index += 1  # 1-indexed ranking
        df.index.name = "rank"

        if top_n:
            return df.head(top_n)
        return df

    def get_team_history(self, team: str) -> pd.DataFrame:
        """Return ELO history for a specific team."""
        if team not in self.history:
            return pd.DataFrame()

        df = pd.DataFrame(self.history[team], columns=["date", "elo"])
        df["team"] = team
        return df

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, game_log: Optional[pd.DataFrame] = None) -> dict:
        """
        Evaluate prediction accuracy of the ELO system.
        
        Metrics:
            - accuracy: % of games where the favorite won
            - log_loss: negative log-likelihood (lower = better calibration)
            - brier_score: mean squared error of probabilities (lower = better)
            - calibration: binned predicted prob vs actual win rate
        """
        if game_log is None:
            game_log = pd.DataFrame(self.game_log)

        if game_log.empty:
            return {}

        probs = game_log["home_win_prob"].values
        actuals = game_log["home_win"].values

        # Accuracy: did the higher-prob team win?
        predicted_home_win = (probs >= 0.5).astype(int)
        accuracy = np.mean(predicted_home_win == actuals)

        # Log loss
        eps = 1e-15
        clipped = np.clip(probs, eps, 1 - eps)
        log_loss = -np.mean(
            actuals * np.log(clipped) + (1 - actuals) * np.log(1 - clipped)
        )

        # Brier score
        brier = np.mean((probs - actuals) ** 2)

        # Calibration bins
        bins = np.arange(0, 1.05, 0.1)
        calibration = []
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                calibration.append(
                    {
                        "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                        "predicted_mean": probs[mask].mean(),
                        "actual_mean": actuals[mask].mean(),
                        "count": int(mask.sum()),
                    }
                )

        return {
            "accuracy": round(accuracy, 4),
            "log_loss": round(log_loss, 4),
            "brier_score": round(brier, 4),
            "n_games": len(game_log),
            "calibration": calibration,
        }

    def evaluate_by_season(
        self, game_log: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Break down evaluation metrics by season."""
        if game_log is None:
            game_log = pd.DataFrame(self.game_log)

        results = []
        for season, group in game_log.groupby("season"):
            metrics = self.evaluate(group)
            metrics["season"] = season
            results.append(metrics)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save ratings and game log to disk."""
        import json
        
        data = {
            "ratings": self.ratings,
            "params": {
                "base_k": self.base_k,
                "home_advantage": self.home_advantage,
                "season_regression": self.season_regression,
                "mov_multiplier": self.mov_multiplier,
                "mov_cap": self.mov_cap,
                "initial_elo": self.initial_elo,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"OK Saved ELO ratings -> {path}")

    def load(self, path: str):
        """Load ratings from disk."""
        import json
        
        with open(path) as f:
            data = json.load(f)
        
        self.ratings = data["ratings"]
        params = data.get("params", {})
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        
        print(f"OK Loaded {len(self.ratings)} team ratings from {path}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_elo(
    games: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    param_grid: Optional[dict] = None,
) -> dict:
    """
    Grid search over ELO hyperparameters to minimize log loss.
    
    Args:
        games: Full game DataFrame
        train_seasons: Seasons to use for training (building up ratings)
        val_seasons: Seasons to evaluate on
        param_grid: Dict of param_name -> list of values to try.
                    Defaults to a reasonable CBB search space.
    
    Returns:
        Dict with best parameters and their validation log loss.
    """
    if param_grid is None:
        param_grid = {
            "base_k": [15, 20, 25, 30],
            "home_advantage": [75, 100, 125],
            "season_regression": [0.35, 0.40, 0.45, 0.50],
            "mov_cap": [20, 25, 30],
        }

    from itertools import product

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))

    best_loss = float("inf")
    best_params = {}
    results = []

    all_seasons = sorted(set(train_seasons) | set(val_seasons))
    start_year = min(all_seasons)
    end_year = max(all_seasons)

    print(f"Tuning ELO: {len(combos)} combinations")
    print(f"  Train: {train_seasons}")
    print(f"  Val:   {val_seasons}")
    print()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        elo = EloRater(**params)
        game_log = elo.rate_seasons(games, start_year, end_year)

        # Evaluate only on validation seasons
        val_log = game_log[game_log["season"].isin(val_seasons)]
        
        if val_log.empty:
            continue
            
        metrics = elo.evaluate(val_log)
        log_loss = metrics["log_loss"]

        results.append({**params, "log_loss": log_loss, "accuracy": metrics["accuracy"]})

        if log_loss < best_loss:
            best_loss = log_loss
            best_params = params.copy()

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(combos)}] Best so far: {best_loss:.4f} | {best_params}")

    print(f"\nOK Best params: {best_params}")
    print(f"  Validation log loss: {best_loss:.4f}")

    return {
        "best_params": best_params,
        "best_log_loss": best_loss,
        "all_results": pd.DataFrame(results).sort_values("log_loss"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demo with synthetic data
    print("ELO System Demo")
    print("=" * 40)

    elo = EloRater()

    # Simulate a few games
    games = [
        ("Duke", "UNC", 78, 72, False),
        ("Kansas", "Kentucky", 85, 80, True),  # neutral
        ("Gonzaga", "Duke", 70, 65, False),
        ("UNC", "Kansas", 82, 79, False),
        ("Kentucky", "Gonzaga", 75, 88, False),
    ]

    for home, away, hs, as_, neutral in games:
        result = elo.update(home, away, hs, as_, neutral=neutral, date="2024-01-01")
        print(
            f"  {home} {hs} - {away} {as_} "
            f"(pred: {result['home_win_prob']:.1%}) "
            f"-> delta {result['home_delta']:+.1f}"
        )

    print("\nRatings:")
    print(elo.get_ratings())

    print("\nEvaluation:")
    metrics = elo.evaluate()
    for k, v in metrics.items():
        if k != "calibration":
            print(f"  {k}: {v}")
