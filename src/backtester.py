"""
backtester.py - Simulate betting performance using historical data.

Implements:
    - Kelly Criterion bankroll sizing (fractional, 0.25x)
    - Flat betting baseline
    - Edge threshold filtering
    - Full P&L tracking with drawdown analysis
    - Separate tournament vs regular season analysis

Usage:
    from src.backtester import Backtester
    
    bt = Backtester(bankroll=10000, kelly_fraction=0.25, min_edge=0.03)
    results = bt.run(predictions_df)
    bt.summary()
"""

import numpy as np
import pandas as pd
from typing import Optional


class Backtester:
    """
    Simulate betting performance with Kelly Criterion sizing.
    
    Kelly Criterion: f* = (b * p - q) / b
        - f* = fraction of bankroll to wager
        - b  = decimal odds - 1 (payout ratio)
        - p  = model's estimated probability of winning
        - q  = 1 - p
    
    We use fractional Kelly (0.25x) because:
        - Full Kelly is too aggressive, especially in CBB with high variance
        - 0.25x Kelly reduces variance ~75% while keeping ~50% of growth rate
        - Protects against model overconfidence
    """

    def __init__(
        self,
        bankroll: float = 10_000,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.03,
        max_bet_pct: float = 0.05,
        min_bet: float = 10.0,
    ):
        """
        Args:
            bankroll: Starting bankroll ($)
            kelly_fraction: Fraction of full Kelly to bet (0.25 = quarter Kelly)
            min_edge: Minimum edge to place a bet (model_prob - market_prob)
            max_bet_pct: Maximum bet as % of current bankroll (risk cap)
            min_bet: Minimum bet size ($)
        """
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct
        self.min_bet = min_bet

        # Tracking
        self.bet_log: list[dict] = []
        self.bankroll_history: list[float] = [bankroll]

    # ------------------------------------------------------------------
    # Kelly math
    # ------------------------------------------------------------------
    @staticmethod
    def kelly_bet_size(
        model_prob: float,
        market_prob: float,
        bankroll: float,
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.05,
        min_bet: float = 10.0,
    ) -> dict:
        """
        Compute optimal bet size using fractional Kelly Criterion.
        
        Args:
            model_prob: Our model's probability for this outcome
            market_prob: Market's implied probability (from odds)
            bankroll: Current bankroll
            kelly_fraction: Fraction of full Kelly
            max_bet_pct: Max bet as % of bankroll
            min_bet: Minimum bet
        
        Returns:
            Dict with bet_size, edge, kelly_pct, full_kelly_pct.
        """
        edge = model_prob - market_prob

        # Decimal odds from market probability
        if market_prob > 0:
            decimal_odds = 1.0 / market_prob
        else:
            return {"bet_size": 0, "edge": 0, "kelly_pct": 0, "skip_reason": "invalid_odds"}

        b = decimal_odds - 1  # payout ratio
        p = model_prob
        q = 1 - p

        # Full Kelly
        if b > 0:
            full_kelly = (b * p - q) / b
        else:
            full_kelly = 0

        # Apply fractional Kelly
        kelly_pct = max(full_kelly * kelly_fraction, 0)

        # Cap at max_bet_pct
        kelly_pct = min(kelly_pct, max_bet_pct)

        bet_size = kelly_pct * bankroll

        # Apply minimum
        if bet_size < min_bet:
            bet_size = 0  # don't bet below minimum

        return {
            "bet_size": round(bet_size, 2),
            "edge": round(edge, 4),
            "full_kelly_pct": round(full_kelly, 4),
            "kelly_pct": round(kelly_pct, 4),
            "decimal_odds": round(decimal_odds, 3),
        }

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    def run(
        self,
        predictions: pd.DataFrame,
        model_prob_col: str = "model_prob",
        market_prob_col: str = "market_prob",
        actual_col: str = "home_win",
        bet_on_col: str = "bet_on",  # "home" or "away"
    ) -> pd.DataFrame:
        """
        Run the backtest across all predictions.
        
        Args:
            predictions: DataFrame with model probabilities, market probabilities,
                         and actual outcomes. Each row = one game.
            model_prob_col: Column with model's P(home_win)
            market_prob_col: Column with market's P(home_win)
            actual_col: Column with actual result (1 = home win)
        
        For each game:
            1. Compare model prob vs market prob
            2. If edge > min_edge, compute Kelly bet size
            3. Determine if bet won or lost
            4. Update bankroll
        """
        self.bankroll = self.initial_bankroll
        self.bet_log = []
        self.bankroll_history = [self.bankroll]

        for _, row in predictions.iterrows():
            model_home_prob = row[model_prob_col]
            market_home_prob = row[market_prob_col]

            if pd.isna(model_home_prob) or pd.isna(market_home_prob):
                continue

            # Check both sides: could bet home or away
            home_edge = model_home_prob - market_home_prob
            away_edge = (1 - model_home_prob) - (1 - market_home_prob)
            # Note: home_edge == -away_edge, so we just check one direction

            # Decide which side to bet (if any)
            if home_edge >= self.min_edge:
                bet_side = "home"
                bet_prob = model_home_prob
                bet_market_prob = market_home_prob
                bet_won = int(row[actual_col]) == 1
            elif -home_edge >= self.min_edge:  # away has edge
                bet_side = "away"
                bet_prob = 1 - model_home_prob
                bet_market_prob = 1 - market_home_prob
                bet_won = int(row[actual_col]) == 0
            else:
                # No edge, skip game
                continue

            # Compute bet size
            sizing = self.kelly_bet_size(
                bet_prob,
                bet_market_prob,
                self.bankroll,
                self.kelly_fraction,
                self.max_bet_pct,
                self.min_bet,
            )

            bet_size = sizing["bet_size"]
            if bet_size == 0:
                continue

            # Compute payout
            decimal_odds = sizing["decimal_odds"]
            if bet_won:
                profit = bet_size * (decimal_odds - 1)
            else:
                profit = -bet_size

            self.bankroll += profit

            # Log
            record = {
                "date": row.get("date", ""),
                "season": row.get("season", ""),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "bet_side": bet_side,
                "model_prob": round(bet_prob, 4),
                "market_prob": round(bet_market_prob, 4),
                "edge": sizing["edge"],
                "kelly_pct": sizing["kelly_pct"],
                "bet_size": bet_size,
                "decimal_odds": decimal_odds,
                "won": bet_won,
                "profit": round(profit, 2),
                "bankroll": round(self.bankroll, 2),
            }
            
            # Add tournament flag if available
            if "is_tournament" in row:
                record["is_tournament"] = row["is_tournament"]

            self.bet_log.append(record)
            self.bankroll_history.append(self.bankroll)

        return pd.DataFrame(self.bet_log)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Comprehensive backtest summary."""
        if not self.bet_log:
            return {"error": "No bets placed. Check min_edge threshold."}

        log = pd.DataFrame(self.bet_log)
        history = np.array(self.bankroll_history)

        total_bets = len(log)
        wins = log["won"].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets

        total_profit = self.bankroll - self.initial_bankroll
        roi = total_profit / self.initial_bankroll

        # Total wagered
        total_wagered = log["bet_size"].sum()
        yield_pct = total_profit / total_wagered if total_wagered > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(history)
        drawdowns = (history - peak) / peak
        max_drawdown = drawdowns.min()

        # Sharpe ratio (daily returns, annualized)
        if len(history) > 1:
            returns = np.diff(history) / history[:-1]
            sharpe = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )
        else:
            sharpe = 0

        # Average edge on bets placed
        avg_edge = log["edge"].mean()

        # CLV (closing line value) - if we have it
        avg_profit_per_bet = total_profit / total_bets

        summary = {
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": round(self.bankroll, 2),
            "total_profit": round(total_profit, 2),
            "roi": f"{roi:.1%}",
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": f"{win_rate:.1%}",
            "total_wagered": round(total_wagered, 2),
            "yield": f"{yield_pct:.1%}",
            "avg_edge": f"{avg_edge:.1%}",
            "avg_profit_per_bet": round(avg_profit_per_bet, 2),
            "max_drawdown": f"{max_drawdown:.1%}",
            "sharpe_ratio": round(sharpe, 2),
            "kelly_fraction": self.kelly_fraction,
            "min_edge_threshold": self.min_edge,
        }

        print("\n" + "=" * 50)
        print("BACKTEST SUMMARY")
        print("=" * 50)
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")

        return summary

    def summary_by_season(self) -> pd.DataFrame:
        """Break down P&L by season."""
        if not self.bet_log:
            return pd.DataFrame()

        log = pd.DataFrame(self.bet_log)
        
        by_season = log.groupby("season").agg(
            bets=("won", "count"),
            wins=("won", "sum"),
            profit=("profit", "sum"),
            wagered=("bet_size", "sum"),
            avg_edge=("edge", "mean"),
        )
        by_season["win_rate"] = by_season["wins"] / by_season["bets"]
        by_season["yield"] = by_season["profit"] / by_season["wagered"]

        return by_season

    def tournament_summary(self) -> dict:
        """Separate analysis for March Madness games."""
        if not self.bet_log:
            return {}

        log = pd.DataFrame(self.bet_log)
        
        if "is_tournament" not in log.columns:
            return {"note": "No tournament flag in data"}

        tourney = log[log["is_tournament"] == True]
        if tourney.empty:
            return {"note": "No tournament bets"}

        wins = tourney["won"].sum()
        total = len(tourney)

        return {
            "tournament_bets": total,
            "tournament_wins": wins,
            "tournament_win_rate": f"{wins/total:.1%}" if total > 0 else "N/A",
            "tournament_profit": round(tourney["profit"].sum(), 2),
            "tournament_wagered": round(tourney["bet_size"].sum(), 2),
        }


# ---------------------------------------------------------------------------
# Flat betting baseline (for comparison)
# ---------------------------------------------------------------------------
class FlatBetBacktester:
    """Simple flat-bet strategy for baseline comparison."""

    def __init__(
        self,
        bankroll: float = 10_000,
        bet_size: float = 100,
        min_edge: float = 0.03,
    ):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.bet_size = bet_size
        self.min_edge = min_edge
        self.bet_log = []

    def run(self, predictions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Same interface as Backtester.run but with flat bets."""
        self.bankroll = self.initial_bankroll
        self.bet_log = []

        model_prob_col = kwargs.get("model_prob_col", "model_prob")
        market_prob_col = kwargs.get("market_prob_col", "market_prob")
        actual_col = kwargs.get("actual_col", "home_win")

        for _, row in predictions.iterrows():
            model_p = row[model_prob_col]
            market_p = row[market_prob_col]

            if pd.isna(model_p) or pd.isna(market_p):
                continue

            edge = model_p - market_p

            if abs(edge) < self.min_edge:
                continue

            if edge > 0:
                bet_side, bet_prob, bet_market = "home", model_p, market_p
                won = int(row[actual_col]) == 1
            else:
                bet_side = "away"
                bet_prob = 1 - model_p
                bet_market = 1 - market_p
                won = int(row[actual_col]) == 0

            decimal_odds = 1.0 / bet_market if bet_market > 0 else 2.0
            profit = self.bet_size * (decimal_odds - 1) if won else -self.bet_size
            self.bankroll += profit

            self.bet_log.append({
                "bet_side": bet_side,
                "edge": abs(edge),
                "won": won,
                "profit": profit,
                "bankroll": self.bankroll,
            })

        return pd.DataFrame(self.bet_log)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Backtester Demo")
    print("=" * 40)

    # Kelly sizing examples
    print("\nKelly Bet Sizing Examples:")
    examples = [
        (0.60, 0.55),  # small edge
        (0.70, 0.55),  # big edge
        (0.55, 0.53),  # tiny edge
        (0.80, 0.60),  # huge edge
    ]
    for model_p, market_p in examples:
        sizing = Backtester.kelly_bet_size(
            model_p, market_p, bankroll=10000, kelly_fraction=0.25
        )
        print(
            f"  Model: {model_p:.0%} vs Market: {market_p:.0%} "
            f"-> Edge: {sizing['edge']:.1%}, "
            f"Bet: ${sizing['bet_size']:.0f} ({sizing['kelly_pct']:.1%})"
        )
