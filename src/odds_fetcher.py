"""
odds_fetcher.py - Fetch betting odds from The Odds API.

Free tier: 500 requests/month (plenty for March Madness).
Docs: https://the-odds-api.com/liveapi/guides/v4/

Provides:
    - Historical odds (moneylines, spreads, totals) from 2020+
    - Live odds for upcoming games (DraftKings, FanDuel, BetMGM, etc.)
    - Consensus line computation (de-vig and average across books)

Usage:
    from src.odds_fetcher import OddsFetcher
    
    fetcher = OddsFetcher(api_key="your_key")  # or set ODDS_API_KEY env var
    
    # Live odds for upcoming games
    upcoming = fetcher.get_upcoming_odds()
    
    # Historical odds
    historical = fetcher.get_historical_odds(date="2024-03-21")
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "odds"

# The Odds API endpoints
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_ncaab"  # NCAA Men's Basketball


class OddsFetcher:
    """Interface to The Odds API for college basketball odds."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            print(
                "WARNING: No API key set. Set ODDS_API_KEY env var or pass api_key. "
                "Get a free key at https://the-odds-api.com"
            )
        self.requests_used = 0
        self.requests_remaining = None

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        """Make a GET request and track API usage."""
        if not self.api_key:
            raise ValueError("API key required. Set ODDS_API_KEY env var.")

        params = params or {}
        params["apiKey"] = self.api_key

        url = f"{BASE_URL}/{endpoint}"
        resp = requests.get(url, params=params, timeout=15)

        # Track usage from headers
        self.requests_used = resp.headers.get("x-requests-used", self.requests_used)
        self.requests_remaining = resp.headers.get("x-requests-remaining")

        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Live / upcoming odds
    # ------------------------------------------------------------------
    def get_upcoming_odds(
        self,
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
        regions: str = "us",
    ) -> pd.DataFrame:
        """
        Get odds for upcoming NCAA basketball games.
        
        Args:
            markets: Comma-separated markets (h2h=moneyline, spreads, totals)
            bookmakers: Filter to specific books (e.g. "draftkings,fanduel")
            regions: us, uk, eu, au
        
        Returns:
            DataFrame with one row per game, odds from each bookmaker.
        """
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        data = self._get(f"sports/{SPORT}/odds", params)

        if not data:
            print("No upcoming games found.")
            return pd.DataFrame()

        rows = self._parse_odds_response(data)
        df = pd.DataFrame(rows)

        print(f"OK {len(df)} upcoming games with odds")
        print(f"  API requests remaining: {self.requests_remaining}")

        return df

    def get_live_odds(self) -> pd.DataFrame:
        """Get odds for currently live games."""
        params = {
            "regions": "us",
            "markets": "h2h,spreads",
            "oddsFormat": "american",
            "eventIds": "",  # empty = all live
        }
        # Live odds endpoint
        data = self._get(f"sports/{SPORT}/odds", params)
        rows = self._parse_odds_response(data)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Historical odds
    # ------------------------------------------------------------------
    def get_historical_odds(
        self,
        date: str,
        markets: str = "h2h,spreads",
        regions: str = "us",
    ) -> pd.DataFrame:
        """
        Get historical odds snapshot for a specific date.
        
        Args:
            date: ISO format date string (e.g. "2024-03-21")
            markets: Comma-separated markets
            regions: Region filter
        
        Returns:
            DataFrame with odds as they were on the specified date.
        
        Note: Historical odds available from late 2020 onward.
              Each historical request costs more API credits.
        """
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "date": f"{date}T00:00:00Z",
        }

        data = self._get(f"sports/{SPORT}/odds-history", params)

        # Historical endpoint wraps data differently
        if isinstance(data, dict) and "data" in data:
            odds_data = data["data"]
        else:
            odds_data = data

        if not odds_data:
            print(f"No historical odds for {date}")
            return pd.DataFrame()

        rows = self._parse_odds_response(odds_data)
        df = pd.DataFrame(rows)
        df["snapshot_date"] = date

        return df

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_odds_response(self, data: list[dict]) -> list[dict]:
        """
        Parse The Odds API response into flat rows.
        
        Each game becomes one row with columns for each bookmaker's odds.
        """
        rows = []

        for event in data:
            base = {
                "event_id": event.get("id", ""),
                "commence_time": event.get("commence_time", ""),
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
            }

            # Parse each bookmaker's odds
            for book in event.get("bookmakers", []):
                book_key = book["key"]

                for market in book.get("markets", []):
                    market_key = market["key"]

                    for outcome in market.get("outcomes", []):
                        team_type = "home" if outcome["name"] == base["home_team"] else "away"
                        
                        col_prefix = f"{book_key}_{market_key}_{team_type}"
                        base[f"{col_prefix}_price"] = outcome.get("price")
                        
                        if "point" in outcome:
                            base[f"{col_prefix}_point"] = outcome["point"]

            rows.append(base)

        return rows

    # ------------------------------------------------------------------
    # Consensus line computation
    # ------------------------------------------------------------------
    @staticmethod
    def american_to_implied_prob(odds: float) -> float:
        """
        Convert American odds to implied probability.
        
        +150 -> 100 / (150 + 100) = 0.400
        -200 -> 200 / (200 + 100) = 0.667
        """
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    @staticmethod
    def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
        """
        Remove the vig (overround) from two-way odds.
        
        Sportsbooks set odds so probabilities sum to >1 (their edge).
        We normalize back to 1.0 to get "true" implied probabilities.
        
        Example:
            -110 / -110 -> implied 52.4% + 52.4% = 104.8%
            De-vigged: 50% / 50%
        """
        total = prob_a + prob_b
        if total == 0:
            return 0.5, 0.5
        return prob_a / total, prob_b / total

    def compute_consensus(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute consensus (market average) implied probabilities.
        
        For each game:
            1. Find all bookmakers' moneyline (h2h) odds
            2. Convert each to implied probability
            3. Remove each book's vig independently
            4. Average de-vigged probabilities across all books
        
        This IS the consensus closing line - the most efficient estimator
        of true probability. Building it yourself from raw odds is better
        than using a pre-computed one because you can explain the methodology.
        
        Returns:
            DataFrame with consensus_home_prob, consensus_away_prob per game.
        """
        results = []

        # Find all h2h (moneyline) columns
        h2h_home_cols = [c for c in odds_df.columns if "h2h_home_price" in c]
        h2h_away_cols = [c for c in odds_df.columns if "h2h_away_price" in c]

        for _, row in odds_df.iterrows():
            home_probs = []
            away_probs = []

            # Get each book's de-vigged probabilities
            for home_col in h2h_home_cols:
                book = home_col.replace("_h2h_home_price", "")
                away_col = f"{book}_h2h_away_price"

                if away_col not in odds_df.columns:
                    continue

                home_odds = row.get(home_col)
                away_odds = row.get(away_col)

                if pd.isna(home_odds) or pd.isna(away_odds):
                    continue

                # Convert to implied probability
                home_imp = self.american_to_implied_prob(home_odds)
                away_imp = self.american_to_implied_prob(away_odds)

                # Remove this book's vig
                home_fair, away_fair = self.remove_vig(home_imp, away_imp)

                home_probs.append(home_fair)
                away_probs.append(away_fair)

            # Average across all books = consensus
            consensus_home = np.mean(home_probs) if home_probs else np.nan
            consensus_away = np.mean(away_probs) if away_probs else np.nan

            results.append(
                {
                    "event_id": row.get("event_id", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "consensus_home_prob": round(consensus_home, 4),
                    "consensus_away_prob": round(consensus_away, 4),
                    "n_books": len(home_probs),
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Edge calculation
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_edge(
        model_prob: float, market_prob: float
    ) -> dict:
        """
        Calculate betting edge: model probability vs market probability.
        
        edge = model_prob - market_prob
        
        Positive edge = model thinks team is more likely to win than the
        market does -> potential value bet.
        
        Also computes expected value (EV) for a $100 bet:
            EV = (model_prob * payout) - (1 - model_prob) * stake
        """
        edge = model_prob - market_prob

        # Implied American odds from market probability
        if market_prob >= 0.5:
            implied_odds = -(market_prob / (1 - market_prob)) * 100
        else:
            implied_odds = ((1 - market_prob) / market_prob) * 100

        # Payout on $100 bet
        if implied_odds > 0:
            payout = 100 * (implied_odds / 100)
        else:
            payout = 100 * (100 / abs(implied_odds))

        ev = model_prob * payout - (1 - model_prob) * 100

        return {
            "model_prob": round(model_prob, 4),
            "market_prob": round(market_prob, 4),
            "edge": round(edge, 4),
            "implied_odds": round(implied_odds, 1),
            "ev_per_100": round(ev, 2),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_odds(self, df: pd.DataFrame, filename: str):
        """Save odds data to CSV."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"OK Saved odds -> {path}")

    def usage(self) -> dict:
        """Return current API usage stats."""
        return {
            "requests_used": self.requests_used,
            "requests_remaining": self.requests_remaining,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo without API key
    print("Odds Fetcher Demo")
    print("=" * 40)

    # Show conversion examples
    examples = [-200, -150, -110, +100, +150, +200, +300]
    print("\nAmerican Odds -> Implied Probability:")
    for odds in examples:
        prob = OddsFetcher.american_to_implied_prob(odds)
        print(f"  {odds:+d} -> {prob:.1%}")

    # Vig removal example
    print("\nVig Removal (-110 / -110):")
    h = OddsFetcher.american_to_implied_prob(-110)
    a = OddsFetcher.american_to_implied_prob(-110)
    print(f"  Raw implied: {h:.1%} + {a:.1%} = {h+a:.1%} (overround)")
    h_fair, a_fair = OddsFetcher.remove_vig(h, a)
    print(f"  De-vigged:   {h_fair:.1%} + {a_fair:.1%} = {h_fair+a_fair:.1%}")

    # Edge calculation example
    print("\nEdge Calculation:")
    edge = OddsFetcher.calculate_edge(model_prob=0.65, market_prob=0.58)
    for k, v in edge.items():
        print(f"  {k}: {v}")
