"""
rolling_stats.py - Compute rolling team efficiency metrics from box scores.

Replaces Barttorvik with leakage-free, game-by-game rolling stats.
Each game only uses stats from games played BEFORE that date.

Metrics computed (all per 100 possessions where applicable):
    - Offensive Efficiency (OE): points per 100 possessions
    - Defensive Efficiency (DE): opponent points per 100 possessions
    - Net Efficiency: OE - DE
    - Effective FG% (eFG): (FGM + 0.5 * 3PM) / FGA
    - Opponent eFG%: same for opponent
    - Turnover Rate: TO / possessions
    - Forced Turnover Rate: opponent TO / opponent possessions
    - Offensive Rebound %: OREB / (OREB + Opp DREB)
    - Defensive Rebound %: DREB / (DREB + Opp OREB)
    - Free Throw Rate: FTA / FGA
    - 3-Point Rate: 3PA / FGA
    - Assist Rate: AST / FGM
    - Tempo: possessions per 40 minutes

Usage:
    from src.rolling_stats import compute_rolling_features

    features = compute_rolling_features(games_df, boxscores_df, window=12)
"""

import numpy as np
import pandas as pd


def estimate_possessions(fga, oreb, to, fta):
    """
    Estimate possessions using the standard formula.

    Possessions = FGA - OREB + TO + 0.475 * FTA

    The 0.475 accounts for and-1s and technical free throws
    where a possession doesn't actually end.
    """
    return fga - oreb + to + 0.475 * fta


def compute_game_stats(boxscores_flat):
    """
    Compute per-game efficiency stats from flat box score data.

    Input: boxscores_flat.csv (1 row per game, home_ and away_ prefixed)
    Output: DataFrame with per-game stats for each team appearance
    """
    df = boxscores_flat.copy()

    rows = []

    for _, game in df.iterrows():
        game_id = game["game_id"]

        # Process both sides
        for side, opp_side in [("home", "away"), ("away", "home")]:
            try:
                # Extract stats
                fgm = _safe_int(game, "{}_fieldGoalsMade".format(side))
                fga = _safe_int(game, "{}_fieldGoalsAttempted".format(side))
                tpm = _safe_int(game, "{}_threePointFieldGoalsMade".format(side))
                tpa = _safe_int(game, "{}_threePointFieldGoalsAttempted".format(side))
                ftm = _safe_int(game, "{}_freeThrowsMade".format(side))
                fta = _safe_int(game, "{}_freeThrowsAttempted".format(side))
                oreb = _safe_int(game, "{}_offensiveRebounds".format(side))
                dreb = _safe_int(game, "{}_defensiveRebounds".format(side))
                ast = _safe_int(game, "{}_assists".format(side))
                to = _safe_int(game, "{}_turnovers".format(side))
                stl = _safe_int(game, "{}_steals".format(side))
                blk = _safe_int(game, "{}_blocks".format(side))
                # Compute points: 2pt FG + 3pt bonus + FT
                pts = (fgm - tpm) * 2 + tpm * 3 + ftm

                # Opponent stats
                opp_fgm = _safe_int(game, "{}_fieldGoalsMade".format(opp_side))
                opp_fga = _safe_int(game, "{}_fieldGoalsAttempted".format(opp_side))
                opp_tpm = _safe_int(game, "{}_threePointFieldGoalsMade".format(opp_side))
                opp_fta = _safe_int(game, "{}_freeThrowsAttempted".format(opp_side))
                opp_oreb = _safe_int(game, "{}_offensiveRebounds".format(opp_side))
                opp_dreb = _safe_int(game, "{}_defensiveRebounds".format(opp_side))
                opp_to = _safe_int(game, "{}_turnovers".format(opp_side))
                opp_ftm = _safe_int(game, "{}_freeThrowsMade".format(opp_side))
                opp_pts = (opp_fgm - opp_tpm) * 2 + opp_tpm * 3 + opp_ftm

                if fga == 0 or opp_fga == 0:
                    continue

                # Possessions
                poss = estimate_possessions(fga, oreb, to, fta)
                opp_poss = estimate_possessions(opp_fga, opp_oreb, opp_to, opp_fta)

                if poss <= 0 or opp_poss <= 0:
                    continue

                # Use average of both teams' possession estimates for consistency
                avg_poss = (poss + opp_poss) / 2

                row = {
                    "game_id": game_id,
                    "team_name": game.get("{}_team_name".format(side), ""),
                    "team_id": game.get("{}_team_id".format(side), ""),
                    "opp_team_name": game.get("{}_team_name".format(opp_side), ""),
                    "side": side,

                    # Raw counts (for aggregation)
                    "pts": pts,
                    "opp_pts": opp_pts,
                    "fgm": fgm,
                    "fga": fga,
                    "tpm": tpm,
                    "tpa": tpa,
                    "ftm": ftm,
                    "fta": fta,
                    "oreb": oreb,
                    "dreb": dreb,
                    "ast": ast,
                    "to": to,
                    "stl": stl,
                    "blk": blk,
                    "poss": avg_poss,

                    # Opponent raw
                    "opp_fgm": opp_fgm,
                    "opp_fga": opp_fga,
                    "opp_tpm": opp_tpm,
                    "opp_fta": opp_fta,
                    "opp_oreb": opp_oreb,
                    "opp_dreb": opp_dreb,
                    "opp_to": opp_to,
                    "opp_poss": opp_poss,

                    # Per-game efficiency
                    "oe": pts / avg_poss * 100,
                    "de": opp_pts / avg_poss * 100,
                    "net_eff": (pts - opp_pts) / avg_poss * 100,
                    "efg": (fgm + 0.5 * tpm) / fga if fga > 0 else np.nan,
                    "opp_efg": (opp_fgm + 0.5 * opp_tpm) / opp_fga if opp_fga > 0 else np.nan,
                    "to_rate": to / avg_poss if avg_poss > 0 else np.nan,
                    "forced_to_rate": opp_to / avg_poss if avg_poss > 0 else np.nan,
                    "orb_pct": oreb / (oreb + opp_dreb) if (oreb + opp_dreb) > 0 else np.nan,
                    "drb_pct": dreb / (dreb + opp_oreb) if (dreb + opp_oreb) > 0 else np.nan,
                    "ft_rate": fta / fga if fga > 0 else np.nan,
                    "tpr": tpa / fga if fga > 0 else np.nan,
                    "ast_rate": ast / fgm if fgm > 0 else np.nan,
                    "tempo": avg_poss / 40 * 40,  # possessions per 40 min (standard game)
                }

                rows.append(row)

            except Exception:
                continue

    result = pd.DataFrame(rows)
    print("Computed per-game stats: {} team-game rows".format(len(result)))
    return result


def compute_rolling_features(
    games,
    game_stats,
    window=12,
    min_games=3,
):
    """
    Compute rolling team stats and attach them to each game as features.

    For each game, looks up both teams' rolling averages BEFORE that game
    (no data leakage).

    Args:
        games: all_games.csv with game_id, date, season, home_team, away_team
        game_stats: per-game stats from compute_game_stats()
        window: number of recent games to average (default 12)
        min_games: minimum games needed before producing stats (default 3)

    Returns:
        DataFrame with one row per game, rolling stats for both teams
    """
    games = games.copy()
    games["date_dt"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.sort_values("date_dt").reset_index(drop=True)

    # Also sort game_stats by joining date
    game_stats = game_stats.copy()
    game_dates = games[["game_id", "date_dt"]].drop_duplicates("game_id")
    game_stats["game_id"] = game_stats["game_id"].astype(str)
    game_dates["game_id"] = game_dates["game_id"].astype(str)
    game_stats = game_stats.merge(game_dates, on="game_id", how="left")
    game_stats = game_stats.sort_values("date_dt").reset_index(drop=True)

    # Stats to compute rolling averages for
    rolling_cols = [
        "oe", "de", "net_eff", "efg", "opp_efg",
        "to_rate", "forced_to_rate", "orb_pct", "drb_pct",
        "ft_rate", "tpr", "ast_rate", "tempo",
    ]

    # Build rolling stats per team using a dict for speed
    # team_id -> list of recent game stats (ordered by date)
    team_history = {}

    # Pre-build lookup: game_id -> {team_id: stats_dict}
    game_team_stats = {}
    for _, row in game_stats.iterrows():
        gid = str(row["game_id"])
        tid = str(row["team_id"])
        if gid not in game_team_stats:
            game_team_stats[gid] = {}
        game_team_stats[gid][tid] = row

    # Process games chronologically
    home_rolling = []
    away_rolling = []

    print("Computing rolling stats (window={})...".format(window))

    for idx, game in games.iterrows():
        gid = str(game["game_id"])
        home_id = str(game.get("home_id", ""))
        away_id = str(game.get("away_id", ""))

        # Get rolling averages BEFORE this game
        home_stats = _get_rolling_avg(team_history, home_id, rolling_cols, window, min_games)
        away_stats = _get_rolling_avg(team_history, away_id, rolling_cols, window, min_games)

        home_rolling.append(home_stats)
        away_rolling.append(away_stats)

        # Now update history with this game's stats
        if gid in game_team_stats:
            for tid, stats in game_team_stats[gid].items():
                if tid not in team_history:
                    team_history[tid] = []
                team_history[tid].append({col: stats.get(col, np.nan) for col in rolling_cols})
                # Keep only recent window
                if len(team_history[tid]) > window * 2:
                    team_history[tid] = team_history[tid][-window * 2:]

        if (idx + 1) % 5000 == 0:
            print("  [{}/{}] {:.0f}%".format(idx + 1, len(games), 100 * (idx + 1) / len(games)))

    # Build output DataFrame
    home_df = pd.DataFrame(home_rolling)
    away_df = pd.DataFrame(away_rolling)

    home_df.columns = ["home_roll_{}".format(c) for c in home_df.columns]
    away_df.columns = ["away_roll_{}".format(c) for c in away_df.columns]

    result = pd.concat([games.reset_index(drop=True), home_df, away_df], axis=1)

    # Compute differentials
    for col in rolling_cols:
        hcol = "home_roll_{}".format(col)
        acol = "away_roll_{}".format(col)
        if hcol in result.columns and acol in result.columns:
            result["roll_{}_diff".format(col)] = result[hcol] - result[acol]

    # Drop temp columns
    if "date_dt" in result.columns:
        result = result.drop(columns=["date_dt"])

    # Report coverage
    coverage = result["home_roll_oe"].notna().mean()
    print("Rolling stats coverage: {:.1f}% of games".format(100 * coverage))

    return result


def _get_rolling_avg(team_history, team_id, cols, window, min_games):
    """Get rolling average stats for a team (pre-game, no leakage)."""
    if team_id not in team_history or len(team_history[team_id]) < min_games:
        return {col: np.nan for col in cols}

    recent = team_history[team_id][-window:]
    avgs = {}
    for col in cols:
        vals = [g[col] for g in recent if not np.isnan(g.get(col, np.nan))]
        avgs[col] = np.mean(vals) if vals else np.nan

    return avgs


def _safe_int(row, key):
    """Safely extract an integer from a row, handling missing/string values."""
    val = row.get(key, 0)
    if pd.isna(val):
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    DATA_DIR = Path("data")
    PROCESSED_DIR = DATA_DIR / "processed"

    # Load data
    print("Loading box scores...")
    boxscores_flat = pd.read_csv(PROCESSED_DIR / "boxscores_flat.csv")

    print("Loading games...")
    games = pd.read_csv(PROCESSED_DIR / "all_games.csv")

    # Step 1: Compute per-game stats
    print("\n--- Per-Game Stats ---")
    game_stats = compute_game_stats(boxscores_flat)
    game_stats.to_csv(PROCESSED_DIR / "game_stats.csv", index=False)
    print("Saved per-game stats")

    # Step 2: Compute rolling features
    print("\n--- Rolling Features ---")
    features = compute_rolling_features(games, game_stats, window=12)
    features.to_csv(PROCESSED_DIR / "rolling_features.csv", index=False)
    print("Saved rolling features")

    # Summary
    print("\n--- Summary ---")
    roll_cols = [c for c in features.columns if c.startswith("roll_") and c.endswith("_diff")]
    print("Differential features: {}".format(len(roll_cols)))
    for c in roll_cols:
        pct_avail = features[c].notna().mean()
        print("  {}: {:.1f}% available".format(c, 100 * pct_avail))
