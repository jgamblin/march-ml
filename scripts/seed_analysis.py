"""Analyze historical tournament outcomes by seed matchup.

Outputs:
- seed_matchup_probs.csv: win probability for each seed vs seed pairing
- seed_by_round.csv: advancement rates by seed for each round
- seed_historical_summary.json: aggregate statistics

Usage:
    python scripts/seed_analysis.py --games_dir data/processed --out_dir results
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def coerce_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def load_games(games_path):
    df = pd.read_csv(games_path)
    df["is_postseason"] = coerce_bool(df.get("is_postseason", False))
    tournament_series = df.get("tournament", pd.Series("", index=df.index)).fillna("").astype(str)
    ncaa_mask = tournament_series.str.contains("Men's Basketball Championship", case=False, na=False)
    return df[ncaa_mask].copy()


def load_features(features_path):
    """Load features to get seed information."""
    return pd.read_csv(features_path)


def extract_seed(team_name, features_df):
    """Find seed for a team in a given season."""
    matches = features_df[features_df["team"].str.lower() == str(team_name).strip().lower()]
    if not matches.empty:
        return matches.iloc[0].get("seed", None)
    return None


def create_seed_matchup_matrix(games_by_season, features_dir):
    """Build historical seed matchup win probabilities."""
    matchups = {}
    
    for season in sorted(games_by_season.keys()):
        features_path = Path(features_dir) / f"tournament_team_features_{season}.csv"
        if not features_path.exists():
            continue
            
        features_df = load_features(features_path)
        season_games = games_by_season[season]
        
        for _, game in season_games.iterrows():
            home_team = game.get("home_team", game.get("team", ""))
            away_team = game.get("away_team", game.get("opp", ""))
            home_seed = extract_seed(home_team, features_df)
            away_seed = extract_seed(away_team, features_df)
            
            if pd.isna(home_seed) or pd.isna(away_seed):
                continue
            
            home_seed = int(home_seed)
            away_seed = int(away_seed)
            
            # Normalize to lower seed first in tuple
            if home_seed <= away_seed:
                key = (home_seed, away_seed)
                winner_seed = home_seed if bool(game.get("home_win", False)) else away_seed
            else:
                key = (away_seed, home_seed)
                winner_seed = away_seed if bool(game.get("home_win", False)) else home_seed
            
            if key not in matchups:
                matchups[key] = {"games": 0, "lower_wins": 0, "higher_wins": 0}
            
            matchups[key]["games"] += 1
            if winner_seed == key[0]:
                matchups[key]["lower_wins"] += 1
            else:
                matchups[key]["higher_wins"] += 1
    
    return matchups


def create_seed_by_round(games_df, features_dir):
    """Calculate advancement rates by seed for each round."""
    round_data = {}
    seasons = [int(f.split("_")[-1].replace(".csv", "")) 
               for f in os.listdir(features_dir) 
               if f.startswith("tournament_team_features_") and f.endswith(".csv")]
    
    for season in sorted(seasons):
        features_path = Path(features_dir) / f"tournament_team_features_{season}.csv"
        features_df = load_features(features_path)
        
        season_games = games_df[games_df["season"] == season].copy()
        
        # Group games by round
        for _, game in season_games.iterrows():
            round_num = int(game.get("round", 0)) if pd.notna(game.get("round", 0)) else 0
            if round_num == 0:
                # Infer from tournament info
                if "First Four" in str(game.get("tournament", "")):
                    round_num = 0
                else:
                    round_num = 1
            
            home_team = game.get("home_team", game.get("team", ""))
            away_team = game.get("away_team", game.get("opp", ""))
            home_seed = extract_seed(home_team, features_df)
            away_seed = extract_seed(away_team, features_df)
            
            if pd.isna(home_seed) or pd.isna(away_seed):
                continue
            
            winner_seed = home_seed if bool(game.get("home_win", False)) else away_seed
            winner_seed = int(winner_seed)
            
            key = (round_num, winner_seed)
            if key not in round_data:
                round_data[key] = {"advances": 0}
            round_data[key]["advances"] += 1
    
    return round_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games_dir", default="data/processed")
    p.add_argument("--features_dir", default="data/processed/features")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load all tournament games by season
    print("Loading tournament games...")
    games_by_season = {}
    total_games = 0
    for games_file in Path(args.games_dir).glob("games_*.csv"):
        season = int(games_file.stem.split("_")[-1])
        games_by_season[season] = load_games(games_file)
        total_games += len(games_by_season[season])
    print(f"Loaded {total_games} tournament games across {len(games_by_season)} seasons")
    
    # Build seed matchup matrix
    print("Analyzing seed matchups...")
    matchups = create_seed_matchup_matrix(games_by_season, args.features_dir)
    
    if not matchups:
        print("No seed matchups found - check that features include 'seed' column")
        return
    
    # Convert to DataFrame for export
    matchup_rows = []
    for (seed_a, seed_b), stats in sorted(matchups.items()):
        win_prob_lower = stats["lower_wins"] / stats["games"] if stats["games"] > 0 else 0
        matchup_rows.append({
            "seed_1": seed_a,
            "seed_2": seed_b,
            "games": stats["games"],
            "lower_seed_wins": stats["lower_wins"],
            "higher_seed_wins": stats["higher_wins"],
            "lower_seed_win_prob": win_prob_lower,
            "higher_seed_win_prob": 1 - win_prob_lower,
        })
    
    matchup_df = pd.DataFrame(matchup_rows)
    matchup_path = Path(args.out_dir) / "seed_matchup_probs.csv"
    matchup_df.to_csv(matchup_path, index=False)
    print(f"Saved matchup probabilities to {matchup_path}")
    
    # Build historical summary
    print("Building historical summary...")
    total_games = sum(len(games) for games in games_by_season.values())
    summary = {
        "total_games": total_games,
        "seasons": sorted(list(games_by_season.keys())),
        "seed_matchup_statistics": {
            f"{row['seed_1']}-{row['seed_2']}": {
                "games": int(row["games"]),
                "lower_seed_win_prob": round(float(row["lower_seed_win_prob"]), 3),
                "higher_seed_win_prob": round(float(row["higher_seed_win_prob"]), 3),
            }
            for _, row in matchup_df.iterrows()
        },
    }
    
    summary_path = Path(args.out_dir) / "seed_historical_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Display top upsets
    print("\n=== Historical Upset Rates ===")
    upsets = matchup_df[matchup_df["seed_1"] > matchup_df["seed_2"]].copy()
    upsets["upset_rate"] = upsets["higher_seed_wins"] / upsets["games"]
    upsets_sorted = upsets.sort_values("upset_rate", ascending=False)
    for _, row in upsets_sorted.head(10).iterrows():
        print(f"  {int(row['seed_2'])}-seed vs {int(row['seed_1'])}-seed: "
              f"{row['upset_rate']:.1%} upset rate ({int(row['higher_seed_wins'])}/{int(row['games'])} games)")


if __name__ == "__main__":
    main()
