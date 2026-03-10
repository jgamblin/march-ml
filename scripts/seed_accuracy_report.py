"""Generate historical seed accuracy report comparing actual vs predicted.

Outputs:
- seed_accuracy_report.csv: actual advancement rates by seed
- seed_accuracy_report.json: detailed analysis with predictions

Usage:
    python scripts/seed_accuracy_report.py --games_dir data/processed --features_dir data/processed/features --out_dir results
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
    return pd.read_csv(features_path)


def extract_seed(team_name, features_df):
    """Find seed for a team."""
    matches = features_df[features_df["team"].str.lower() == str(team_name).strip().lower()]
    if not matches.empty:
        return int(matches.iloc[0].get("seed", 0))
    return 0


def infer_round(game_row):
    """Infer tournament round from game info."""
    tournament_info = str(game_row.get("tournament", ""))
    if "First Four" in tournament_info:
        return 0
    if "Round of 64" in tournament_info or tournament_info.count("Round") == 1:
        return 1
    if "Round of 32" in tournament_info or tournament_info.count("Round") == 2:
        return 2
    if "Sweet 16" in tournament_info or "Elite Eight" in tournament_info:
        return 3
    if "Final Four" in tournament_info or "Championship" in tournament_info:
        return 4
    # Infer from explicit round field
    if "round" in game_row.index:
        return int(game_row.get("round", 1))
    return 1


def analyze_seed_performance(games_df, features_dir):
    """Analyze how often each seed advances by round."""
    rounds = {0: "First Four", 1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8+"}
    
    seed_advances = {}
    seed_appearances = {}
    
    seasons = []
    for f in os.listdir(features_dir):
        if f.startswith("tournament_team_features_") and f.endswith(".csv"):
            season = int(f.split("_")[-1].replace(".csv", ""))
            seasons.append(season)
    
    for season in sorted(seasons):
        features_path = Path(features_dir) / f"tournament_team_features_{season}.csv"
        features_df = load_features(features_path)
        
        season_games = games_df[games_df["season"] == season].copy()
        
        # Track which seeds play in each round
        for _, game in season_games.iterrows():
            round_num = infer_round(game)
            
            home_team = game.get("home_team", game.get("team", ""))
            away_team = game.get("away_team", game.get("opp", ""))
            home_seed = extract_seed(home_team, features_df)
            away_seed = extract_seed(away_team, features_df)
            
            home_won = bool(coerce_bool(pd.Series([game.get("home_win", False)]))[0])
            
            # Track appearance and advancement
            for seed in [home_seed, away_seed]:
                if seed == 0:
                    continue
                
                key = seed
                if key not in seed_appearances:
                    seed_appearances[key] = {r: 0 for r in rounds.keys()}
                    seed_advances[key] = {r: 0 for r in rounds.keys()}
                
                seed_appearances[key][round_num] += 1
                
                if (seed == home_seed and home_won) or (seed == away_seed and not home_won):
                    seed_advances[key][round_num] += 1
    
    return seed_appearances, seed_advances


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games_dir", default="data/processed")
    p.add_argument("--features_dir", default="data/processed/features")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading tournament games...")
    all_games = []
    for games_file in Path(args.games_dir).glob("games_*.csv"):
        all_games.append(load_games(games_file))
    games_df = pd.concat(all_games, ignore_index=True)
    print(f"Loaded {len(games_df)} tournament games")
    
    print("Analyzing historical seed performance...")
    seed_appearances, seed_advances = analyze_seed_performance(games_df, args.features_dir)
    
    # Build report
    rounds = {0: "First Four", 1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8+"}
    report_rows = []
    
    for seed in sorted(seed_appearances.keys()):
        for round_num, round_name in rounds.items():
            appearances = seed_appearances[seed].get(round_num, 0)
            advances = seed_advances[seed].get(round_num, 0)
            win_rate = advances / appearances if appearances > 0 else 0
            
            report_rows.append({
                "seed": seed,
                "round": round_name,
                "appearances": appearances,
                "advances": advances,
                "win_rate": win_rate,
            })
    
    report_df = pd.DataFrame(report_rows)
    csv_path = Path(args.out_dir) / "seed_accuracy_report.csv"
    report_df.to_csv(csv_path, index=False)
    print(f"Saved report to {csv_path}")
    
    # Build JSON with analysis
    json_data = {
        "analysis": "Historical seed performance in NCAA tournaments (2021-2025)",
        "seed_performance": {},
    }
    
    for seed in sorted(seed_appearances.keys()):
        seed_data = {}
        for round_num, round_name in rounds.items():
            appearances = seed_appearances[seed].get(round_num, 0)
            advances = seed_advances[seed].get(round_num, 0)
            win_rate = advances / appearances if appearances > 0 else 0
            
            if appearances > 0:
                seed_data[round_name] = {
                    "appearances": int(appearances),
                    "advances": int(advances),
                    "win_rate": round(win_rate, 3),
                }
        
        json_data["seed_performance"][f"seed_{seed}"] = seed_data
    
    json_path = Path(args.out_dir) / "seed_accuracy_report.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON report to {json_path}")
    
    # Print summary
    print("\n=== Historical Seed Performance ===")
    for seed in sorted(seed_appearances.keys())[:8]:
        print(f"\n{seed}-Seed:")
        for round_num in [1, 2, 3, 4]:
            appearances = seed_appearances[seed].get(round_num, 0)
            if appearances > 0:
                advances = seed_advances[seed].get(round_num, 0)
                win_rate = advances / appearances
                round_name = rounds[round_num]
                print(f"  {round_name:20s}: {advances}/{appearances} games = {win_rate:.1%} advance rate")


if __name__ == "__main__":
    main()
