"""Generate game-by-game matchup probabilities for all possible tournament games.

Outputs:
- matchup_probabilities.csv: win probabilities for all seed vs seed matchups
- round_matchup_guide.json: structured guide by tournament round

Usage:
    python scripts/matchup_probabilities.py --models_dir models --features_path data/processed/features/tournament_teams.csv --season 2025 --out_dir results
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer


def coerce_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def load_models_and_features(models_dir):
    """Load trained models and feature metadata."""
    lr_model = joblib.load(Path(models_dir) / "lr_model.joblib")
    lr_cal = joblib.load(Path(models_dir) / "lr_cal.joblib")
    
    try:
        xgb_model = joblib.load(Path(models_dir) / "xgb_model.joblib")
        xgb_cal = joblib.load(Path(models_dir) / "xgb_cal.joblib")
        has_xgb = True
    except:
        xgb_model = None
        xgb_cal = None
        has_xgb = False
    
    feature_names = joblib.load(Path(models_dir) / "model_features.joblib")
    return lr_model, lr_cal, xgb_model, xgb_cal, feature_names, has_xgb


def load_tournament_teams(features_path, season):
    """Load pre-tournament team features."""
    df = pd.read_csv(features_path)
    season_df = df[df["season"] == season].copy()
    return season_df


def compute_matchup_features(team1_row, team2_row, feature_names):
    """Build feature vector for matchup (team1 as home, team2 as away)."""
    features = []
    for feat in feature_names:
        if feat == "neutral_site":
            features.append(1.0)  # Tournament games are neutral site
        elif feat == "home_edge":
            features.append(0.0)  # No home edge at neutral site
        elif feat.startswith("diff_"):
            base_feat = feat.replace("diff_", "")
            t1_val = float(team1_row.get(base_feat, 0))
            t2_val = float(team2_row.get(base_feat, 0))
            features.append(t1_val - t2_val)
        else:
            features.append(0.0)
    return np.array(features).reshape(1, -1)


def predict_win_prob(features, lr_model, lr_cal, xgb_model, xgb_cal, has_xgb):
    """Predict win probability for a matchup."""
    # LR prediction
    lr_prob = lr_cal.predict_proba(features)[0][1]
    
    if has_xgb:
        # XGBoost prediction
        xgb_prob = xgb_cal.predict_proba(features)[0][1]
        # Ensemble average
        prob = (lr_prob + xgb_prob) / 2
    else:
        prob = lr_prob
    
    return float(prob)


def generate_seed_matchup_grid(teams_df):
    """Generate all possible seed matchups."""
    matchups = []
    seeds = sorted(teams_df["seed"].dropna().unique())
    
    for seed1 in seeds:
        for seed2 in seeds:
            if seed1 <= seed2:
                t1 = teams_df[teams_df["seed"] == seed1].iloc[0] if len(teams_df[teams_df["seed"] == seed1]) > 0 else None
                t2 = teams_df[teams_df["seed"] == seed2].iloc[0] if len(teams_df[teams_df["seed"] == seed2]) > 0 else None
                
                if t1 is not None and t2 is not None:
                    matchups.append((int(seed1), int(seed2), str(t1.get("team", "Team" + str(seed1))), str(t2.get("team", "Team" + str(seed2)))))
    
    return matchups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="models")
    p.add_argument("--features_path", default="data/processed/features/tournament_teams.csv")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading models and features...")
    lr_model, lr_cal, xgb_model, xgb_cal, feature_names, has_xgb = load_models_and_features(args.models_dir)
    teams_df = load_tournament_teams(args.features_path, args.season)
    
    print(f"Generating matchup predictions for {len(teams_df)} teams...")
    
    matchup_rows = []
    
    # Generate seed matchup grid
    matchups = generate_seed_matchup_grid(teams_df)
    
    for seed1, seed2, team1_name, team2_name in matchups:
        # Get representative team for each seed
        team1_row = teams_df[teams_df["seed"] == seed1]
        team2_row = teams_df[teams_df["seed"] == seed2]
        
        if len(team1_row) == 0 or len(team2_row) == 0:
            continue
        
        team1_row = team1_row.iloc[0]
        team2_row = team2_row.iloc[0]
        
        # Compute features for matchup
        features = compute_matchup_features(team1_row, team2_row, feature_names)
        
        # Predict win probability
        team1_win_prob = predict_win_prob(features, lr_model, lr_cal, xgb_model, xgb_cal, has_xgb)
        team2_win_prob = 1 - team1_win_prob
        
        matchup_rows.append({
            "seed_1": seed1,
            "team_1": str(team1_row.get("team", "")),
            "seed_2": seed2,
            "team_2": str(team2_row.get("team", "")),
            "seed1_win_prob": team1_win_prob,
            "seed2_win_prob": team2_win_prob,
            "predicted_winner": str(team1_row.get("team", "")) if team1_win_prob > 0.5 else str(team2_row.get("team", "")),
            "confidence": max(team1_win_prob, team2_win_prob),
        })
    
    # Export to CSV
    matchup_df = pd.DataFrame(matchup_rows)
    csv_path = Path(args.out_dir) / "matchup_probabilities.csv"
    matchup_df.to_csv(csv_path, index=False)
    print(f"\nSaved matchup probabilities to {csv_path}")
    
    # Export to JSON by round
    json_data = {
        "season": args.season,
        "total_matchups": len(matchup_df),
        "first_four_matchups": matchup_df[
            ((matchup_df["seed_1"] == 16) & (matchup_df["seed_2"] == 16)) |
            ((matchup_df["seed_1"] == 11) & (matchup_df["seed_2"] == 11))
        ].to_dict("records"),
        "round_of_64": matchup_df[
            (matchup_df["seed_1"] <= 8) & (matchup_df["seed_2"] <= 8)
        ].to_dict("records"),
        "round_of_32": matchup_df[
            (matchup_df["seed_1"] <= 4) & (matchup_df["seed_2"] <= 4)
        ].to_dict("records"),
    }
    
    json_path = Path(args.out_dir) / "matchup_probabilities.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Saved JSON matchup guide to {json_path}")
    
    # Print sample
    print("\n=== Sample Matchup Probabilities ===")
    print("\nClosest Matchups (50/50 predictions):")
    matchup_df["upset_differential"] = (matchup_df["seed1_win_prob"] - 0.5).abs()
    closest = matchup_df.nsmallest(5, "upset_differential")
    for _, row in closest.iterrows():
        print(f"  {int(row['seed_1'])}-seed vs {int(row['seed_2'])}-seed: "
              f"{row['seed1_win_prob']:.1%} vs {row['seed2_win_prob']:.1%}")
    
    print("\nBiggest Upsets (worst favorite prediction):")
    matchup_df["is_upset"] = matchup_df["seed_1"] > matchup_df["seed_2"]
    upsets = matchup_df[matchup_df["is_upset"]].copy()
    upsets["upset_prob"] = upsets["seed2_win_prob"]
    biggest = upsets.nlargest(5, "upset_prob")
    for _, row in biggest.iterrows():
        print(f"  {int(row['seed_1'])}-seed vs {int(row['seed_2'])}-seed: "
              f"{row['seed2_win_prob']:.1%} chance of upset")


if __name__ == "__main__":
    main()
