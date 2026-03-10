"""Validate core pipeline artifacts and output schemas.

Checks:
- feature outputs exist and include required columns
- training summary exists and contains expected keys
- simulation JSON exists and matches expected schema (when provided)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REQUIRED_FEATURE_COLUMNS = {
    "season",
    "team",
    "team_id",
    "games_played",
    "wins",
    "losses",
    "win_pct",
    "avg_points_for",
    "avg_points_against",
    "avg_margin",
    "last5_win_pct",
    "sos_win_pct",
    "opp_avg_margin",
    "adj_margin",
    "is_d1",
}

REQUIRED_TRAINING_KEYS = {
    "features_path",
    "games_dir",
    "game_scope",
    "rows",
    "feature_columns",
    "seasons",
    "holdout_results",
    "overall_holdout_ensemble",
    "models",
}

REQUIRED_SIM_KEYS = {
    "season",
    "sims",
    "teams",
    "bracket_source",
    "bracket",
    "champion_probs",
    "round_probs",
    "model_metadata",
}


def fail(message):
    raise ValueError(message)


def validate_feature_file(path):
    if not path.exists():
        fail(f"Missing feature file: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_FEATURE_COLUMNS - set(df.columns)
    if missing:
        fail(f"Feature file {path} missing columns: {sorted(missing)}")
    if df.empty:
        fail(f"Feature file is empty: {path}")
    return df


def validate_training_summary(path):
    if not path.exists():
        fail(f"Missing training summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_TRAINING_KEYS - set(payload.keys())
    if missing:
        fail(f"Training summary missing keys: {sorted(missing)}")
    if not isinstance(payload["feature_columns"], list) or not payload["feature_columns"]:
        fail("Training summary has empty or invalid `feature_columns`")
    if not isinstance(payload["seasons"], list) or len(payload["seasons"]) < 2:
        fail("Training summary `seasons` must contain at least two seasons")
    return payload


def validate_simulation(path, tournament_teams_df=None, allow_nd=False):
    if not path.exists():
        fail(f"Missing simulation output: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_SIM_KEYS - set(payload.keys())
    if missing:
        fail(f"Simulation output missing keys: {sorted(missing)}")

    teams = payload.get("teams", [])
    if not isinstance(teams, list) or not teams:
        fail("Simulation `teams` must be a non-empty list")
    if len(teams) & (len(teams) - 1) != 0:
        fail("Simulation team count must be a power of two")

    champion_probs = payload.get("champion_probs", [])
    if not isinstance(champion_probs, list) or not champion_probs:
        fail("Simulation `champion_probs` must be a non-empty list")
    total_prob = 0.0
    for item in champion_probs:
        if not isinstance(item, list) or len(item) != 2:
            fail("Each `champion_probs` item must be [team, probability]")
        total_prob += float(item[1])
    if abs(total_prob - 1.0) > 0.05:
        fail(f"Champion probabilities should sum to ~1.0, got {total_prob:.4f}")

    if tournament_teams_df is not None and not allow_nd:
        season = int(payload.get("season"))
        season_df = tournament_teams_df[tournament_teams_df["season"] == season].copy()
        if not season_df.empty and "is_d1" in season_df.columns:
            season_df["team_key"] = season_df["team"].astype(str).str.strip().str.lower()
            d1_lookup = dict(zip(season_df["team_key"], season_df["is_d1"].astype(bool)))
            non_d1 = [team for team in teams if not bool(d1_lookup.get(str(team).strip().lower(), True))]
            if non_d1:
                fail(f"Simulation includes non-D1 teams while allow_nd is false: {non_d1[:5]}")

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/processed/features")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--sim_out", default=None, help="optional simulation output JSON path")
    parser.add_argument("--allow_nd", action="store_true", help="allow non-D1 teams in simulation validation")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)

    teams_df = validate_feature_file(features_dir / "teams.csv")
    tournament_df = validate_feature_file(features_dir / "tournament_teams.csv")
    _ = teams_df  # retained for future extension
    validate_training_summary(models_dir / "training_summary.json")

    if args.sim_out:
        validate_simulation(Path(args.sim_out), tournament_teams_df=tournament_df, allow_nd=args.allow_nd)

    print("Artifact validation passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Artifact validation failed: {exc}")
        sys.exit(1)