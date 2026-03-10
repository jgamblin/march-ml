"""Hyperparameter tuning for logistic regression and XGBoost models.

Grid searches over regularization parameters to find optimal validation accuracy.

Outputs:
- hyperparameter_tuning_results.csv: all grid search results
- hyperparameter_tuning_summary.json: best parameters and metrics

Usage:
    python scripts/hyperparameter_tuning.py --features data/processed/features/tournament_teams.csv --games_dir data/processed --out_dir results
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False


def coerce_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def load_features(path):
    return pd.read_csv(path)


def load_games(path):
    df = pd.read_csv(path)
    df["is_postseason"] = coerce_bool(df.get("is_postseason", False))
    tournament_series = df.get("tournament", pd.Series("", index=df.index)).fillna("").astype(str)
    ncaa_mask = tournament_series.str.contains("Men's Basketball Championship", case=False, na=False)
    return df[ncaa_mask].copy()


def build_feature_lookup(features_df):
    by_id = {}
    by_name = {}
    for _, row in features_df.iterrows():
        season = int(row["season"])
        team_id = str(row.get("team_id", "")).strip()
        team_name = str(row.get("team", "")).strip().lower()
        if team_id:
            by_id[(season, team_id)] = row
        if team_name:
            by_name[(season, team_name)] = row
    return by_id, by_name


def feature_columns_from_df(features_df):
    cols = [
        "games_played", "wins", "losses", "win_pct", "avg_points_for",
        "avg_points_against", "avg_margin", "last5_win_pct", "last10_wins",
        "last10_losses", "offense_trend", "defense_trend", "sos_win_pct",
        "opp_avg_margin", "adj_margin"
    ]
    return [col for col in cols if col in features_df.columns]


def build_matchup_dataset(games_df, features_df):
    """Build feature matrix and target vector."""
    by_id, by_name = build_feature_lookup(features_df)
    feature_cols = feature_columns_from_df(features_df)
    
    X_list = []
    y_list = []
    seasons_list = []
    
    for _, game in games_df.iterrows():
        season = int(game["season"])
        home_team = str(game.get("home_team", game.get("team", "")))
        away_team = str(game.get("away_team", game.get("opp", "")))
        home_id = str(game.get("home_id", game.get("team_id", ""))).strip()
        away_id = str(game.get("away_id", game.get("opp_id", ""))).strip()
        
        home_row = None
        away_row = None
        
        if home_id and (season, home_id) in by_id:
            home_row = by_id[(season, home_id)]
        elif (season, home_team.lower()) in by_name:
            home_row = by_name[(season, home_team.lower())]
        
        if away_id and (season, away_id) in by_id:
            away_row = by_id[(season, away_id)]
        elif (season, away_team.lower()) in by_name:
            away_row = by_name[(season, away_team.lower())]
        
        if home_row is None or away_row is None:
            continue
        
        home_features = [float(home_row.get(col, 0)) for col in feature_cols]
        away_features = [float(away_row.get(col, 0)) for col in feature_cols]
        
        diff_features = [h - a for h, a in zip(home_features, away_features)]
        diff_features.extend([1.0, 0.0])  # neutral_site=1, home_edge=0
        
        X_list.append(diff_features)
        y_list.append(1 if bool(game.get("home_win", False)) else 0)
        seasons_list.append(season)
    
    return np.array(X_list), np.array(y_list), np.array(seasons_list), feature_cols


def grid_search_lr(X, y, seasons, C_range=None, penalty_range=None):
    """Grid search logistic regression hyperparameters."""
    if C_range is None:
        C_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    if penalty_range is None:
        penalty_range = ["l2", "l1"]
    
    results = []
    
    for C in C_range:
        for penalty in penalty_range:
            # Skip l1 with lbfgs
            if penalty == "l1":
                solver = "liblinear"
            else:
                solver = "lbfgs"
            
            # Season-aware evaluation
            for test_season in sorted(np.unique(seasons)):
                if len(seasons[seasons == test_season]) < 10:
                    continue
                
                train_mask = seasons != test_season
                test_mask = seasons == test_season
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                
                # Train model
                try:
                    lr = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
                    lr.fit(X_train, y_train)
                    
                    # Calibrate
                    cal_lr = CalibratedClassifierCV(lr, cv="prefit", method="sigmoid")
                    cal_lr.fit(X_test, y_test)
                    
                    # Evaluate
                    y_pred_proba = cal_lr.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    logloss = log_loss(y_test, y_pred_proba)
                    brier = brier_score_loss(y_test, y_pred_proba)
                    
                    results.append({
                        "model": "logistic_regression",
                        "C": C,
                        "penalty": penalty,
                        "solver": solver,
                        "test_season": test_season,
                        "games": len(y_test),
                        "accuracy": accuracy,
                        "log_loss": logloss,
                        "brier_score": brier,
                    })
                except Exception as e:
                    continue
    
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features/tournament_teams.csv")
    p.add_argument("--games_dir", default="data/processed")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading data for hyperparameter tuning...")
    features_df = load_features(args.features)
    
    games_list = []
    for games_file in Path(args.games_dir).glob("games_*.csv"):
        games_list.append(load_games(games_file))
    games_df = pd.concat(games_list, ignore_index=True)
    
    print("Building matchup dataset...")
    X, y, seasons, feature_cols = build_matchup_dataset(games_df, features_df)
    print(f"  Dataset: {len(X)} matchups, {len(feature_cols)} features")
    
    print("Running grid search for logistic regression...")
    lr_results = grid_search_lr(X, y, seasons,
                                 C_range=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                                 penalty_range=["l2", "l1"])
    
    print(f"  Evaluated {len(lr_results)} parameter combinations")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(lr_results)
    
    # Find best parameters by average accuracy
    best_idx = results_df.groupby(["C", "penalty"])["accuracy"].mean().idxmax()
    best_C, best_penalty = best_idx
    best_results = results_df[(results_df["C"] == best_C) & (results_df["penalty"] == best_penalty)]
    best_accuracy = best_results["accuracy"].mean()
    best_logloss = best_results["log_loss"].mean()
    
    print(f"\n=== Best Hyperparameters ===")
    print(f"  C: {best_C}")
    print(f"  Penalty: {best_penalty}")
    print(f"  Average accuracy: {best_accuracy:.4f}")
    print(f"  Average log-loss: {best_logloss:.4f}")
    
    # Export results
    csv_path = Path(args.out_dir) / "hyperparameter_tuning_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    # Export summary
    summary = {
        "best_model": "logistic_regression",
        "best_hyperparameters": {
            "C": best_C,
            "penalty": best_penalty,
            "solver": "liblinear" if best_penalty == "l1" else "lbfgs",
        },
        "best_performance": {
            "accuracy": round(float(best_accuracy), 4),
            "log_loss": round(float(best_logloss), 4),
        },
        "current_baseline": {
            "accuracy": 0.664,
            "log_loss": 0.634,
        },
        "improvement": {
            "accuracy_delta": round(float(best_accuracy - 0.664), 4),
            "log_loss_delta": round(float(best_logloss - 0.634), 4),
        },
        "recommendation": (
            f"Update model with C={best_C}, penalty={best_penalty}" 
            if best_accuracy > 0.664 
            else "Baseline parameters already optimal or close"
        ),
    }
    
    summary_path = Path(args.out_dir) / "hyperparameter_tuning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Print top 10 parameter combinations
    print(f"\n=== Top 10 Parameter Combinations ===")
    top_results = results_df.groupby(["C", "penalty"])[["accuracy", "log_loss"]].agg({
        "accuracy": "mean",
        "log_loss": "mean",
    }).sort_values("accuracy", ascending=False).head(10)
    
    for (C, penalty), row in top_results.iterrows():
        print(f"  C={C:8.4f}, penalty={penalty:2s}: accuracy={row['accuracy']:.4f}, log_loss={row['log_loss']:.4f}")


if __name__ == "__main__":
    main()
