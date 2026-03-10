"""Extract feature importance from trained models.

Outputs:
- feature_importance.csv: importance scores for all features
- feature_importance.json: detailed feature analysis

Usage:
    python scripts/feature_importance.py --models_dir models --out_dir results
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd


def extract_lr_importance(lr_model, feature_names):
    """Extract feature importance from logistic regression coefficients."""
    coef = lr_model.coef_[0]
    abs_coef = [abs(c) for c in coef]
    return {
        "coefficient": float(coef),
        "abs_importance": float(abs(coef)),
        "direction": "positive" if coef > 0 else "negative",
    }


def extract_xgb_importance(xgb_model, feature_names):
    """Extract feature importance from XGBoost."""
    try:
        importance = xgb_model.feature_importances_
        return float(importance)
    except:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="models")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load models
    print("Loading trained models...")
    lr_model = joblib.load(Path(args.models_dir) / "lr_model.joblib")
    feature_names = joblib.load(Path(args.models_dir) / "model_features.joblib")
    
    try:
        xgb_model = joblib.load(Path(args.models_dir) / "xgb_model.joblib")
        has_xgb = True
    except:
        xgb_model = None
        has_xgb = False
    
    training_summary = json.load(open(Path(args.models_dir) / "training_summary.json"))
    
    # Extract importance
    print(f"Analyzing {len(feature_names)} features...")
    importance_rows = []
    
    # Get baseline metrics
    holdout_ensemble = training_summary.get("overall_holdout_ensemble", {})
    baseline_accuracy = holdout_ensemble.get("accuracy", 0.664) if holdout_ensemble else 0.664
    baseline_logloss = holdout_ensemble.get("log_loss", 0.634) if holdout_ensemble else 0.634
    
    for i, feat_name in enumerate(feature_names):
        row = {
            "rank": i + 1,
            "feature": feat_name,
            "lr_coefficient": float(lr_model.coef_[0][i]),
            "lr_abs_importance": abs(float(lr_model.coef_[0][i])),
        }
        
        if has_xgb:
            row["xgb_importance"] = float(xgb_model.feature_importances_[i])
        
        importance_rows.append(row)
    
    importance_df = pd.DataFrame(importance_rows)
    
    # Normalize scores
    max_coef = importance_df["lr_coefficient"].abs().max()
    importance_df["lr_coefficient_normalized"] = importance_df["lr_coefficient"] / max_coef if max_coef > 0 else 0
    
    if has_xgb:
        max_xgb = importance_df["xgb_importance"].max()
        importance_df["xgb_importance_normalized"] = importance_df["xgb_importance"] / max_xgb if max_xgb > 0 else 0
    
    # Sort by absolute LR coefficient
    importance_df = importance_df.sort_values("lr_abs_importance", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    
    # Export CSV
    csv_path = Path(args.out_dir) / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance to {csv_path}")
    
    # Export JSON with detailed analysis
    json_data = {
        "analysis_date": "2025-03-09",
        "model_info": {
            "feature_count": len(feature_names),
            "holdout_accuracy": baseline_accuracy,
            "holdout_log_loss": baseline_logloss,
        },
        "top_features": [
            {
                "rank": int(row["rank"]),
                "feature": row["feature"],
                "lr_coefficient": round(float(row["lr_coefficient"]), 6),
                "direction": "positive" if row["lr_coefficient"] > 0 else "negative",
                "xgb_importance": round(float(row["xgb_importance"]), 6) if has_xgb else None,
                "interpretation": _interpret_feature(row["feature"]),
            }
            for _, row in importance_df.head(10).iterrows()
        ],
        "all_features": [
            {
                "rank": int(row["rank"]),
                "feature": row["feature"],
                "lr_coefficient": round(float(row["lr_coefficient"]), 6),
                "lr_normalized": round(float(row["lr_coefficient_normalized"]), 4),
                "xgb_importance": round(float(row["xgb_importance"]), 6) if has_xgb else None,
            }
            for _, row in importance_df.iterrows()
        ],
    }
    
    json_path = Path(args.out_dir) / "feature_importance.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved detailed analysis to {json_path}")
    
    # Print summary
    print("\n=== Top 10 Most Important Features ===")
    for _, row in importance_df.head(10).iterrows():
        direction = "↑" if row["lr_coefficient"] > 0 else "↓"
        print(f"  {int(row['rank']):2d}. {row['feature']:25s} {direction} ({row['lr_coefficient']:+.4f})")


def _interpret_feature(feat_name):
    """Provide interpretation of what the feature means."""
    interpretations = {
        "diff_wins": "Home team's additional wins vs opponent",
        "diff_losses": "Home team's additional losses vs opponent",
        "diff_win_pct": "Home team's win % advantage",
        "diff_avg_points_for": "Home team's scoring advantage",
        "diff_avg_points_against": "Home team's defense advantage (lower opponent scoring)",
        "diff_avg_margin": "Home team's point margin advantage",
        "diff_last5_win_pct": "Home team's recent (5-game) momentum advantage",
        "diff_last10_wins": "Home team's recent (10-game) wins advantage",
        "diff_last10_losses": "Home team's recent (10-game) losses disadvantage",
        "diff_offense_trend": "Home team's recent offensive trend (positive=improving)",
        "diff_defense_trend": "Home team's recent defensive trend (positive=improving)",
        "diff_sos_win_pct": "Home team's opponent strength advantage",
        "diff_opp_avg_margin": "Home team's adjusted margin advantage",
        "diff_adj_margin": "Home team's strength of schedule adjusted margin",
        "neutral_site": "Game is at neutral location (reduces home advantage)",
        "home_edge": "Home team's inherent home court advantage",
    }
    return interpretations.get(feat_name, "Team stat advantage")


if __name__ == "__main__":
    main()
