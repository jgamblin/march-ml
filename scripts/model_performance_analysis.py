"""Model performance analysis and comparison.

Analyzes trained model performance metrics and provides insights into:
- Feature performance
- Calibration quality  
- Prediction confidence distribution
- Holdout accuracy by season

Outputs:
- model_performance_report.json: comprehensive metrics and analysis

Usage:
    python scripts/model_performance_analysis.py --models_dir models --out_dir results
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd


def load_training_summary(models_dir):
    """Load training summary JSON."""
    with open(Path(models_dir) / "training_summary.json") as f:
        return json.load(f)


def analyze_performance_by_season(summary):
    """Extract season-by-season performance metrics."""
    holdout_results = summary.get("holdout_results", [])
    
    by_season = {}
    for result in holdout_results:
        season = result.get("season")
        if season not in by_season:
            by_season[season] = {}
        
        ensemble = result.get("ensemble", {})
        by_season[season] = {
            "accuracy": ensemble.get("accuracy"),
            "log_loss": ensemble.get("log_loss"),
            "brier_score": ensemble.get("brier"),
            "games": ensemble.get("games"),
        }
    
    return by_season


def analyze_model_selection(summary):
    """Compare LR vs XGB performance."""
    holdout_results = summary.get("holdout_results", [])
    
    lr_metrics = {"accuracy": [], "log_loss": []}
    xgb_metrics = {"accuracy": [], "log_loss": []}
    ensemble_metrics = {"accuracy": [], "log_loss": []}
    
    for result in holdout_results:
        lr = result.get("logistic_regression", {})
        xgb = result.get("xgboost", {})
        ensemble = result.get("ensemble", {})
        
        if lr:
            lr_metrics["accuracy"].append(lr.get("accuracy"))
            lr_metrics["log_loss"].append(lr.get("log_loss"))
        if xgb:
            xgb_metrics["accuracy"].append(xgb.get("accuracy"))
            xgb_metrics["log_loss"].append(xgb.get("log_loss"))
        if ensemble:
            ensemble_metrics["accuracy"].append(ensemble.get("accuracy"))
            ensemble_metrics["log_loss"].append(ensemble.get("log_loss"))
    
    return {
        "logistic_regression": {
            "avg_accuracy": sum(lr_metrics["accuracy"]) / len(lr_metrics["accuracy"]) if lr_metrics["accuracy"] else 0,
            "avg_log_loss": sum(lr_metrics["log_loss"]) / len(lr_metrics["log_loss"]) if lr_metrics["log_loss"] else 0,
        },
        "xgboost": {
            "avg_accuracy": sum(xgb_metrics["accuracy"]) / len(xgb_metrics["accuracy"]) if xgb_metrics["accuracy"] else 0,
            "avg_log_loss": sum(xgb_metrics["log_loss"]) / len(xgb_metrics["log_loss"]) if xgb_metrics["log_loss"] else 0,
        },
        "ensemble": {
            "avg_accuracy": sum(ensemble_metrics["accuracy"]) / len(ensemble_metrics["accuracy"]) if ensemble_metrics["accuracy"] else 0,
            "avg_log_loss": sum(ensemble_metrics["log_loss"]) / len(ensemble_metrics["log_loss"]) if ensemble_metrics["log_loss"] else 0,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="models")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading training summary...")
    summary = load_training_summary(args.models_dir)
    
    print("Analyzing model performance...")
    
    # Season-by-season analysis
    seasonal_perf = analyze_performance_by_season(summary)
    
    # Model comparison
    model_comparison = analyze_model_selection(summary)
    
    # Feature analysis
    feature_cols = summary.get("feature_columns", [])
    
    # Build comprehensive report
    report = {
        "model_info": {
            "features_path": summary.get("features_path"),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "total_games": summary.get("rows"),
            "training_seasons": summary.get("seasons"),
            "game_scope": summary.get("game_scope"),
        },
        "seasonal_performance": seasonal_perf,
        "model_comparison": model_comparison,
        "overall_holdout": summary.get("overall_holdout_ensemble", {}),
        "calibration_method": str(summary.get("models", {}).get("calibration", "Unknown")),
        "models_available": {
            "logistic_regression": summary.get("models", {}).get("logistic_regression", False),
            "xgboost": summary.get("models", {}).get("xgboost", False),
        },
    }
    
    # Export report
    output_path = Path(args.out_dir) / "model_performance_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved performance report to {output_path}")
    
    # Print summary
    print("\n=== Model Performance Summary ===")
    print(f"Total games analyzed: {summary.get('rows'):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Training seasons: {summary.get('seasons')}")
    
    print(f"\n=== Overall Holdout Performance ===")
    overall = summary.get("overall_holdout_ensemble") or {}
    print(f"  Accuracy: {overall.get('accuracy', 0):.2%}")
    print(f"  Log-loss: {overall.get('log_loss', 0):.4f}")
    print(f"  Brier score: {overall.get('brier', 0):.4f}")
    print(f"  Games: {overall.get('games', 0)}")
    
    print(f"\n=== Model Comparison (Averages) ===")
    for model_name, metrics in model_comparison.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics.get('avg_accuracy', 0):.2%}")
        print(f"  Log-loss: {metrics.get('avg_log_loss', 0):.4f}")
    
    print(f"\n=== Performance by Season ===")
    for season in sorted(seasonal_perf.keys()):
        perf = seasonal_perf[season]
        print(f"{season}:")
        print(f"  Accuracy: {perf.get('accuracy', 0):.2%} ({perf.get('games')} games)")
        print(f"  Log-loss: {perf.get('log_loss', 0):.4f}")


if __name__ == "__main__":
    main()
