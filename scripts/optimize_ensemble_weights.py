"""Optimize ensemble weights for LR + XGBoost model combination.

Tests different weight combinations to find optimal balance between
logistic regression and XGBoost predictions for tournament games.

Output: ensemble_weight_optimization.json with best weights and performance
"""
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss


def load_models(models_dir='models'):
    """Load trained models and calibrators."""
    models_dir = Path(models_dir)
    
    # Load base models
    lr_model = joblib.load(models_dir / 'lr_model.joblib')
    xgb_model = joblib.load(models_dir / 'xgb_model.joblib')
    
    # Load calibrators if available
    lr_cal = None
    xgb_cal = None
    if (models_dir / 'lr_cal.joblib').exists():
        lr_cal = joblib.load(models_dir / 'lr_cal.joblib')
    if (models_dir / 'xgb_cal.joblib').exists():
        xgb_cal = joblib.load(models_dir / 'xgb_cal.joblib')
    
    # Load feature names
    feature_names = joblib.load(models_dir / 'model_features.joblib')
    
    return lr_model, xgb_model, lr_cal, xgb_cal, feature_names


def build_tournament_dataset(games_dir, features_path, feature_names):
    """Build feature matrix from tournament games matching model training."""
    games_dir = Path(games_dir)
    
    # Load features
    features_df = pd.read_csv(features_path)
    
    # Load tournament games
    all_games = []
    for year in [2021, 2022, 2023, 2024, 2025]:
        games_path = games_dir / f'games_{year}.csv'
        if games_path.exists():
            games = pd.read_csv(games_path)
            postseason_mask = (games.get('is_postseason', False).astype(str).str.lower() == 'true') | \
                            (games.get('tournament', '').astype(str).str.contains('Championship', case=False, na=False))
            all_games.append(games[postseason_mask].copy())
    
    if not all_games:
        return None, None, None
    
    games_df = pd.concat(all_games, ignore_index=True)
    
    # Base feature columns that exist in features_df
    base_cols = ['games_played', 'wins', 'losses', 'win_pct', 'avg_points_for', 
                 'avg_points_against', 'avg_margin', 'last5_win_pct', 'last10_wins',
                 'last10_losses', 'offense_trend', 'defense_trend', 'sos_win_pct',
                 'opp_avg_margin', 'adj_margin']
    
    X_list = []
    y_list = []
    seasons_list = []
    
    for _, game in games_df.iterrows():
        home_team = str(game.get('home_team', '')).strip()
        away_team = str(game.get('away_team', '')).strip()
        home_win = game.get('home_win')
        
        if pd.isna(home_win) or not home_team or not away_team:
            continue
        
        season = int(game.get('season', 2025))
        is_neutral = game.get('is_neutral', False)
        
        # Find team features
        home_match = features_df[(features_df['team'].str.strip() == home_team) & 
                                (features_df['season'] == season)]
        away_match = features_df[(features_df['team'].str.strip() == away_team) & 
                                (features_df['season'] == season)]
        
        if home_match.empty or away_match.empty:
            continue
        
        # Build feature vector matching training format
        feature_vec = []
        for col in base_cols:
            home_val = float(home_match.iloc[0].get(col, 0))
            away_val = float(away_match.iloc[0].get(col, 0))
            feature_vec.append(home_val - away_val)  # diff_*
        
        # Add neutral_site and home_edge indicators
        neutral_site = 1.0 if str(is_neutral).lower() == 'true' else 0.0
        home_edge = 0.0 if neutral_site else 1.0
        feature_vec.extend([neutral_site, home_edge])
        
        X_list.append(feature_vec)
        y_list.append(1 if bool(home_win) else 0)
        seasons_list.append(season)
    
    return np.array(X_list), np.array(y_list), np.array(seasons_list)


def get_predictions(model, calibrator, X):
    """Get calibrated predictions from a model."""
    if calibrator is not None:
        # Use calibrated predictions
        return calibrator.predict_proba(X)[:, 1]
    else:
        # Use raw model predictions
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        else:
            # For models without predict_proba
            return model.predict(X)


def optimize_ensemble_weights(lr_model, xgb_model, lr_cal, xgb_cal, X, y, seasons):
    """Test different ensemble weight combinations."""
    
    # Get base predictions
    print("  Generating predictions from both models...")
    lr_probs = get_predictions(lr_model, lr_cal, X)
    xgb_probs = get_predictions(xgb_model, xgb_cal, X)
    
    results = []
    weight_range = np.arange(0, 1.05, 0.05)  # 0% to 100% in 5% increments
    
    print("  Testing weight combinations...")
    for lr_weight in weight_range:
        xgb_weight = 1 - lr_weight
        
        # Ensemble predictions
        ensemble_probs = (lr_weight * lr_probs) + (xgb_weight * xgb_probs)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, ensemble_preds)
        logloss = log_loss(y, ensemble_probs)
        
        results.append({
            'lr_weight': round(lr_weight, 2),
            'xgb_weight': round(xgb_weight, 2),
            'accuracy': round(accuracy, 4),
            'log_loss': round(logloss, 4),
        })
        
        if lr_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            print(f"    LR:{lr_weight:.0%} / XGB:{xgb_weight:.0%} -> "
                  f"Acc: {accuracy:.4f}, LogLoss: {logloss:.4f}")
    
    # Also test by season
    print("\n  Testing by season...")
    season_results = {}
    for season in sorted(np.unique(seasons)):
        season_mask = seasons == season
        if np.sum(season_mask) < 10:
            continue
        
        X_season = X[season_mask]
        y_season = y[season_mask]
        lr_probs_season = get_predictions(lr_model, lr_cal, X_season)
        xgb_probs_season = get_predictions(xgb_model, xgb_cal, X_season)
        
        season_weights = []
        for lr_weight in weight_range:
            xgb_weight = 1 - lr_weight
            ensemble_probs = (lr_weight * lr_probs_season) + (xgb_weight * xgb_probs_season)
            ensemble_preds = (ensemble_probs >= 0.5).astype(int)
            accuracy = accuracy_score(y_season, ensemble_preds)
            season_weights.append({'lr_weight': lr_weight, 'accuracy': accuracy})
        
        best_season_weight = max(season_weights, key=lambda x: x['accuracy'])
        season_results[int(season)] = {
            'best_lr_weight': round(best_season_weight['lr_weight'], 2),
            'best_xgb_weight': round(1 - best_season_weight['lr_weight'], 2),
            'best_accuracy': round(best_season_weight['accuracy'], 4),
            'games': int(np.sum(season_mask)),
        }
        print(f"    {season}: Best weights LR:{best_season_weight['lr_weight']:.0%} / "
              f"XGB:{1-best_season_weight['lr_weight']:.0%} -> "
              f"Acc: {best_season_weight['accuracy']:.4f}")
    
    return results, season_results


def main():
    parser = argparse.ArgumentParser(description='Optimize ensemble model weights')
    parser.add_argument('--models_dir', default='models', help='Directory with trained models')
    parser.add_argument('--games_dir', default='data/processed', help='Directory with games files')
    parser.add_argument('--features', default='data/processed/features/tournament_teams.csv')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    
    print("\n=== Ensemble Weight Optimization ===\n")
    
    # Load models
    print("Loading models...")
    lr_model, xgb_model, lr_cal, xgb_cal, feature_names = load_models(args.models_dir)
    
    # Build dataset
    print("Building tournament dataset...")
    X, y, seasons = build_tournament_dataset(args.games_dir, args.features, feature_names)
    
    if X is None or len(X) == 0:
        print("⚠️  No tournament data found")
        return
    
    print(f"  Dataset: {len(X)} tournament games, {len(feature_names)} features\n")
    
    # Optimize weights
    print("Optimizing ensemble weights...")
    results, season_results = optimize_ensemble_weights(
        lr_model, xgb_model, lr_cal, xgb_cal, X, y, seasons
    )
    
    # Find best overall weights
    results_df = pd.DataFrame(results)
    best_by_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_by_logloss = results_df.loc[results_df['log_loss'].idxmin()]
    
    # Current baseline (50/50)
    baseline = results_df[results_df['lr_weight'] == 0.5].iloc[0]
    
    print(f"\n=== Results ===\n")
    print(f"Current baseline (50/50 ensemble):")
    print(f"  Accuracy: {baseline['accuracy']:.4f}")
    print(f"  Log-loss: {baseline['log_loss']:.4f}")
    
    print(f"\nBest weights by accuracy:")
    print(f"  LR: {best_by_accuracy['lr_weight']:.0%} / XGB: {best_by_accuracy['xgb_weight']:.0%}")
    print(f"  Accuracy: {best_by_accuracy['accuracy']:.4f} "
          f"({best_by_accuracy['accuracy'] - baseline['accuracy']:+.4f})")
    print(f"  Log-loss: {best_by_accuracy['log_loss']:.4f}")
    
    print(f"\nBest weights by log-loss:")
    print(f"  LR: {best_by_logloss['lr_weight']:.0%} / XGB: {best_by_logloss['xgb_weight']:.0%}")
    print(f"  Accuracy: {best_by_logloss['accuracy']:.4f}")
    print(f"  Log-loss: {best_by_logloss['log_loss']:.4f} "
          f"({best_by_logloss['log_loss'] - baseline['log_loss']:+.4f})")
    
    # Save results
    summary = {
        'dataset': {
            'total_games': len(X),
            'features': len(feature_names),
            'seasons': sorted([int(s) for s in np.unique(seasons)]),
        },
        'baseline_5050': {
            'lr_weight': 0.5,
            'xgb_weight': 0.5,
            'accuracy': float(baseline['accuracy']),
            'log_loss': float(baseline['log_loss']),
        },
        'best_by_accuracy': {
            'lr_weight': float(best_by_accuracy['lr_weight']),
            'xgb_weight': float(best_by_accuracy['xgb_weight']),
            'accuracy': float(best_by_accuracy['accuracy']),
            'log_loss': float(best_by_accuracy['log_loss']),
            'accuracy_improvement': float(best_by_accuracy['accuracy'] - baseline['accuracy']),
        },
        'best_by_logloss': {
            'lr_weight': float(best_by_logloss['lr_weight']),
            'xgb_weight': float(best_by_logloss['xgb_weight']),
            'accuracy': float(best_by_logloss['accuracy']),
            'log_loss': float(best_by_logloss['log_loss']),
            'logloss_improvement': float(best_by_logloss['log_loss'] - baseline['log_loss']),
        },
        'by_season': season_results,
        'all_weight_combinations': results,
        'recommendation': (
            f"Use LR:{best_by_accuracy['lr_weight']:.0%} / XGB:{best_by_accuracy['xgb_weight']:.0%} "
            f"for {best_by_accuracy['accuracy'] - baseline['accuracy']:+.2%} accuracy improvement"
        ),
    }
    
    out_path = Path(args.out_dir) / 'ensemble_weight_optimization.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved to {out_path}")
    
    # Save CSV
    csv_path = Path(args.out_dir) / 'ensemble_weight_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")


if __name__ == '__main__':
    main()
