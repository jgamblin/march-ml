#!/usr/bin/env python3
"""
Prediction Confidence Intervals
Estimates prediction uncertainty using bootstrap resampling
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from sklearn.utils import resample

def load_models(models_dir='models'):
    """Load trained models and features"""
    models_dir = Path(models_dir)
    
    lr_model = joblib.load(models_dir / 'lr_model.joblib')
    xgb_model = joblib.load(models_dir / 'xgb_model.joblib')
    lr_platt = joblib.load(models_dir / 'lr_platt.joblib')
    xgb_platt = joblib.load(models_dir / 'xgb_platt.joblib')
    feature_names = joblib.load(models_dir / 'model_features.joblib')
    
    return {
        'lr_model': lr_model,
        'xgb_model': xgb_model,
        'lr_platt': lr_platt,
        'xgb_platt': xgb_platt,
        'feature_names': feature_names
    }

def build_tournament_dataset(games_dir='data/processed', features_path='data/processed/features', feature_names=None):
    """Build dataset of tournament games"""
    games_dir = Path(games_dir)
    features_path = Path(features_path)
    
    tournament_games = []
    
    for year in [2021, 2022, 2023, 2024, 2025]:
        games_file = games_dir / f'games_{year}.csv'
        season_agg = features_path / f'season_aggregates_{year}.csv'
        
        if not games_file.exists() or not season_agg.exists():
            continue
        
        games = pd.read_csv(games_file)
        stats = pd.read_csv(season_agg)
        
        # Filter NCAA tournament games (Men's Basketball Championship)
        tourney = games[games['tournament'].str.contains("Men's Basketball Championship", na=False)].copy()
        
        for _, game in tourney.iterrows():
            # Use home_team as team1, away_team as team2
            team1 = game['home_team']
            team2 = game['away_team']
            is_neutral = game.get('is_neutral', True)
            
            # Get team stats
            t1_stats = stats[stats['team'] == team1]
            t2_stats = stats[stats['team'] == team2]
            
            if len(t1_stats) == 0 or len(t2_stats) == 0:
                continue
            
            t1_stats = t1_stats.iloc[0]
            t2_stats = t2_stats.iloc[0]
            
            # Build feature vector
            feature_vec = []
            
            # Base features
            for feat in ['games_played', 'wins', 'losses', 'win_pct',
                        'avg_points_for', 'avg_points_against', 'avg_margin',
                        'last5_win_pct', 'last10_wins', 'last10_losses',
                        'offense_trend', 'defense_trend', 'sos_win_pct',
                        'opp_avg_margin', 'adj_margin']:
                t1_val = t1_stats.get(feat, 0)
                t2_val = t2_stats.get(feat, 0)
                feature_vec.append(float(t1_val - t2_val))
            
            # Add location indicators
            neutral_site = 1.0 if str(is_neutral).lower() == 'true' else 0.0
            home_edge = 0.0 if neutral_site else 1.0
            feature_vec.extend([neutral_site, home_edge])
            
            # Determine outcome (home team win = 1, away win = 0)
            score1 = game.get('home_score', 0)
            score2 = game.get('away_score', 0)
            outcome = 1 if score1 > score2 else 0
            
            tournament_games.append({
                'season': year,
                'team1': team1,
                'team2': team2,
                'features': feature_vec,
                'outcome': outcome,
                'is_neutral': is_neutral
            })
    
    print(f"Loaded {len(tournament_games)} tournament games")
    return tournament_games

def get_predictions(model, calibrator, X):
    """Get calibrated predictions"""
    raw_probs = model.predict_proba(X)[:, 1]
    cal_probs = calibrator.predict(raw_probs.reshape(-1, 1)).flatten()
    return cal_probs

def bootstrap_confidence(models, X, y, n_bootstrap=1000, lr_weight=0.65):
    """
    Estimate prediction confidence using bootstrap resampling
    
    Returns:
        confidence_scores: Array of confidence values (0-1) for each prediction
        prediction_intervals: List of (lower, upper) bounds for each prediction
    """
    n_samples = len(X)
    xgb_weight = 1.0 - lr_weight
    
    # Storage for bootstrap predictions
    bootstrap_preds = np.zeros((n_bootstrap, n_samples))
    
    print(f"Running {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        # Resample training data with replacement
        indices = resample(range(n_samples), n_samples=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Get predictions on original dataset (out-of-bag)
        lr_probs = get_predictions(models['lr_model'], models['lr_platt'], X)
        xgb_probs = get_predictions(models['xgb_model'], models['xgb_platt'], X)
        ensemble_probs = lr_weight * lr_probs + xgb_weight * xgb_probs
        
        bootstrap_preds[i, :] = ensemble_probs
        
        if (i + 1) % 200 == 0:
            print(f"  Completed {i+1}/{n_bootstrap} iterations")
    
    # Calculate confidence metrics
    pred_mean = bootstrap_preds.mean(axis=0)
    pred_std = bootstrap_preds.std(axis=0)
    
    # Confidence score: inverse of prediction variance
    # Higher variance = lower confidence
    # Normalize to 0-1 scale
    max_std = pred_std.max()
    confidence_scores = 1.0 - (pred_std / max_std) if max_std > 0 else np.ones(n_samples)
    
    # 95% confidence intervals
    pred_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
    pred_upper = np.percentile(bootstrap_preds, 97.5, axis=0)
    prediction_intervals = list(zip(pred_lower, pred_upper))
    
    return confidence_scores, prediction_intervals, pred_mean, pred_std

def categorize_confidence(confidence_score):
    """Categorize confidence level"""
    if confidence_score >= 0.80:
        return "Very High"
    elif confidence_score >= 0.65:
        return "High"
    elif confidence_score >= 0.50:
        return "Moderate"
    elif confidence_score >= 0.35:
        return "Low"
    else:
        return "Very Low"

def main():
    print("=== Prediction Confidence Analysis ===\n")
    
    # Load models
    print("Loading models...")
    models = load_models()
    feature_names = models['feature_names']
    print(f"Model features: {len(feature_names)}")
    
    # Build tournament dataset
    print("\nBuilding tournament dataset...")
    tournament_games = build_tournament_dataset(feature_names=feature_names)
    
    if len(tournament_games) == 0:
        print("No tournament games found!")
        return
    
    # Convert to arrays
    X = np.array([g['features'] for g in tournament_games])
    y = np.array([g['outcome'] for g in tournament_games])
    
    print(f"Dataset: {len(X)} games, {X.shape[1]} features")
    
    # Run bootstrap analysis
    print("\nRunning bootstrap confidence estimation...")
    confidence_scores, pred_intervals, pred_mean, pred_std = bootstrap_confidence(
        models, X, y, n_bootstrap=1000, lr_weight=0.65
    )
    
    # Add confidence metrics to games
    for i, game in enumerate(tournament_games):
        game['pred_prob'] = float(pred_mean[i])
        game['pred_std'] = float(pred_std[i])
        game['confidence_score'] = float(confidence_scores[i])
        game['confidence_level'] = categorize_confidence(confidence_scores[i])
        game['pred_interval_lower'] = float(pred_intervals[i][0])
        game['pred_interval_upper'] = float(pred_intervals[i][1])
        game['interval_width'] = float(pred_intervals[i][1] - pred_intervals[i][0])
    
    # Analyze confidence patterns
    print("\n=== Confidence Analysis ===\n")
    
    confidence_levels = {}
    for game in tournament_games:
        level = game['confidence_level']
        if level not in confidence_levels:
            confidence_levels[level] = {
                'count': 0,
                'correct': 0,
                'avg_prob': 0,
                'avg_std': 0
            }
        
        confidence_levels[level]['count'] += 1
        confidence_levels[level]['avg_prob'] += game['pred_prob']
        confidence_levels[level]['avg_std'] += game['pred_std']
        
        # Check if prediction was correct
        predicted_winner = 1 if game['pred_prob'] > 0.5 else 0
        if predicted_winner == game['outcome']:
            confidence_levels[level]['correct'] += 1
    
    # Calculate averages
    for level in confidence_levels:
        count = confidence_levels[level]['count']
        confidence_levels[level]['accuracy'] = confidence_levels[level]['correct'] / count
        confidence_levels[level]['avg_prob'] /= count
        confidence_levels[level]['avg_std'] /= count
    
    # Print results by confidence level
    print("Accuracy by Confidence Level:")
    for level in ['Very High', 'High', 'Moderate', 'Low', 'Very Low']:
        if level in confidence_levels:
            stats = confidence_levels[level]
            print(f"  {level:12s}: {stats['count']:3d} games, "
                  f"Accuracy: {stats['accuracy']:.3f}, "
                  f"Avg Prob: {stats['avg_prob']:.3f}, "
                  f"Avg Std: {stats['avg_std']:.4f}")
    
    # Identify high-confidence predictions
    high_conf_games = [g for g in tournament_games if g['confidence_score'] >= 0.80]
    print(f"\nHigh confidence predictions: {len(high_conf_games)} games")
    
    if len(high_conf_games) > 0:
        high_conf_correct = sum(1 for g in high_conf_games 
                               if (g['pred_prob'] > 0.5) == g['outcome'])
        print(f"  Accuracy on high-confidence games: {high_conf_correct/len(high_conf_games):.3f}")
    
    # Identify coin-flip games (low confidence)
    coin_flip_games = [g for g in tournament_games 
                      if 0.45 <= g['pred_prob'] <= 0.55]
    print(f"\nCoin-flip predictions (45-55%): {len(coin_flip_games)} games")
    
    # Season-specific analysis
    print("\n=== Confidence by Season ===\n")
    for season in [2021, 2022, 2023, 2024, 2025]:
        season_games = [g for g in tournament_games if g['season'] == season]
        if len(season_games) == 0:
            continue
        
        avg_conf = np.mean([g['confidence_score'] for g in season_games])
        avg_std = np.mean([g['pred_std'] for g in season_games])
        
        print(f"{season}: {len(season_games):3d} games, "
              f"Avg Confidence: {avg_conf:.3f}, "
              f"Avg Std: {avg_std:.4f}")
    
    # Save detailed results
    results = {
        'summary': {
            'total_games': len(tournament_games),
            'bootstrap_iterations': 1000,
            'ensemble_weights': {'lr': 0.65, 'xgb': 0.35},
            'confidence_distribution': {
                level: {
                    'count': stats['count'],
                    'accuracy': float(stats['accuracy']),
                    'avg_probability': float(stats['avg_prob']),
                    'avg_std_dev': float(stats['avg_std'])
                }
                for level, stats in confidence_levels.items()
            }
        },
        'games': [
            {
                'season': g['season'],
                'team1': g['team1'],
                'team2': g['team2'],
                'outcome': g['outcome'],
                'pred_prob': g['pred_prob'],
                'pred_std': g['pred_std'],
                'confidence_score': g['confidence_score'],
                'confidence_level': g['confidence_level'],
                'interval_lower': g['pred_interval_lower'],
                'interval_upper': g['pred_interval_upper']
            }
            for g in tournament_games
        ]
    }
    
    output_file = Path('results/prediction_confidence_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved confidence analysis to {output_file}")
    
    # Save CSV for easy analysis
    df = pd.DataFrame([
        {
            'season': g['season'],
            'team1': g['team1'],
            'team2': g['team2'],
            'outcome': g['outcome'],
            'pred_prob': g['pred_prob'],
            'pred_std': g['pred_std'],
            'confidence_score': g['confidence_score'],
            'confidence_level': g['confidence_level'],
            'interval_lower': g['pred_interval_lower'],
            'interval_upper': g['pred_interval_upper'],
            'interval_width': g['interval_width']
        }
        for g in tournament_games
    ])
    
    csv_file = Path('results/prediction_confidence_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved confidence results to {csv_file}")
    
    print("\n=== Key Insights ===\n")
    print(f"Average confidence score: {np.mean(confidence_scores):.3f}")
    print(f"Average prediction uncertainty (std): {np.mean(pred_std):.4f}")
    print(f"Games with confidence > 0.80: {len(high_conf_games)} ({100*len(high_conf_games)/len(tournament_games):.1f}%)")
    print(f"Coin-flip games (45-55%): {len(coin_flip_games)} ({100*len(coin_flip_games)/len(tournament_games):.1f}%)")

if __name__ == '__main__':
    main()
