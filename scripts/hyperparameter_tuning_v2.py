"""Fixed hyperparameter tuning that actually runs on tournament data.

Simpler version that:
1. Uses the standardized games schema with season column
2. Runs 5-fold cross-validation
3. Grid searches over reasonable parameter ranges
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss


def run_hyperparameter_tuning(games_dir, features_path, out_dir='results'):
    """Run hyperparameter tuning with standardized games schema."""
    
    games_dir = Path(games_dir)
    feature_cols = [
        'win_pct', 'opp_avg_margin', 'avg_margin', 'adj_margin', 'avg_points_against',
        'avg_points_for', 'last5_win_pct', 'sos_win_pct', 'last10_wins', 'last10_losses',
        'offense_trend', 'defense_trend', 'avg_points_for', 'avg_points_against'
    ]
    
    print("\n=== Hyperparameter Tuning ===\n")
    
    # Load features
    print("Loading tournament features...")
    features_df = pd.read_csv(features_path)
    
    # Load all tournament games
    print("Loading tournament games...")
    all_games = []
    for year in [2021, 2022, 2023, 2024, 2025]:
        games_path = games_dir / f'games_{year}.csv'
        if games_path.exists():
            games = pd.read_csv(games_path)
            postseason_mask = (games.get('is_postseason', False).astype(str).str.lower() == 'true') | \
                            (games.get('tournament', '').astype(str).str.contains('Championship', case=False, na=False))
            all_games.append(games[postseason_mask].copy())
    
    if not all_games:
        print("⚠️  No tournament games found")
        return
    
    games_df = pd.concat(all_games, ignore_index=True)
    print(f"Found {len(games_df)} tournament games\n")
    
    # Build matchup dataset
    print("Building matchup dataset...")
    X_list = []
    y_list = []
    
    feature_cols_found = []
    for col in feature_cols:
        if col in features_df.columns:
            feature_cols_found.append(col)
    
    for _, game in games_df.iterrows():
        home_team = str(game.get('home_team', '')).strip()
        away_team = str(game.get('away_team', '')).strip()
        home_win = game.get('home_win')
        
        if pd.isna(home_win) or not home_team or not away_team:
            continue
        
        season = int(game.get('season', 2025))
        
        # Find team features
        home_match = features_df[(features_df['team'].str.strip() == home_team) & (features_df['season'] == season)]
        away_match = features_df[(features_df['team'].str.strip() == away_team) & (features_df['season'] == season)]
        
        if home_match.empty or away_match.empty:
            continue
        
        # Build feature difference
        home_feats = [float(home_match.iloc[0].get(col, 0)) for col in feature_cols_found]
        away_feats = [float(away_match.iloc[0].get(col, 0)) for col in feature_cols_found]
        diff_feats = [h - a for h, a in zip(home_feats, away_feats)]
        
        X_list.append(diff_feats)
        y_list.append(1 if bool(home_win) else 0)
    
    if not X_list:
        print("⚠️  No valid matchups created")
        return
    
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Created {len(X)} training matchups with {len(feature_cols_found)} features\n")
    
    # Run hyperparameter grid search
    print("Grid searching hyperparameters...")
    results = []
    
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        for penalty in ['l2']:
            try:
                lr = LogisticRegression(C=C, penalty=penalty, max_iter=1000, solver='lbfgs')
                scores = cross_val_score(lr, X, y, cv=min(5, len(X) // 3), scoring='accuracy')
                
                results.append({
                    'C': C,
                    'penalty': penalty,
                    'cv_accuracy_mean': scores.mean(),
                    'cv_accuracy_std': scores.std(),
                    'cv_fold_count': len(scores),
                })
                
                print(f"  C={C:7.3f}, penalty={penalty:2s} -> {scores.mean():.3f} (+/- {scores.std():.3f})")
            except Exception as e:
                print(f"  C={C:7.3f}, penalty={penalty:2s} -> error: {str(e)[:50]}")
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("⚠️  No successful grid search results")
        return
    
    best_result = results_df.loc[results_df['cv_accuracy_mean'].idxmax()]
    
    print(f"\n=== Best Hyperparameters ===")
    print(f"  C: {best_result['C']}")
    print(f"  Penalty: {best_result['penalty']}")
    print(f"  Cross-validation accuracy: {best_result['cv_accuracy_mean']:.4f}")
    print(f"  Baseline (current model): 0.6640")
    print(f"  Delta: {best_result['cv_accuracy_mean'] - 0.6640:+.4f}")
    
    # Save results
    out_path = Path(out_dir) / 'hyperparameter_tuning_summary.json'
    summary = {
        'dataset': {
            'games': len(games_df),
            'tournament_games': len(X),
            'features': len(feature_cols_found),
        },
        'best_hyperparameters': {
            'C': float(best_result['C']),
            'penalty': str(best_result['penalty']),
        },
        'best_performance': {
            'cv_accuracy': round(float(best_result['cv_accuracy_mean']), 4),
            'cv_std': round(float(best_result['cv_accuracy_std']), 4),
        },
        'baseline_performance': {
            'accuracy': 0.6640,
            'method': 'ensemble_model',
        },
        'improvement': {
            'accuracy_delta': round(float(best_result['cv_accuracy_mean'] - 0.6640), 4),
        },
        'all_results': results_df.to_dict('records'),
    }
    
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save CSV
    csv_path = Path(out_dir) / 'hyperparameter_tuning_results.csv'
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Saved to {out_path}")
    print(f"✓ Saved to {csv_path}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal hyperparameters for tournament predictions')
    parser.add_argument('--games_dir', default='data/processed')
    parser.add_argument('--features', default='data/processed/features/tournament_teams.csv')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    
    run_hyperparameter_tuning(args.games_dir, args.features, args.out_dir)
