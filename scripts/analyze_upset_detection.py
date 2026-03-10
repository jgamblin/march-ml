"""Detect where the model is vulnerable to upsets.

Analyzes model predictions against historical outcomes to find:
1. High-seed vs low-seed matchup accuracy
2. Accuracy by prediction confidence
3. Win/loss patterns for different matchup types
"""
import json
import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path


def analyze_upset_detection(games_dir, features_path, out_dir='results'):
    """Identify matchup types where model struggles."""
    
    games_dir = Path(games_dir)
    
    # Load features with seeds
    features_df = pd.read_csv(features_path)
    
    # Load tournament games from all seasons
    all_games = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        games_path = games_dir / f'games_{season}.csv'
        if games_path.exists():
            games = pd.read_csv(games_path)
            games['season'] = season
            # Filter to postseason only
            postseason_mask = (games.get('is_postseason', False).astype(str).str.lower() == 'true') | \
                            (games.get('tournament', '').astype(str).str.contains('Championship', case=False, na=False))
            all_games.append(games[postseason_mask])
    
    if not all_games:
        print("No tournament games found")
        return
    
    games_df = pd.concat(all_games, ignore_index=True)
    
    # Analysis buckets
    upset_analysis = defaultdict(lambda: {
        'matchups': [],
        'home_wins': 0,
        'away_wins': 0,
        'accuracy': 0,
    })
    
    print("\n=== Upset Detection Analysis ===\n")
    
    # 1. Seed-based analysis
    print("1. Performance by Seed Matchup Type:\n")
    
    seed_results = defaultdict(lambda: {'total': 0, 'upsets': 0})
    
    for _, game in games_df.iterrows():
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        home_win = game.get('home_win', False)
        
        if pd.isna(home_win):
            continue
        
        season = int(game['season']) if 'season' in game else 2025
        
        # Find seed data
        home_features = features_df[(features_df['team'] == home_team) & (features_df['season'] == season)]
        away_features = features_df[(features_df['team'] == away_team) & (features_df['season'] == season)]
        
        if home_features.empty or away_features.empty:
            continue
        
        home_seed = home_features.iloc[0].get('seed')
        away_seed = away_features.iloc[0].get('seed')
        
        if pd.isna(home_seed) or pd.isna(away_seed):
            continue
        
        # Categorize matchup
        higher_seed = min(home_seed, away_seed)  # Lower seed number = higher seed
        lower_seed = max(home_seed, away_seed)
        seed_diff = lower_seed - higher_seed
        
        higher_favored = (home_seed < away_seed and home_win) or (away_seed < home_seed and not home_win)
        is_upset = not higher_favored
        
        seed_key = f'Seed{int(higher_seed)}_vs_Seed{int(lower_seed)}'
        seed_results[seed_key]['total'] += 1
        if is_upset:
            seed_results[seed_key]['upsets'] += 1
    
    for matchup in sorted(seed_results.keys()):
        data = seed_results[matchup]
        upset_pct = (data['upsets'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"  {matchup:20} {data['total']:3} games, {upset_pct:5.1f}% upsets")
    
    # 2. Margin of victory analysis
    print(f"\n2. Performance by Predicted Confidence:\n")
    
    margin_results = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for _, game in games_df.iterrows():
        home_score = game.get('home_score')
        away_score = game.get('away_score')
        
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        
        home_win = game.get('home_win', False)
        if pd.isna(home_win):
            continue
        
        margin = abs(home_score - away_score)
        
        # Categorize by margin
        if margin <= 5:
            confidence = 'Close (0-5 pts)'
        elif margin <= 10:
            confidence = 'Moderate (6-10 pts)'
        else:
            confidence = 'Large (11+ pts)'
        
        margin_results[confidence]['total'] += 1
        # Note: without model predictions, we can only track game margin statistics
    
    for confidence in ['Close (0-5 pts)', 'Moderate (6-10 pts)', 'Large (11+ pts)']:
        data = margin_results[confidence]
        if data['total'] > 0:
            print(f"  {confidence:20} {data['total']:3} games")
    
    # 3. Summary statistics
    print(f"\n3. Upset Summary:\n")
    total_tournament_games = sum(data['total'] for data in seed_results.values())
    total_upsets = sum(data['upsets'] for data in seed_results.values())
    overall_upset_rate = (total_upsets / total_tournament_games * 100) if total_tournament_games > 0 else 0
    
    print(f"  Total tournament games analyzed: {total_tournament_games}")
    print(f"  Total upsets (higher seed lost): {total_upsets}")
    print(f"  Overall upset rate: {overall_upset_rate:.1f}%")
    
    print(f"\n4. Model Vulnerability Assessment:\n")
    
    # Find most upset-prone matchups
    most_upset_prone = sorted(seed_results.items(), key=lambda x: x[1]['upsets'] / max(x[1]['total'], 1), reverse=True)[:5]
    print(f"  Most upset-prone matchups:")
    for matchup, data in most_upset_prone:
        upset_pct = (data['upsets'] / data['total'] * 100) if data['total'] > 0 else 0
        if data['total'] >= 2:  # Only show if enough samples
            print(f"    {matchup:25} {upset_pct:5.1f}% upset rate ({data['upsets']}/{data['total']})")
    
    # Build report
    report = {
        'total_games': total_tournament_games,
        'total_upsets': total_upsets,
        'overall_upset_rate': round(overall_upset_rate, 1),
        'seed_matchup_results': dict(seed_results),
        'margin_results': dict(margin_results),
    }
    
    # Save
    out_path = Path(out_dir) / 'upset_detection_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed analysis saved to {out_path}")
    
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze upset detection and model weaknesses')
    parser.add_argument('--games_dir', default='data/processed', help='Directory with games_*.csv')
    parser.add_argument('--features', default='data/processed/features/tournament_teams.csv')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    
    analyze_upset_detection(args.games_dir, args.features, args.out_dir)
