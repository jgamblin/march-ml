"""Analyze regional imbalance in tournament predictions.

Checks if model bias toward certain regions is due to:
1. Seeding (some regions get better seeds)
2. Actual regional strength (better teams in some regions)
3. Model bias (systematic overweighting of some regions)
"""
import json
import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path


def analyze_regional_imbalance(sim_json_path, features_path, out_dir='results'):
    """Analyze why model favors certain regions."""
    
    # Load simulation results
    with open(sim_json_path) as f:
        sim_data = json.load(f)
    
    champion_probs = {team: prob for team, prob in sim_data.get('champion_probs', [])}
    
    # Load tournament features with seeds
    features_df = pd.read_csv(features_path)
    season = sim_data.get('season', 2025)
    season_features = features_df[features_df['season'] == season].copy()
    
    # Filter to tournament-seeded teams only
    seeded_features = season_features[season_features['seed'].notna()].copy()
    
    # Add champion probabilities
    seeded_features['champion_prob'] = seeded_features['team'].map(champion_probs)
    seeded_features = seeded_features[seeded_features['champion_prob'].notna()]
    
    # Group by region
    regional_stats = defaultdict(lambda: {
        'teams': [],
        'total_prob': 0,
        'avg_prob': 0,
        'avg_seed': 0,
        'avg_adj_margin': 0,
    })
    
    for _, row in seeded_features.iterrows():
        region = row['region']
        regional_stats[region]['teams'].append({
            'team': row['team'],
            'seed': row['seed'],
            'champion_prob': row['champion_prob'],
            'adj_margin': row['adj_margin'],
            'win_pct': row['win_pct'],
        })
        regional_stats[region]['total_prob'] += row['champion_prob']
    
    # Calculate regional metrics
    for region, stats in regional_stats.items():
        stats['num_teams'] = len(stats['teams'])
        if stats['num_teams'] > 0:
            stats['avg_prob'] = stats['total_prob'] / stats['num_teams']
            stats['avg_seed'] = sum(t['seed'] for t in stats['teams']) / stats['num_teams']
            stats['avg_adj_margin'] = sum(t['adj_margin'] for t in stats['teams']) / stats['num_teams']
    
    # Build report
    report = {
        'season': season,
        'regions': {},
        'analysis': {},
    }
    
    # Regional breakdown
    total_prob = sum(stats['total_prob'] for stats in regional_stats.values())
    for region in sorted(regional_stats.keys()):
        stats = regional_stats[region]
        report['regions'][region] = {
            'teams': stats['num_teams'],
            'total_champion_prob': round(stats['total_prob'], 4),
            'pct_of_total': round(stats['total_prob'] / total_prob * 100, 1) if total_prob > 0 else 0,
            'avg_champion_prob': round(stats['avg_prob'], 4),
            'avg_seed': round(stats['avg_seed'], 1),
            'avg_adjusted_margin': round(stats['avg_adj_margin'], 2),
            'top_team': max(stats['teams'], key=lambda x: x['champion_prob'])['team'],
            'top_team_prob': round(max(t['champion_prob'] for t in stats['teams']), 4),
        }
    
    # Analyze sources of imbalance
    sorted_regions = sorted(report['regions'].items(), key=lambda x: x[1]['pct_of_total'], reverse=True)
    
    print("\n=== Regional Imbalance Analysis ===\n")
    print("Regional Championship Probability Distribution:")
    for region, stats in sorted_regions:
        print(f"  {region:12} {stats['pct_of_total']:5.1f}% ({stats['total_champion_prob']:.3f})")
    
    # Coefficient of variation
    probs = [stats['pct_of_total'] for _, stats in sorted_regions]
    mean_prob = sum(probs) / len(probs)
    variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
    std_dev = variance ** 0.5
    cv = (std_dev / mean_prob) * 100
    
    print(f"\nRegional Probability Distribution:")
    print(f"  Expected (uniform): 25% per region")
    print(f"  Actual range: {min(probs):.1f}% - {max(probs):.1f}%")
    print(f"  Coefficient of variation: {cv:.1f}%")
    
    # Root cause analysis
    print(f"\n=== Root Cause Analysis ===\n")
    
    # Check if seeding explains imbalance
    seed_variance = 0
    for region, stats in report['regions'].items():
        seed_variance += (stats['avg_seed'] - 8.5) ** 2
    seed_variance = (seed_variance / len(report['regions'])) ** 0.5
    
    print(f"1. Seeding Influence:")
    for region, stats in sorted(report['regions'].items(), key=lambda x: x[1]['avg_seed']):
        print(f"   {region:12} avg seed: {stats['avg_seed']:.1f} (lower = better)")
    
    # Check if team quality explains imbalance
    print(f"\n2. Team Quality (Adjusted Margin):")
    for region, stats in sorted(report['regions'].items(), key=lambda x: x[1]['avg_adjusted_margin'], reverse=True):
        print(f"   {region:12} avg adj margin: {stats['avg_adjusted_margin']:+.2f}")
    
    # Assess model bias
    print(f"\n3. Model Bias Assessment:")
    qual_rank = sorted([(r, s['avg_adjusted_margin']) for r, s in report['regions'].items()], key=lambda x: x[1], reverse=True)
    prob_rank = sorted([(r, s['pct_of_total']) for r, s in report['regions'].items()], key=lambda x: x[1], reverse=True)
    
    qual_order = [r for r, _ in qual_rank]
    prob_order = [r for r, _ in prob_rank]
    
    if qual_order == prob_order:
        print("   ✓ Model ranking matches team quality ranking - no obvious bias")
    else:
        print(f"   ⚠️  Model ranking differs from quality:")
        print(f"      Quality ranking: {' > '.join(qual_order)}")
        print(f"      Prob ranking:    {' > '.join(prob_order)}")
    
    # Save report
    out_path = Path(out_dir) / 'regional_imbalance_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed analysis saved to {out_path}")
    
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze regional imbalance in tournament predictions')
    parser.add_argument('--sim_json', default='results/sim_5000_final.json', help='Simulation output')
    parser.add_argument('--features', default='data/processed/features/tournament_teams.csv', help='Tournament features')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    
    analyze_regional_imbalance(args.sim_json, args.features, args.out_dir)
