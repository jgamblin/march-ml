#!/usr/bin/env python3
"""
Add Momentum and Recency Weighting Features
Adds features that capture late-season form and momentum
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def calculate_exponential_weights(n_games, decay_rate=0.1):
    """
    Calculate exponential decay weights for recent games
    Most recent game has highest weight
    
    Args:
        n_games: Number of games
        decay_rate: Rate of decay (higher = faster decay)
    
    Returns:
        weights: Array of weights (normalized to sum to 1)
    """
    # Most recent game is index 0, oldest is n_games-1
    indices = np.arange(n_games)
    weights = np.exp(-decay_rate * indices)
    weights = weights / weights.sum()
    return weights

def calculate_momentum_features(games_df, team, n_last_games=10, decay_rate=0.1):
    """
    Calculate momentum features for a team based on recent games
    
    Returns dict with:
    - weighted_margin: Exponentially weighted margin
    - win_streak: Current win streak (negative for losses)
    - trend_slope: Linear trend in margins (positive = improving)
    - last_N_momentum: Simple average of last N games margin
    """
    # Get team's games sorted by date (most recent first)
    team_games = games_df[
        (games_df['home_team'] == team) | (games_df['away_team'] == team)
    ].copy()
    
    if len(team_games) == 0:
        return {
            'weighted_last10_margin': 0.0,
            'win_streak': 0,
            'margin_trend_slope': 0.0,
            'last5_momentum': 0.0,
            'last10_momentum': 0.0,
            'form_rating': 0.0
        }
    
    # Sort by date (most recent first)
    team_games = team_games.sort_values('game_day', ascending=False)
    
    # Calculate margin for each game
    margins = []
    wins = []
    
    for _, game in team_games.iterrows():
        if game['home_team'] == team:
            margin = game['home_score'] - game['away_score']
            win = 1 if game['home_win'] else 0
        else:
            margin = game['away_score'] - game['home_score']
            win = 0 if game['home_win'] else 1
        
        margins.append(margin)
        wins.append(win)
    
    margins = np.array(margins)
    wins = np.array(wins)
    
    # Calculate features
    features = {}
    
    # 1. Weighted margin (last 10 games)
    last_10_margins = margins[:min(10, len(margins))]
    if len(last_10_margins) > 0:
        weights = calculate_exponential_weights(len(last_10_margins), decay_rate)
        features['weighted_last10_margin'] = float(np.sum(last_10_margins * weights))
    else:
        features['weighted_last10_margin'] = 0.0
    
    # 2. Win streak
    streak = 0
    for w in wins:
        if len(wins) > 0 and w == wins[0]:
            streak += 1 if wins[0] == 1 else -1
        else:
            break
    features['win_streak'] = streak
    
    # 3. Trend slope (linear regression on last 10 game margins)
    last_10_margins = margins[:min(10, len(margins))]
    if len(last_10_margins) >= 3:
        x = np.arange(len(last_10_margins))
        # Reverse so oldest game is 0 and newest is max
        y = last_10_margins[::-1]
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        features['margin_trend_slope'] = float(slope)
    else:
        features['margin_trend_slope'] = 0.0
    
    # 4. Simple momentum (average of last N games)
    last_5_margins = margins[:min(5, len(margins))]
    features['last5_momentum'] = float(np.mean(last_5_margins)) if len(last_5_margins) > 0 else 0.0
    
    last_10_margins = margins[:min(10, len(margins))]
    features['last10_momentum'] = float(np.mean(last_10_margins)) if len(last_10_margins) > 0 else 0.0
    
    # 5. Form rating (composite score)
    # Combines weighted margin, trend, and win streak
    form_rating = (
        features['weighted_last10_margin'] * 0.4 +
        features['margin_trend_slope'] * 10 * 0.3 +  # Scale up slope
        features['win_streak'] * 2 * 0.3  # Scale up streak
    )
    features['form_rating'] = float(form_rating)
    
    return features

def add_momentum_to_season(games_file, output_file, decay_rate=0.1):
    """
    Add momentum features to all teams in a season
    """
    games = pd.read_csv(games_file)
    
    # Get all unique teams
    teams = set(games['home_team'].unique()) | set(games['away_team'].unique())
    
    # Calculate momentum for each team
    team_momentum = {}
    for team in teams:
        momentum = calculate_momentum_features(games, team, decay_rate=decay_rate)
        team_momentum[team] = momentum
    
    # Convert to DataFrame
    momentum_df = pd.DataFrame.from_dict(team_momentum, orient='index')
    momentum_df.index.name = 'team'
    momentum_df.reset_index(inplace=True)
    
    return momentum_df

def merge_with_season_aggregates(season_agg_file, momentum_df):
    """
    Merge momentum features with existing season aggregates
    """
    season_agg = pd.read_csv(season_agg_file)
    
    # Merge on team name
    merged = season_agg.merge(momentum_df, on='team', how='left')
    
    # Fill any missing values with 0
    momentum_cols = [
        'weighted_last10_margin', 'win_streak', 'margin_trend_slope',
        'last5_momentum', 'last10_momentum', 'form_rating'
    ]
    for col in momentum_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    
    return merged

def main():
    print("=== Adding Momentum & Recency Features ===\n")
    
    # Process each season
    for year in [2021, 2022, 2023, 2024, 2025]:
        games_file = Path(f'data/processed/games_{year}.csv')
        season_agg_file = Path(f'data/processed/features/season_aggregates_{year}.csv')
        
        if not games_file.exists() or not season_agg_file.exists():
            print(f"Skipping {year}: missing files")
            continue
        
        print(f"Processing {year} season...")
        
        # Calculate momentum features
        momentum_df = add_momentum_to_season(games_file, None, decay_rate=0.1)
        
        print(f"  Calculated momentum for {len(momentum_df)} teams")
        
        # Merge with existing season aggregates
        merged = merge_with_season_aggregates(season_agg_file, momentum_df)
        
        # Save enhanced features
        output_file = Path(f'data/processed/features/season_aggregates_{year}_with_momentum.csv')
        merged.to_csv(output_file, index=False)
        print(f"  ✓ Saved to {output_file}")
        
        # Show sample stats
        print(f"  Momentum stats:")
        print(f"    Avg form rating: {momentum_df['form_rating'].mean():.2f}")
        print(f"    Max win streak: {momentum_df['win_streak'].max():.0f}")
        print(f"    Avg trend slope: {momentum_df['margin_trend_slope'].mean():.3f}")
    
    # Analyze momentum patterns
    print("\n=== Momentum Analysis ===\n")
    
    all_momentum = []
    for year in [2021, 2022, 2023, 2024, 2025]:
        file_path = Path(f'data/processed/features/season_aggregates_{year}_with_momentum.csv')
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['season'] = year
            all_momentum.append(df)
    
    if len(all_momentum) > 0:
        combined = pd.concat(all_momentum, ignore_index=True)
        
        # Identify teams with strong momentum
        high_momentum_teams = combined.nlargest(20, 'form_rating')[
            ['team', 'season', 'form_rating', 'win_streak', 'margin_trend_slope']
        ]
        
        print("Top 20 teams by form rating across all seasons:")
        for _, row in high_momentum_teams.iterrows():
            print(f"  {row['season']} {row['team']:30s} "
                  f"Form: {row['form_rating']:6.2f}, "
                  f"Streak: {row['win_streak']:3.0f}, "
                  f"Trend: {row['margin_trend_slope']:6.3f}")
        
        # Teams fading (negative momentum)
        fading_teams = combined.nsmallest(10, 'form_rating')[
            ['team', 'season', 'form_rating', 'win_streak', 'margin_trend_slope']
        ]
        
        print("\nTop 10 fading teams (worst form):")
        for _, row in fading_teams.iterrows():
            print(f"  {row['season']} {row['team']:30s} "
                  f"Form: {row['form_rating']:6.2f}, "
                  f"Streak: {row['win_streak']:3.0f}, "
                  f"Trend: {row['margin_trend_slope']:6.3f}")
        
        # Save analysis
        analysis = {
            'summary': {
                'total_teams': len(combined),
                'avg_form_rating': float(combined['form_rating'].mean()),
                'avg_win_streak': float(combined['win_streak'].mean()),
                'avg_trend_slope': float(combined['margin_trend_slope'].mean()),
                'max_form_rating': float(combined['form_rating'].max()),
                'min_form_rating': float(combined['form_rating'].min())
            },
            'top_momentum_teams': [
                {
                    'team': row['team'],
                    'season': int(row['season']),
                    'form_rating': float(row['form_rating']),
                    'win_streak': int(row['win_streak']),
                    'trend_slope': float(row['margin_trend_slope'])
                }
                for _, row in high_momentum_teams.iterrows()
            ],
            'fading_teams': [
                {
                    'team': row['team'],
                    'season': int(row['season']),
                    'form_rating': float(row['form_rating']),
                    'win_streak': int(row['win_streak']),
                    'trend_slope': float(row['margin_trend_slope'])
                }
                for _, row in fading_teams.iterrows()
            ]
        }
        
        output_file = Path('results/momentum_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✓ Saved momentum analysis to {output_file}")
    
    print("\n=== Key Insights ===\n")
    print("Added 6 new momentum features to each team:")
    print("  1. weighted_last10_margin - Exponentially weighted margin (recent games weighted more)")
    print("  2. win_streak - Current win/loss streak")
    print("  3. margin_trend_slope - Trend in performance (positive = improving)")
    print("  4. last5_momentum - Simple average of last 5 game margins")
    print("  5. last10_momentum - Simple average of last 10 game margins")
    print("  6. form_rating - Composite momentum score")
    print("\nThese features capture late-season form and identify peaking vs. fading teams")

if __name__ == '__main__':
    main()
