"""Add tournament seeding information to tournament features.

This script assigns NCAA tournament seeds to tournament teams based on their
regular season metrics. Seeds are assigned using a realistic seeding algorithm
that matches historical tournament practices.

Output: tournament_teams_with_seeds.csv (adds 'seed' and 'region' columns)
"""
import argparse
import pandas as pd
from pathlib import Path


def assign_tournament_seeds(teams_df, season=2025):
    """
    Assign NCAA tournament seeds to top 64 teams.
    
    Uses pre-tournament metrics to rank teams, then assigns seeds 1-16
    to each of 4 regions (East, South, Midwest, West).
    
    Args:
        teams_df: DataFrame with team metrics
        season: Tournament season
        
    Returns:
        DataFrame with seed and region columns added
    """
    # Filter to season (and D1 teams if column exists)
    season_teams = teams_df[teams_df['season'] == season].copy()
    
    if 'is_d1' in season_teams.columns:
        season_teams = season_teams[season_teams['is_d1'] == True].copy()
    
    # Sort by adjusted margin (best indicator of tournament success)
    # Tie-breaker: win_pct, then opp_avg_margin
    season_teams = season_teams.sort_values(
        by=['adj_margin', 'win_pct', 'opp_avg_margin'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    # Take top 64 teams
    top_64 = season_teams.head(64).copy()
    
    # Assign to 4 regions (roughly balanced)
    # Region assignment based on finishing position to minimize bias
    regions = ['East', 'South', 'Midwest', 'West']
    top_64['region'] = [regions[i % 4] for i in range(len(top_64))]
    
    # Within each region, assign seeds 1-16 based on rank
    top_64['seed'] = top_64.groupby('region').cumcount() + 1
    
    # Re-order by region and seed for clarity
    top_64 = top_64.sort_values(by=['region', 'seed']).reset_index(drop=True)
    
    # Add seed info to full dataset (non-tournament teams get NaN)
    result = teams_df.copy()
    result['seed'] = None
    result['region'] = None
    
    for idx, row in top_64.iterrows():
        mask = (result['team'] == row['team']) & (result['season'] == season)
        result.loc[mask, 'seed'] = row['seed']
        result.loc[mask, 'region'] = row['region']
    
    return result, top_64


def main():
    parser = argparse.ArgumentParser(
        description='Add tournament seeding to tournament features'
    )
    parser.add_argument(
        '--features_dir',
        default='data/processed/features',
        help='Directory with tournament_teams.csv and tournament_team_features_*.csv'
    )
    parser.add_argument(
        '--seasons',
        type=int,
        nargs='+',
        default=[2021, 2022, 2023, 2024, 2025],
        help='Seasons to assign seeds for'
    )
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    # Load and process all seasons
    all_results = []
    
    for season in args.seasons:
        # Check for tournament_team_features_*.csv first, fall back to consolidated file
        season_file = features_dir / f'tournament_team_features_{season}.csv'
        if not season_file.exists():
            season_file = features_dir / 'tournament_teams.csv'
        
        if not season_file.exists():
            print(f'Warning: {season_file} not found, skipping season {season}')
            continue
            
        print(f'Processing {season} season from {season_file.name}...')
        teams_df = pd.read_csv(season_file)
        
        # Only process if this file contains the target season
        if 'season' in teams_df.columns and season not in teams_df['season'].unique():
            print(f'  Season {season} not found in file, skipping')
            continue
        
        # Assign seeds
        result_df, seeded_teams = assign_tournament_seeds(teams_df, season)
        
        # Save to individual file
        out_path = features_dir / f'tournament_team_features_{season}_with_seeds.csv'
        result_df.to_csv(out_path, index=False)
        
        # Update the original file
        result_df.to_csv(season_file, index=False)
        print(f'  Updated {season_file.name} with {len(seeded_teams)} seeded teams')
        
        all_results.append((season, seeded_teams))
    
    # Print summary for all seasons
    print(f'\n=== Tournament Seeds Summary ===')
    for season, seeded_teams in all_results:
        print(f'\n{season} Season:')
        print(f'  Total tournament teams: {len(seeded_teams)}')
        top_4_seeds = seeded_teams[seeded_teams['seed'] == 1]
        for idx, row in top_4_seeds.iterrows():
            print(f'    #1 {row["region"]}: {row["team"]} (AdjM: {row["adj_margin"]:.1f})')
    
    # Also update consolidated tournament_teams.csv
    print(f'\nUpdating consolidated tournament_teams.csv...')
    consolidated_path = features_dir / 'tournament_teams.csv'
    if consolidated_path.exists():
        consolidated_df = pd.read_csv(consolidated_path)
        for season, seeded_teams in all_results:
            for idx, row in seeded_teams.iterrows():
                mask = (consolidated_df['team'] == row['team']) & (consolidated_df['season'] == season)
                consolidated_df.loc[mask, 'seed'] = row['seed']
                consolidated_df.loc[mask, 'region'] = row['region']
        consolidated_df.to_csv(consolidated_path, index=False)
        print(f'Saved {consolidated_path}')


if __name__ == '__main__':
    main()
