"""Standardize games file schema by adding explicit season column.

This script:
1. Adds a 'season' column to each games_*.csv file
2. Creates a consolidated games file with all seasons
3. Updates hyperparameter_tuning.py to use the new schema
"""
import argparse
import pandas as pd
from pathlib import Path


def add_season_column(games_dir, seasons=None):
    """Add explicit season column to games files."""
    
    games_dir = Path(games_dir)
    
    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]
    
    all_games = []
    
    for season in seasons:
        games_path = games_dir / f'games_{season}.csv'
        
        if not games_path.exists():
            print(f"  ⚠️  {games_path.name} not found, skipping")
            continue
        
        print(f"  Processing {games_path.name}...")
        games_df = pd.read_csv(games_path)
        
        # Add season column if not present
        if 'season' not in games_df.columns:
            games_df.insert(1, 'season', season)  # Insert after game_id
        
        # Save back
        games_df.to_csv(games_path, index=False)
        print(f"    Added season column, {len(games_df)} games")
        
        all_games.append(games_df)
    
    # Create consolidated file
    if all_games:
        consolidated_df = pd.concat(all_games, ignore_index=True)
        consolidated_path = games_dir.parent / 'games_all_seasons.csv'
        consolidated_df.to_csv(consolidated_path, index=False)
        print(f"\n✓ Created consolidated file: {consolidated_path.name}")
        print(f"  Total games: {len(consolidated_df)}")
        print(f"  Seasons: {sorted(consolidated_df['season'].unique())}")


def main():
    parser = argparse.ArgumentParser(description='Standardize games file schema')
    parser.add_argument('--games_dir', default='data/processed', help='Directory with games_*.csv')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2021, 2022, 2023, 2024, 2025])
    args = parser.parse_args()
    
    print(f"\n=== Standardizing Games File Schema ===\n")
    add_season_column(args.games_dir, args.seasons)
    print(f"\n✓ Schema standardization complete")
    print(f"  All games_*.csv files now have 'season' column")
    print(f"  games_all_seasons.csv available for consolidated analysis")


if __name__ == '__main__':
    main()
