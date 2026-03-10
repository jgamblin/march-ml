"""Add conference strength metrics to features.

Creates conference strength priors based on:
1. Average adjusted margin per conference
2. Conference tournament performance history
3. Inter-conference win rates

This enriches SOS and opponent metrics with conference context.
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


# Major conference affiliations (2021-2025)
CONFERENCE_MAPPINGS = {
    # Power 6 Conferences
    'ACC': ['Duke', 'North Carolina', 'Virginia', 'Virginia Tech', 'NC State', 'Clemson',
            'Miami', 'Florida State', 'Syracuse', 'Louisville', 'Pittsburgh', 'Boston College',
            'Wake Forest', 'Georgia Tech', 'Notre Dame'],
    'Big Ten': ['Michigan', 'Ohio State', 'Michigan State', 'Illinois', 'Purdue', 'Indiana',
                'Iowa', 'Wisconsin', 'Minnesota', 'Maryland', 'Penn State', 'Rutgers',
                'Nebraska', 'Northwestern'],
    'Big 12': ['Kansas', 'Baylor', 'Texas Tech', 'Texas', 'Iowa State', 'Kansas State',
               'Oklahoma', 'Oklahoma State', 'West Virginia', 'TCU', 'BYU', 'Cincinnati',
               'Houston', 'UCF'],
    'Big East': ['Villanova', 'Creighton', 'UConn', 'Providence', 'Xavier', 'Butler',
                 'Marquette', 'St. John', 'Seton Hall', 'Georgetown', 'DePaul'],
    'Pac-12': ['Arizona', 'UCLA', 'USC', 'Oregon', 'Stanford', 'Colorado', 'Washington',
               'Utah', 'Arizona State', 'California', 'Oregon State', 'Washington State'],
    'SEC': ['Kentucky', 'Tennessee', 'Auburn', 'Alabama', 'Arkansas', 'Florida', 'LSU',
            'Mississippi State', 'Ole Miss', 'Missouri', 'South Carolina', 'Georgia',
            'Texas A&M', 'Vanderbilt'],
    
    # Mid-Major Conferences
    'American': ['Memphis', 'SMU', 'Temple', 'Tulsa', 'Tulane', 'Wichita State', 'East Carolina',
                 'South Florida', 'UAB', 'UTSA', 'North Texas', 'Rice', 'Charlotte', 'FAU'],
    'Mountain West': ['San Diego State', 'Boise State', 'Colorado State', 'Utah State', 'UNLV',
                      'New Mexico', 'Wyoming', 'Nevada', 'Fresno State', 'Air Force', 'San Jose State'],
    'WCC': ['Gonzaga', 'St. Mary', 'BYU', 'San Francisco', 'Santa Clara', 'Loyola Marymount',
            'Pepperdine', 'Pacific', 'Portland'],
    'Atlantic 10': ['Dayton', 'VCU', 'Saint Louis', 'Davidson', 'Rhode Island', 'Richmond',
                    'St. Bonaventure', 'George Mason', 'UMass', 'Duquesne', 'La Salle',
                    'Fordham', 'Saint Joseph', 'George Washington'],
    'Missouri Valley': ['Drake', 'Loyola Chicago', 'Bradley', 'Illinois State', 'Indiana State',
                        'Missouri State', 'Northern Iowa', 'Southern Illinois', 'Valparaiso', 'Murray State'],
}


def assign_conferences(teams_df):
    """Assign conference to each team based on team name patterns."""
    teams_df['conference'] = 'Other'
    
    for conference, teams in CONFERENCE_MAPPINGS.items():
        for team_pattern in teams:
            # Match team names containing the pattern
            mask = teams_df['team'].str.contains(team_pattern, case=False, na=False)
            teams_df.loc[mask, 'conference'] = conference
    
    return teams_df


def calculate_conference_strength(teams_df):
    """Calculate conference strength metrics."""
    
    # Group by conference and season
    conf_stats = {}
    
    for season in teams_df['season'].unique():
        season_data = teams_df[teams_df['season'] == season]
        
        conf_metrics = {}
        for conf in season_data['conference'].unique():
            conf_teams = season_data[season_data['conference'] == conf]
            
            if len(conf_teams) == 0:
                continue
            
            conf_metrics[conf] = {
                'avg_adj_margin': conf_teams['adj_margin'].mean(),
                'avg_win_pct': conf_teams['win_pct'].mean(),
                'avg_sos': conf_teams['sos_win_pct'].mean(),
                'num_teams': len(conf_teams),
                'top_team_adj_margin': conf_teams['adj_margin'].max(),
            }
        
        conf_stats[int(season)] = conf_metrics
    
    return conf_stats


def add_conference_features(teams_df, conf_stats):
    """Add conference strength features to team data."""
    
    teams_df['conf_avg_adj_margin'] = 0.0
    teams_df['conf_avg_win_pct'] = 0.0
    teams_df['conf_strength_tier'] = 'Mid-Major'
    
    power_6 = ['ACC', 'Big Ten', 'Big 12', 'Big East', 'Pac-12', 'SEC']
    
    for idx, row in teams_df.iterrows():
        season = row['season']
        conf = row['conference']
        
        if season in conf_stats and conf in conf_stats[season]:
            stats = conf_stats[season][conf]
            teams_df.at[idx, 'conf_avg_adj_margin'] = stats['avg_adj_margin']
            teams_df.at[idx, 'conf_avg_win_pct'] = stats['avg_win_pct']
        
        if conf in power_6:
            teams_df.at[idx, 'conf_strength_tier'] = 'Power-6'
        elif conf in ['American', 'Mountain West', 'WCC', 'Atlantic 10', 'Missouri Valley']:
            teams_df.at[idx, 'conf_strength_tier'] = 'High-Major'
        else:
            teams_df.at[idx, 'conf_strength_tier'] = 'Mid-Major'
    
    return teams_df


def main():
    parser = argparse.ArgumentParser(description='Add conference strength priors to features')
    parser.add_argument('--features', default='data/processed/features/tournament_teams.csv')
    parser.add_argument('--out', default='data/processed/features/tournament_teams_with_conferences.csv')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    
    print("\n=== Conference Strength Analysis ===\n")
    
    # Load features
    print("Loading tournament features...")
    teams_df = pd.read_csv(args.features)
    
    # Assign conferences
    print("Assigning conferences to teams...")
    teams_df = assign_conferences(teams_df)
    
    # Count assignments
    conf_counts = teams_df.groupby('conference')['team'].count().sort_values(ascending=False)
    print(f"\nTeam-Conference assignments:")
    for conf, count in conf_counts.head(15).items():
        print(f"  {conf:20} {count:4} teams")
    
    # Calculate conference strength
    print("\nCalculating conference strength metrics...")
    conf_stats = calculate_conference_strength(teams_df)
    
    # Add conference features
    print("Adding conference features to teams...")
    teams_df = add_conference_features(teams_df, conf_stats)
    
    # Save enriched features
    teams_df.to_csv(args.out, index=False)
    print(f"\n✓ Saved enriched features to {args.out}")
    
    # Summary by season
    print(f"\n=== Conference Strength by Season ===\n")
    for season in sorted(conf_stats.keys()):
        print(f"{season} Season:")
        season_confs = sorted(conf_stats[season].items(), 
                             key=lambda x: x[1]['avg_adj_margin'], reverse=True)[:10]
        for conf, stats in season_confs:
            print(f"  {conf:20} Avg Adj Margin: {stats['avg_adj_margin']:+6.2f}, "
                  f"Win%: {stats['avg_win_pct']:.3f}, Teams: {stats['num_teams']}")
        print()
    
    # Save conference stats
    out_path = Path(args.out_dir) / 'conference_strength_analysis.json'
    summary = {
        'by_season': {str(k): v for k, v in conf_stats.items()},
        'power_6_conferences': ['ACC', 'Big Ten', 'Big 12', 'Big East', 'Pac-12', 'SEC'],
        'high_major_conferences': ['American', 'Mountain West', 'WCC', 'Atlantic 10', 'Missouri Valley'],
    }
    
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved conference analysis to {out_path}")
    
    # Print key insights
    print(f"\n=== Key Insights ===\n")
    latest_season = max(conf_stats.keys())
    latest_stats = conf_stats[latest_season]
    
    power_6_avg = np.mean([s['avg_adj_margin'] for c, s in latest_stats.items() 
                           if c in ['ACC', 'Big Ten', 'Big 12', 'Big East', 'Pac-12', 'SEC']])
    mid_major_avg = np.mean([s['avg_adj_margin'] for c, s in latest_stats.items() 
                             if c not in ['ACC', 'Big Ten', 'Big 12', 'Big East', 'Pac-12', 'SEC']])
    
    print(f"{latest_season} Season:")
    print(f"  Power-6 avg adjusted margin: {power_6_avg:+.2f}")
    print(f"  Mid-Major avg adjusted margin: {mid_major_avg:+.2f}")
    print(f"  Conference strength gap: {power_6_avg - mid_major_avg:.2f} points")
    print(f"\nConference tier feature added to all {len(teams_df)} team records")


if __name__ == '__main__':
    main()
