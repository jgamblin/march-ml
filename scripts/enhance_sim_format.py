"""Transform old simulation format to optimizer-compatible format.

Takes sim output with only champion_probs and adds teams/round_probs
needed by the optimizer.
"""
import json
import argparse
from pathlib import Path


def enhance_sim_format(sim_json_path, out_path=None):
    """Add teams and round_probs to simulation output."""
    with open(sim_json_path) as f:
        old_data = json.load(f)
    
    # Extract teams from champion_probs list
    champion_list = old_data.get('champion_probs', [])
    teams = [team for team, prob in champion_list]
    
    # Build round_probs structure from champion probs
    round_probs = {}
    for team in teams:
        # Get champion probability
        champ_prob = next((prob for t, prob in champion_list if t == team), 0.0)
        
        # Estimate round advancement based on champion probability
        # Higher champion prob = higher round advancement probs
        round_probs[team] = {
            'round_1': 1.0,
            'round_2': 0.5 + (champ_prob * 0.5),  # Strong favorites more likely to advance
            'round_3': 0.25 + (champ_prob * 0.4),
            'round_4': 0.125 + (champ_prob * 0.3),
            'round_5': 0.05 + (champ_prob * 0.2),
            'round_6': 0.02 + (champ_prob * 0.1),
            'finalist': 0.01 + (champ_prob * 0.05),
            'champion': champ_prob,
        }
    
    # Create enhanced format
    enhanced_data = {
        'season': old_data.get('season'),
        'sims': old_data.get('sims'),
        'teams': teams,
        'champion_probs': champion_list,
        'round_probs': round_probs,
        'bracket_source': old_data.get('bracket_source'),
        'schema_version': '2025-03-09',
    }
    
    # Determine output path
    if out_path is None:
        out_path = Path(sim_json_path).stem + '_optimized.json'
    
    # Save
    with open(out_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    return enhanced_data, out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhance simulation format for optimizer')
    parser.add_argument('--sim_in', default='results/sim_5000_final.json', help='Input simulation')
    parser.add_argument('--out', help='Output path')
    args = parser.parse_args()
    
    enhanced, out_path = enhance_sim_format(args.sim_in, args.out)
    print(f'✓ Enhanced {args.sim_in}')
    print(f'  Teams: {len(enhanced["teams"])}')
    print(f'  Simulations: {enhanced["sims"]}')
    print(f'  Saved to: {out_path}')
