"""Bracket optimizer: generate strategy-aware entry portfolios.

Strategies:
  - chalk: maximize expected points (greedy)
  - balanced: high expected points with reduced volatility (target 30th+ percentile)
  - contrarian: high percentile outlier brackets (aim for 70th+ percentile)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np

from pool_scorer import score_bracket, parse_scoring_profile


StrategyType = Literal['chalk', 'balanced', 'contrarian']


def optimize_entries(
    sim_json_path: str,
    profile_name: str = 'espn',
    strategy: StrategyType = 'chalked',
    num_entries: int = 10,
    custom_profile: Optional[Dict] = None,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Generate optimized bracket entries under a strategy.
    
    Args:
        sim_json_path: Path to simulation JSON
        profile_name: Scoring profile (espn, cbs, simple, custom)
        strategy: One of 'chalk', 'balanced', 'contrarian'
        num_entries: Number of bracket entries to generate
        custom_profile: Custom scoring dict if profile_name='custom'
        random_seed: Optional seed for reproducibility
    
    Returns:
        Dict with strategy info and ranked bracket entries
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    with open(sim_json_path, 'r') as f:
        sim_data = json.load(f)
    
    teams = sim_data['teams']
    round_probs = sim_data['round_probs']
    
    profile = parse_scoring_profile(profile_name, custom_profile)
    
    # Generate candidate bracket pool
    num_candidates_to_sample = min(50000, max(1000, len(teams) * 100))
    candidates = []
    
    for _ in range(num_candidates_to_sample):
        shuffled = np.random.permutation(teams).tolist()
        expected = score_bracket(shuffled, round_probs, profile)
        candidates.append((shuffled, expected))
    
    # Compute percentiles for each candidate
    scores = np.array([score for _, score in candidates])
    candidates_with_percentiles = []
    for bracket, score in candidates:
        percentile = (np.sum(scores < score) / len(scores)) * 100  # percentage of scores below this one
        candidates_with_percentiles.append((bracket, score, percentile))
    
    # Filter by strategy
    if strategy == 'chalk':
        # Top expected value
        filtered = sorted(candidates_with_percentiles, key=lambda x: x[1], reverse=True)[:num_entries * 5]
    
    elif strategy == 'balanced':
        # High EV but reasonable percentile (30th+)
        filtered = [
            c for c in candidates_with_percentiles
            if c[2] >= 30  # percentile >= 30th
        ]
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:num_entries * 5]
    
    elif strategy == 'contrarian':
        # High percentile (70th+) even if lower EV
        filtered = [
            c for c in candidates_with_percentiles
            if c[2] >= 70  # percentile >= 70th
        ]
        filtered = sorted(filtered, key=lambda x: x[2], reverse=True)[:num_entries * 5]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Diversity selection: pick diverse brackets
    selected = []
    for bracket, score, percentile in filtered:
        if len(selected) >= num_entries:
            break
        
        # Check diversity with existing selections
        is_diverse = True
        for sel_bracket, _, _ in selected:
            overlap = len(set(bracket) & set(sel_bracket)) / 64.0
            if overlap > 0.95:  # Too similar
                is_diverse = False
                break
        
        if is_diverse:
            selected.append((bracket, score, percentile))
    
    # Format output
    result = {
        'strategy': strategy,
        'profile': profile_name,
        'profile_definition': profile,
        'season': sim_data.get('season'),
        'sims': sim_data.get('sims'),
        'bracket_source': sim_data.get('bracket_source'),
        'entries': []
    }
    
    for i, (bracket, score, percentile) in enumerate(selected):
        result['entries'].append({
            'entry_number': i + 1,
            'bracket_seeds': [
                # Find seed for each team if available
                next(
                    (str(rec.get('seed', '?')) for rec in sim_data.get('bracket', []) if rec.get('team') == team),
                    '?'
                )
                for team in bracket
            ],
            'bracket_teams': bracket,
            'expected_score': float(score),
            'percentile_rank': float(percentile),
        })
    
    return result


def generate_portfolio(
    sim_json_path: str,
    profile_name: str = 'espn',
    chalk_entries: int = 1,
    balanced_entries: int = 3,
    contrarian_entries: int = 1,
    custom_profile: Optional[Dict] = None,
) -> Dict:
    """
    Generate multi-strategy portfolio of bracket entries.
    
    Args:
        sim_json_path: Path to simulation JSON
        profile_name: Scoring profile
        chalk_entries: Number of chalk entries
        balanced_entries: Number of balanced entries
        contrarian_entries: Number of contrarian entries
        custom_profile: Custom scoring if needed
    
    Returns:
        Combined portfolio with all strategies
    """
    portfolio = {
        'strategies': {},
        'total_entries': chalk_entries + balanced_entries + contrarian_entries,
    }
    
    if chalk_entries > 0:
        portfolio['strategies']['chalk'] = optimize_entries(
            sim_json_path,
            profile_name=profile_name,
            strategy='chalk',
            num_entries=chalk_entries,
            custom_profile=custom_profile,
        )
    
    if balanced_entries > 0:
        portfolio['strategies']['balanced'] = optimize_entries(
            sim_json_path,
            profile_name=profile_name,
            strategy='balanced',
            num_entries=balanced_entries,
            custom_profile=custom_profile,
        )
    
    if contrarian_entries > 0:
        portfolio['strategies']['contrarian'] = optimize_entries(
            sim_json_path,
            profile_name=profile_name,
            strategy='contrarian',
            num_entries=contrarian_entries,
            custom_profile=custom_profile,
        )
    
    return portfolio


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sim_out', required=True, help='Simulation output JSON')
    p.add_argument('--profile', default='espn', help='Scoring profile')
    p.add_argument('--strategy', default='balanced', choices=['chalk', 'balanced', 'contrarian'])
    p.add_argument('--num_entries', type=int, default=10)
    p.add_argument('--out', default=None, help='Output JSON file (default: stdout)')
    args = p.parse_args()
    
    result = optimize_entries(
        args.sim_out,
        profile_name=args.profile,
        strategy=args.strategy,
        num_entries=args.num_entries,
    )
    
    output = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(output)
        print(f"Wrote {len(result['entries'])} entries to {args.out}")
    else:
        print(output)
