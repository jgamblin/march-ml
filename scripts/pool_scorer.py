"""Pool scoring engine for bracket evaluation and strategy-aware entry generation.

Supports multiple scoring profiles:
  - ESPN: 10 (win), 40 (E8), 80 (F4), 140 (Final)
  - CBS: 10 (win), 40 (E8), 80 (F4), 170 (Final)
  - Custom: user-defined point values per round

Integrates with simulation output to compute expected points and percentiles
for bracket optimization.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# Standard pool scoring profiles
SCORING_PROFILES = {
    'espn': {
        'round_of_64': 10,
        'round_of_32': 10,
        'sweet_16': 40,
        'elite_8': 40,
        'final_4': 80,
        'title_game': 140,
        'champion': 140,
    },
    'cbs': {
        'round_of_64': 10,
        'round_of_32': 10,
        'sweet_16': 40,
        'elite_8': 40,
        'final_4': 80,
        'title_game': 160,
        'champion': 170,
    },
    'simple': {
        'round_of_64': 1,
        'round_of_32': 2,
        'sweet_16': 4,
        'elite_8': 8,
        'final_4': 16,
        'title_game': 32,
        'champion': 32,
    },
}


def parse_scoring_profile(profile_name: str, custom_profile: Optional[Dict] = None) -> Dict[str, int]:
    """
    Get scoring profile by name or return custom profile.
    
    Args:
        profile_name: One of 'espn', 'cbs', 'simple', or 'custom'
        custom_profile: Custom scoring dict if profile_name is 'custom'
    
    Returns:
        Dict mapping round labels to point values
    """
    if profile_name == 'custom':
        if custom_profile is None:
            raise ValueError("custom_profile required when profile_name='custom'")
        return custom_profile
    
    if profile_name not in SCORING_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Options: {list(SCORING_PROFILES.keys())}")
    
    return SCORING_PROFILES[profile_name]


def score_bracket(
    bracket_teams: List[str],
    round_probs: Dict[str, Dict[str, float]],
    profile: Dict[str, int]
) -> float:
    """
    Compute expected score for a bracket given simulation probabilities.
    
    For each team in the bracket, multiply the probability of reaching each round
    by the points for that round, then sum all expected points.
    
    Args:
        bracket_teams: List of team names in bracket
        round_probs: Dict from simulate_bracket output {team: {round: prob}}
        profile: Scoring profile {round_label: points}
    
    Returns:
        Expected points for this bracket
    """
    total_expected = 0.0
    
    for team in bracket_teams:
        if team not in round_probs:
            # Skip teams not in simulation (shouldn't happen)
            continue
        
        team_rounds = round_probs[team]
        for round_name, points in profile.items():
            prob = team_rounds.get(round_name, 0.0)
            total_expected += prob * points
    
    return total_expected


def generate_all_bracket_permutations_sample(
    teams: List[str],
    round_probs: Dict[str, Dict[str, float]],
    profile: Dict[str, int],
    num_samples: int = 10000
) -> List[Tuple[List[str], float]]:
    """
    Generate random bracket samples and score them.
    
    This is a placeholder for true optimizer. For now, sample different
    bracket orderings and score each one.
    
    Args:
        teams: Available teams to form brackets from
        round_probs: Simulation round reach probabilities
        profile: Scoring profile
        num_samples: Number of random brackets to generate and score
    
    Returns:
        List of (bracket_teams, expected_points) sorted by expected points (descending)
    """
    if len(teams) != 64:
        raise ValueError(f"Expected 64 teams, got {len(teams)}")
    
    candidates = []
    
    for _ in range(num_samples):
        # Randomly shuffle bracket ordering
        shuffled = np.random.permutation(teams).tolist()
        expected_pts = score_bracket(shuffled, round_probs, profile)
        candidates.append((shuffled, expected_pts))
    
    # Sort by expected points descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def rank_brackets_by_percentile(
    bracket: List[str],
    round_probs: Dict[str, Dict[str, float]],
    profile: Dict[str, int],
    baseline_score: float,
    all_scores: List[float]
) -> float:
    """
    Compute percentile rank for a bracket among all simulated outcomes.
    
    Args:
        bracket: Bracket teams
        round_probs: Simulation results
        profile: Scoring profile
        baseline_score: Score of the bracket itself
        all_scores: All sampled scores for percentile computation
    
    Returns:
        Percentile rank (0-100)
    """
    if not all_scores:
        return 50.0
    
    lower = sum(1 for s in all_scores if s < baseline_score)
    percentile = (lower / len(all_scores)) * 100
    return percentile


def score_simulation_output(
    sim_json_path: str,
    profile_name: str = 'espn',
    custom_profile: Optional[Dict] = None,
    num_recommendations: int = 10
) -> Dict[str, Any]:
    """
    Score all bracket possibilities from a simulation output.
    
    Args:
        sim_json_path: Path to simulation JSON output
        profile_name: Scoring profile name
        custom_profile: Custom scoring profile if needed
        num_recommendations: Top N candidates to return
    
    Returns:
        Dict with:
          - profile: scoring profile used
          - chalk_bracket: highest-EV bracket
          - chalk_score: expected points
          - candidates: top N brackets with scores and percentiles
    """
    with open(sim_json_path, 'r') as f:
        sim_data = json.load(f)
    
    teams = sim_data['teams']
    round_probs = sim_data['round_probs']
    
    profile = parse_scoring_profile(profile_name, custom_profile)
    
    # Compute chalk bracket (highest expected value)
    chalk_score = score_bracket(teams, round_probs, profile)
    
    # Generate candidate brackets by sampling
    candidates = generate_all_bracket_permutations_sample(
        teams, round_probs, profile, num_samples=min(10000, max(100, len(teams) * 10))
    )
    
    # Get top candidates
    top_candidates = candidates[:num_recommendations]
    
    # Collect all scores for percentile computation
    all_scores = [score for _, score in candidates]
    
    # Build output
    result = {
        'profile': profile_name,
        'profile_definition': profile,
        'chalk_bracket': teams,
        'chalk_expected_score': float(chalk_score),
        'chalk_percentile': rank_brackets_by_percentile(teams, round_probs, profile, chalk_score, all_scores),
        'candidates': []
    }
    
    for i, (bracket, score) in enumerate(top_candidates):
        percentile = rank_brackets_by_percentile(bracket, round_probs, profile, score, all_scores)
        result['candidates'].append({
            'rank': i + 1,
            'bracket': bracket,
            'expected_score': float(score),
            'percentile': float(percentile),
            'diversity_from_chalk': float(len(set(bracket) - set(teams)) / 64)  # Frame change
        })
    
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sim_out', required=True, help='Simulation output JSON')
    p.add_argument('--profile', default='espn', choices=list(SCORING_PROFILES.keys()))
    p.add_argument('--top_n', type=int, default=10)
    args = p.parse_args()
    
    result = score_simulation_output(args.sim_out, profile_name=args.profile, num_recommendations=args.top_n)
    print(f"Chalk bracket expected: {result['chalk_expected_score']:.1f} ({result['chalk_percentile']:.1f}th percentile)")
    print(f"Top {args.top_n} candidates:")
    for cand in result['candidates']:
        print(f"  #{cand['rank']}: {cand['expected_score']:.1f} pts ({cand['percentile']:.1f}th %ile)")
