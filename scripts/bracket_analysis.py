"""Comprehensive bracket analysis and insights from simulations.

Analyzes simulation results to provide:
- Champion probability distribution
- Round advancement statistics
- Upset potential (lower seeds advancing far)
- Bracket strength metrics

Outputs:
- bracket_analysis.json: detailed statistics and rankings
- champion_analysis.json: champion probability analysis
-advance_odds.csv: round advancement by team strength

Usage:
    python scripts/bracket_analysis.py --sim_out results/sim_5000_final.json --out_dir results
"""
import argparse
import json
import os
from pathlib import Path
from collections import Counter

import pandas as pd


def load_simulation(sim_path):
    """Load simulation output."""
    with open(sim_path) as f:
        return json.load(f)


def analyze_champion_probs(sim_data):
    """Analyze champion probability distribution."""
    champion_list = sim_data.get("champion_probs", [])
    
    # Convert to sorted list if needed
    if isinstance(champion_list, dict):
        champion_prob_sorted = sorted(champion_list.items(), key=lambda x: x[1], reverse=True)
    else:
        champion_prob_sorted = sorted(champion_list, key=lambda x: x[1], reverse=True)
    
    return {
        "total_teams": len(champion_prob_sorted),
        "top_10": champion_prob_sorted[:10],
        "top_25": champion_prob_sorted[:25],
        "probability_distribution": {
            ">10%": len([p for _, p in champion_prob_sorted if p >0.1]),
            "5-10%": len([p for _, p in champion_prob_sorted if 0.05 <= p <= 0.1]),
            "1-5%": len([p for _, p in champion_prob_sorted if 0.01 <= p < 0.05]),
            "<1%": len([p for _, p in champion_prob_sorted if p < 0.01]),
        },
        "concentration": {
            "top_1_pct_of_champ_prob": sum([p for _, p in champion_prob_sorted[:1]]),
            "top_5_pct_of_champ_prob": sum([p for _, p in champion_prob_sorted[:5]]),
            "top_10_pct_of_champ_prob": sum([p for _, p in champion_prob_sorted[:10]]),
        },
    }


def analyze_round_odds(sim_data):
    """Analyze round advancement probabilities."""
    round_probs = sim_data.get("round_probs", {})
    
    analysis = {}
    for round_name, team_odds in round_probs.items():
        team_odds_list = [(team, float(prob)) for team, prob in team_odds.items()]
        teams_in_round = len(team_odds_list)
        
        analysis[round_name] = {
            "expected_teams": teams_in_round,
            "top_5": sorted(team_odds_list, key=lambda x: x[1], reverse=True)[:5],
            "favorites": len([t for _, p in team_odds_list if p > 0.5]),
            "contenders": len([t for _, p in team_odds_list if 0.1 < p <= 0.5]),
            "dark_horse": len([t for _, p in team_odds_list if p <= 0.1]),
        }
    
    return analysis


def analyze_upset_potential(sim_data):
    """Identify potential for surprising results."""
    champion_list = sim_data.get("champion_probs", [])
    
    # Convert to dict
    if isinstance(champion_list, list):
        champion_probs = {team: prob for team, prob in champion_list}
    else:
        champion_probs = champion_list
    
    round_probs = sim_data.get("round_probs", {})
    
    # Find teams with low FF odds but high F4 odds
    final_4_odds = round_probs.get("Final Four", {})
    championship_odds = champion_probs
    
    potential_upsets = []
    for team in final_4_odds.keys():
        f4_prob = float(final_4_odds.get(team, 0))
        champ_prob = float(championship_odds.get(team, 0))
        
        # Team with decent F4 odds but low champ odds
        if f4_prob > 0.10 and champ_prob < 0.05:
            potential_upsets.append({
                "team": team,
                "final_4_odds": f4_prob,
                "champion_odds": champ_prob,
                "potential_ratio": f4_prob / (champ_prob + 0.001),
            })
    
    return sorted(potential_upsets, key=lambda x: x["potential_ratio"], reverse=True)[:10]


def analyze_bracket_balance(sim_data):
    """Analyze regional balance in predictions."""
    champion_list = sim_data.get("champion_probs", [])
    
    # Convert to dict
    if isinstance(champion_list, list):
        champion_probs = {team: prob for team, prob in champion_list}
    else:
        champion_probs = champion_list
    
    # Try to identify regional clustering in names
    regions = {"East": 0, "West": 0, "South": 0, "Midwest": 0}
    region_keywords = {
        "East": ["Carolina", "Florida", "Georgia", "Virginia", "Maryland", "Boston"],
        "West": ["Arizona", "UCLA", "USC", "Colorado", "Utah", "Oregon"],
        "South": ["Texas", "Duke", "Wake", "Clemson", "Miami", "Louisville"],
        "Midwest": ["Kansas", "Iowa", "Illinois", "Wisconsin", "Michigan", "Ohio"],
    }
    
    for team, prob in champion_probs.items():
        for region, keywords in region_keywords.items():
            if any(keyword in team for keyword in keywords):
                regions[region] += prob
                break
    
    return {
        "regional_champion_odds": regions,
        "balanced": sum([(regions[r] - 0.25)**2 for r in regions]) < 0.05,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_out", required=True, help="Simulation output JSON")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading simulation from {args.sim_out}...")
    sim_data = load_simulation(args.sim_out)
    
    print("Analyzing simulation results...")
    
    # Champion analysis
    champion_analysis = analyze_champion_probs(sim_data)
    
    # Round advancement analysis
    round_analysis = analyze_round_odds(sim_data)
    
    # Upset potential
    upset_teams = analyze_upset_potential(sim_data)
    
    # Bracket balance
    bracket_balance = analyze_bracket_balance(sim_data)
    
    # Build comprehensive analysis
    analysis = {
        "simulation_info": {
            "season": sim_data.get("season"),
            "simulations": sim_data.get("sims"),
            "bracket_source": sim_data.get("bracket_source"),
        },
        "champion_analysis": champion_analysis,
        "round_advancement": round_analysis,
        "upset_potential": upset_teams,
        "bracket_balance": bracket_balance,
    }
    
    # Export to JSON
    output_path = Path(args.out_dir) / "bracket_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved comprehensive analysis to {output_path}")
    
    # Print summary
    print("\n=== Bracket Analysis Summary ===")
    print(f"Season: {sim_data.get('season')}")
    print(f"Simulations: {sim_data.get('sims'):,}")
    
    print(f"\n=== Top 10 Champion Favorites ===")
    for i, (team, prob) in enumerate(champion_analysis["top_10"], 1):
        print(f"  {i:2d}. {team:30s} {prob:6.2%}")
    
    print(f"\n=== Concentration of Probability ===")
    for label, value in champion_analysis["concentration"].items():
        print(f"  {label}: {value:.1%}")
    
    print(f"\n=== Biggest Upset Potential ===")
    for team_data in upset_teams[:5]:
        print(f"  {team_data['team']:30s} F4: {team_data['final_4_odds']:5.1%}, "
              f"Champ: {team_data['champion_odds']:5.1%}")
    
    if bracket_balance["balanced"]:
        print(f"\n✓ Bracket appears balanced across regions")
    else:
        print(f"\n⚠️ Bracket skewed toward certain regions")
        for region, prob in bracket_balance["regional_champion_odds"].items():
            print(f"   {region}: {prob:.1%}")


if __name__ == "__main__":
    main()
