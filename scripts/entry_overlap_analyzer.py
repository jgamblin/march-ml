"""Analyze overlap in generated bracket entries.

Outputs:
- entry_overlap_matrix.csv: overlap % between each pair of entries
- entry_overlap_summary.json: diversity metrics and recommendations

Usage:
    python scripts/entry_overlap_analyzer.py --opt_out results/optimizer_output.json --out_dir results
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def load_optimizer_output(opt_path):
    """Load optimizer output file."""
    with open(opt_path, "r") as f:
        data = json.load(f)
    return data


def extract_bracket_teams(bracket_data):
    """Extract list of teams from bracket data."""
    if isinstance(bracket_data, dict) and "bracket_teams" in bracket_data:
        return set(bracket_data["bracket_teams"])
    elif isinstance(bracket_data, list):
        return set(bracket_data)
    return set()


def calculate_overlap(teams1, teams2):
    """Calculate overlap percentage between two bracket team lists."""
    if len(teams1) == 0 or len(teams2) == 0:
        return 0.0
    intersection = len(teams1 & teams2)
    max_teams = max(len(teams1), len(teams2))
    return intersection / max_teams if max_teams > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opt_out", required=True, help="Optimizer output JSON file")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading optimizer output from {args.opt_out}...")
    opt_data = load_optimizer_output(args.opt_out)
    
    entries = opt_data.get("entries", [])
    if not entries:
        print("No entries found in optimizer output")
        return
    
    print(f"Analyzing {len(entries)} bracket entries...")
    
    # Extract bracket teams for each entry
    bracket_teams = []
    entry_info = []
    
    for i, entry in enumerate(entries):
        teams = extract_bracket_teams(entry)
        bracket_teams.append(teams)
        entry_info.append({
            "entry_number": entry.get("entry_number", i + 1),
            "strategy": entry.get("strategy", "unknown"),
            "expected_score": entry.get("expected_score", 0),
            "percentile_rank": entry.get("percentile_rank", 0),
            "team_count": len(teams),
        })
    
    # Build overlap matrix
    overlap_matrix = []
    for i, teams1 in enumerate(bracket_teams):
        row = {"entry": i + 1}
        for j, teams2 in enumerate(bracket_teams):
            if i <= j:
                overlap = calculate_overlap(teams1, teams2)
                row[f"entry_{j+1}"] = overlap
        overlap_matrix.append(row)
    
    # Export overlap matrix
    overlap_df = pd.DataFrame(overlap_matrix)
    matrix_path = Path(args.out_dir) / "entry_overlap_matrix.csv"
    overlap_df.to_csv(matrix_path, index=False)
    print(f"Saved overlap matrix to {matrix_path}")
    
    # Calculate summary statistics
    overlaps = []
    for i, teams1 in enumerate(bracket_teams):
        for j, teams2 in enumerate(bracket_teams):
            if i < j:
                overlap = calculate_overlap(teams1, teams2)
                overlaps.append(overlap)
    
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    min_overlap = min(overlaps) if overlaps else 0
    max_overlap = max(overlaps) if overlaps else 0
    
    # Calculate correlation between entries
    entry_differences = []
    for i, teams1 in enumerate(bracket_teams):
        for j, teams2 in enumerate(bracket_teams):
            if i < j:
                diff = 1 - calculate_overlap(teams1, teams2)
                entry_differences.append({
                    "entry_pair": f"{i+1}-{j+1}",
                    "overlap": 1 - diff,
                    "difference": diff,
                })
    
    difference_df = pd.DataFrame(entry_differences)
    difference_df = difference_df.sort_values("overlap")
    
    # Identify most unique entries
    uniqueness_scores = {}
    for i, teams1 in enumerate(bracket_teams):
        total_overlap = 0
        for j, teams2 in enumerate(bracket_teams):
            if i != j:
                total_overlap += calculate_overlap(teams1, teams2)
        uniqueness_scores[i] = 1 - (total_overlap / (len(bracket_teams) - 1) if len(bracket_teams) > 1 else 0)
    
    # Build JSON summary
    summary = {
        "entries_count": len(entries),
        "strategy": opt_data.get("strategy", "unknown"),
        "profile": opt_data.get("profile", "unknown"),
        "overlap_statistics": {
            "average_overlap": round(avg_overlap, 3),
            "minimum_overlap": round(min_overlap, 3),
            "maximum_overlap": round(max_overlap, 3),
            "recommended_pairs_for_diversity": difference_df.head(5)[["entry_pair", "overlap"]].to_dict("records"),
        },
        "entry_diversity_scores": {
            f"entry_{i+1}": round(uniqueness_scores[i], 3)
            for i in range(len(bracket_teams))
        },
        "most_unique_entry": f"entry_{max(uniqueness_scores, key=uniqueness_scores.get) + 1}",
        "uniqueness_recommendation": (
            "Portfolio has good diversity" if avg_overlap < 0.7
            else "Consider mixing strategies or adjusting optimizer parameters"
        ),
        "entry_details": entry_info,
    }
    
    summary_path = Path(args.out_dir) / "entry_overlap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved overlap summary to {summary_path}")
    
    # Print summary
    print(f"\n=== Portfolio Diversity Analysis ===")
    print(f"Number of entries: {len(entries)}")
    print(f"Average overlap between entries: {avg_overlap:.1%}")
    print(f"Range: {min_overlap:.1%} to {max_overlap:.1%}")
    print(f"\nMost unique entry: Entry {max(uniqueness_scores, key=uniqueness_scores.get) + 1} "
          f"(diverges {uniqueness_scores[max(uniqueness_scores, key=uniqueness_scores.get)]:.1%} from others)")
    
    print(f"\nLeast overlapping pairs (most diverse):")
    for idx, row in difference_df.head(5).iterrows():
        print(f"  {row['entry_pair']}: {row['overlap']:.1%} overlap ({row['difference']:.1%} different)")
    
    if avg_overlap > 0.8:
        print(f"\n⚠️  WARNING: High portfolio overlap ({avg_overlap:.1%}). Consider:")
        print(f"   - Using different strategies (chalk vs balanced vs contrarian)")
        print(f"   - Adjusting number of entries")
        print(f"   - Manually tweaking contrarian brackets")
    else:
        print(f"\n✓ Portfolio diversity is good.")


if __name__ == "__main__":
    main()
