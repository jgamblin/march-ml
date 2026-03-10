"""Compare optimizer strategies across chalk, balanced, and contrarian approaches."""
import json
import argparse
from pathlib import Path


def compare_strategies(chalk_file, balanced_file, contrarian_file, out_dir='results'):
    """Load and compare all 3 optimizer strategies."""
    
    # Load strategy files
    with open(chalk_file) as f:
        chalk = json.load(f)
    with open(balanced_file) as f:
        balanced = json.load(f)
    with open(contrarian_file) as f:
        contrarian = json.load(f)
    
    # Extract bracket data
    chalk_entry = chalk['entries'][0] if chalk['entries'] else {}
    balanced_entry = balanced['entries'][0] if balanced['entries'] else {}
    contrarian_entry = contrarian['entries'][0] if contrarian['entries'] else {}
    
    chalk_teams = set(chalk_entry.get('bracket_teams', []))
    balanced_teams = set(balanced_entry.get('bracket_teams', []))
    contrarian_teams = set(contrarian_entry.get('bracket_teams', []))
    
    # Build comparison report
    report = {
        'simulation_info': chalk.get('season'),
        'strategies': {
            'chalk': {
                'type': 'Maximum Expected Value',
                'entries': len(chalk['entries']),
                'expected_score': chalk_entry.get('expected_score'),
                'percentile_rank': chalk_entry.get('percentile_rank'),
                'sample_picks': chalk_entry.get('bracket_teams', [])[:5] if chalk_entry else [],
            },
            'balanced': {
                'type': 'EV + Risk Balance',
                'entries': len(balanced['entries']),
                'expected_score': balanced_entry.get('expected_score'),
                'percentile_rank': balanced_entry.get('percentile_rank'),
                'sample_picks': balanced_entry.get('bracket_teams', [])[:5] if balanced_entry else [],
            },
            'contrarian': {
                'type': '70th+ Percentile (Low Pick %)',
                'entries': len(contrarian['entries']),
                'expected_score': contrarian_entry.get('expected_score'),
                'percentile_rank': contrarian_entry.get('percentile_rank'),
                'sample_picks': contrarian_entry.get('bracket_teams', [])[:5] if contrarian_entry else [],
            },
        },
        'strategy_overlap': {
            'chalk_vs_balanced': {
                'overlap_count': len(chalk_teams & balanced_teams),
                'overlap_pct': round(len(chalk_teams & balanced_teams) / 64 * 100, 1) if chalk_teams else 0,
            },
            'chalk_vs_contrarian': {
                'overlap_count': len(chalk_teams & contrarian_teams),
                'overlap_pct': round(len(chalk_teams & contrarian_teams) / 64 * 100, 1) if chalk_teams else 0,
            },
            'balanced_vs_contrarian': {
                'overlap_count': len(balanced_teams & contrarian_teams),
                'overlap_pct': round(len(balanced_teams & contrarian_teams) / 64 * 100, 1) if balanced_teams else 0,
            },
        },
    }
    
    # Summary
    print("\n=== Optimizer Strategy Comparison ===\n")
    
    for strategy, data in report['strategies'].items():
        print(f"{strategy.upper()} ({data['type']}):")
        print(f"  Entries generated: {data['entries']}")
        print(f"  Expected Score: {data['expected_score']:.1f}" if data['expected_score'] else "  Expected Score: N/A")
        print(f"  Percentile: {data['percentile_rank']:.1f}%" if data['percentile_rank'] else "  Percentile: N/A")
        print(f"  First 5 picks: {', '.join(data['sample_picks'])}")
        print()
    
    # Overlap summary
    print("\n=== Bracket Diversification ===")
    for pair, overlap in report['strategy_overlap'].items():
        strat_a, strat_b = pair.split('_vs_')
        print(f"{strat_a.upper()} vs {strat_b.upper()}: {overlap['overlap_count']}/64 "
              f"teams overlap ({overlap['overlap_pct']}%)")
    
    # Diversity assessment
    print("\n=== Diversity Assessment ===")
    avg_overlap = sum(o['overlap_pct'] for o in report['strategy_overlap'].values()) / 3
    if avg_overlap > 60:
        print(f"⚠️  Strategies are highly correlated (avg {avg_overlap:.1f}% overlap)")
        print("    Consider different scoring profiles or risk parameters")
    elif avg_overlap > 40:
        print(f"✓  Moderate diversification (avg {avg_overlap:.1f}% overlap)")
        print("    Good balance between coherent picks and varied approaches")
    else:
        print(f"✓  High diversification (avg {avg_overlap:.1f}% overlap)")
        print("    Strategies differ significantly - strong portfolio hedge")
    
    # Save report
    out_path = Path(out_dir) / 'optimizer_strategy_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to {out_path}")
    
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare optimizer strategies')
    parser.add_argument('--chalk', default='results/optimizer_chalk_5000.json')
    parser.add_argument('--balanced', default='results/optimizer_balanced_5000.json')
    parser.add_argument('--contrarian', default='results/optimizer_contrarian_5000.json')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    
    compare_strategies(args.chalk, args.balanced, args.contrarian, args.out_dir)
