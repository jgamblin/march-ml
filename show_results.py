#!/usr/bin/env python3
"""Display top championship probabilities from a simulation output file.

Usage:
    python show_results.py
    python show_results.py results/sim_2026_5000.json
"""
import json
import sys
from pathlib import Path

def find_latest_sim():
    candidates = sorted(Path('results').glob('sim_*.json'), key=lambda p: p.stat().st_mtime)
    candidates = [c for c in candidates if 'smoke' not in c.name]
    return str(candidates[-1]) if candidates else None

sim_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_sim()
if not sim_path:
    print("No simulation output found. Run: python scripts/run_pipeline.py --mode simulate")
    sys.exit(1)

d = json.load(open(sim_path))
season = d.get('season', '?')
sims = d.get('sims', '?')
bracket_source = d.get('bracket_source', 'unknown')
feature_count = d.get('model_metadata', {}).get('feature_count', '?')

print("\n" + "="*75)
print(f"  NCAA {season} TOURNAMENT PROJECTIONS".center(75))
print("="*75)
print(f"\nFile: {sim_path}")
print(f"Sims: {sims:,} | Teams: {len(d.get('teams', []))} | Bracket: {bracket_source}")
if bracket_source == 'generated_top64':
    print("  ⚠  adj_margin selection — not official bracket (re-run after Selection Sunday)")
print()

if 'champion_probs' in d:
    champs = sorted(d['champion_probs'], key=lambda x: x[1], reverse=True)
    print("  TOP 15 CHAMPIONSHIP FAVORITES:")
    print("  " + "-"*71)
    for i, (team, prob) in enumerate(champs[:15], 1):
        bar = "█" * int(prob * 40)
        pct = prob * 100
        print(f"  {i:2d}. {team:38s} {pct:5.1f}% {bar}")
    print("  " + "-"*71)

meta = d.get('model_metadata', {})
print(f"\n  MODEL:")
print(f"    Ensemble: LR ({meta.get('ensemble_weights',{}).get('lr_weight',0.65):.0%}) + XGBoost ({meta.get('ensemble_weights',{}).get('xgb_weight',0.35):.0%})")
print(f"    Features: {feature_count}")
print(f"    Calibration: LR=sigmoid, XGBoost=isotonic")
print(f"\n  VALIDATION (LOSO 2021-2025):")
print(f"    Accuracy: 64.7%  95% CI [59.3%, 70.1%]")
print(f"    Lower-seed baseline: 70.4%")
print(f"\n  To regenerate charts:")
print(f"    python scripts/generate_charts.py --sim {sim_path}")
print("\n" + "="*75 + "\n")
