#!/usr/bin/env python3
"""Display championship and round-by-round probabilities from a simulation file.

Usage:
    python show_results.py
    python show_results.py results/sim_2026_5000.json
"""
import json
import sys
from pathlib import Path


ROUNDS = [
    ('round_of_32',  'R32  '),
    ('sweet_16',     'S16  '),
    ('elite_8',      'E8   '),
    ('final_4',      'F4   '),
    ('title_game',   'Final'),
    ('champion',     'Champ'),
]


def find_latest_sim():
    candidates = sorted(Path('results').glob('sim_*.json'), key=lambda p: p.stat().st_mtime)
    candidates = [c for c in candidates if 'smoke' not in c.name]
    return str(candidates[-1]) if candidates else None


def load_training_summary():
    path = Path('models') / 'training_summary.json'
    if path.exists():
        return json.loads(path.read_text())
    return {}


sim_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_sim()
if not sim_path:
    print("No simulation output found. Run: python scripts/run_pipeline.py --mode simulate")
    sys.exit(1)

d = json.load(open(sim_path))
season = d.get('season', '?')
sims = d.get('sims', '?')
bracket_source = d.get('bracket_source', 'unknown')
feature_count = d.get('model_metadata', {}).get('feature_count', '?')
round_probs = d.get('round_probs', {})

print("\n" + "="*75)
print(f"  NCAA {season} TOURNAMENT PROJECTIONS".center(75))
print("="*75)
print(f"\nFile: {sim_path}")
print(f"Sims: {sims:,} | Teams: {len(d.get('teams', []))} | Bracket: {bracket_source}")
if bracket_source == 'generated_top64':
    print("  ⚠  Efficiency-based selection — not official bracket")
    print("     Re-run after Selection Sunday for real bracket odds")
print()

# ── Champion odds ──────────────────────────────────────────────────────────
if 'champion_probs' in d:
    champs = sorted(d['champion_probs'], key=lambda x: x[1], reverse=True)
    print("  TOP 15 CHAMPIONSHIP FAVORITES:")
    print("  " + "-"*71)
    for i, (team, prob) in enumerate(champs[:15], 1):
        bar = "█" * int(prob * 40)
        pct = prob * 100
        print(f"  {i:2d}. {team:38s} {pct:5.1f}% {bar}")
    print("  " + "-"*71)

# ── Path to championship (top 8 by Final Four odds) ────────────────────────
if round_probs:
    by_f4 = sorted(round_probs.items(), key=lambda kv: kv[1].get('final_4', 0), reverse=True)
    top8 = by_f4[:8]
    hdr = "  {:<30s}  {}".format("", "  ".join(r[1] for r in ROUNDS))
    print(f"\n  PATH TO CHAMPIONSHIP (top 8 by Final Four odds):")
    print("  " + "-"*71)
    print(hdr)
    for team, rp in top8:
        row = f"  {team:<30s}"
        for key, _ in ROUNDS:
            p = rp.get(key, 0)
            row += f"  {p:>4.0%} "
        print(row)
    print("  " + "-"*71)

# ── Model metadata ─────────────────────────────────────────────────────────
summary = load_training_summary()
meta = d.get('model_metadata', {})
ew = meta.get('ensemble_weights', {})
lr_w = ew.get('lr_weight', 0)
xgb_w = ew.get('xgb_weight', 1)

print(f"\n  MODEL:")
print(f"    Ensemble: LR ({lr_w:.0%}) + XGBoost ({xgb_w:.0%})  |  Features: {feature_count}")
print(f"    Calibration: LR=sigmoid, XGBoost=isotonic")

# Dynamic accuracy from training_summary.json
rcv = summary.get('rolling_cv_overall') or {}
loso = summary.get('loso_overall') or {}
baselines = summary.get('baselines', {})
seed_bl = baselines.get('lower_seed')

if rcv.get('accuracy'):
    acc = rcv['accuracy']
    ci = rcv.get('accuracy_ci_95', [acc - 0.04, acc + 0.04])
    # Derive seasons range from training summary
    all_seasons = summary.get('seasons', [])
    loso_seasons = [s['season'] for s in summary.get('loso_per_season', [])]
    if loso_seasons:
        seasons_range = f"{min(loso_seasons)}–{max(loso_seasons)}"
    elif all_seasons:
        seasons_range = f"{min(all_seasons)}–{max(all_seasons)}"
    else:
        seasons_range = '2015–2025'
    print(f"\n  VALIDATION:")
    print(f"    Rolling CV accuracy ({seasons_range}): {acc:.1%}  95% CI [{ci[0]:.1%}, {ci[1]:.1%}]")
    if loso.get('accuracy'):
        la = loso['accuracy']
        lci = loso.get('accuracy_ci_95', [la - 0.04, la + 0.04])
        print(f"    LOSO accuracy:                    {la:.1%}  95% CI [{lci[0]:.1%}, {lci[1]:.1%}]")
    if seed_bl:
        delta = acc - seed_bl
        sign = "+" if delta >= 0 else ""
        print(f"    Lower-seed baseline:              {seed_bl:.1%}  (model {sign}{delta:+.1%})")
elif loso.get('accuracy'):
    la = loso['accuracy']
    lci = loso.get('accuracy_ci_95', [la - 0.04, la + 0.04])
    print(f"\n  VALIDATION:")
    print(f"    LOSO accuracy: {la:.1%}  95% CI [{lci[0]:.1%}, {lci[1]:.1%}]")
    if seed_bl:
        print(f"    Lower-seed baseline: {seed_bl:.1%}")

print(f"\n  To regenerate charts:")
print(f"    python scripts/generate_charts.py --sim {sim_path}")
print("\n" + "="*75 + "\n")
