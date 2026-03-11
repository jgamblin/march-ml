#!/usr/bin/env python3
"""Generate charts from training summary and simulation output.

Usage:
    python scripts/generate_charts.py
    python scripts/generate_charts.py --sim results/sim_2026_5000.json
    python scripts/generate_charts.py --out_dir results/charts
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CHART_STYLE = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',
    'grid.color': '#21262d',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
}

NCAA_BLUE = '#1a73e8'
NCAA_GOLD = '#f5a623'
BASELINE_COLOR = '#8b949e'
SUCCESS_GREEN = '#3fb950'
WARNING_RED = '#f85149'


def apply_style():
    plt.rcParams.update(CHART_STYLE)
    plt.rcParams['font.family'] = 'DejaVu Sans'


def load_training_summary(models_dir='models'):
    path = Path(models_dir) / 'training_summary.json'
    if not path.exists():
        raise FileNotFoundError(f"training_summary.json not found in {models_dir}")
    return json.loads(path.read_text())


def load_sim(sim_path):
    path = Path(sim_path)
    if not path.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")
    return json.loads(path.read_text())


def chart_loso_per_season(summary, out_dir):
    """Bar chart of LOSO accuracy per season with overall CI band."""
    seasons_data = summary.get('loso_per_season', [])
    if not seasons_data:
        print("  Skipping LOSO chart: no loso_per_season data")
        return

    seasons = [str(d['season']) for d in seasons_data]
    accs = [d['accuracy'] for d in seasons_data]
    overall = summary.get('loso_overall', {})
    overall_acc = overall.get('accuracy', np.mean(accs))
    ci_lo = overall.get('ci_low', overall_acc - 0.05)
    ci_hi = overall.get('ci_high', overall_acc + 0.05)
    baselines = summary.get('baselines', {})
    seed_baseline = baselines.get('lower_seed', None)

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [SUCCESS_GREEN if a >= overall_acc else NCAA_BLUE for a in accs]
    bars = ax.bar(seasons, accs, color=colors, width=0.55, zorder=3, edgecolor='#21262d', linewidth=0.8)

    # Overall accuracy line
    ax.axhline(overall_acc, color=NCAA_GOLD, linewidth=2, linestyle='-', zorder=4, label=f'LOSO Overall {overall_acc:.1%}')
    # CI band
    ax.axhspan(ci_lo, ci_hi, alpha=0.15, color=NCAA_GOLD, zorder=2, label=f'95% CI [{ci_lo:.1%}, {ci_hi:.1%}]')
    # Seed baseline
    if seed_baseline:
        ax.axhline(seed_baseline, color=WARNING_RED, linewidth=1.5, linestyle='--', zorder=4, label=f'Lower-seed baseline {seed_baseline:.1%}')

    # Value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.005, f'{acc:.1%}',
                ha='center', va='bottom', fontsize=10, color='#c9d1d9', fontweight='bold')

    ax.set_ylim(0.40, min(1.0, max(accs) + 0.12))
    ax.set_xlabel('Season', fontsize=12, labelpad=8)
    ax.set_ylabel('Accuracy', fontsize=12, labelpad=8)
    ax.set_title('LOSO Accuracy by Season', fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.grid(axis='y', zorder=1)

    out_path = Path(out_dir) / 'loso_by_season.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def chart_model_vs_baselines(summary, out_dir):
    """Horizontal bar chart comparing model accuracy to all baselines."""
    overall = summary.get('loso_overall', {})
    model_acc = overall.get('accuracy')
    if model_acc is None:
        print("  Skipping baselines chart: no loso_overall.accuracy")
        return
    baselines = summary.get('baselines', {})
    if not baselines:
        print("  Skipping baselines chart: no baselines in summary")
        return

    label_map = {
        'always_team_a': 'Always stronger team (adj_margin)',
        'adj_margin_sign': 'Sign of adj_margin diff',
        'win_pct_sign': 'Sign of win% diff',
        'lower_seed': 'Lower seed wins',
    }
    items = [(label_map.get(k, k), v) for k, v in baselines.items()]
    items.append(('Model (LR + XGBoost ensemble)', model_acc))
    items.sort(key=lambda x: x[1])

    labels, values = zip(*items)
    colors = [NCAA_GOLD if 'Model' in l else NCAA_BLUE for l in labels]

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor='#21262d', linewidth=0.8, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', fontsize=10, color='#c9d1d9', fontweight='bold')

    ax.set_xlim(0.0, max(values) + 0.10)
    ax.set_xlabel('Accuracy', fontsize=12, labelpad=8)
    ax.set_title('Model vs. Baselines (LOSO Tournament Accuracy)', fontsize=14, fontweight='bold', pad=12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.grid(axis='x', zorder=1)

    model_patch = mpatches.Patch(color=NCAA_GOLD, label='Model')
    baseline_patch = mpatches.Patch(color=NCAA_BLUE, label='Baselines')
    ax.legend(handles=[model_patch, baseline_patch], fontsize=9, loc='lower right', framealpha=0.3)

    out_path = Path(out_dir) / 'model_vs_baselines.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def chart_champion_probs(sim, out_dir, top_n=15):
    """Horizontal bar chart of top-N champion probabilities."""
    champs = sim.get('champion_probs', [])
    if not champs:
        print("  Skipping champion chart: no champion_probs in sim")
        return

    champs_sorted = sorted(champs, key=lambda x: x[1], reverse=True)[:top_n]
    champs_sorted = list(reversed(champs_sorted))  # bottom = lowest for horizontal
    teams, probs = zip(*champs_sorted)
    season = sim.get('season', '')
    sims = sim.get('sims', '')

    apply_style()
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))

    norm = plt.Normalize(vmin=min(probs), vmax=max(probs))
    cmap = plt.cm.Blues
    colors = [cmap(0.4 + 0.6 * norm(p)) for p in probs]
    # Highlight top team
    colors[-1] = NCAA_GOLD

    bars = ax.barh(teams, probs, color=colors, height=0.65, edgecolor='#21262d', linewidth=0.6, zorder=3)
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{prob:.1%}', va='center', fontsize=9, color='#c9d1d9')

    ax.set_xlim(0, max(probs) * 1.25)
    ax.set_xlabel('Championship Probability', fontsize=12, labelpad=8)
    ax.set_title(f'{season} Tournament Championship Probabilities\n({sims:,} simulations)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.grid(axis='x', zorder=1)

    note = sim.get('bracket_source', '')
    if note == 'generated_top64':
        ax.text(0.99, 0.01, '⚠ adj_margin selection — not official bracket',
                transform=ax.transAxes, fontsize=8, color=WARNING_RED,
                ha='right', va='bottom', style='italic')

    out_path = Path(out_dir) / f'champion_probs_{season}.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def chart_shap_importance(shap_data, out_dir):
    """Horizontal bar chart of mean absolute SHAP values per feature."""
    mean_abs = shap_data.get("mean_abs_shap", {})
    if not mean_abs:
        print("  Skipping SHAP importance chart: no mean_abs_shap data")
        return

    feats = sorted(mean_abs.keys(), key=lambda k: mean_abs[k])
    vals = [mean_abs[f] for f in feats]
    clean_labels = [f.replace("diff_", "Δ ").replace("_", " ") for f in feats]

    apply_style()
    fig, ax = plt.subplots(figsize=(9, max(4, len(feats) * 0.45)))
    colors = [NCAA_GOLD if v == max(vals) else NCAA_BLUE for v in vals]
    bars = ax.barh(clean_labels, vals, color=colors, height=0.6, edgecolor='#21262d', linewidth=0.8, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color='#c9d1d9')
    ax.set_xlabel('Mean |SHAP value|', fontsize=12, labelpad=8)
    ax.set_title('Feature Importance (SHAP)', fontsize=14, fontweight='bold', pad=12)
    ax.grid(axis='x', zorder=1)
    out_path = Path(out_dir) / 'shap_importance.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def chart_shap_beeswarm(shap_data, out_dir):
    """Beeswarm-style scatter: each dot is a game, x=SHAP value, color=feature value magnitude."""
    shap_vals = shap_data.get("shap_values")
    x_vals = shap_data.get("x_values")
    feat_cols = shap_data.get("feature_columns")
    if not shap_vals or not x_vals or not feat_cols:
        print("  Skipping SHAP beeswarm: missing data")
        return

    sv = np.array(shap_vals)      # (n_games, n_features)
    xv = np.array(x_vals)         # (n_games, n_features)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)  # ascending — bottom = least important

    apply_style()
    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_cols) * 0.55)))
    cmap = plt.cm.RdBu_r

    for plot_idx, feat_idx in enumerate(order):
        sv_col = sv[:, feat_idx]
        xv_col = xv[:, feat_idx]
        # Normalize feature values to [0,1] for color mapping
        xv_range = xv_col.max() - xv_col.min()
        xv_norm = (xv_col - xv_col.min()) / (xv_range + 1e-9)
        # Jitter y-axis for beeswarm effect
        rng = np.random.default_rng(feat_idx)
        y_jitter = plot_idx + rng.uniform(-0.3, 0.3, size=len(sv_col))
        sc = ax.scatter(sv_col, y_jitter, c=xv_norm, cmap=cmap,
                        alpha=0.6, s=14, linewidths=0, zorder=3)

    clean_labels = [feat_cols[i].replace("diff_", "Δ ").replace("_", " ") for i in order]
    ax.set_yticks(range(len(feat_cols)))
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.axvline(0, color='#8b949e', linewidth=1, linestyle='--', zorder=2)
    ax.set_xlabel('SHAP value (impact on win probability)', fontsize=11, labelpad=8)
    ax.set_title('SHAP Beeswarm — Per-Game Feature Impact', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', zorder=1)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label('Feature value\n(low → high)', fontsize=8, color='#c9d1d9')
    cb.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e')

    out_path = Path(out_dir) / 'shap_beeswarm.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description='Generate charts from model training and simulation output')
    p.add_argument('--models_dir', default='models', help='directory containing training_summary.json')
    p.add_argument('--sim', default=None, help='path to simulation JSON (default: latest in results/)')
    p.add_argument('--out_dir', default='results/charts', help='output directory for chart PNGs')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve sim path
    sim_path = args.sim
    if sim_path is None:
        candidates = sorted(Path('results').glob('sim_*.json'))
        candidates = [c for c in candidates if 'smoke' not in c.name]
        if not candidates:
            print("No simulation JSON found in results/. Run simulate first.")
            sim_path = None
        else:
            sim_path = str(candidates[-1])
            print(f"Using latest sim: {sim_path}")

    print("Generating charts...")

    try:
        summary = load_training_summary(args.models_dir)
        chart_loso_per_season(summary, out_dir)
        chart_model_vs_baselines(summary, out_dir)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")

    # SHAP charts (generated during training; optional)
    shap_path = Path(args.models_dir) / 'shap_summary.json'
    if shap_path.exists():
        try:
            shap_data = json.loads(shap_path.read_text())
            chart_shap_importance(shap_data, out_dir)
            chart_shap_beeswarm(shap_data, out_dir)
        except Exception as e:
            print(f"  Warning: SHAP charts failed: {e}")
    else:
        print(f"  Skipping SHAP charts: {shap_path} not found (run train first)")

    if sim_path:
        try:
            sim = load_sim(sim_path)
            chart_champion_probs(sim, out_dir)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    print(f"Done. Charts saved to {out_dir}/")


if __name__ == '__main__':
    main()
