#!/usr/bin/env python3
"""Generate publication-ready charts from training summary and simulation output.

Usage:
    python scripts/generate_charts.py
    python scripts/generate_charts.py --sim results/sim_2026_5000.json
    python scripts/generate_charts.py --out_dir results/charts
    python scripts/generate_charts.py --highlight "Missouri Tigers,Kansas Jayhawks"
"""
import argparse
import json
import re as _re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# ProfessionalVisualizer
# ---------------------------------------------------------------------------

class ProfessionalVisualizer:
    """Produces FiveThirtyEight-inspired, publication-ready charts.

    All charts share:
    - Light gray background, white axes, Helvetica Neue typography
    - Seaborn 'mako' sequential palette for gradient encoding
    - No top/right spines, no tick marks, axis-aligned gridlines only
    - Branding footer: '@jgamblin | march-ml · Updated <timestamp>'
    - Optional per-team highlight color for user-interest teams
    """

    # ── palette constants ──────────────────────────────────────────────────
    FIG_BG         = '#F0F0F0'   # FiveThirtyEight-style light gray
    AXES_BG        = '#FFFFFF'
    TEXT_DARK      = '#1A1A2E'
    TEXT_MID       = '#555555'
    TEXT_LIGHT     = '#888888'
    GRID_COLOR     = '#E0E0E0'
    SPINE_COLOR    = '#CCCCCC'
    ACCENT_GOLD    = '#E8A838'   # top-ranked / winner
    ACCENT_CORAL   = '#E05263'   # user-interest highlight
    BASELINE_GRAY  = '#AAAAAA'
    SUCCESS_GREEN  = '#3BB273'
    WARNING_ORANGE = '#E07B39'
    SIGNATURE      = '@jgamblin | march-ml'
    SOCIAL_TOP     = 0.85

    _HISTORICAL_CHAMP_PCT = {
        1: 0.575, 2: 0.200, 3: 0.100, 4: 0.050,
        5: 0.025, 6: 0.025, 7: 0.000, 8: 0.025,
        9: 0.000, 10: 0.000, 11: 0.025, 12: 0.000,
        13: 0.000, 14: 0.000, 15: 0.000, 16: 0.000,
    }

    def __init__(self, out_dir='results/charts', highlight_teams=None, dpi=180):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.highlight_teams = [t.strip().lower() for t in (highlight_teams or [])]

    # ── internal helpers ───────────────────────────────────────────────────

    def _apply_base_style(self):
        available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        font = next((f for f in ('Helvetica Neue', 'Helvetica', 'DejaVu Sans')
                     if f in available_fonts), 'sans-serif')
        plt.rcParams.update({
            'figure.facecolor':  self.FIG_BG,
            'axes.facecolor':    self.AXES_BG,
            'axes.edgecolor':    self.SPINE_COLOR,
            'axes.labelcolor':   self.TEXT_MID,
            'xtick.color':       self.TEXT_LIGHT,
            'ytick.color':       self.TEXT_DARK,
            'text.color':        self.TEXT_DARK,
            'grid.color':        self.GRID_COLOR,
            'grid.linestyle':    '-',
            'grid.linewidth':    0.8,
            'grid.alpha':        1.0,
            'font.family':       font,
            'axes.titlepad':     14,
            'figure.dpi':        self.dpi,
            'xtick.bottom':      False,
            'ytick.left':        False,
            'xtick.major.size':  0,
            'ytick.major.size':  0,
        })

    def _make_figure(self, figsize):
        self._apply_base_style()
        return plt.subplots(figsize=figsize)

    def _clean_axes(self, ax, grid_axis='x'):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.SPINE_COLOR)
        ax.spines['bottom'].set_color(self.SPINE_COLOR)
        ax.tick_params(length=0)
        ax.set_axisbelow(True)
        ax.grid(axis=grid_axis, color=self.GRID_COLOR, linewidth=0.8)

    def _set_title(self, ax, title, subtitle=None):
        """Left-aligned title + subtitle with guaranteed vertical separation."""
        if subtitle:
            # Title sits higher; subtitle is a smaller, lighter second line
            ax.text(0, 1.13, title, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', color=self.TEXT_DARK, va='bottom')
            ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                    fontsize=9, color=self.TEXT_LIGHT, va='bottom')
        else:
            ax.text(0, 1.04, title, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', color=self.TEXT_DARK, va='bottom')

    def _add_footer(self, fig, generated_at=None):
        if generated_at:
            try:
                # Parse ISO format (e.g. "2026-03-13T14:28:25.304598Z") → readable
                dt = datetime.fromisoformat(generated_at.rstrip('Z'))
                ts = dt.strftime('%Y-%m-%d %H:%M UTC')
            except (ValueError, AttributeError):
                ts = str(generated_at)
        else:
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        fig.text(0.01, 0.005, f'{self.SIGNATURE}  ·  Updated {ts}',
                 transform=fig.transFigure,
                 fontsize=7.5, color=self.TEXT_LIGHT, style='italic', va='bottom')
        fig.text(0.99, 0.005, '  PROJECTED BY march-ml  ',
                 transform=fig.transFigure, fontsize=7.5, fontweight='bold',
                 color=self.ACCENT_GOLD, va='bottom', ha='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333',
                           edgecolor=self.ACCENT_GOLD, linewidth=0.9))

    def _add_social_header(self, fig, title, subtitle=None, generated_at=None):
        """Dark header bar with title, subtitle, branding badge, and timestamp footer."""
        from matplotlib.patches import Rectangle as _Rect
        rect = _Rect((0, self.SOCIAL_TOP), 1.0, 1.0 - self.SOCIAL_TOP,
                     transform=fig.transFigure, facecolor='#222222',
                     edgecolor='none', zorder=10, clip_on=False)
        fig.add_artist(rect)
        header_mid = (self.SOCIAL_TOP + 1.0) / 2
        fig.text(0.015, header_mid + 0.025, title,
                 transform=fig.transFigure, fontsize=13, fontweight='bold',
                 color='white', va='center', zorder=11)
        if subtitle:
            fig.text(0.015, header_mid - 0.025, subtitle,
                     transform=fig.transFigure, fontsize=8,
                     color=self.ACCENT_GOLD, va='center', zorder=11)
        fig.text(0.985, header_mid, '  PROJECTED BY march-ml  ',
                 transform=fig.transFigure, fontsize=7.5, fontweight='bold',
                 color=self.ACCENT_GOLD, va='center', ha='right', zorder=11,
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#333333',
                           edgecolor=self.ACCENT_GOLD, linewidth=1.0))
        if generated_at:
            try:
                dt = datetime.fromisoformat(generated_at.rstrip('Z'))
                ts = dt.strftime('%Y-%m-%d %H:%M UTC')
            except (ValueError, AttributeError):
                ts = str(generated_at)
        else:
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        fig.text(0.01, 0.005, f'Updated {ts}  ·  {self.SIGNATURE}',
                 transform=fig.transFigure, fontsize=7, color=self.TEXT_LIGHT,
                 style='italic', va='bottom')

    def _save(self, fig, filename, top_pad=0.88, social=False):
        """Save figure; top_pad leaves room for title block (use lower value for subtitles)."""
        fig.subplots_adjust(top=top_pad, bottom=0.10)
        path = self.out_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                    facecolor=self.FIG_BG, edgecolor='none')
        if social:
            self._save_social(fig, path)
        plt.close(fig)
        print(f"  Saved {path}")

    def _save_social(self, fig, original_path: Path):
        """Save a 1200×675 (16:9) social-media-optimized version alongside the original."""
        social_path = original_path.parent / (original_path.stem + '_social.png')
        try:
            from PIL import Image
            img = Image.open(original_path).convert('RGB')
            w, h = img.size
            target_w, target_h = 1200, 675
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            bg_hex = self.FIG_BG.lstrip('#')
            bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
            canvas = Image.new('RGB', (target_w, target_h), bg_rgb)
            offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
            canvas.paste(img_resized, offset)
            canvas.save(social_path, dpi=(150, 150))
            print(f"  Saved {social_path} (16:9 social)")
        except ImportError:
            fig.savefig(social_path, dpi=100, bbox_inches='tight',
                        facecolor=self.FIG_BG, edgecolor='none')
            print(f"  Saved {social_path} (16:9 social, PIL not available)")

    def _bar_colors(self, values, names=None):
        """Map values to mako palette; override top item and user highlights."""
        palette = sns.color_palette('mako', 256)
        lo, hi  = min(values), max(values)
        span    = hi - lo or 1.0
        # Sample from [index 25..85] — skip very dark + very light extremes
        colors  = [list(palette[int(25 + 60 * (v - lo) / span)]) for v in values]
        # Highlight top value (last element after ascending sort to index -1)
        colors[-1] = list(matplotlib.colors.to_rgb(self.ACCENT_GOLD))
        if names:
            for i, name in enumerate(names):
                if name.lower() in self.highlight_teams:
                    colors[i] = list(matplotlib.colors.to_rgb(self.ACCENT_CORAL))
        return colors

    @staticmethod
    def _wilson_ci(probs, n, z=1.96):
        ci_lo, ci_hi = [], []
        for p in probs:
            center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
            margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
            ci_lo.append(max(0.0, p - (center - margin)))
            ci_hi.append(min(1.0, (center + margin) - p))
        return ci_lo, ci_hi

    # ── public chart methods ───────────────────────────────────────────────

    def chart_champion_probs(self, sim, top_n=15):
        """Horizontal bar chart with 95% Wilson CI error bars."""
        champs = sim.get('champion_probs', [])
        if not champs:
            print("  Skipping champion chart: no champion_probs in sim")
            return

        champs_sorted = list(reversed(
            sorted(champs, key=lambda x: x[1], reverse=True)[:top_n]
        ))
        teams, probs  = zip(*champs_sorted)
        season        = sim.get('season', '')
        n_sims        = sim.get('sims', 1)
        generated_at  = sim.get('generated_at')

        ci_lo, ci_hi  = self._wilson_ci(probs, n_sims)
        colors        = self._bar_colors(list(probs), names=list(teams))

        fig, ax = self._make_figure((11, max(5.5, top_n * 0.54)))

        bars = ax.barh(teams, probs, xerr=[ci_lo, ci_hi],
                       color=colors, height=0.62, edgecolor='white', linewidth=0.5,
                       error_kw=dict(ecolor=self.TEXT_LIGHT, elinewidth=1.0,
                                     capsize=3, capthick=1.0))

        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + ci_hi[i] + 0.004,
                    bar.get_y() + bar.get_height() / 2,
                    f'{prob:.1%}', va='center', ha='left',
                    fontsize=9, color=self.TEXT_DARK, fontweight='semibold')

        team_seeds = {entry['team']: entry.get('seed') for entry in sim.get('bracket', [])}

        cinderellas = [
            (i, team, prob, team_seeds.get(team))
            for i, (team, prob) in enumerate(champs_sorted)
            if team_seeds.get(team) and team_seeds.get(team) > 10 and prob > 0.02
        ]
        for i, team, prob, seed in cinderellas:
            bar_y_center = bars[i].get_y() + bars[i].get_height() / 2
            ax.annotate(
                f"★ Cinderella?\nSeed #{seed}",
                xy=(prob, bar_y_center),
                xytext=(prob + 0.08, bar_y_center),
                fontsize=8, color=self.ACCENT_CORAL, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0',
                          edgecolor=self.ACCENT_CORAL, linewidth=1.2, alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=self.ACCENT_CORAL, lw=1.2,
                                connectionstyle='arc3,rad=0.1'),
            )

        xlim_mult = 1.55 if cinderellas else 1.32
        ax.set_xlim(0, max(p + h for p, h in zip(probs, ci_hi)) * xlim_mult)
        ax.set_xlabel('Championship Probability', fontsize=10, labelpad=8, color=self.TEXT_MID)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        self._clean_axes(ax, grid_axis='x')

        legend_items = [mpatches.Patch(color=self.ACCENT_GOLD, label='#1 Favorite')]
        if self.highlight_teams:
            legend_items.append(mpatches.Patch(color=self.ACCENT_CORAL, label='Your team'))
        ax.legend(handles=legend_items, fontsize=8, frameon=False,
                  loc='lower right', labelcolor=self.TEXT_MID)

        if sim.get('bracket_source') == 'generated_top64':
            ax.text(0.99, 1.02, '⚠ Efficiency-based selection — not official bracket',
                    transform=ax.transAxes, fontsize=7.5, color=self.WARNING_ORANGE,
                    ha='right', va='bottom', style='italic')

        title = f'{season} Championship Favorites'
        subtitle = (f'Top {top_n} by win probability  ·  {n_sims:,} Monte Carlo sims  '
                    f'·  Error bars = 95% CI')
        self._add_social_header(fig, title, subtitle, generated_at)
        self._save(fig, f'champion_probs_{season}.png', top_pad=self.SOCIAL_TOP, social=True)

    def chart_round_probs(self, sim, top_n=5):
        """Heatmap of round-by-round reach probabilities (teams × rounds)."""
        round_probs = sim.get('round_probs', {})
        if not round_probs:
            print("  Skipping round-probs chart: no round_probs in sim")
            return

        round_keys   = ['sweet_16', 'elite_8', 'final_4', 'title_game', 'champion']
        round_labels = ['Sweet 16', 'Elite 8', 'Final Four', 'Title Game', 'Champion']

        # Sort by champion probability, highest at top
        by_champ = sorted(round_probs.items(),
                          key=lambda kv: kv[1].get('champion', 0), reverse=True)[:top_n]
        teams  = [t for t, _ in by_champ]
        matrix = np.array([[rp.get(rk, 0) for rk in round_keys] for _, rp in by_champ])
        season       = sim.get('season', '')
        n_sims       = sim.get('sims', '')
        generated_at = sim.get('generated_at')

        row_h = 0.52
        fig, ax = self._make_figure((9, max(6.5, top_n * row_h)))

        # mako_r: high probability → dark navy, low probability → light mint
        sns.heatmap(
            matrix, ax=ax,
            xticklabels=round_labels,
            yticklabels=teams,
            cmap='mako_r',
            vmin=0, vmax=1,
            annot=False,          # draw custom annotations below for color control
            linewidths=0.8,
            linecolor=self.FIG_BG,
            cbar=True,
            cbar_kws=dict(shrink=0.55, aspect=22, pad=0.02),
        )

        # Adaptive text color: white on dark cells, dark on light cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                txt_color = 'white' if val > 0.28 else self.TEXT_DARK
                weight    = 'bold'  if val > 0.45 else 'normal'
                ax.text(j + 0.5, i + 0.5, f'{val:.0%}',
                        ha='center', va='center',
                        fontsize=9, color=txt_color, fontweight=weight)

        # Most Likely Path: golden border on the highest-confidence round per team
        for i in range(matrix.shape[0]):
            candidates = [j for j in range(matrix.shape[1]) if matrix[i, j] >= 0.50]
            best_j = max(candidates) if candidates else int(np.argmax(matrix[i]))
            ax.add_patch(plt.Rectangle((best_j, i), 1, 1, fill=False,
                                       edgecolor=self.ACCENT_GOLD, linewidth=2.8, zorder=5))

        # Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(length=0, labelsize=7)
        cbar.set_label('Probability', fontsize=8, color=self.TEXT_LIGHT, labelpad=6)
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        cbar.outline.set_visible(False)

        # Style axes — labels on bottom so they don't collide with title block
        ax.tick_params(left=False, bottom=False, top=False, length=0)
        ax.set_yticklabels(teams, fontsize=9.5, color=self.TEXT_DARK, va='center')
        ax.set_xticklabels(round_labels, fontsize=10.5, fontweight='bold',
                           color=self.TEXT_DARK, rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Source note below chart if pre-bracket
        if sim.get('bracket_source') == 'generated_top64':
            ax.text(0.99, -0.04, '⚠ Efficiency-based selection — not official bracket',
                    transform=ax.transAxes, fontsize=7.5, color=self.WARNING_ORANGE,
                    ha='right', va='top', style='italic')

        self._add_social_header(fig, f'{season} Road to the Championship',
                                subtitle=f'Probability of reaching each round  ·  Top {top_n} teams by championship odds  ·  {n_sims:,} simulations',
                                generated_at=generated_at)
        self._save(fig, f'round_probs_{season}.png', top_pad=self.SOCIAL_TOP, social=True)

    def chart_loso_per_season(self, summary):
        """Bar chart of rolling CV accuracy by season with CI band."""
        seasons_data = summary.get('rolling_cv_per_season') or summary.get('loso_per_season', [])
        cv_label     = 'Rolling CV' if summary.get('rolling_cv_per_season') else 'LOSO'
        if not seasons_data:
            print("  Skipping accuracy-by-season chart: no per-season data")
            return

        seasons = [str(d['season']) for d in seasons_data]
        accs    = [d['accuracy'] for d in seasons_data]
        overall = summary.get('rolling_cv_overall') or summary.get('loso_overall', {})
        overall_acc  = overall.get('accuracy', float(np.mean(accs)))
        ci           = overall.get('accuracy_ci_95', [overall_acc - 0.05, overall_acc + 0.05])
        seed_baseline = summary.get('baselines', {}).get('lower_seed')

        fig, ax = self._make_figure((10, 5.5))

        bar_colors = [self.SUCCESS_GREEN if a >= overall_acc
                      else sns.color_palette('mako', 1)[0] for a in accs]
        bars = ax.bar(seasons, accs, color=bar_colors, width=0.55, zorder=3,
                      edgecolor='white', linewidth=0.6)

        ax.axhline(overall_acc, color=self.ACCENT_GOLD, linewidth=2.0, zorder=4,
                   label=f'{cv_label} avg {overall_acc:.1%}')
        ax.axhspan(ci[0], ci[1], alpha=0.12, color=self.ACCENT_GOLD, zorder=2,
                   label=f'95% CI [{ci[0]:.1%} – {ci[1]:.1%}]')
        if seed_baseline:
            ax.axhline(seed_baseline, color=self.ACCENT_CORAL, linewidth=1.5,
                       linestyle='--', zorder=4, label=f'Seed baseline {seed_baseline:.1%}')

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.006,
                    f'{acc:.0%}', ha='center', va='bottom',
                    fontsize=9, color=self.TEXT_DARK, fontweight='bold')

        ax.set_ylim(0.45, min(1.0, max(accs) + 0.13))
        ax.set_xlabel('Season (test year)', fontsize=10, labelpad=8, color=self.TEXT_MID)
        ax.set_ylabel('Accuracy', fontsize=10, labelpad=8, color=self.TEXT_MID)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        self._clean_axes(ax, grid_axis='y')
        self._set_title(ax, f'{cv_label} Accuracy by Season',
                        subtitle='Green bars exceed average  ·  Dashed line = lower-seed baseline')
        ax.legend(fontsize=8, frameon=False, loc='lower right', labelcolor=self.TEXT_MID)

        self._add_footer(fig)
        self._save(fig, 'loso_by_season.png')

    def chart_model_vs_baselines(self, summary):
        """Horizontal bar chart: model accuracy vs. naive baselines."""
        overall   = summary.get('rolling_cv_overall') or summary.get('loso_overall', {})
        cv_label  = 'Rolling CV' if summary.get('rolling_cv_overall') else 'LOSO'
        model_acc = overall.get('accuracy')
        baselines = summary.get('baselines', {})
        if model_acc is None or not baselines:
            print("  Skipping baselines chart: missing data")
            return

        label_map = {
            'always_team_a':   'Always stronger team (adj_margin)',
            'adj_margin_sign': 'Sign of adj_margin difference',
            'win_pct_sign':    'Sign of win-rate difference',
            'lower_seed':      'Lower seed always wins',
        }
        items = [(label_map.get(k, k), v) for k, v in baselines.items()]
        items.append(('ML Model  (LR + XGBoost ensemble)', model_acc))
        items.sort(key=lambda x: x[1])

        labels, values = zip(*items)
        colors = [self.ACCENT_GOLD if 'ML Model' in lbl else self.BASELINE_GRAY
                  for lbl in labels]

        fig, ax = self._make_figure((10, 4.5))
        bars = ax.barh(labels, values, color=colors, height=0.55,
                       edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1%}', va='center', ha='left',
                    fontsize=9.5, color=self.TEXT_DARK, fontweight='bold')

        ax.set_xlim(0.0, max(values) + 0.12)
        ax.set_xlabel('Accuracy', fontsize=10, labelpad=8, color=self.TEXT_MID)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        self._clean_axes(ax, grid_axis='x')
        self._set_title(ax, 'Model vs. Baselines',
                        subtitle=f'{cv_label} accuracy on held-out tournament games  ·  Gold bar = our model')
        ax.legend(handles=[
            mpatches.Patch(color=self.ACCENT_GOLD, label='ML Model'),
            mpatches.Patch(color=self.BASELINE_GRAY, label='Naive baselines'),
        ], fontsize=8, frameon=False, loc='lower right', labelcolor=self.TEXT_MID)

        self._add_footer(fig)
        self._save(fig, 'model_vs_baselines.png')

    def chart_shap_importance(self, shap_data):
        """Horizontal bar chart of mean |SHAP| per feature."""
        mean_abs = shap_data.get('mean_abs_shap', {})
        if not mean_abs:
            print("  Skipping SHAP importance chart: no mean_abs_shap data")
            return

        feats  = sorted(mean_abs, key=lambda k: mean_abs[k])
        vals   = [mean_abs[f] for f in feats]

        def _label(f):
            name = f.replace('diff_', '\u0394 ').replace('_', ' ').title()
            for old, new in (('Sos ', 'SOS '), ('Pom ', 'POM '), (' Net ', ' NET '),
                             (' Em', ' EM'), ('Adj Em', 'Adj EM')):
                name = name.replace(old, new)
            return name.strip()

        clean  = [_label(f) for f in feats]
        colors = list(sns.color_palette('mako_r', len(feats)))
        colors[-1] = list(matplotlib.colors.to_rgb(self.ACCENT_GOLD))

        n = len(feats)
        fig, ax = self._make_figure((10, max(5.0, n * 0.52)))
        bars = ax.barh(clean, vals, color=colors, height=0.65,
                       edgecolor='white', linewidth=0.5)

        max_val = max(vals)
        for bar, val in zip(bars, vals):
            ax.text(val + max_val * 0.012, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', ha='left',
                    fontsize=8.5, color=self.TEXT_DARK)

        # Thin separator line above the gold (#1) bar
        sep_y = bars[-1].get_y() + bars[-1].get_height() + 0.45
        ax.axhline(sep_y, color=self.GRID_COLOR, linewidth=1.2)

        ax.set_xlim(0, max_val * 1.20)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:.1f}'
        ))
        ax.set_xlabel('Mean |SHAP value|  (impact on win probability)',
                      fontsize=10, labelpad=8, color=self.TEXT_MID)
        self._clean_axes(ax, grid_axis='x')

        # Inline compact title/subtitle — avoids proportional gap on tall figures
        ax.text(0, 1.05, 'Feature Importance', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=self.TEXT_DARK, va='bottom')
        ax.text(0, 1.01, 'SHAP values  \u00b7  Higher = more influential  \u00b7  Gold = top feature',
                transform=ax.transAxes, fontsize=9, color=self.TEXT_LIGHT, va='bottom')

        self._add_footer(fig)
        self._save(fig, 'shap_importance.png', top_pad=0.92)

    def chart_shap_beeswarm(self, shap_data):
        """Beeswarm scatter: each dot = one game, x = SHAP impact, color = feature value."""
        shap_vals = shap_data.get('shap_values')
        x_vals    = shap_data.get('x_values')
        feat_cols = shap_data.get('feature_columns')
        if not shap_vals or not x_vals or not feat_cols:
            print("  Skipping SHAP beeswarm: missing data")
            return

        sv       = np.array(shap_vals)
        xv       = np.array(x_vals)
        mean_abs = np.abs(sv).mean(axis=0)
        order    = np.argsort(mean_abs)

        fig, ax = self._make_figure((11, max(5.5, len(feat_cols) * 0.58)))
        cmap = plt.cm.RdBu_r

        for plot_idx, feat_idx in enumerate(order):
            sv_col  = sv[:, feat_idx]
            xv_col  = xv[:, feat_idx]
            xv_span = xv_col.max() - xv_col.min()
            xv_norm = (xv_col - xv_col.min()) / (xv_span + 1e-9)
            rng     = np.random.default_rng(feat_idx)
            y_jit   = plot_idx + rng.uniform(-0.28, 0.28, size=len(sv_col))
            ax.scatter(sv_col, y_jit, c=xv_norm, cmap=cmap,
                       alpha=0.55, s=12, linewidths=0)

        clean = [feat_cols[i].replace('diff_', '\u0394 ').replace('_', ' ').title()
                 for i in order]
        ax.set_yticks(range(len(feat_cols)))
        ax.set_yticklabels(clean, fontsize=8.5)
        ax.axvline(0, color=self.SPINE_COLOR, linewidth=1.2, linestyle='--')
        ax.set_xlabel('SHAP value  (positive = favors team A winning)',
                      fontsize=10, labelpad=8, color=self.TEXT_MID)
        self._clean_axes(ax, grid_axis='x')
        self._set_title(ax, 'SHAP Beeswarm \u2014 Per-Game Feature Impact',
                        subtitle='Each dot = one game  \u00b7  Color = feature value (blue low to red high)')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.5, aspect=20)
        cb.set_label('Feature value\n(low to high)', fontsize=8, color=self.TEXT_LIGHT)
        cb.ax.tick_params(labelsize=7, length=0)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=self.TEXT_LIGHT)
        cb.outline.set_visible(False)

        self._add_footer(fig)
        self._save(fig, 'shap_beeswarm.png')

    def chart_chaos_index(self, sim):
        """Grouped bar chart: current sim championship % by seed vs historical (1985-2025)."""
        bracket = sim.get('bracket', [])
        if not bracket:
            print("  Skipping chaos chart: no bracket data")
            return
        team_seeds = {e['team']: e.get('seed', 0) for e in bracket}
        champion_probs = sim.get('champion_probs', [])
        season = sim.get('season', '')
        n_sims = sim.get('sims', 1)
        generated_at = sim.get('generated_at')

        sim_by_seed = {}
        for team, prob in champion_probs:
            seed = team_seeds.get(team, 0)
            if seed:
                sim_by_seed[seed] = sim_by_seed.get(seed, 0) + prob

        show_seeds = sorted(s for s in sim_by_seed if s <= 12)
        if not show_seeds:
            print("  Skipping chaos chart: insufficient seed data")
            return

        hist_vals = [self._HISTORICAL_CHAMP_PCT.get(s, 0) for s in show_seeds]
        sim_vals  = [sim_by_seed.get(s, 0) for s in show_seeds]

        chaos_score = sum(s * sim_by_seed.get(s, 0) for s in range(1, 17))
        hist_chaos  = sum(s * self._HISTORICAL_CHAMP_PCT.get(s, 0) for s in range(1, 17))
        chaos_delta = chaos_score - hist_chaos

        x = np.arange(len(show_seeds))
        width = 0.38
        fig, ax = self._make_figure((12, 6))

        ax.bar(x - width/2, hist_vals, width, label='Historical avg (1985–2025)',
               color=self.BASELINE_GRAY, alpha=0.75, edgecolor='white', linewidth=0.5)
        bars_sim = ax.bar(x + width/2, sim_vals, width, label=f'{season} Simulation',
                          color=[self.ACCENT_CORAL if s > 5 else self.SUCCESS_GREEN
                                 for s in show_seeds],
                          edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars_sim, sim_vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
                        f'{val:.0%}', ha='center', va='bottom',
                        fontsize=8, color=self.TEXT_DARK)

        chaos_label = 'MORE CHALK ↓' if chaos_delta < 0 else 'MORE CHAOS ↑'
        chaos_color = self.ACCENT_GOLD if chaos_delta < 0 else self.ACCENT_CORAL
        ax.text(0.98, 0.97,
                f'Chaos Score: {chaos_score:.1f}  ({chaos_label})\n'
                f'Historical avg: {hist_chaos:.1f}  |  Δ {chaos_delta:+.1f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
                color=chaos_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.AXES_BG,
                          edgecolor=chaos_color, linewidth=1.5))

        ax.set_xticks(x)
        ax.set_xticklabels([f'Seed {s}' for s in show_seeds], fontsize=9)
        ax.set_ylabel('Championship Probability', fontsize=10, labelpad=8, color=self.TEXT_MID)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, max(max(hist_vals), max(sim_vals)) * 1.4)
        self._clean_axes(ax, grid_axis='y')
        ax.legend(fontsize=9, frameon=False, loc='upper right', labelcolor=self.TEXT_MID)

        if sim.get('bracket_source') == 'generated_top64':
            ax.text(0.99, -0.04, '⚠ Efficiency-based selection — not official bracket',
                    transform=ax.transAxes, fontsize=7.5, color=self.WARNING_ORANGE,
                    ha='right', va='top', style='italic')

        self._add_social_header(fig, f'{season} Chaos Index',
                                subtitle='Championship probability by seed vs. historical averages  ·  Green = favored seeds  ·  Red = upsets',
                                generated_at=generated_at)
        self._save(fig, f'chaos_index_{season}.png', top_pad=self.SOCIAL_TOP, social=True)

    def chart_team_profiles(self, sim, out_subdir='teams'):
        """Generate one round-probability chart per team in results/charts/teams/.

        Each chart shows the team's probability of reaching every tournament round,
        styled consistently with the other charts and saved as a sanitized filename.
        """
        round_probs = sim.get('round_probs', {})
        if not round_probs:
            print("  Skipping team charts: no round_probs in sim")
            return

        round_keys   = ['round_of_32', 'sweet_16',
                        'elite_8', 'final_4', 'title_game', 'champion']
        round_labels = ['Round of 32', 'Sweet 16',
                        'Elite 8', 'Final Four', 'Title Game', 'Champion']
        season       = sim.get('season', '')
        n_sims       = sim.get('sims', '')
        generated_at = sim.get('generated_at')

        # Championship rank per team (1 = most likely champion)
        champ_rank = {team: rank + 1 for rank, (team, _) in enumerate(
            sorted(sim.get('champion_probs', []), key=lambda x: x[1], reverse=True))}

        teams_out = self.out_dir / out_subdir
        teams_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for team, rp in sorted(round_probs.items(),
                                key=lambda kv: champ_rank.get(kv[0], 999)):
            # Build ordered list of (label, value) for rounds present in this team's data
            rows = [(lbl, rp[key]) for key, lbl in zip(round_keys, round_labels) if key in rp]
            if not rows:
                continue
            labels, probs = zip(*rows)
            colors = self._bar_colors(list(probs))

            fig, ax = self._make_figure((8, 4.5))
            bars = ax.barh(labels, probs, color=colors, height=0.62,
                           edgecolor='white', linewidth=0.5)

            for bar, prob in zip(bars, probs):
                ax.text(min(prob + 0.015, 0.95), bar.get_y() + bar.get_height() / 2,
                        f'{prob:.0%}', va='center', ha='left',
                        fontsize=9.5, color=self.TEXT_DARK, fontweight='semibold')

            ax.set_xlim(0, 1.0)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            self._clean_axes(ax, grid_axis='x')

            rank      = champ_rank.get(team, '?')
            champ_pct = rp.get('champion', 0)
            self._set_title(ax, team,
                            subtitle=(f'#{rank} overall  ·  Champion odds: {champ_pct:.1%}'
                                      f'  ·  {season}  ·  {n_sims:,} simulations'))
            self._add_footer(fig, generated_at)

            safe = _re.sub(r'[^a-z0-9]+', '_', team.lower()).strip('_') + '.png'
            fig.subplots_adjust(top=0.84, bottom=0.10)
            fig.savefig(teams_out / safe, dpi=self.dpi, bbox_inches='tight',
                        facecolor=self.FIG_BG, edgecolor='none')
            plt.close(fig)
            count += 1

        print(f"  Saved {count} team charts to {teams_out}/")


# ---------------------------------------------------------------------------
# Standalone loaders
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Generate publication-ready charts from model training and simulation output')
    p.add_argument('--models_dir', default='models',
                   help='directory containing training_summary.json')
    p.add_argument('--sim', default=None,
                   help='path to simulation JSON (default: latest in results/)')
    p.add_argument('--out_dir', default='results/charts',
                   help='output directory for chart PNGs')
    p.add_argument('--highlight', default=None,
                   help='comma-separated team names to highlight e.g. "Missouri Tigers,Kansas"')
    p.add_argument('--dpi', type=int, default=180, help='output DPI (default: 180)')
    p.add_argument('--teams', action='store_true',
                   help='also generate individual team charts in results/charts/teams/')
    args = p.parse_args()

    highlight_teams = [t.strip() for t in args.highlight.split(',')] if args.highlight else []
    viz = ProfessionalVisualizer(out_dir=args.out_dir,
                                 highlight_teams=highlight_teams,
                                 dpi=args.dpi)

    sim_path = args.sim
    if sim_path is None:
        candidates = sorted(Path('results').glob('sim_*.json'))
        candidates = [c for c in candidates if 'smoke' not in c.name]
        if candidates:
            sim_path = str(candidates[-1])
            print(f"Using latest sim: {sim_path}")
        else:
            print("No simulation JSON found in results/. Run simulate first.")

    print("Generating charts...")

    try:
        summary = load_training_summary(args.models_dir)
        viz.chart_loso_per_season(summary)
        viz.chart_model_vs_baselines(summary)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")

    shap_path = Path(args.models_dir) / 'shap_summary.json'
    if shap_path.exists():
        try:
            shap_data = json.loads(shap_path.read_text())
            viz.chart_shap_importance(shap_data)
            viz.chart_shap_beeswarm(shap_data)
        except Exception as e:
            print(f"  Warning: SHAP charts failed: {e}")
    else:
        print(f"  Skipping SHAP charts: {shap_path} not found (run train first)")

    if sim_path:
        try:
            sim = load_sim(sim_path)
            viz.chart_champion_probs(sim)
            viz.chart_round_probs(sim)
            viz.chart_chaos_index(sim)
            if args.teams:
                viz.chart_team_profiles(sim)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    print(f"Done. Charts saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
