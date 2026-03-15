#!/usr/bin/env python3
"""Generate a shareable NCAA tournament bracket with predicted picks and win probabilities.

Reads simulation JSON and produces a 4200×2025 px PNG showing all 64 teams,
predicted picks advancing through each round, and each team's per-game win %.

Usage:
    python scripts/generate_bracket_visual.py
    python scripts/generate_bracket_visual.py --sim results/sim_results.json
    python scripts/generate_bracket_visual.py --out results/charts/bracket_2026.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── layout ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H, DPI = 28.0, 13.5, 150   # → 4200 × 2025 px
CX = 14.0                              # figure x-centre
BOX_W = 2.2                            # box width (all boxes same size)
BOX_H = 0.250                          # box height

# Left-side round columns (left edge of box); rounds go L→R
L_R64, L_R32, L_S16, L_E8 = 0.15, 2.60, 5.05, 7.50
# Right-side round columns (left edge of box); mirrored
R_R64 = FIG_W - 0.15 - BOX_W      # 25.65
R_R32 = FIG_W - L_R32 - BOX_W     # 23.20
R_S16 = FIG_W - L_S16 - BOX_W     # 20.75
R_E8  = FIG_W - L_E8  - BOX_W     # 18.30

# Centre columns
L_FF  = 10.45   # left  Final Four matchup
CHAMP = 12.90   # championship game  (centred at CX=14.0)
R_FF  = 15.35   # right Final Four matchup

# Region y bounds
TOP_Y0, TOP_Y1 = 6.75, 13.25   # East / South
BOT_Y0, BOT_Y1 = 0.25,  6.75   # West / Midwest

# Natural E8 midpoints become FF entry y-positions
TOP_MID = (TOP_Y0 + TOP_Y1) / 2   # ≈ 10.0
BOT_MID = (BOT_Y0 + BOT_Y1) / 2   # ≈  3.5
FF_CTR  = (TOP_MID + BOT_MID) / 2  # ≈  6.75  (where left/right FF lines meet)

# Championship game y-positions
FIN_Y_A = FF_CTR + 0.55   # top finalist  ≈ 7.30
FIN_Y_B = FF_CTR - 0.55   # bottom finalist ≈ 6.20
CHAMP_Y = 4.50             # champion box y-centre

# ── colours ───────────────────────────────────────────────────────────────────
BG     = '#F7F7F7'
BX_DEF = '#FFFFFF'
BX_WIN = '#E8F0FE'
BX_FIN = '#D2E3FC'
BX_CHP = '#FFF8E1'
FG     = '#1A1A2E'
DIM    = '#888888'
LINE   = '#CCCCCC'
WLINE  = '#1A73E8'
GOLD   = '#E8A000'
GSEED  = '#666666'
TC     = '#444444'
FONT   = 'Helvetica Neue'

# ── helpers ───────────────────────────────────────────────────────────────────

def _cp(team, cp_list):
    for t, p in cp_list:
        if t == team:
            return p
    return 0.0


def _game_prob(team, this_rnd, prev_rnd, rp):
    """Conditional win probability for this_rnd given team reached it."""
    if not team or team not in rp:
        return 0.0
    trp = rp[team]
    this_val = trp.get(this_rnd, 0.0)
    if prev_rnd is None:
        return this_val
    prev_val = trp.get(prev_rnd, 0.0)
    return (this_val / prev_val) if prev_val > 0 else 0.0


def _pair_probs(team_a, team_b, next_rnd, prev_rnd, rp):
    """Win probabilities for a head-to-head matchup, normalized to sum to 1.0."""
    pa = _game_prob(team_a, next_rnd, prev_rnd, rp)
    pb = _game_prob(team_b, next_rnd, prev_rnd, rp)
    total = pa + pb
    if total <= 0:
        return 0.5, 0.5
    return pa / total, pb / total


def _win(a, b, rnd, rp):
    pa = rp.get(a, {}).get(rnd, 0.0)
    pb = rp.get(b, {}).get(rnd, 0.0)
    return a if pa >= pb else b


def _trim(name, n=16):
    if len(name) <= n:
        return name
    parts = name.split()
    for k in range(len(parts), 0, -1):
        s = ' '.join(parts[:k])
        if len(s) <= n:
            return s
    return name[:n]


def _slot_ys(y0, y1, n=16):
    """n y-centres from y1 (top) down to y0 (bottom)."""
    pitch = (y1 - y0) / n
    return [y1 - (i + 0.5) * pitch for i in range(n)]


def _mids(ys):
    return [(ys[i * 2] + ys[i * 2 + 1]) / 2 for i in range(len(ys) // 2)]


def _box(ax, x, y, team, seed, prob, win=False, finalist=False, champ=False):
    if champ:
        bg, ec, lw = BX_CHP, GOLD, 2.0
    elif finalist:
        bg, ec, lw = BX_FIN, WLINE, 1.5
    elif win:
        bg, ec, lw = BX_WIN, WLINE, 1.2
    else:
        bg, ec, lw = BX_DEF, LINE, 0.8
    ax.add_patch(FancyBboxPatch(
        (x, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle='round,pad=0.003', fc=bg, ec=ec, lw=lw, zorder=3))
    sc = GOLD if champ else (WLINE if win or finalist else GSEED)
    ax.text(x + 0.10, y, str(seed) if seed else '?',
            ha='left', va='center', fontsize=5.6, fontweight='bold', color=sc, zorder=4)
    nc = GOLD if champ else (FG if win or finalist else DIM)
    ax.text(x + 0.32, y, _trim(team),
            ha='left', va='center', fontsize=6.0,
            fontweight='bold' if (win or finalist or champ) else 'normal',
            color=nc, zorder=4)
    if prob >= 0.005:
        pc = GOLD if champ else (WLINE if (win or finalist) else '#AAAAAA')
        ax.text(x + BOX_W - 0.07, y, f'{prob * 100:.0f}%',
                ha='right', va='center', fontsize=5.5, color=pc, zorder=4)


def _elbow(ax, x_edge, ya, yb, ym, left=True, win=False):
    """
    Draw bracket elbow from two team boxes exiting at x_edge.
    Returns the x coordinate where the exit horizontal ends.
    """
    col, lw = (WLINE, 0.85) if win else (LINE, 0.5)
    G, EX = 0.07, 0.15
    if left:
        cx = x_edge + G
        ax.plot([x_edge, cx], [ya, ya], color=col, lw=lw, zorder=2)
        ax.plot([x_edge, cx], [yb, yb], color=col, lw=lw, zorder=2)
        ax.plot([cx, cx],     [ya, yb], color=col, lw=lw, zorder=2)
        ax.plot([cx, cx + EX], [ym, ym], color=col, lw=lw, zorder=2)
        return cx + EX
    else:
        cx = x_edge - G
        ax.plot([x_edge, cx], [ya, ya], color=col, lw=lw, zorder=2)
        ax.plot([x_edge, cx], [yb, yb], color=col, lw=lw, zorder=2)
        ax.plot([cx, cx],     [ya, yb], color=col, lw=lw, zorder=2)
        ax.plot([cx - EX, cx], [ym, ym], color=col, lw=lw, zorder=2)
        return cx - EX


def _hline(ax, x0, x1, y, win=False):
    col, lw = (WLINE, 0.85) if win else (LINE, 0.5)
    ax.plot([x0, x1], [y, y], color=col, lw=lw, zorder=2)


# ── region ────────────────────────────────────────────────────────────────────

def _draw_region(ax, teams16, rp, x_r64, x_r32, x_s16, x_e8,
                 y0, y1, left, label):
    """Draw one 16-team region. Returns (ff_team, e8_mid_y)."""
    seed_map = {t['team']: (t.get('seed') or 0) for t in teams16}
    names    = [t['team'] for t in teams16]

    ys64 = _slot_ys(y0, y1)
    ys32 = _mids(ys64)
    ys16 = _mids(ys32)
    yse8 = _mids(ys16)
    e8_mid = (yse8[0] + yse8[1]) / 2

    r32 = [_win(names[i*2], names[i*2+1], 'round_of_32', rp) for i in range(8)]
    s16 = [_win(r32[i*2],   r32[i*2+1],   'sweet_16',    rp) for i in range(4)]
    e8  = [_win(s16[i*2],   s16[i*2+1],   'elite_8',     rp) for i in range(2)]
    ff  = _win(e8[0], e8[1], 'final_4', rp)

    # helper: which edge to use for elbow exit
    xe = lambda x: (x + BOX_W) if left else x
    # helper: which edge to connect TO for next round
    nx = lambda x: x if left else (x + BOX_W)

    # R64 — draw as 8 head-to-head pairs
    for i in range(0, 16, 2):
        ta, tb = names[i], names[i + 1]
        pa, pb = _pair_probs(ta, tb, 'round_of_32', None, rp)
        _box(ax, x_r64, ys64[i],     ta, seed_map[ta], pa, win=(r32[i // 2] == ta))
        _box(ax, x_r64, ys64[i + 1], tb, seed_map[tb], pb, win=(r32[i // 2] == tb))
    for i in range(8):
        end = _elbow(ax, xe(x_r64), ys64[i*2], ys64[i*2+1], ys32[i], left=left, win=True)
        _hline(ax, end, nx(x_r32), ys32[i], win=True)

    # R32 — draw as 4 pairs
    for i in range(0, 8, 2):
        ta, tb = r32[i], r32[i + 1]
        pa, pb = _pair_probs(ta, tb, 'sweet_16', 'round_of_32', rp)
        _box(ax, x_r32, ys32[i],     ta, seed_map.get(ta, 0), pa, win=(s16[i // 2] == ta))
        _box(ax, x_r32, ys32[i + 1], tb, seed_map.get(tb, 0), pb, win=(s16[i // 2] == tb))
    for i in range(4):
        end = _elbow(ax, xe(x_r32), ys32[i*2], ys32[i*2+1], ys16[i], left=left, win=True)
        _hline(ax, end, nx(x_s16), ys16[i], win=True)

    # S16 — draw as 2 pairs
    for i in range(0, 4, 2):
        ta, tb = s16[i], s16[i + 1]
        pa, pb = _pair_probs(ta, tb, 'elite_8', 'sweet_16', rp)
        _box(ax, x_s16, ys16[i],     ta, seed_map.get(ta, 0), pa, win=(e8[i // 2] == ta))
        _box(ax, x_s16, ys16[i + 1], tb, seed_map.get(tb, 0), pb, win=(e8[i // 2] == tb))
    for i in range(2):
        end = _elbow(ax, xe(x_s16), ys16[i*2], ys16[i*2+1], yse8[i], left=left, win=True)
        _hline(ax, end, nx(x_e8), yse8[i], win=True)

    # E8 — 1 pair
    ta, tb = e8[0], e8[1]
    pa, pb = _pair_probs(ta, tb, 'final_4', 'elite_8', rp)
    _box(ax, x_e8, yse8[0], ta, seed_map.get(ta, 0), pa, win=(ff == ta))
    _box(ax, x_e8, yse8[1], tb, seed_map.get(tb, 0), pb, win=(ff == tb))
    end = _elbow(ax, xe(x_e8), yse8[0], yse8[1], e8_mid, left=left, win=True)

    # E8 → FF column connector
    ff_target = L_FF if left else (R_FF + BOX_W)
    _hline(ax, end, ff_target, e8_mid, win=True)

    # Region label — top regions go above, bottom regions go below
    if y0 == TOP_Y0:
        ax.text(x_r64 + BOX_W / 2, y1 + 0.20, label,
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=TC, family=FONT, zorder=5)
    else:
        ax.text(x_r64 + BOX_W / 2, y0 - 0.20, label,
                ha='center', va='top', fontsize=8.5, fontweight='bold',
                color=TC, family=FONT, zorder=5)

    # Round headers (above top regions only)
    if y0 == TOP_Y0:
        for lbl, xc in [('R64', x_r64), ('R32', x_r32), ('S16', x_s16), ('E8', x_e8)]:
            ax.text(xc + BOX_W / 2, TOP_Y1 + 0.06, lbl,
                    ha='center', va='bottom', fontsize=6.5,
                    color=TC, fontweight='bold', family=FONT, zorder=5)

    return ff, e8_mid


# ── centre ────────────────────────────────────────────────────────────────────

def _draw_centre(ax, east_ff, east_ff_y, west_ff, west_ff_y,
                 south_ff, south_ff_y, midwest_ff, midwest_ff_y,
                 rp, seed_map):

    champ_l  = _win(east_ff,  west_ff,    'title_game', rp)
    champ_r  = _win(south_ff, midwest_ff, 'title_game', rp)
    champion = _win(champ_l, champ_r, 'champion', rp)

    # FF boxes — two pairs, each normalized to sum to 100%
    pa_l, pb_l = _pair_probs(east_ff, west_ff,    'title_game', 'final_4', rp)
    pa_r, pb_r = _pair_probs(south_ff, midwest_ff, 'title_game', 'final_4', rp)
    _box(ax, L_FF, east_ff_y,    east_ff,    seed_map.get(east_ff, 0),
         pa_l, win=(champ_l == east_ff))
    _box(ax, L_FF, west_ff_y,    west_ff,    seed_map.get(west_ff, 0),
         pb_l, win=(champ_l == west_ff))
    _box(ax, R_FF, south_ff_y,   south_ff,   seed_map.get(south_ff, 0),
         pa_r, win=(champ_r == south_ff))
    _box(ax, R_FF, midwest_ff_y, midwest_ff, seed_map.get(midwest_ff, 0),
         pb_r, win=(champ_r == midwest_ff))

    # Left FF elbow (East + West → championship left edge)
    end_l = _elbow(ax, L_FF + BOX_W, east_ff_y, west_ff_y, FF_CTR, left=True, win=True)
    _hline(ax, end_l, CHAMP, FF_CTR, win=True)
    # Route up from FF_CTR to top finalist y
    ax.plot([CHAMP, CHAMP], [FF_CTR, FIN_Y_A], color=WLINE, lw=0.85, zorder=2)

    # Right FF elbow (South + Midwest → championship right edge)
    end_r = _elbow(ax, R_FF, south_ff_y, midwest_ff_y, FF_CTR, left=False, win=True)
    _hline(ax, end_r, CHAMP + BOX_W, FF_CTR, win=True)
    # Route down from FF_CTR to bottom finalist y
    ax.plot([CHAMP + BOX_W, CHAMP + BOX_W], [FF_CTR, FIN_Y_B], color=WLINE, lw=0.85, zorder=2)

    # Championship finalist boxes — normalized pair
    p_fin_l, p_fin_r = _pair_probs(champ_l, champ_r, 'champion', 'title_game', rp)
    _box(ax, CHAMP, FIN_Y_A, champ_l, seed_map.get(champ_l, 0),
         p_fin_l, finalist=True, win=(champion == champ_l))
    _box(ax, CHAMP, FIN_Y_B, champ_r, seed_map.get(champ_r, 0),
         p_fin_r, finalist=True, win=(champion == champ_r))

    # Finalist elbow → routes down to champion box
    cx_fin = CHAMP + BOX_W + 0.07           # ≈ 15.17  (fits in gap before R_FF=15.35)
    fin_exit = cx_fin + 0.15                 # ≈ 15.32
    ax.plot([CHAMP + BOX_W, cx_fin], [FIN_Y_A, FIN_Y_A], color=GOLD, lw=1.1, zorder=2)
    ax.plot([CHAMP + BOX_W, cx_fin], [FIN_Y_B, FIN_Y_B], color=GOLD, lw=1.1, zorder=2)
    ax.plot([cx_fin, cx_fin],        [FIN_Y_A, FIN_Y_B], color=GOLD, lw=1.1, zorder=2)
    fin_mid = (FIN_Y_A + FIN_Y_B) / 2       # ≈ 6.75
    ax.plot([cx_fin, fin_exit], [fin_mid, fin_mid], color=GOLD, lw=1.1, zorder=2)
    # Vertical from fin_exit down to champion box top
    champ_top = CHAMP_Y + BOX_H / 2
    ax.plot([fin_exit, fin_exit], [fin_mid, champ_top], color=GOLD, lw=1.1, zorder=2)
    # Horizontal back to CHAMP right edge (enters champion box top-right)
    ax.plot([fin_exit, CHAMP + BOX_W], [champ_top, champ_top], color=GOLD, lw=1.1, zorder=2)

    # Champion box — show their normalized championship win probability
    p_champ = p_fin_l if champion == champ_l else p_fin_r
    _box(ax, CHAMP, CHAMP_Y, champion, seed_map.get(champion, 0),
         p_champ, champ=True)
    ax.text(CX, CHAMP_Y - BOX_H / 2 - 0.22, 'CHAMPION',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color=GOLD, family=FONT, zorder=5)

    # Column headers
    ax.text(L_FF + BOX_W / 2, TOP_Y1 + 0.06, 'F4',
            ha='center', va='bottom', fontsize=6.5,
            color=TC, fontweight='bold', family=FONT, zorder=5)
    ax.text(R_FF + BOX_W / 2, TOP_Y1 + 0.06, 'F4',
            ha='center', va='bottom', fontsize=6.5,
            color=TC, fontweight='bold', family=FONT, zorder=5)
    ax.text(CHAMP + BOX_W / 2, FIN_Y_A + BOX_H / 2 + 0.10, 'CHAMPIONSHIP',
            ha='center', va='bottom', fontsize=6.5,
            color=TC, fontweight='bold', family=FONT, zorder=5)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sim', default='results/sim_results.json',
                    help='simulation JSON (default: results/sim_results.json)')
    ap.add_argument('--out', default='results/charts/bracket_2026.png',
                    help='output PNG path')
    args = ap.parse_args()

    with open(args.sim) as f:
        data = json.load(f)

    bracket  = sorted(data['bracket'], key=lambda x: x['slot'])
    rp       = data['round_probs']
    cp_list  = data['champion_probs']
    season   = data.get('season', 2026)
    sims     = data.get('sims', 0)
    seed_map = {b['team']: (b.get('seed') or 0) for b in bracket}

    def reg(s1, s2):
        return [b for b in bracket if s1 <= b['slot'] <= s2]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')

    # Title
    ax.text(CX, FIG_H - 0.18, f'{season} NCAA Tournament — Predicted Bracket',
            ha='center', va='top', fontsize=14, fontweight='bold',
            color=FG, family=FONT, zorder=5)
    ax.text(CX, FIG_H - 0.68,
            f'{sims:,} Monte Carlo simulations  ·  % = win probability for that game  ·  @jgamblin',
            ha='center', va='top', fontsize=7, color=DIM, family=FONT, zorder=5)

    # Subtle divider between top and bottom halves
    mid_y = (TOP_Y0 + BOT_Y1) / 2
    ax.axhline(y=mid_y, xmin=0, xmax=1, color='#1e1e38', lw=0.8, zorder=1)

    # ── draw all four regions ─────────────────────────────────────────────────
    east_ff, east_ff_y = _draw_region(
        ax, reg(1, 16),  rp, L_R64, L_R32, L_S16, L_E8,
        TOP_Y0, TOP_Y1, True, 'EAST')
    west_ff, west_ff_y = _draw_region(
        ax, reg(17, 32), rp, L_R64, L_R32, L_S16, L_E8,
        BOT_Y0, BOT_Y1, True, 'WEST')
    south_ff, south_ff_y = _draw_region(
        ax, reg(33, 48), rp, R_R64, R_R32, R_S16, R_E8,
        TOP_Y0, TOP_Y1, False, 'SOUTH')
    midwest_ff, midwest_ff_y = _draw_region(
        ax, reg(49, 64), rp, R_R64, R_R32, R_S16, R_E8,
        BOT_Y0, BOT_Y1, False, 'MIDWEST')

    # ── centre: Final Four + Championship ─────────────────────────────────────
    _draw_centre(ax, east_ff, east_ff_y, west_ff, west_ff_y,
                 south_ff, south_ff_y, midwest_ff, midwest_ff_y,
                 rp, seed_map)

    # ── save ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.1)
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved → {out}  ({out.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
