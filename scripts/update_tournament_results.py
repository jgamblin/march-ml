#!/usr/bin/env python3
"""
update_tournament_results.py

Reads scraped game data (games_2026.csv) and sim_results.json, then writes
tournament_results.json with actual game winners for the web bracket display.

tournament_results.json format:
  first_four:   list of winner names indexed by first_four game order
  round_of_64:  list of winner names indexed by bracket game order (0-31)
  round_of_32:  list of winner names indexed by bracket game order (0-15)
  ... etc.

Run after scrape step so game data is current.
"""
import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


ROUND_KEYS = ['round_of_64', 'round_of_32', 'sweet_16', 'elite_8', 'final_4', 'champion']

# cbbpy tournament name patterns → our round keys
TOURNAMENT_ROUND_MAP = {
    'first four':    'first_four',
    'first round':   'round_of_64',
    'second round':  'round_of_32',
    'sweet 16':      'sweet_16',
    'sweet sixteen': 'sweet_16',
    'elite 8':       'elite_8',
    'elite eight':   'elite_8',
    'final four':    'final_4',
    'semifinal':     'final_4',
    'championship':  'champion',
    'national championship': 'champion',
}


def normalize(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def match_team(name: str, candidates: list[str]) -> str | None:
    """Find the best matching team from candidates list."""
    n = normalize(name)
    for c in candidates:
        if normalize(c) == n:
            return c
    # Substring match fallback
    for c in candidates:
        if n in normalize(c) or normalize(c) in n:
            return c
    return None


def classify_round(tournament_str: str) -> str | None:
    """Map a tournament string from cbbpy to our round key."""
    t = tournament_str.lower()
    for key, val in TOURNAMENT_ROUND_MAP.items():
        if key in t:
            return val
    return None


def load_tournament_games(games_csv: Path, season: int) -> pd.DataFrame:
    """Load and filter to completed NCAA tournament games for the given season."""
    df = pd.read_csv(games_csv)
    # Filter to NCAA tournament games with final scores
    ncaa = df[
        df['tournament'].str.contains('NCAA Men', na=False) &
        (df['game_status'] == 'Final') &
        (df['home_score'] > 0)
    ].copy()
    return ncaa


def build_results(sim_path: Path, games_csv: Path, season: int) -> dict:
    with open(sim_path) as f:
        sim = json.load(f)

    bracket = sorted(sim['bracket'], key=lambda x: x['slot'])
    ff_games = sim.get('first_four', [])
    all_bracket_teams = [t['team'] for t in bracket]
    # All teams including FF losers
    all_teams = all_bracket_teams + [t for g in ff_games for t in [g['teamA'], g['teamB']]]

    games = load_tournament_games(games_csv, season)

    results = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'first_four': [],
        'round_of_64': [],
        'round_of_32': [],
        'sweet_16': [],
        'elite_8': [],
        'final_4': [],
        'champion': [],
    }

    # ── First Four ────────────────────────────────────────────────────────────
    ff_results = {}  # slot -> winner name
    for _, row in games.iterrows():
        rnd = classify_round(row['tournament'])
        if rnd != 'first_four':
            continue
        home = match_team(row['home_team'], all_teams)
        away = match_team(row['away_team'], all_teams)
        if not home or not away:
            continue
        winner = home if row['home_score'] > row['away_score'] else away
        # Find which FF game this is
        for g in ff_games:
            if {normalize(g['teamA']), normalize(g['teamB'])} == {normalize(home), normalize(away)}:
                ff_results[g['slot']] = winner
                break

    # Output FF results in the same order as sim's first_four array
    results['first_four'] = [ff_results.get(g['slot']) for g in ff_games]

    # ── Main bracket rounds ───────────────────────────────────────────────────
    # Build game index lookup: (teamA, teamB) -> (round_key, game_idx)
    # We simulate what computeBracket does: pair teams in sorted slot order
    current = [t['team'] for t in bracket]
    game_lookup = {}  # frozenset({tA, tB}) -> (round_key, game_idx)

    for rnd_key in ROUND_KEYS:
        for gi in range(len(current) // 2):
            tA = current[gi * 2]
            tB = current[gi * 2 + 1]
            game_lookup[frozenset({normalize(tA), normalize(tB)})] = (rnd_key, gi)
        # Advance: predicted winner moves on (we'll fill in actuals below,
        # but we need to iterate so later rounds reference actual winners)
        current = [current[i * 2] for i in range(len(current) // 2)]  # placeholder

    # Now match scraped games to game indices
    round_game_results = {k: {} for k in ROUND_KEYS}

    for _, row in games.iterrows():
        rnd = classify_round(row['tournament'])
        if rnd not in ROUND_KEYS:
            continue
        # Try to find both teams in the bracket
        home = match_team(row['home_team'], all_bracket_teams)
        away = match_team(row['away_team'], all_bracket_teams)
        if not home or not away:
            continue
        key = frozenset({normalize(home), normalize(away)})
        if key not in game_lookup:
            continue
        lookup_rnd, gi = game_lookup[key]
        winner = home if row['home_score'] > row['away_score'] else away
        round_game_results[lookup_rnd][gi] = winner

    # Convert dicts to ordered lists (None for games not yet played)
    for rnd_key in ROUND_KEYS:
        game_dict = round_game_results[rnd_key]
        if not game_dict:
            continue
        max_idx = max(game_dict.keys())
        results[rnd_key] = [game_dict.get(i) for i in range(max_idx + 1)]

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim', default='results/sim_results.json')
    p.add_argument('--games_csv', default='data/processed/games_2026.csv')
    p.add_argument('--out', default='results/tournament_results.json')
    p.add_argument('--season', type=int, default=2026)
    args = p.parse_args()

    results = build_results(Path(args.sim), Path(args.games_csv), args.season)

    # Count filled entries
    filled = sum(1 for v in results['first_four'] if v) + \
             sum(1 for rnd in ROUND_KEYS for v in results[rnd] if v)
    print(f"Found {filled} completed game results")
    for key in ['first_four'] + ROUND_KEYS:
        vals = results[key]
        done = [v for v in vals if v]
        if done:
            print(f"  {key}: {len(done)} results → {done}")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Written to {args.out}")


if __name__ == '__main__':
    main()
