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
    '1st round':     'round_of_64',
    'second round':  'round_of_32',
    '2nd round':     'round_of_32',
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


def _find_matching_pairing(teams: list[str], actual_games: list[tuple]) -> list[tuple] | None:
    """
    For a small set of teams (4 or fewer), try every possible way to pair them
    into games and return the pairing that matches the actual games from CSV.

    Returns a list of (teamA, teamB, game_idx) triples in game_idx order,
    or None if no exact match is found (bracket not yet set / no data).
    """
    n = len(teams)
    if n == 2:
        # Only one possible pairing
        pair = frozenset({normalize(teams[0]), normalize(teams[1])})
        actual_pairs = {frozenset({normalize(h), normalize(a)}) for h, a, w in actual_games}
        if pair in actual_pairs:
            return [(teams[0], teams[1], 0)]
        return None
    if n == 4:
        # Three possible pairings of 4 teams into 2 games
        for (i, j), (k, l) in [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]:
            p0 = frozenset({normalize(teams[i]), normalize(teams[j])})
            p1 = frozenset({normalize(teams[k]), normalize(teams[l])})
            actual_pairs = {frozenset({normalize(h), normalize(a)}) for h, a, w in actual_games}
            if p0 in actual_pairs and p1 in actual_pairs:
                return [(teams[i], teams[j], 0), (teams[k], teams[l], 1)]
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
    # Drop duplicate rows (cbbpy sometimes stores each game twice)
    ncaa = ncaa.drop_duplicates(subset=['home_team', 'away_team', 'home_score', 'away_score'])
    return ncaa


def build_results(sim_path: Path, games_csv: Path, season: int) -> dict:
    with open(sim_path) as f:
        sim = json.load(f)

    bracket = sorted(sim['bracket'], key=lambda x: x['slot'])
    ff_games = sim.get('first_four', [])
    all_bracket_teams = [t['team'] for t in bracket]
    all_teams = all_bracket_teams + [t for g in ff_games for t in [g['teamA'], g['teamB']]]

    games = load_tournament_games(games_csv, season)

    # Index scraped games by round: round_key -> list of (home, away, winner)
    games_by_round: dict[str, list[tuple[str, str, str]]] = {k: [] for k in ['first_four'] + ROUND_KEYS}
    for _, row in games.iterrows():
        rnd = classify_round(row['tournament'])
        if rnd not in games_by_round:
            continue
        home = match_team(row['home_team'], all_teams)
        away = match_team(row['away_team'], all_teams)
        if not home or not away:
            continue
        winner = home if row['home_score'] > row['away_score'] else away
        games_by_round[rnd].append((home, away, winner))

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
    ff_results: dict[int, str] = {}
    for home, away, winner in games_by_round['first_four']:
        for g in ff_games:
            if {normalize(g['teamA']), normalize(g['teamB'])} == {normalize(home), normalize(away)}:
                ff_results[g['slot']] = winner
                break
    results['first_four'] = [ff_results.get(g['slot']) for g in ff_games]

    # ── Main bracket rounds — process iteratively so each round's expected
    #   matchups are built from actual winners of the previous round. ─────────
    current = [t['team'] for t in bracket]  # 64 teams in slot order

    # Replace First Four placeholder teams with their actual winners so that
    # round_of_64 expected matchups use the correct opponent names.
    for g in ff_games:
        actual_winner = ff_results.get(g['slot'])
        if not actual_winner:
            continue
        # The slot of a First Four game IS the bracket slot of the winner
        idx = next((i for i, t in enumerate(bracket) if t['slot'] == g['slot']), None)
        if idx is not None:
            current[idx] = actual_winner

    for rnd_key in ROUND_KEYS:
        n_games = len(current) // 2

        # ── Special handling for Final 4 and Championship: the bracket may cross
        #   regions in a way that doesn't match sequential slot order. Try all
        #   possible pairings and use the one that matches actual CSV games. ───
        if n_games <= 2 and games_by_round[rnd_key]:
            matched = _find_matching_pairing(current, games_by_round[rnd_key])
            if matched:
                # Look up actual winners for the matched pairings
                actual_pairs_map = {}
                for home, away, winner in games_by_round[rnd_key]:
                    actual_pairs_map[frozenset({normalize(home), normalize(away)})] = winner

                game_dict = {}
                reordered = []
                for tA, tB, gi in matched:
                    key = frozenset({normalize(tA), normalize(tB)})
                    winner = actual_pairs_map.get(key)
                    if winner:
                        game_dict[gi] = winner
                    reordered.append((tA, tB))

                if game_dict:
                    results[rnd_key] = [game_dict.get(i) for i in range(n_games)]

                # Advance current using actual winners from the matched pairing
                next_current = []
                for tA, tB, gi in matched:
                    key = frozenset({normalize(tA), normalize(tB)})
                    actual_winner = actual_pairs_map.get(key)
                    next_current.append(actual_winner or tA)
                current = next_current
                continue

        # Build lookup: frozenset({normA, normB}) -> game_index for THIS round
        expected: dict[frozenset, int] = {}
        for gi in range(n_games):
            tA = current[gi * 2]
            tB = current[gi * 2 + 1]
            expected[frozenset({normalize(tA), normalize(tB)})] = gi

        # Match scraped games against expected pairings
        game_dict: dict[int, str] = {}
        for home, away, winner in games_by_round[rnd_key]:
            key = frozenset({normalize(home), normalize(away)})
            gi = expected.get(key)
            if gi is not None:
                game_dict[gi] = winner

        # Write ordered results list
        if game_dict:
            max_idx = max(game_dict.keys())
            results[rnd_key] = [game_dict.get(i) for i in range(max(max_idx + 1, n_games))]
        else:
            results[rnd_key] = []

        # Advance current using ACTUAL winners (fall back to left-team prediction)
        next_current = []
        for gi in range(n_games):
            actual_winner = game_dict.get(gi)
            predicted = current[gi * 2]  # higher slot / left side
            next_current.append(actual_winner or predicted)
        current = next_current

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
