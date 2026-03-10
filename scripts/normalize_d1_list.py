#!/usr/bin/env python3
"""
Normalize a raw D1 team list (one school per line) to the repo's `team`/`team_id` values
by fuzzy-matching against `data/processed/features/teams.csv` and producing a CSV
with columns `team` and `team_id` suitable for `prepare_features.py --d1_list`.

Usage:
  python3 scripts/normalize_d1_list.py --raw data/mappings/d1_list.txt --teams data/processed/features/teams.csv --out data/mappings/d1_list_normalized.csv

Dependencies: python stdlib + pandas
"""
import argparse
import difflib
import pandas as pd


def best_match(name, choices, cutoff=0.6):
    name_norm = name.strip().lower()
    matches = difflib.get_close_matches(name_norm, choices.keys(), n=1, cutoff=cutoff)
    if matches:
        k = matches[0]
        return choices[k]
    return None


def normalize(raw_path, teams_csv, out_path):
    teams = pd.read_csv(teams_csv)
    # build mapping from normalized team name -> team_id
    # Prefer matching against known Division-I candidates: team_id not starting with 'nd-'
    mask = ~teams['team_id'].astype(str).str.startswith('nd-')
    choices = {t.strip().lower(): tid for t, tid in zip(teams.loc[mask, 'team'].astype(str), teams.loc[mask, 'team_id'].astype(str))}
    # deduplicate choices (keep first occurrence)
    # read raw list
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_lines = [l.strip() for l in f.readlines() if l.strip()]

    rows = []
    unmatched = []
    for r in raw_lines:
        match = best_match(r, choices, cutoff=0.6)
        if match is not None:
            rows.append({'team': r, 'team_id': match})
        else:
            # try looser cutoff
            match2 = best_match(r, choices, cutoff=0.45)
            if match2 is not None:
                rows.append({'team': r, 'team_id': match2})
            else:
                rows.append({'team': r, 'team_id': ''})
                unmatched.append(r)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for u in unmatched[:20]:
            print("  ", u)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--teams', default='data/processed/features/teams.csv')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    normalize(args.raw, args.teams, args.out)
