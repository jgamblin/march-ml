#!/usr/bin/env python3
"""
Apply default D1 tagging to `data/processed/features/teams.csv` by marking
`is_d1 = not team_id.startswith('nd-')` and rewrite the file.
"""
import pandas as pd
from pathlib import Path

p = Path('data/processed/features/teams.csv')
if not p.exists():
    raise SystemExit('Missing teams.csv')

df = pd.read_csv(p)
df['is_d1'] = ~df['team_id'].astype(str).str.startswith('nd-')
df.to_csv(p, index=False)
print(f'Updated {p} with default is_d1 (rows={len(df)})')
