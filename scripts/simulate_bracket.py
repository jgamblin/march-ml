"""Simulate tournaments using trained models.

By default this script selects the top 64 Division-I teams by pre-tournament
`adj_margin` from the latest season in `data/processed/features/tournament_teams.csv`
and runs Monte Carlo simulations. Output includes champion odds, round reach
probabilities, and bracket metadata.

Supported custom bracket formats:
- plain text: one team per line in bracket order
- CSV: `team` plus optional `slot`, `seed`, and `region`
- JSON: either a list of team records or an object with a `teams` list
"""
import argparse
import gc
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import bracket parsing module
try:
    from parse_bracket import parse_ncaa_bracket, validate_bracket
except ImportError:
    from scripts.parse_bracket import parse_ncaa_bracket, validate_bracket


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_models(models_dir='models'):
    lr_path = Path(models_dir) / 'lr_model.joblib'
    xgb_path = Path(models_dir) / 'xgb_model.joblib'
    feat_path = Path(models_dir) / 'model_features.joblib'
    if not lr_path.exists() or not feat_path.exists():
        raise FileNotFoundError('Required model artifacts not found in models/')
    # load base models
    base_lr = joblib.load(lr_path)
    base_xgb = None
    if xgb_path.exists():
        try:
            base_xgb = joblib.load(xgb_path)
        except Exception:
            base_xgb = None
    # load calibrators if present
    lr_cal = None
    xgb_cal = None
    if (Path(models_dir) / 'lr_cal.joblib').exists():
        lr_cal = joblib.load(Path(models_dir) / 'lr_cal.joblib')
    elif (Path(models_dir) / 'lr_platt.joblib').exists():
        lr_cal = joblib.load(Path(models_dir) / 'lr_platt.joblib')
    if (Path(models_dir) / 'xgb_cal.joblib').exists():
        xgb_cal = joblib.load(Path(models_dir) / 'xgb_cal.joblib')
    elif (Path(models_dir) / 'xgb_platt.joblib').exists():
        xgb_cal = joblib.load(Path(models_dir) / 'xgb_platt.joblib')
    features = joblib.load(feat_path)
    return base_lr, base_xgb, lr_cal, xgb_cal, features


def load_ensemble_weights(models_dir='models', default_lr=0.5, default_xgb=0.5):
    """Load optimized ensemble weights from models/ensemble_weights.json if present."""
    weights_path = Path(models_dir) / 'ensemble_weights.json'
    if weights_path.exists():
        try:
            data = json.loads(weights_path.read_text())
            lr_w = float(data.get('lr_weight', default_lr))
            xgb_w = float(data.get('xgb_weight', default_xgb))
            print(f"  Loaded ensemble weights: LR={lr_w:.0%} / XGB={xgb_w:.0%} "
                  f"(acc={data.get('accuracy', '?'):.4f}, from {weights_path})")
            return lr_w, xgb_w
        except Exception as exc:
            print(f"  Warning: could not read {weights_path}: {exc}")
    return default_lr, default_xgb



    strata = ['chalk', 'competitive', 'balanced']
    loaded = {}
    base = Path(models_dir)
    for stratum in strata:
        lr_path = base / f'{stratum}_lr_model.joblib'
        feat_path = base / f'{stratum}_model_features.joblib'
        if not lr_path.exists() or not feat_path.exists():
            continue

        xgb_path = base / f'{stratum}_xgb_model.joblib'
        lr_cal_path = base / f'{stratum}_lr_cal.joblib'
        xgb_cal_path = base / f'{stratum}_xgb_cal.joblib'

        loaded[stratum] = {
            'base_lr': joblib.load(lr_path),
            'base_xgb': joblib.load(xgb_path) if xgb_path.exists() else None,
            'lr_cal': joblib.load(lr_cal_path) if lr_cal_path.exists() else None,
            'xgb_cal': joblib.load(xgb_cal_path) if xgb_cal_path.exists() else None,
            'feat_names': joblib.load(feat_path),
        }

    return loaded


def load_team_features(path='data/processed/features/tournament_teams.csv'):
    feature_path = Path(path)
    if not feature_path.exists() and feature_path.name == 'tournament_teams.csv':
        feature_path = feature_path.with_name('teams.csv')
    return pd.read_csv(feature_path)


def coerce_slot(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def normalize_bracket_records(records):
    normalized = []
    seen = set()
    for idx, record in enumerate(records, start=1):
        if isinstance(record, str):
            team = record.strip()
            slot = idx
            seed = None
            region = None
        else:
            team = str(record.get('team', '') or record.get('name', '')).strip()
            slot = coerce_slot(record.get('slot', idx), idx)
            seed = record.get('seed')
            region = record.get('region')
        if not team:
            continue
        if team in seen:
            raise ValueError(f'Duplicate team in bracket input: {team}')
        seen.add(team)
        normalized.append(
            {
                'slot': slot,
                'team': team,
                'seed': None if pd.isna(seed) else seed,
                'region': None if pd.isna(region) else region,
            }
        )

    normalized = sorted(normalized, key=lambda item: item['slot'])
    teams = [item['team'] for item in normalized]
    if not teams:
        raise ValueError('No teams were found in the bracket input')
    if len(teams) != len(set(teams)):
        raise ValueError('Bracket input contains duplicate team names')
    if len(teams) & (len(teams) - 1) != 0:
        raise ValueError('Bracket size must be a power of two (e.g. 4, 8, 16, 32, 64)')
    return teams, normalized


def load_bracket(bracket_file):
    path = Path(bracket_file)
    if not path.exists():
        raise FileNotFoundError(f'Bracket file not found: {bracket_file}')

    suffix = path.suffix.lower()
    if suffix == '.json':
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        records = payload.get('teams', payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError('JSON bracket must be a list or an object with a `teams` list')
        teams, normalized = normalize_bracket_records(records)
        return teams, normalized, 'json'

    if suffix == '.csv':
        frame = pd.read_csv(path)
        if 'team' not in frame.columns and 'name' not in frame.columns:
            raise ValueError('CSV bracket must contain a `team` column')
        teams, normalized = normalize_bracket_records(frame.to_dict(orient='records'))
        return teams, normalized, 'csv'

    with open(path, 'r', encoding='utf-8') as handle:
        records = [line.strip() for line in handle if line.strip()]
    teams, normalized = normalize_bracket_records(records)
    return teams, normalized, 'text'


def compute_sos_for_season(team_feats, season, games_dir='data/processed'):
    # compute simple SOS = avg opponent final win_pct using team_feats win_pct as final
    df = team_feats[team_feats['season'] == season].copy()
    win_map = {(season, str(r['team']).strip()): float(r['win_pct']) for _, r in df.iterrows()}
    opps = {k: [] for k in win_map.keys()}
    games_file = Path(games_dir) / f'games_{season}.csv'
    if not games_file.exists():
        return team_feats
    gdf = pd.read_csv(games_file)
    for _, g in gdf.iterrows():
        home = str(g.get('home_team','')).strip()
        away = str(g.get('away_team','')).strip()
        hkey = (season, home)
        akey = (season, away)
        if hkey in opps and akey in win_map:
            opps[hkey].append(win_map[akey])
        if akey in opps and hkey in win_map:
            opps[akey].append(win_map[hkey])
    # compute mean
    sos_vals = {}
    for k, lst in opps.items():
        sos_vals[k[1]] = float(np.mean(lst)) if len(lst) else 0.0
    # attach to team_feats
    team_feats = team_feats.copy()
    team_feats['sos'] = team_feats.apply(lambda r: sos_vals.get(str(r['team']).strip(), 0.0) if r['season']==season else 0.0, axis=1)
    return team_feats


def get_team_row(df, season, team_name):
    # exact match first
    # Normalize team names for comparison
    team_name_clean = team_name.strip().lower()
    df_season = df[df['season'] == season]
    
    if len(df_season) == 0:
        return None
    
    # Use apply to avoid pandas string API compatibility issues
    def match_team(t):
        return str(t).strip().lower() == team_name_clean
    
    # Exact match first
    mask = df_season['team'].apply(match_team)
    row = df_season[mask]
    
    if len(row) == 0:
        # Fallback: contains match
        def contains_team(t):
            return team_name_clean in str(t).strip().lower()
        mask = df_season['team'].apply(contains_team)
        row = df_season[mask]
    
    if len(row) == 0:
        return None
    return row.iloc[0]


def apply_team_overrides(row, overrides):
    if row is None or not overrides:
        return row
    updated = row.copy()
    for key in ['seed', 'region', 'slot']:
        value = overrides.get(key)
        if value is not None and not pd.isna(value):
            updated[key] = value
    return updated


def coerce_numeric_feature(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        tier_map = {
            'power-6': 3.0,
            'high-major': 2.0,
            'mid-major': 1.0,
        }
        if lowered in tier_map:
            return tier_map[lowered]
    try:
        return float(value)
    except Exception:
        return 0.0


def add_interaction_features(values):
    diff_seed = float(values.get('diff_seed', 0.0))
    diff_adj_margin = float(values.get('diff_adj_margin', 0.0))
    diff_form_rating = float(values.get('diff_form_rating', 0.0))
    diff_conf_avg_adj_margin = float(values.get('diff_conf_avg_adj_margin', 0.0))
    diff_sos_win_pct = float(values.get('diff_sos_win_pct', 0.0))
    diff_last10_momentum = float(values.get('diff_last10_momentum', 0.0))
    neutral_site = float(values.get('neutral_site', 1.0))

    values.update(
        {
            'seed_diff_abs': abs(diff_seed),
            'seed_adj_margin_interaction': diff_seed * diff_adj_margin,
            'seed_form_interaction': diff_seed * diff_form_rating,
            'seed_conf_interaction': diff_seed * diff_conf_avg_adj_margin,
            'seed_sos_interaction': diff_seed * diff_sos_win_pct,
            'adj_margin_form_interaction': diff_adj_margin * diff_form_rating,
            'momentum_conf_interaction': diff_last10_momentum * diff_conf_avg_adj_margin,
            'momentum_sos_interaction': diff_form_rating * diff_sos_win_pct,
            'neutral_form_interaction': neutral_site * diff_form_rating,
            'neutral_seed_interaction': neutral_site * diff_seed,
        }
    )
    return values


def make_feature_vector(home_row, away_row, feat_cols, neutral_site=True):
    values = {}
    for col in feat_cols:
        if col.startswith('diff_'):
            base_col = col[len('diff_'):]
            values[col] = coerce_numeric_feature(home_row.get(base_col, 0.0)) - coerce_numeric_feature(away_row.get(base_col, 0.0))
        elif col == 'neutral_site':
            values[col] = float(neutral_site)
        elif col == 'is_tournament':
            values[col] = 1.0  # all simulated games are tournament games
        elif col == 'home_edge':
            values[col] = 0.0 if neutral_site else 1.0
        else:
            values[col] = 0.0
    values = add_interaction_features(values)
    return pd.DataFrame([values], columns=feat_cols)


def make_feature_matrix(home_rows, away_rows, feat_cols, neutral_site=True):
    """Build a 2D numpy feature matrix for a batch of matchups.

    Returns a (n, len(feat_cols)) float64 array, avoiding per-row DataFrame
    overhead so the calibrators can be called once for the entire batch.
    """
    col_idx = {col: i for i, col in enumerate(feat_cols)}
    n = len(home_rows)
    matrix = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for j, (hr, ar) in enumerate(zip(home_rows, away_rows)):
        values = {}
        for col in feat_cols:
            if col.startswith('diff_'):
                base_col = col[len('diff_'):]
                values[col] = coerce_numeric_feature(hr.get(base_col, 0.0)) - coerce_numeric_feature(ar.get(base_col, 0.0))
            elif col == 'neutral_site':
                values[col] = float(neutral_site)
            elif col == 'is_tournament':
                values[col] = 1.0
            elif col == 'home_edge':
                values[col] = 0.0 if neutral_site else 1.0
            else:
                values[col] = 0.0
        values = add_interaction_features(values)
        for col, idx in col_idx.items():
            matrix[j, idx] = values.get(col, 0.0)
    return matrix


def _batch_model_predict(base_model, calibrator, mat):
    """Predict probabilities for a batch feature matrix using calibrator or base model."""
    if calibrator is not None:
        try:
            return calibrator.predict_proba(mat)[:, 1]
        except Exception:
            pass
    if base_model is not None:
        try:
            return base_model.predict_proba(mat)[:, 1]
        except Exception:
            pass
    return None


def batch_predict_prob(models_dict, feat_names, home_rows, away_rows, lr_weight=0.65, xgb_weight=0.35):
    """Predict win probabilities for a batch of matchups with a single predict_proba call per model."""
    n = len(home_rows)
    if n == 0:
        return np.array([])
    mat = make_feature_matrix(home_rows, away_rows, feat_names)
    ps_lr = _batch_model_predict(models_dict['base_lr'], models_dict['lr_cal'], mat)
    ps_xgb = _batch_model_predict(models_dict['base_xgb'], models_dict['xgb_cal'], mat)
    if ps_lr is None and ps_xgb is None:
        return np.full(n, 0.5)
    if ps_lr is None:
        return ps_xgb
    if ps_xgb is None:
        return ps_lr
    total = max(1e-9, float(lr_weight) + float(xgb_weight))
    return (float(lr_weight) / total) * ps_lr + (float(xgb_weight) / total) * ps_xgb


def batch_predict_prob_routed(default_models, seed_models, feat_names, home_rows, away_rows, lr_weight, xgb_weight):
    """Batch prediction with optional seed-stratified model routing."""
    n = len(home_rows)
    if n == 0:
        return np.array([])
    if not seed_models:
        return batch_predict_prob(default_models, feat_names, home_rows, away_rows, lr_weight, xgb_weight)
    # Group matchups by seed stratum then batch-predict each group
    strata_indices = {}
    for i, (hr, ar) in enumerate(zip(home_rows, away_rows)):
        stratum = get_seed_stratum(hr, ar)
        strata_indices.setdefault(stratum, []).append(i)
    probs = np.full(n, 0.5)
    for stratum, indices in strata_indices.items():
        selected = (seed_models.get(stratum) if stratum else None) or default_models
        batch_h = [home_rows[i] for i in indices]
        batch_a = [away_rows[i] for i in indices]
        batch_probs = batch_predict_prob(selected, feat_names, batch_h, batch_a, lr_weight, xgb_weight)
        for idx, p in zip(indices, batch_probs):
            probs[idx] = p
    return probs


def predict_model_probability(base_model, calibrator, vec, vec_np):
    raw_prob = None
    if base_model is not None:
        try:
            raw_prob = base_model.predict_proba(vec)[:, 1]
        except Exception:
            try:
                raw_prob = base_model.predict_proba(vec_np)[:, 1]
            except Exception:
                raw_prob = None

    if calibrator is not None:
        for candidate in (vec, vec_np):
            try:
                return calibrator.predict_proba(candidate)[:, 1]
            except Exception:
                continue
        if raw_prob is not None:
            try:
                return calibrator.predict_proba(np.asarray(raw_prob).reshape(-1, 1))[:, 1]
            except Exception:
                pass

    return raw_prob


def predict_prob(base_lr, base_xgb, lr_cal, xgb_cal, feat_names, home_row, away_row, lr_weight=0.65, xgb_weight=0.35):
    vec = make_feature_vector(home_row, away_row, feat_names, neutral_site=True)
    vec_np = vec.values
    ps = []

    p_lr = predict_model_probability(base_lr, lr_cal, vec, vec_np)
    if p_lr is not None:
        ps.append(p_lr)

    p_xgb = predict_model_probability(base_xgb, xgb_cal, vec, vec_np)
    if p_xgb is not None:
        ps.append(p_xgb)

    if not ps:
        return 0.5

    if len(ps) == 1:
        p = float(ps[0][0])
        return p

    total_weight = max(1e-9, float(lr_weight) + float(xgb_weight))
    lr_w = float(lr_weight) / total_weight
    xgb_w = float(xgb_weight) / total_weight
    p = float((lr_w * p_lr[0]) + (xgb_w * p_xgb[0]))
    return p


def get_seed_stratum(home_row, away_row):
    try:
        home_seed = coerce_numeric_feature(home_row.get('seed', np.nan))
        away_seed = coerce_numeric_feature(away_row.get('seed', np.nan))
        seed_diff_abs = abs(home_seed - away_seed)
    except Exception:
        return None

    if seed_diff_abs >= 9:
        return 'chalk'
    if seed_diff_abs <= 4:
        return 'competitive'
    return 'balanced'


def predict_prob_with_seed_routing(
    default_models,
    seed_models,
    home_row,
    away_row,
    lr_weight=0.65,
    xgb_weight=0.35,
):
    selected_models = None
    if seed_models:
        stratum = get_seed_stratum(home_row, away_row)
        if stratum is not None:
            selected_models = seed_models.get(stratum)

    model_set = selected_models or default_models
    return predict_prob(
        model_set['base_lr'],
        model_set['base_xgb'],
        model_set['lr_cal'],
        model_set['xgb_cal'],
        model_set['feat_names'],
        home_row,
        away_row,
        lr_weight=lr_weight,
        xgb_weight=xgb_weight,
    )


def round_label(team_count):
    labels = {
        64: 'round_of_64',
        32: 'round_of_32',
        16: 'sweet_16',
        8: 'elite_8',
        4: 'final_4',
        2: 'title_game',
        1: 'champion',
    }
    return labels.get(team_count, f'round_of_{team_count}')


def simulate_once(
    bracket_teams,
    season,
    base_lr,
    base_xgb,
    lr_cal,
    xgb_cal,
    feat_names,
    team_feats,
    team_overrides=None,
    lr_weight=0.65,
    xgb_weight=0.35,
    seed_models=None,
):
    teams = list(bracket_teams)
    reach_counts = {team: set([round_label(len(teams))]) for team in teams}
    rounds = int(np.log2(len(teams)))
    for _ in range(rounds):
        next_round = []
        next_label = round_label(len(teams) // 2)
        for i in range(0, len(teams), 2):
            home = teams[i]
            away = teams[i + 1]
            home_row = apply_team_overrides(get_team_row(team_feats, season, home), (team_overrides or {}).get(home))
            away_row = apply_team_overrides(get_team_row(team_feats, season, away), (team_overrides or {}).get(away))
            if home_row is None or away_row is None:
                winner = home if away_row is None else away
                if home_row is not None and away_row is not None:
                    winner = home if float(home_row.get('win_pct', 0.0)) >= float(away_row.get('win_pct', 0.0)) else away
            else:
                # Orient so team_A (home argument) = stronger team by adj_margin or seed
                home_adj = float(home_row.get('adj_margin', 0.0)) if home_row is not None else 0.0
                away_adj = float(away_row.get('adj_margin', 0.0)) if away_row is not None else 0.0
                home_seed_val = float(home_row.get('seed', 99)) if home_row is not None else 99
                away_seed_val = float(away_row.get('seed', 99)) if away_row is not None else 99
                # Use seed to orient if available, else adj_margin
                if home_seed_val != 99 or away_seed_val != 99:
                    # Lower seed number = better team = team_A
                    if away_seed_val < home_seed_val:
                        home_row, away_row = away_row, home_row
                        flipped = True
                    else:
                        flipped = False
                else:
                    if away_adj > home_adj:
                        home_row, away_row = away_row, home_row
                        flipped = True
                    else:
                        flipped = False
                p_home = predict_prob_with_seed_routing(
                    {
                        'base_lr': base_lr,
                        'base_xgb': base_xgb,
                        'lr_cal': lr_cal,
                        'xgb_cal': xgb_cal,
                        'feat_names': feat_names,
                    },
                    seed_models,
                    home_row,
                    away_row,
                    lr_weight=lr_weight,
                    xgb_weight=xgb_weight,
                )
                # If we flipped, p_home is now P(original away wins); invert to get P(original home wins)
                if flipped:
                    p_home = 1.0 - p_home
                winner = home if np.random.rand() < p_home else away
            next_round.append(winner)
            reach_counts[winner].add(next_label)
        teams = next_round
    champ = teams[0]
    return champ, reach_counts


def demo_bracket_from_top64_with_options(team_feats, season, min_games=10, allow_nd=False):
    """Selects top 64 teams by adj_margin (schedule-adjusted margin, more reliable than raw win%)."""
    df = team_feats[team_feats['season'] == season].copy()
    df = df.sort_values('adj_margin', ascending=False)
    # if `is_d1` exists, prefer those; otherwise prefer team_id not starting with 'nd-'
    if 'is_d1' in df.columns:
        primary = df[(df['is_d1']) & (df['games_played'] >= min_games)].sort_values('adj_margin', ascending=False)
        if len(primary) >= 64:
            return primary['team'].astype(str).tolist()[:64]
        if not allow_nd:
            secondary = df[(df['is_d1'])].sort_values('adj_margin', ascending=False)
            if len(secondary) >= 64:
                return secondary['team'].astype(str).tolist()[:64]
    else:
        df['team_id_str'] = df['team_id'].astype(str)
        df['is_nd'] = df['team_id_str'].str.startswith('nd-')
        primary = df[(~df['is_nd']) & (df['games_played'] >= min_games)].sort_values('adj_margin', ascending=False)
        if len(primary) >= 64:
            return primary['team'].astype(str).tolist()[:64]
        if not allow_nd:
            secondary = df[~df['is_nd']].sort_values('adj_margin', ascending=False)
            if len(secondary) >= 64:
                return secondary['team'].astype(str).tolist()[:64]

    # tertiary: include teams with enough games regardless of D1 tag
    tertiary = df[df['games_played'] >= min_games].sort_values('adj_margin', ascending=False)
    if len(tertiary) >= 64:
        return tertiary['team'].astype(str).tolist()[:64]

    # fallback: allow ND teams if allowed, else pure top-by-adj_margin
    if allow_nd:
        fallback = df.sort_values('adj_margin', ascending=False)
        if len(fallback) >= 64:
            return fallback['team'].astype(str).tolist()[:64]
    else:
        if 'is_d1' in df.columns:
            fallback = df[df['is_d1']].sort_values('adj_margin', ascending=False)
        else:
            fallback = df[~df['team_id'].astype(str).str.startswith('nd-')].sort_values('adj_margin', ascending=False)
        if len(fallback) >= 64:
            return fallback['team'].astype(str).tolist()[:64]

    raise ValueError('Not enough teams for 64-team demo bracket')


def simulate_once_fast(
    bracket_teams,
    season,
    base_lr,
    base_xgb,
    lr_cal,
    xgb_cal,
    feat_names,
    team_feats_dict,
    team_overrides=None,
    lr_weight=0.65,
    xgb_weight=0.35,
    seed_models=None,
):
    """Fast version using dict-based team lookups instead of DataFrame operations."""
    teams = list(bracket_teams)
    reach_counts = {team: set([round_label(len(teams))]) for team in teams}
    rounds = int(np.log2(len(teams)))
    for _ in range(rounds):
        next_round = []
        next_label = round_label(len(teams) // 2)
        for i in range(0, len(teams), 2):
            home = teams[i]
            away = teams[i + 1]
            # Fast dict-based lookup
            home_key = (season, home.strip().lower())
            away_key = (season, away.strip().lower())
            home_row = team_feats_dict.get(home_key)
            away_row = team_feats_dict.get(away_key)
            
            # Apply overrides if provided
            if team_overrides and home in team_overrides:
                home_row = {**home_row, **{k: v for k, v in team_overrides[home].items() if v is not None and not pd.isna(v)}} if home_row else None
            if team_overrides and away in team_overrides:
                away_row = {**away_row, **{k: v for k, v in team_overrides[away].items() if v is not None and not pd.isna(v)}} if away_row else None
            
            if home_row is None or away_row is None:
                winner = home if away_row is None else away
                if home_row is not None and away_row is not None:
                    winner = home if float(home_row.get('win_pct', 0.0)) >= float(away_row.get('win_pct', 0.0)) else away
            else:
                # Orient so team_A (home argument) = stronger team by adj_margin or seed
                home_adj = float(home_row.get('adj_margin', 0.0)) if home_row is not None else 0.0
                away_adj = float(away_row.get('adj_margin', 0.0)) if away_row is not None else 0.0
                home_seed_val = float(home_row.get('seed', 99)) if home_row is not None else 99
                away_seed_val = float(away_row.get('seed', 99)) if away_row is not None else 99
                # Use seed to orient if available, else adj_margin
                if home_seed_val != 99 or away_seed_val != 99:
                    # Lower seed number = better team = team_A
                    if away_seed_val < home_seed_val:
                        home_row, away_row = away_row, home_row
                        flipped = True
                    else:
                        flipped = False
                else:
                    if away_adj > home_adj:
                        home_row, away_row = away_row, home_row
                        flipped = True
                    else:
                        flipped = False
                p_home = predict_prob_with_seed_routing(
                    {
                        'base_lr': base_lr,
                        'base_xgb': base_xgb,
                        'lr_cal': lr_cal,
                        'xgb_cal': xgb_cal,
                        'feat_names': feat_names,
                    },
                    seed_models,
                    home_row,
                    away_row,
                    lr_weight=lr_weight,
                    xgb_weight=xgb_weight,
                )
                # If we flipped, p_home is now P(original away wins); invert to get P(original home wins)
                if flipped:
                    p_home = 1.0 - p_home
                winner = home if np.random.rand() < p_home else away
            next_round.append(winner)
            reach_counts[winner].add(next_label)
        teams = next_round
    champ = teams[0]
    return champ, reach_counts


def precompute_matchup_probs(
    teams, season, default_models, seed_models, feat_names,
    team_feats_dict, team_overrides, lr_weight, xgb_weight,
):
    """Pre-compute win probability for every possible pair among bracket teams.

    Makes a single batch predict_proba call per model (instead of one per game),
    reducing sklearn validation overhead by ~64x and cutting simulation time ~10x.

    Returns a dict: (team_a, team_b) -> float = P(team_a beats team_b).
    """
    home_rows_list = []
    away_rows_list = []
    valid_triples = []   # (i, j, flipped) for matched pairs
    fallback = {}        # (i, j) -> prob for teams missing features

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_i, team_j = teams[i], teams[j]
            hk = (season, team_i.strip().lower())
            ak = (season, team_j.strip().lower())
            hr = team_feats_dict.get(hk)
            ar = team_feats_dict.get(ak)

            if team_overrides:
                if hr and team_i in team_overrides:
                    ov = team_overrides[team_i]
                    hr = {**hr, **{k: v for k, v in ov.items() if v is not None and not pd.isna(v)}}
                if ar and team_j in team_overrides:
                    ov = team_overrides[team_j]
                    ar = {**ar, **{k: v for k, v in ov.items() if v is not None and not pd.isna(v)}}

            if hr is None or ar is None:
                fallback[(i, j)] = 1.0 if ar is None else (0.0 if hr is None else 0.5)
                continue

            hs = float(hr.get('seed', 99))
            js = float(ar.get('seed', 99))
            ha = float(hr.get('adj_margin', 0.0))
            ja = float(ar.get('adj_margin', 0.0))
            if hs != 99 or js != 99:
                flipped = js < hs   # team_j has lower (better) seed → orient as home
            else:
                flipped = ja > ha   # team_j has higher adj_margin
            if flipped:
                home_rows_list.append(ar)
                away_rows_list.append(hr)
            else:
                home_rows_list.append(hr)
                away_rows_list.append(ar)
            valid_triples.append((i, j, flipped))

    prob_lookup = {}

    if valid_triples:
        raw_probs = batch_predict_prob_routed(
            default_models, seed_models, feat_names,
            home_rows_list, away_rows_list, lr_weight, xgb_weight,
        )
        for (i, j, flipped), p in zip(valid_triples, raw_probs):
            p_i_beats_j = 1.0 - float(p) if flipped else float(p)
            prob_lookup[(teams[i], teams[j])] = p_i_beats_j
            prob_lookup[(teams[j], teams[i])] = 1.0 - p_i_beats_j

    for (i, j), p in fallback.items():
        prob_lookup[(teams[i], teams[j])] = p
        prob_lookup[(teams[j], teams[i])] = 1.0 - p

    return prob_lookup


def simulate_once_precomputed(bracket_teams, prob_lookup):
    """Run one Monte Carlo simulation using pre-computed matchup probabilities.

    O(63) dict lookups per sim instead of O(63) model inference calls.
    """
    teams = list(bracket_teams)
    reach_counts = {team: {round_label(len(teams))} for team in teams}
    rounds = int(np.log2(len(teams)))
    for _ in range(rounds):
        n = len(teams)
        next_label = round_label(n // 2)
        rands = np.random.rand(n // 2)
        next_round = []
        for k in range(0, n, 2):
            home, away = teams[k], teams[k + 1]
            p = prob_lookup.get((home, away), 0.5)
            winner = home if rands[k // 2] < p else away
            next_round.append(winner)
            reach_counts[winner].add(next_label)
        teams = next_round
    return teams[0], reach_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sims', type=int, default=1000)
    p.add_argument('--season', type=int, default=None)
    p.add_argument('--bracket_file', default=None, help='optional bracket file (.txt, .csv, or .json)')
    p.add_argument('--official_bracket', action='store_true', help='parse bracket as official NCAA format (with seeding/metadata)')
    p.add_argument('--out', default='results/sim_results.json')
    p.add_argument('--models_dir', default='models', help='directory containing trained model artifacts')
    p.add_argument('--seed_stratified_models_dir', default=None, help='optional directory containing seed-stratified model artifacts')
    p.add_argument('--features_path', default='data/processed/features/tournament_teams.csv', help='team feature CSV used for simulation lookups')
    p.add_argument('--min_games', type=int, default=10, help='minimum games required to consider a team for demo bracket')
    p.add_argument('--allow_nd', action='store_true', help='allow non-D1 (nd-) teams in demo bracket selection')
    p.add_argument('--lr_weight', type=float, default=None, help='ensemble weight for Logistic Regression (default: read from models/ensemble_weights.json or 0.5)')
    p.add_argument('--xgb_weight', type=float, default=None, help='ensemble weight for XGBoost (default: read from models/ensemble_weights.json or 0.5)')
    args = p.parse_args()

    ensure_dir(Path(args.out).parent)

    base_lr, base_xgb, lr_cal, xgb_cal, feat_names = load_models(args.models_dir)

    # Load optimized ensemble weights; CLI args override if provided
    default_lr, default_xgb = load_ensemble_weights(args.models_dir)
    lr_weight = args.lr_weight if args.lr_weight is not None else default_lr
    xgb_weight = args.xgb_weight if args.xgb_weight is not None else default_xgb
    seed_models = None
    if args.seed_stratified_models_dir:
        seed_models = load_seed_stratified_models(args.seed_stratified_models_dir)
        if not seed_models:
            print(f'Warning: no usable seed-stratified models found in {args.seed_stratified_models_dir}; using default models')
    team_feats = load_team_features(args.features_path)

    if args.season is None:
        args.season = int(team_feats['season'].max())
    season = args.season

    # Convert team features to dict for faster lookups and to avoid frequent DataFrame operations
    team_feats_dict = {}
    for _, row in team_feats.iterrows():
        key = (int(row['season']), str(row['team']).strip().lower())
        team_feats_dict[key] = row.to_dict()

    if args.bracket_file:
        if args.official_bracket:
            # Parse as official NCAA bracket with seeding/metadata
            bracket_dict = parse_ncaa_bracket(args.bracket_file)
            teams = bracket_dict['teams']
            bracket_records = bracket_dict['bracket']
            bracket_source = 'official_ncaa'
        else:
            # Use legacy load_bracket function
            teams, bracket_records, bracket_source = load_bracket(args.bracket_file)
    else:
        if not args.bracket_file:
            print(
                "\n⚠️  WARNING: No --bracket_file provided. Selecting teams by adj_margin rank.\n"
                "   This does NOT reflect actual NCAA tournament selection (committee criteria,\n"
                "   conference tournament results, at-large bids).\n"
                "   Use --bracket_file data/brackets/official_YYYY.csv --official_bracket for real predictions.\n"
            )
        teams = demo_bracket_from_top64_with_options(team_feats, season, min_games=args.min_games, allow_nd=args.allow_nd)
        bracket_records = [
            {'slot': idx, 'team': team, 'seed': None, 'region': None}
            for idx, team in enumerate(teams, start=1)
        ]
        bracket_source = 'generated_top64'

    if len(teams) & (len(teams) - 1) != 0:
        raise ValueError('Bracket size must be power of two (e.g., 64)')

    team_overrides = {record['team']: record for record in bracket_records}

    if lr_weight < 0 or xgb_weight < 0 or (lr_weight + xgb_weight) <= 0:
        raise ValueError('Ensemble weights must be non-negative and sum to a positive value')

    default_models_dict = {
        'base_lr': base_lr,
        'base_xgb': base_xgb,
        'lr_cal': lr_cal,
        'xgb_cal': xgb_cal,
    }

    # Pre-compute win probabilities for every possible matchup among bracket teams.
    # This makes one batch predict_proba call per model (64*63/2 = 2016 rows) instead
    # of a separate call for each of the ~63,000 individual game predictions, cutting
    # sklearn validation overhead by ~30x and reducing simulation time ~10x.
    n_pairs = len(teams) * (len(teams) - 1) // 2
    print(f'Pre-computing {n_pairs} matchup probabilities...')
    prob_lookup = precompute_matchup_probs(
        teams, season, default_models_dict, seed_models, feat_names,
        team_feats_dict, team_overrides, lr_weight, xgb_weight,
    )

    sims = args.sims
    champions = Counter()
    round_counts = {team: Counter() for team in teams}
    for i in range(sims):
        champ, reaches = simulate_once_precomputed(teams, prob_lookup)
        champions[champ] += 1
        for team, labels in reaches.items():
            for label in labels:
                round_counts.setdefault(team, Counter())[label] += 1
        if (i+1) % max(1, sims//10) == 0:
            print(f'Simulated {i+1}/{sims}...')

    champion_probs = {team: champions.get(team, 0) / sims for team in teams}
    round_probs = {
        team: {label: count / sims for label, count in sorted(counts.items())}
        for team, counts in round_counts.items()
    }
    sorted_champs = sorted(champion_probs.items(), key=lambda x: x[1], reverse=True)

    # Determine model version string
    model_components = []
    if base_lr is not None:
        model_components.append('lr')
    if base_xgb is not None:
        model_components.append('xgb')
    if lr_cal is not None or xgb_cal is not None:
        model_components.append('calibrated')
    model_version = '+'.join(model_components) if model_components else 'unknown'

    # Build output with schema versioning
    output = {
        'schema_version': '2025-03-09',
        'generated_at': datetime.now().isoformat() + 'Z',
        'season': season,
        'sims': sims,
        'teams': teams,
        'bracket_source': bracket_source,
        'bracket': bracket_records,
        'champion_probs': sorted_champs,
        'round_probs': round_probs,
        'model_metadata': {
            'model_version': model_version,
            'lr_model': base_lr is not None,
            'xgb_model': base_xgb is not None,
            'lr_calibration': lr_cal is not None,
            'xgb_calibration': xgb_cal is not None,
            'ensemble_weights': {
                'lr_weight': float(lr_weight),
                'xgb_weight': float(xgb_weight),
            },
            'seed_stratified_models': {
                'enabled': bool(seed_models),
                'source_dir': args.seed_stratified_models_dir,
                'loaded_strata': sorted(list(seed_models.keys())) if seed_models else [],
            },
            'feature_count': len(feat_names),
        },
    }

    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)

    print('Top champions:')
    for t, p in sorted_champs[:10]:
        print(f'{t}: {p:.3f}')
    print('Results saved to', args.out)


if __name__ == '__main__':
    main()
