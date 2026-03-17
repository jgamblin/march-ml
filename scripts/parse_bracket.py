"""Parse official NCAA tournament brackets and handle First Four integration.

Supports:
- Official NCAA bracket formats (JSON, CSV)
- First Four play-in game results
- Automatic slot validation and filling
- Backward compatibility with demo/generated brackets
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def parse_ncaa_bracket(source_file: str) -> Dict[str, Any]:
    """
    Parse official NCAA bracket from JSON or CSV file.
    
    Expected structure for CSV:
    - Columns: team, seed, region, slot (optional)
    - One row per team
    
    Expected structure for JSON:
    - List of dicts with keys: team, seed, region, slot (optional)
    - OR dict with 'teams' key containing the list
    
    Returns dict with keys:
    - teams: List[str] of team names in bracket order
    - brackets: List[Dict] with slot, team, seed, region
    - regions: Dict[str, List[int]] mapping region name to slots
    - seeds: Dict[int, List[str]] mapping seed number to teams
    """
    path = Path(source_file)
    if not path.exists():
        raise FileNotFoundError(f"Bracket file not found: {source_file}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("teams", data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError("JSON bracket must be list or object with 'teams' key")
    elif suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if "team" not in df.columns and "name" not in df.columns:
            raise ValueError("CSV bracket must have 'team' or 'name' column")
        records = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported bracket format: {suffix}")
    
    return _normalize_bracket_records(records)


def _normalize_bracket_records(records: List[Any]) -> Dict[str, Any]:
    """Normalize bracket records to standard format."""
    import pandas as pd
    
    normalized = []
    seen = set()
    
    for idx, record in enumerate(records, start=1):
        if isinstance(record, str):
            team = record.strip()
            seed = None
            region = None
            slot = idx
        else:
            team = str(record.get("team", "") or record.get("name", "")).strip()
            seed = record.get("seed")
            region = record.get("region")
            
            # Parse slot if provided, else use enumeration order
            slot_raw = record.get("slot")
            if slot_raw is not None:
                try:
                    slot = int(slot_raw)
                except (ValueError, TypeError):
                    slot = idx
            else:
                slot = idx
        
        if not team:
            continue
        
        if team in seen:
            raise ValueError(f"Duplicate team in bracket: {team}")
        seen.add(team)
        
        normalized.append({
            "slot": slot,
            "team": team,
            "seed": None if pd.isna(seed) else int(seed) if seed else None,
            "region": str(region).strip() if region and not pd.isna(region) else None,
        })
    
    if not normalized:
        raise ValueError("No teams found in bracket")
    
    # Sort by slot
    normalized = sorted(normalized, key=lambda x: x["slot"])
    teams = [r["team"] for r in normalized]
    
    # Validate bracket size is power of 2, or 68 (First Four)
    if len(teams) != 68 and (len(teams) & (len(teams) - 1) != 0):
        raise ValueError(f"Bracket size {len(teams)} is not a power of 2")
    
    # Build region and seed lookup
    regions = {}
    seeds = {}
    for record in normalized:
        if record.get("region"):
            if record["region"] not in regions:
                regions[record["region"]] = []
            regions[record["region"]].append(record["slot"])
        if record.get("seed"):
            if record["seed"] not in seeds:
                seeds[record["seed"]] = []
            seeds[record["seed"]].append(record["team"])
    
    return {
        "teams": teams,
        "bracket": normalized,
        "regions": regions,
        "seeds": seeds,
    }


def integrate_first_four(
    bracket_68: Dict[str, Any],
    first_four_results: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Integrate First Four play-in results into a 68-team bracket.
    
    Args:
        bracket_68: Full 68-team bracket (64 at-large + 4 First Four play-in slots)
        first_four_results: List of (winner, loser) tuples for the 4 play-in games
    
    Returns:
        64-team bracket with First Four winners inserted in place of play-in slots
    """
    if len(bracket_68["teams"]) != 68:
        raise ValueError(f"Expected 68-team bracket, got {len(bracket_68['teams'])}")
    
    if len(first_four_results) != 4:
        raise ValueError(f"Expected 4 First Four results, got {len(first_four_results)}")
    
    # Identify play-in slots (typically slots 65-68 or marked with specific slot numbers)
    # Play-in teams are usually the 4 16-seeds from the First Four
    bracket_records = bracket_68["bracket"].copy()
    playin_slots = [r["slot"] for r in bracket_records if r.get("seed") == 16]
    
    if len(playin_slots) < 4:
        # Fallback: use last 4 slots
        playin_slots = sorted([r["slot"] for r in bracket_records])[-4:]
    
    if len(playin_slots) != 4:
        raise ValueError("Could not identify 4 play-in slots in 68-team bracket")
    
    # Map play-in slots to winners
    slot_to_winner = {}
    for i, (slot, (winner, loser)) in enumerate(zip(sorted(playin_slots), first_four_results)):
        slot_to_winner[slot] = winner
    
    # Replace play-in teams with winners
    new_bracket = []
    new_teams = []
    for record in bracket_records:
        if record["slot"] in slot_to_winner:
            winner = slot_to_winner[record["slot"]]
            new_record = record.copy()
            new_record["team"] = winner
            new_bracket.append(new_record)
            new_teams.append(winner)
        else:
            new_bracket.append(record)
            new_teams.append(record["team"])
    
    return {
        "teams": new_teams,
        "bracket": new_bracket,
        "regions": bracket_68.get("regions", {}),
        "seeds": bracket_68.get("seeds", {}),
    }


def validate_bracket(bracket: Dict[str, Any], expected_size: int = 64) -> Tuple[bool, str]:
    """
    Validate bracket structure and size.
    
    Returns:
        (is_valid, error_message)
    """
    if expected_size not in (4, 8, 16, 32, 64, 68):
        return False, f"Invalid expected size: {expected_size}"
    
    if "teams" not in bracket or not isinstance(bracket["teams"], list):
        return False, "Bracket missing 'teams' list"
    
    if len(bracket["teams"]) != expected_size:
        return False, f"Expected {expected_size} teams, got {len(bracket['teams'])}"
    
    if len(set(bracket["teams"])) != len(bracket["teams"]):
        return False, "Bracket contains duplicate teams"
    
    # Validate bracket records
    if "bracket" in bracket:
        if len(bracket["bracket"]) != expected_size:
            return False, f"Bracket records mismatch: {len(bracket['bracket'])} != {expected_size}"
    
    return True, ""


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bracket", required=True, help="Path to bracket file")
    p.add_argument("--validate_only", action="store_true", help="Only validate, don't parse")
    args = p.parse_args()
    
    try:
        bracket = parse_ncaa_bracket(args.bracket)
        print(f"✓ Parsed bracket with {len(bracket['teams'])} teams")
        if bracket.get("regions"):
            print(f"  Regions: {list(bracket['regions'].keys())}")
        if bracket.get("seeds"):
            print(f"  Seeds: {sorted(bracket['seeds'].keys())}")
        print(f"  Teams: {', '.join(bracket['teams'][:5])}...")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)
