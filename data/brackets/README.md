# Bracket input files

Use this directory for structured tournament bracket inputs.

## Supported fields

The simulator accepts `.txt`, `.csv`, and `.json` bracket files.

Recommended CSV columns:

- `slot`: bracket order slot
- `team`: team name matching the feature file
- `seed`: optional numeric seed
- `region`: optional region label

Adjacent slots are paired in the first round, so slots `1` and `2` play each other, then `3` and `4`, and so on.

## Example command

```bash
python scripts/run_pipeline.py --mode simulate --season 2025 --bracket_file data/brackets/template_64.csv --sim_out results/sim_2025_structured.json
```

## Notes

- Fill in all teams before running a full 64-team simulation.
- If you provide `seed`, that value can be used by future seed-aware models.
- The default fallback simulator still works without a bracket file.
