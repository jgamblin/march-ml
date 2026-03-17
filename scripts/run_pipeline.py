"""Orchestrate the NCAA bracket pipeline.

Examples:
    python scripts/run_pipeline.py --mode features
    python scripts/run_pipeline.py --mode train
    python scripts/run_pipeline.py --mode full --sims 5000
"""
import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_command(cmd):
    logger.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd)


def check_joblib_tracking(models_dir="models"):
    """Warn if any .joblib model artifacts are being silently ignored by git.

    The .gitignore includes ``models/`` to keep large binary files out of the
    repo by default.  Committed exceptions (training_summary.json,
    ensemble_weights.json) are force-added by GitHub Actions.  If someone
    tries to commit the .joblib files, this check surfaces the issue early.
    """
    from pathlib import Path
    import shutil

    if not shutil.which("git"):
        return  # git not available; skip silently

    joblib_files = list(Path(models_dir).glob("*.joblib"))
    if not joblib_files:
        return

    ignored = []
    for jf in joblib_files:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(jf)],
            capture_output=True,
        )
        if result.returncode == 0:
            ignored.append(str(jf))

    if ignored:
        logger.warning(
            "The following model artifacts are gitignored and will NOT be committed:\n"
            "  %s\n"
            "  This is intentional — .joblib files are regenerated on each training run.\n"
            "  To force-track them: git add -f %s",
            "\n  ".join(ignored),
            models_dir + "/*.joblib",
        )
    else:
        logger.info("Model artifacts in %s/ are tracked by git.", models_dir)


def run_scrape(
        fetch_pbp=False, seasons=None, historical=False, since=None, lookback_days=None):
    cmd = [sys.executable, "scripts/scrape_with_cbbpy.py", "--out", "data/processed", "--raw", "data/raw"]
    if seasons:
        cmd.extend(["--seasons", *[str(season) for season in seasons]])
    elif historical:
        cmd.append("--historical")
    elif lookback_days:
        cmd.extend(["--lookback-days", str(lookback_days)])
    elif since:
        cmd.extend(["--since", since])
    if fetch_pbp:
        cmd.append("--pbp")
    run_command(cmd)


def run_features(seasons=None, d1_list=None, seed_map=None, conf_map=None):
    cmd = [sys.executable, "scripts/prepare_features.py", "--input", "data/processed", "--out", "data/processed/features"]
    if seasons:
        cmd.extend(["--seasons", *[str(season) for season in seasons]])
    if d1_list:
        cmd.extend(["--d1_list", d1_list])
    if seed_map:
        cmd.extend(["--seed_map", seed_map])
    if conf_map:
        cmd.extend(["--conf_map", conf_map])
    run_command(cmd)


def run_train(game_scope="ncaa_tourney", interactions=False, include_regular_season=False, regular_season_weight=0.3):
    cmd = [
        sys.executable,
        "scripts/train_baseline.py",
        "--features",
        "data/processed/features/tournament_teams.csv",
        "--games_dir",
        "data/processed",
        "--out_dir",
        "models",
        "--game_scope",
        game_scope,
    ]
    if interactions:
        cmd.append("--interactions")
    if include_regular_season:
        cmd.extend(["--include_regular_season", "--regular_season_weight", str(regular_season_weight)])
    run_command(cmd)


def run_ensemble_optimize(include_regular_season=True, regular_season_weight=0.3):
    """Find optimal LR/XGB blend weights via LOSO and write models/ensemble_weights.json."""
    cmd = [
        sys.executable,
        "scripts/optimize_ensemble_weights.py",
        "--features",
        "data/processed/features/tournament_teams.csv",
        "--games_dir",
        "data/processed",
        "--models_dir",
        "models",
    ]
    if include_regular_season:
        cmd.append("--include_regular_season")
    cmd.extend(["--regular_season_weight", str(regular_season_weight)])
    run_command(cmd)


def run_simulation(sims=1000, season=None, bracket_file=None, official_bracket=False, out_path="results/sim_results.json", min_games=10, allow_nd=False, seed=None):
    cmd = [sys.executable, "scripts/simulate_bracket.py", "--sims", str(sims), "--out", out_path, "--min_games", str(min_games)]
    if season is not None:
        cmd.extend(["--season", str(season)])
    if bracket_file:
        cmd.extend(["--bracket_file", bracket_file])
    if official_bracket:
        cmd.append("--official_bracket")
    if allow_nd:
        cmd.append("--allow_nd")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    run_command(cmd)


def run_bracket_visual(sim_out="results/sim_results.json", out_path=None):
    season = sim_out.replace("results/sim_", "").replace(".json", "").split("_")[0]
    if out_path is None:
        out_path = f"results/charts/bracket_{season}.png"
    cmd = [sys.executable, "scripts/generate_bracket_visual.py",
           "--sim", sim_out, "--out", out_path]
    run_command(cmd)


def run_validate(sim_out=None, allow_nd=False):
    cmd = [sys.executable, "scripts/validate_artifacts.py"]
    if sim_out:
        cmd.extend(["--sim_out", sim_out])
    if allow_nd:
        cmd.append("--allow_nd")
    run_command(cmd)


def run_optimize(sim_out=None, profile="espn", strategy="balanced", num_entries=10, opt_out=None):
    cmd = [sys.executable, "scripts/optimize_entries.py"]
    if sim_out:
        cmd.extend(["--sim_out", sim_out])
    if profile:
        cmd.extend(["--profile", profile])
    if strategy:
        cmd.extend(["--strategy", strategy])
    if num_entries:
        cmd.extend(["--num_entries", str(num_entries)])
    if opt_out:
        cmd.extend(["--out", opt_out])
    run_command(cmd)


def run_smoke(d1_list=None, seed_map=None, conf_map=None):
    """Fast validation: features (2025) → train → simulate (100) → validate."""
    logger.info("=== SMOKE TEST: Fast pipeline validation ===")
    run_features(seasons=[2025], d1_list=d1_list, seed_map=seed_map, conf_map=conf_map)
    run_train(game_scope="ncaa_tourney")
    logger.info("Smoke test: running simulation with 100 sims...")
    run_simulation(sims=100, season=2025, out_path="results/smoke_test.json", min_games=10, allow_nd=False)
    logger.info("Smoke test complete. Results in results/smoke_test.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=["scrape", "features", "train", "simulate", "validate", "optimize", "smoke", "full"])
    p.add_argument("--pbp", action="store_true", help="fetch play-by-play during scraping")
    p.add_argument("--seasons", nargs="*", type=int, help="optional season override")
    p.add_argument("--historical", action="store_true",
                   help="also scrape 2015-2019 historical seasons (used with --mode scrape/full)")
    p.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                   help="incremental scrape: only fetch games from this date forward (mutually exclusive with --lookback-days)")
    p.add_argument("--lookback-days", type=int, default=None, metavar="N",
                   help="incremental scrape: fetch last N days for the current season (default for scheduled runs: 14)")
    p.add_argument("--d1_list", default="data/mappings/d1_list_normalized.csv")
    p.add_argument("--seed_map", default=None)
    p.add_argument("--conf_map", default=None)
    p.add_argument("--game_scope", choices=["ncaa_tourney", "postseason", "all"], default="ncaa_tourney")
    p.add_argument("--interactions", action="store_true", help="enable interaction features in training (off by default)")
    p.add_argument("--include_regular_season", action="store_true", help="include regular-season games in training at reduced weight (expands dataset ~100x)")
    p.add_argument("--regular_season_weight", type=float, default=0.3, help="sample weight for regular-season rows (default 0.3)")
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--season", type=int, default=None, help="season for bracket simulation")
    p.add_argument("--bracket_file", default=None, help="optional bracket file (.txt, .csv, or .json)")
    p.add_argument("--official_bracket", action="store_true", help="parse bracket as official NCAA format (slot/seed/region metadata)")
    p.add_argument("--sim_out", default="results/sim_results.json")
    p.add_argument("--min_games", type=int, default=10)
    p.add_argument("--allow_nd", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible simulations")
    # Optimizer arguments
    p.add_argument("--profile", default="espn", choices=["espn", "cbs", "simple"], help="pool scoring profile")
    p.add_argument("--strategy", default="balanced", choices=["chalk", "balanced", "contrarian"], help="optimization strategy")
    p.add_argument("--num_entries", type=int, default=10, help="number of bracket entries to generate")
    p.add_argument("--opt_out", default=None, help="output file for optimizer results")
    args = p.parse_args()

    if args.mode in {"scrape", "full"}:
        run_scrape(fetch_pbp=args.pbp, seasons=args.seasons,
                   historical=getattr(args, 'historical', False),
                   since=getattr(args, 'since', None),
                   lookback_days=getattr(args, 'lookback_days', None))
    if args.mode in {"features", "full"}:
        run_features(seasons=args.seasons, d1_list=args.d1_list, seed_map=args.seed_map, conf_map=args.conf_map)
    if args.mode in {"train", "full"}:
        run_train(
            game_scope=args.game_scope,
            interactions=args.interactions,
            include_regular_season=args.include_regular_season,
            regular_season_weight=args.regular_season_weight,
        )
        run_ensemble_optimize(
            include_regular_season=args.include_regular_season,
            regular_season_weight=args.regular_season_weight,
        )
        check_joblib_tracking()
    if args.mode in {"simulate", "full"}:
        run_simulation(
            sims=args.sims,
            season=args.season,
            bracket_file=args.bracket_file,
            official_bracket=args.official_bracket,
            out_path=args.sim_out,
            min_games=args.min_games,
            allow_nd=args.allow_nd,
            seed=args.seed,
        )
        run_bracket_visual(sim_out=args.sim_out)
    if args.mode in {"validate", "full"}:
        run_validate(sim_out=args.sim_out if args.mode == "full" else args.sim_out, allow_nd=args.allow_nd)
    if args.mode == "optimize":
        run_optimize(sim_out=args.sim_out, profile=args.profile, strategy=args.strategy, num_entries=args.num_entries, opt_out=args.opt_out)
    if args.mode == "smoke":
        run_smoke(d1_list=args.d1_list, seed_map=args.seed_map, conf_map=args.conf_map)


if __name__ == "__main__":
    main()
