"""Command-line entry point: one subcommand per pipeline stage.

    uv run assignment-2 prep
    uv run assignment-2 train                            # task A (play), every registered model
    uv run assignment-2 train --task engagement --model svd
    uv run assignment-2 models                           # list registered models
    uv run assignment-2 eval                             # not implemented yet
    uv run assignment-2 figures                          # not implemented yet

Every subcommand accepts --data-dir / --art-dir / --results-dir / --seed, so the
Makefile can pass them in any order. Stage code lives in assignment_2.modules and is
imported lazily, so `--help` stays fast even with heavy ML dependencies installed.
"""

import argparse
import pickle
import sys
import time
from collections.abc import Callable

from pydantic import ValidationError

from assignment_2.modules.config import Config, load

# Written out rather than imported from modules.data.tasks (which imports pandas), so the
# parser can be built without the heavy imports. Keep in sync with tasks.TASKS.
TASKS = ("play", "engagement")


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--config", help="extra TOML overrides (default: $A2_CONFIG); config.toml always applies"
    )
    g = p.add_argument_group(
        "paths (default: from config; env A2_PATHS__DATA_DIR etc.)")
    g.add_argument("--data-dir", help="raw datasets (from scripts/get_data)")
    g.add_argument(
        "--art-dir", help="regenerable outputs: splits, trained models")
    g.add_argument("--results-dir",
                   help="committed outputs: metrics.json, figures/")
    p.add_argument("--seed", type=int,
                   help="random seed (default: from config)")
    return p


# ── commands ────────────────────────────────────────────────────────────────
# Each takes the resolved Config plus parsed args. Keep these thin: parse options,
# call into modules/, report. The real work belongs in the modules.


def cmd_prep(cfg: Config, args: argparse.Namespace) -> None:
    from assignment_2.modules import data

    out = data.prep(cfg)
    stats = data.read_meta(cfg)["stats"]
    own, play = stats["own"], stats["play"]
    print(
        f"[prep] {cfg.sample.tag}: {own['all']['users']:,} users, "
        f"{own['all']['items']:,} games, sparsity {own['all']['sparsity']:.4f} | "
        f"play: {play['test']['users']:,} test users, {play['cold_dropped']:,} cold dropped | "
        f"own: {own['test']['users']:,} test users, {own['cold_dropped']:,} cold dropped "
        f"-> {out}"
    )


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    from assignment_2.modules.data import load_prepped, train_tables
    from assignment_2.modules.data.prep import prepped_dir
    from assignment_2.modules.models import MODELS, build
    from assignment_2.modules.models.ranking.base import TrainData

    names = args.model or list(MODELS)
    unknown = sorted(set(names) - set(MODELS))
    if unknown:
        sys.exit(
            f"unknown model(s): {', '.join(unknown)} (see `assignment-2 models`)")
    if not names:
        sys.exit(
            "no models registered yet: add one to MODELS in modules/models/registry.py")

    t = load_prepped(cfg)  # refuses a cache built with different settings

    # The task's training tables, with its held-out pairs removed (see data/tasks.py).
    tables = train_tables(t, args.task)
    n_dropped = tables.pop("n_reviews_dropped")
    print(
        f"[train] {args.task}: dropped {n_dropped:,} reviews of held-out games")
    data = TrainData(**tables)

    # One folder per task: each task trains on its own split, so the models differ.
    out = prepped_dir(cfg) / "models" / args.task
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        start = time.perf_counter()
        model = build(name, cfg).fit(data)
        path = out / f"{name}.pkl"
        with path.open("wb") as f:
            pickle.dump(model, f)
        print(
            f"[train] {args.task}/{name}: fit in {time.perf_counter() - start:.1f}s -> {path}")


def cmd_eval(cfg: Config, args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "evaluation is being settled in notebooks; see core/evaluate.py")


def cmd_figures(cfg: Config, args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "figures are made in the notebooks for now (core/plots.py)")


def cmd_models(cfg: Config, args: argparse.Namespace) -> None:
    from assignment_2.modules import models

    if not models.MODELS:
        print("(none registered)")
    for name, cls in models.MODELS.items():
        doc = (cls.__doc__ or "").strip().splitlines()
        print(f"  {name:20s} {doc[0] if doc else ''}")


# ── parser ──────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="assignment-2", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(
        dest="command", required=True, metavar="command")

    def add(name: str, fn: Callable, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=help, description=help)
        p.set_defaults(func=fn)
        return p

    add("prep", cmd_prep,
        "sample, split, and build candidate tables -> artifacts/steam/<tag>/")
    p = add(
        "train", cmd_train, "fit models on a task's split -> artifacts/steam/<tag>/models/<task>/"
    )
    p.add_argument(
        "--task",
        default="play",
        choices=TASKS,
        help="play (task A, ranking) or engagement (task B, log playtime); default: play",
    )
    p.add_argument("--model", action="append",
                   help="model to train (repeatable; default: all)")
    p = add("eval", cmd_eval,
            "score trained models on the fixed split (not implemented yet)")
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    add("figures", cmd_figures, "plots for the slides (not implemented yet)")
    add("models", cmd_models, "list registered models")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        cfg = load(
            args.config,
            seed=args.seed,
            paths={
                "data_dir": args.data_dir,
                "art_dir": args.art_dir,
                "results_dir": args.results_dir,
            },
        )
    except (ValidationError, FileNotFoundError) as e:
        sys.exit(f"config error: {e}")
    cfg.paths.ensure()
    if cfg.sources:
        print(f"[config] {', '.join(str(p) for p in cfg.sources)}")

    start = time.perf_counter()
    try:
        args.func(cfg, args)
    except NotImplementedError as e:
        sys.exit(f"{args.command}: not implemented yet ({e})")
    except FileNotFoundError as e:
        sys.exit(f"{args.command}: {e}")
    print(f"[{args.command}] done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
