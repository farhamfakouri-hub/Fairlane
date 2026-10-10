"""Command-line entry point: simulate, score, experiments, smoke check, screens, API, planner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ree.config import DEFAULT_CONFIG_ENV, Config, dump_config, load_config


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config file")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one sub-command per task."""
    parser = argparse.ArgumentParser(prog="ree", description="Retail cashier efficiency engine")
    commands = parser.add_subparsers(dest="command", required=True)

    dump = commands.add_parser("dump-config", help="write the default configuration as YAML")
    dump.add_argument("--out", type=str, default="config/default.yaml")

    simulate = commands.add_parser("simulate", help="generate a simulated store")
    _add_config_argument(simulate)
    simulate.add_argument("--name", type=str, default=None)
    simulate.add_argument("--seed", type=int, default=None)
    simulate.add_argument("--weeks", type=int, default=None)
    simulate.add_argument("--cashiers", type=int, default=None)
    simulate.add_argument("--mode", choices=["random", "biased"], default=None)

    score = commands.add_parser("score", help="score the cashiers of a stored store")
    _add_config_argument(score)
    score.add_argument("--name", type=str, default=None)
    score.add_argument("--method", choices=["v1", "v2"], default=None)

    experiments = commands.add_parser("experiments", help="run all experiments and write reports")
    _add_config_argument(experiments)
    experiments.add_argument("--reduced", action="store_true")
    experiments.add_argument("--jobs", type=int, default=None)
    experiments.add_argument("--seeds", type=int, default=None)

    smoke = commands.add_parser("smoke", help="small end-to-end check, under two minutes")
    _add_config_argument(smoke)

    ui = commands.add_parser("ui", help="start the review screen")
    _add_config_argument(ui)
    ui.add_argument("--host", type=str, default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8501)

    api = commands.add_parser("api", help="start the score API")
    _add_config_argument(api)
    api.add_argument("--host", type=str, default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    commands.add_parser("export-openapi", help="write the OpenAPI schema to docs/openapi.json")

    planner = commands.add_parser("planner", help="shift planner (Part 2); see planner --help")
    planner.add_argument("planner_args", nargs=argparse.REMAINDER)

    figures = commands.add_parser("figures", help="regenerate a single figure")
    _add_config_argument(figures)
    figures.add_argument("which", choices=["gaps"])
    return parser


def _apply_simulation_overrides(config: Config, args: argparse.Namespace) -> Config:
    overrides: dict[str, object] = {}
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "weeks", None) is not None:
        overrides["simulator.n_weeks"] = args.weeks
    if getattr(args, "cashiers", None) is not None:
        overrides["simulator.population.n_cashiers"] = args.cashiers
    if getattr(args, "mode", None) is not None:
        overrides["simulator.scheduler.mode"] = args.mode
    if getattr(args, "name", None) is not None:
        overrides["paths.store_name"] = args.name
    return config.with_overrides(overrides) if overrides else config


def run_dump_config(args: argparse.Namespace) -> int:
    """Write the default configuration to a YAML file."""
    dump_config(Config(), Path(args.out))
    print(f"wrote {args.out}")
    return 0


def run_simulate(args: argparse.Namespace) -> int:
    """Simulate a store and write it to the data directory."""
    from ree.simulator.run import simulate_to_directory

    config = _apply_simulation_overrides(load_config(args.config), args)
    directory = simulate_to_directory(config)
    print(f"store written to {directory}")
    return 0


def run_score(args: argparse.Namespace) -> int:
    """Score a stored simulation and print a summary of the run."""
    from ree.engine.pipeline import score_directory

    config = load_config(args.config)
    if args.name is not None:
        config = config.with_overrides({"paths.store_name": args.name})
    if args.method is not None:
        config = config.with_overrides({"engine.speed.method": args.method})
    summary = score_directory(config)
    print(summary)
    return 0


def run_experiments(args: argparse.Namespace) -> int:
    """Run the experiments and write the report and figures."""
    from ree.experiments.run_all import run_all

    config = load_config(args.config)
    run_all(config, reduced=args.reduced, jobs=args.jobs, seeds=args.seeds)
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    """Run the end-to-end smoke check."""
    from ree.experiments.run_all import run_smoke as smoke_run

    return smoke_run(load_config(args.config))


def run_ui(args: argparse.Namespace) -> int:
    """Start the Streamlit review screen in a subprocess."""
    import subprocess

    app_path = Path(__file__).resolve().parent / "ui" / "review_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    environment = dict(os.environ)
    if args.config is not None:
        environment[DEFAULT_CONFIG_ENV] = str(Path(args.config).resolve())
    print(f"review screen at http://{args.host}:{args.port} (Ctrl+C to stop)")
    return subprocess.call(command, env=environment)


def run_api(args: argparse.Namespace) -> int:
    """Serve the score API with uvicorn."""
    import uvicorn

    from ree.api.app import create_app

    uvicorn.run(create_app(load_config(args.config)), host=args.host, port=args.port)
    return 0


def run_export_openapi(args: argparse.Namespace) -> int:
    """Write the score API schema to docs/openapi.json."""
    import json

    from ree.api.app import create_app
    from ree.api.provider import InMemoryScoreProvider

    app = create_app(Config(), provider=InMemoryScoreProvider([]))
    target = Path("docs") / "openapi.json"
    target.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {target}")
    return 0


def run_planner(args: argparse.Namespace) -> int:
    """Hand the remaining arguments to the planner command line."""
    from ree.planner.cli import main as planner_main

    return planner_main(list(args.planner_args))


def run_figures(args: argparse.Namespace) -> int:
    """Write the gap distribution figure."""
    from ree.experiments.figures import make_gap_distribution_figure

    path = make_gap_distribution_figure(load_config(args.config))
    print(f"wrote {path}")
    return 0


HANDLERS = {
    "dump-config": run_dump_config,
    "simulate": run_simulate,
    "score": run_score,
    "experiments": run_experiments,
    "smoke": run_smoke,
    "ui": run_ui,
    "api": run_api,
    "export-openapi": run_export_openapi,
    "figures": run_figures,
    "planner": run_planner,
}


def main(argv: list[str] | None = None) -> int:
    """Run the chosen sub-command; arguments that start with 'planner' go to the planner CLI."""
    given = list(sys.argv[1:] if argv is None else argv)
    if given and given[0] == "planner":
        from ree.planner.cli import main as planner_main

        return planner_main(given[1:])
    parser = build_parser()
    args = parser.parse_args(given)
    return int(HANDLERS[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
