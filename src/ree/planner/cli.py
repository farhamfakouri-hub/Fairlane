"""Command line of the shift planner: python -m ree.cli planner <command>."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ree.planner.config import (
    PlannerConfig,
    dump_planner_config,
    load_planner_config,
)
from ree.planner.daily.daily_plan import board_lines, plan_days
from ree.planner.demand.forecast import default_model
from ree.planner.domain.demand_types import PlannerCalendar
from ree.planner.domain.patterns import (
    PatternSpec,
    dump_pattern_spec,
    full_pattern_set,
    load_pattern_spec,
    solver_pattern_set,
)
from ree.planner.domain.rules import labour_rules_markdown
from ree.planner.domain.shifts import minutes_to_clock
from ree.planner.eval.fairness_audit import (
    AUDIT_SEEDS,
    GUARDED,
    audit_passed,
    audit_table_path,
    run_audit,
    write_audit,
)
from ree.planner.eval.outputs import write_outputs
from ree.planner.eval.run_eval import (
    DEFAULT_STORE,
    FIRST_SEED,
    evaluation_config,
    evaluation_plan,
    results_path,
    run_evaluation,
)
from ree.planner.eval.stability import (
    FIRST_STABILITY_SEED,
    STABILITY_SEEDS,
    run_stability,
    write_stability,
)
from ree.planner.eval.validation import queue_validation
from ree.planner.roster.solver import RosterSolution, plan_roster
from ree.planner.sim.roster_gate import (
    GATE_FIRST_SEED,
    GATE_SCENARIOS,
    gate_passed,
    run_gate,
    scenario_inputs,
    write_gate_report,
)
from ree.planner.sim.scenarios import STORE_SIZES, store_config
from ree.planner.smoke import run_checks
from ree.planner.staffing.requirement import requirement_for_cycle, requirement_summary

QUEUE_CHECK_TOLERANCE = 0.06


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one sub-command for each task."""
    parser = argparse.ArgumentParser(prog="ree planner", description="Shift planner (Part 2)")
    commands = parser.add_subparsers(dest="command", required=True)

    dump = commands.add_parser("dump-config", help="write config/planner.yaml and the patterns")
    dump.add_argument("--out", type=str, default="config/planner.yaml")
    dump.add_argument("--patterns-out", type=str, default="config/shift_patterns.yaml")

    commands.add_parser("labour-rules", help="print the labour rule table as Markdown")
    commands.add_parser(
        "export-openapi", help="write the planner OpenAPI schema to docs/planner_openapi.json"
    )

    serve = commands.add_parser("api", help="serve the planner API")
    serve.add_argument("--host", type=str, default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)

    screens = commands.add_parser("ui", help="open the manager or the worker screen")
    screens.add_argument("--worker", action="store_true")
    screens.add_argument("--host", type=str, default="127.0.0.1")
    screens.add_argument("--port", type=int, default=8502)

    requirement = commands.add_parser(
        "requirement", help="print the staffing requirement of a cycle"
    )
    requirement.add_argument("--config", type=str, default=None)
    requirement.add_argument("--start", type=str, default=None)
    requirement.add_argument("--weeks", type=int, default=1)

    roster = commands.add_parser("roster", help="solve a roster for a simulated store")
    roster.add_argument("--store", type=str, default="tiny", choices=STORE_SIZES)
    roster.add_argument("--weeks", type=int, default=1)
    roster.add_argument("--seed", type=int, default=3)
    roster.add_argument("--mode", type=str, default=None, choices=("reproducible", "fast"))
    roster.add_argument("--budget", type=float, default=None)
    roster.add_argument("--backend", type=str, default=None, choices=("cp_sat", "heuristic"))
    roster.add_argument("--posts-only", action="store_true")

    daily = commands.add_parser("daily", help="solve a roster and plan its days (breaks and posts)")
    daily.add_argument("--store", type=str, default="small", choices=STORE_SIZES)
    daily.add_argument("--seed", type=int, default=3)
    daily.add_argument("--day", type=str, default=None)
    daily.add_argument("--backend", type=str, default=None, choices=("cp_sat", "heuristic"))
    daily.add_argument("--posts-only", action="store_true")

    audit = commands.add_parser("audit", help="audit the fairness of the roster")
    audit.add_argument("--store", type=str, default=DEFAULT_STORE, choices=STORE_SIZES)
    audit.add_argument("--reduced", action="store_true")
    audit.add_argument("--seeds", type=int, default=None)
    audit.add_argument("--jobs", type=int, default=2)
    audit.add_argument("--report-only", action="store_true")

    stability = commands.add_parser(
        "stability", help="measure how many assignments change when the roster is re-solved"
    )
    stability.add_argument("--store", type=str, default="tiny", choices=STORE_SIZES)
    stability.add_argument("--reduced", action="store_true")
    stability.add_argument("--seeds", type=int, default=None)
    stability.add_argument("--jobs", type=int, default=2)

    evaluate = commands.add_parser(
        "evaluate", help="simulate plans A to F and L on the scenarios and write the report"
    )
    evaluate.add_argument("--store", type=str, default=DEFAULT_STORE, choices=STORE_SIZES)
    evaluate.add_argument("--reduced", action="store_true")
    evaluate.add_argument("--seeds", type=int, default=None)
    evaluate.add_argument("--jobs", type=int, default=2)

    report = commands.add_parser(
        "report", help="rebuild the figures and the report from the results"
    )
    report.add_argument("--store", type=str, default=DEFAULT_STORE, choices=STORE_SIZES)
    report.add_argument("--reduced", action="store_true")

    commands.add_parser("queue-check", help="compare the queue simulator with Erlang C")

    gate = commands.add_parser("roster-gate", help="solve many seeded scenarios and check each")
    gate.add_argument("--scenarios", type=int, default=GATE_SCENARIOS)
    gate.add_argument("--first-seed", type=int, default=GATE_FIRST_SEED)
    gate.add_argument("--processes", type=int, default=None)
    gate.add_argument("--out", type=str, default="outputs/planner/reports/roster_gate.json")

    smoke = commands.add_parser("smoke", help="small end-to-end check of the planner")
    smoke.add_argument("--config", type=str, default=None)
    smoke.add_argument("--patterns", type=str, default=None)
    smoke.add_argument("--quick", action="store_true")
    return parser


def _load(config_path: str | None) -> PlannerConfig:
    return load_planner_config(config_path)


def run_dump_config(args: argparse.Namespace) -> int:
    """Write the planner configuration and the shift pattern specification to YAML."""
    dump_planner_config(PlannerConfig(), Path(args.out))
    dump_pattern_spec(PatternSpec(), Path(args.patterns_out))
    print(f"wrote {args.out} and {args.patterns_out}")
    return 0


def foundation_summary(config: PlannerConfig, spec: PatternSpec) -> list[str]:
    """Lines that summarise the calendar, the requirement and the pattern sets."""
    calendar = PlannerCalendar(config)
    saturday = date.fromisoformat(config.start_date) + timedelta(days=6)
    windows = ", ".join(
        f"{minutes_to_clock(start)}-{minutes_to_clock(end)}"
        for start, end in calendar.rush_windows(saturday)
    )
    full = full_pattern_set(config, spec)
    solver = solver_pattern_set(config, spec)
    table = requirement_for_cycle(
        default_model(calendar, config), calendar, config, date.fromisoformat(config.start_date), 1
    )
    staff_hours = sum(table.staff_minutes(day) for day in table.dates()) / 60.0
    peak = max(row.registers for _, row in table.slots(saturday))
    return [
        f"cycle starts {config.start_date}, {config.rules.planning_weeks} weeks",
        f"saturday rush windows: {windows}",
        f"patterns on the {spec.grid_minutes}-minute grid: "
        f"{len(full.weekday)} weekday, {len(full.sunday)} sunday",
        f"patterns for the solver ({spec.solver_grid_minutes}-minute grid): "
        f"{len(solver.weekday)} weekday, {len(solver.sunday)} sunday",
        f"first week needs {staff_hours:.0f} staff hours, {peak} registers at the saturday peak",
    ]


def run_smoke(args: argparse.Namespace) -> int:
    """Run the planner smoke check; --quick prints the foundation summary only."""
    config = _load(args.config)
    spec = load_pattern_spec(args.patterns)
    for line in foundation_summary(config, spec):
        print(line)
    print("planner foundation: ok")
    if args.quick:
        return 0
    failed = 0
    for check in run_checks():
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")
        failed += not check.ok
    print("planner smoke: " + ("ok" if not failed else f"{failed} checks failed"))
    return 1 if failed else 0


def run_requirement(args: argparse.Namespace) -> int:
    """Print the staffing requirement of a cycle."""
    config = _load(args.config)
    calendar = PlannerCalendar(config)
    start = date.fromisoformat(args.start or config.start_date)
    table = requirement_for_cycle(
        default_model(calendar, config), calendar, config, start, args.weeks
    )
    for line in requirement_summary(table):
        print(line)
    return 0


def run_export_openapi(args: argparse.Namespace) -> int:
    """Write the planner API schema to docs/planner_openapi.json."""
    import json

    from ree.planner.api.planner_app import create_planner_app

    target = Path("docs") / "planner_openapi.json"
    schema = create_planner_app().openapi()
    target.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {target}")
    return 0


def run_api(args: argparse.Namespace) -> int:
    """Serve the planner API with uvicorn."""
    import uvicorn

    from ree.planner.api.planner_app import create_planner_app

    uvicorn.run(create_planner_app(), host=args.host, port=args.port)
    return 0


def run_ui(args: argparse.Namespace) -> int:
    """Open the manager or the worker screen with Streamlit."""
    import subprocess
    import sys

    name = "worker_app.py" if args.worker else "manager_app.py"
    app_path = Path(__file__).resolve().parent / "ui" / name
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
    print(f"planner screen at http://{args.host}:{args.port} (Ctrl+C to stop)")
    return subprocess.call(command)


def run_labour_rules(args: argparse.Namespace) -> int:
    """Print the labour rules as a Markdown table."""
    print(labour_rules_markdown(PlannerConfig().rules))
    return 0


def roster_summary(solution: RosterSolution, config: PlannerConfig) -> list[str]:
    """Lines that summarise a solved roster: status, checks, fairness and terms."""
    report = solution.report
    metrics = solution.metrics
    lines = [
        f"status {solution.status} with {solution.backend} in {solution.seconds:.1f} s, "
        f"objective {solution.objective}, relaxed {solution.relaxed or 'nothing'}",
    ]
    if report is not None:
        lines.append(
            f"checker: {len(report.errors)} errors, {len(report.warnings)} warnings, "
            f"{report.shortfall_minutes():.0f} uncovered staff-minutes"
        )
    if metrics is not None:
        limit = config.objective.tier_epsilon
        lines.append(
            f"fairness: burden spread {metrics.burden_spread:.3f}, "
            f"largest tier gap {metrics.max_tier_gap:.1%} (limit {limit:.0%}), "
            f"largest hours deviation {metrics.max_hours_deviation:.2f} h"
        )
    nonzero = {name: round(value) for name, value in solution.terms.items() if value}
    if nonzero:
        lines.append(f"objective terms: {nonzero}")
    lines.extend(f"note: {note}" for note in solution.notes)
    return lines


def run_roster(args: argparse.Namespace) -> int:
    """Solve a roster for a simulated store and print its summary."""
    extra: dict[str, object] = {}
    if args.mode:
        extra["solver.mode"] = args.mode
    if args.backend:
        extra["solver.backend"] = args.backend
    if args.posts_only:
        extra["objective.scores_in_roster"] = False
    if args.budget is not None:
        key = "solver.time_limit_seconds" if args.mode == "fast" else "solver.deterministic_time"
        extra[key] = args.budget
    config = store_config(args.store, **extra)
    config, workers, requirement, scores = scenario_inputs(
        args.seed, config, args.weeks, args.store
    )
    print(
        f"store {args.store}: {len(workers)} workers, {args.weeks} week(s), "
        f"mode {config.solver.mode}, seed {args.seed}"
    )
    solution = plan_roster(workers, requirement, config, scores, weeks=args.weeks)
    for line in roster_summary(solution, config):
        print(line)
    errors = solution.report.errors if solution.report is not None else []
    return 0 if not errors else 1


def run_daily(args: argparse.Namespace) -> int:
    """Solve a roster and plan its days; optionally print the board and relief of one day."""
    extra: dict[str, object] = {}
    if args.backend:
        extra["solver.backend"] = args.backend
    if args.posts_only:
        extra["objective.scores_in_roster"] = False
    config = store_config(args.store, **extra)
    config, workers, requirement, scores = scenario_inputs(args.seed, config, 1, args.store)
    solution = plan_roster(workers, requirement, config, scores, weeks=1)
    print(f"roster: status {solution.status}, {len(solution.plan.assignments)} shifts")
    results = plan_days(solution.plan, requirement, config, scores)
    failed = False
    for day, result in results.items():
        report = result.report
        failed = failed or bool(report.errors)
        print(
            f"{day} {day.strftime('%a')}: {len(report.errors)} errors, "
            f"{len(report.warnings)} warnings, {result.metrics.breaks} breaks "
            f"(at most {result.metrics.peak_simultaneous_breaks} at once), "
            f"{result.metrics.relief_handovers} relief handovers, "
            f"{result.metrics.capacity_gap_minutes} person-minutes below capacity"
        )
    if args.day:
        chosen = results.get(date.fromisoformat(args.day))
        if chosen is None:
            print(f"no shifts on {args.day}")
            return 1
        print(f"board for {args.day}")
        for line in board_lines(chosen.plan):
            print("  " + line)
        print("relief chain")
        for line in chosen.relief_lines:
            print("  " + line)
        for alert in chosen.plan.alerts:
            print(f"  alert: {alert}")
    return 1 if failed else 0


def run_audit_command(args: argparse.Namespace) -> int:
    """Run the fairness audit and write its report; the exit code says whether it passed."""
    config = evaluation_config(args.store)
    if args.report_only:
        saved = audit_table_path(config, args.reduced)
        if not saved.exists():
            print(f"no audit at {saved}; run `planner audit` first")
            return 1
        frame = pd.read_csv(saved)
    else:
        count = args.seeds if args.seeds is not None else (1 if args.reduced else AUDIT_SEEDS)
        seeds = list(range(FIRST_SEED, FIRST_SEED + count))
        frame = run_audit(config, seeds, jobs=args.jobs, progress=True)
    for written in write_audit(frame, config, args.reduced, save_table=not args.report_only):
        print(f"wrote {written}")
    limit = config.objective.tier_epsilon
    guarded = frame[frame["variant"].isin(GUARDED)]
    worst = float(guarded["max_tier_gap"].max())
    print(f"largest tier gap {worst:.3f} (limit {limit:.2f})")
    passed = audit_passed(frame, config)
    print("audit passed" if passed else "audit FAILED")
    return 0 if passed else 1


def run_stability_command(args: argparse.Namespace) -> int:
    """Run the stability study and write its report."""
    config = store_config(args.store)
    count = args.seeds if args.seeds is not None else (2 if args.reduced else STABILITY_SEEDS)
    seeds = list(range(FIRST_STABILITY_SEED, FIRST_STABILITY_SEED + count))
    frame = run_stability(config, seeds, jobs=args.jobs, progress=True)
    for written in write_stability(frame, config, args.reduced):
        print(f"wrote {written}")
    seeds = len(frame)
    print(
        f"assignments changed per roster of {frame['assignments'].mean():.0f}: "
        f"{frame['changed_with_reference'].sum() / seeds:.0f} with the previous roster as "
        f"reference, {frame['changed_without_reference'].sum() / seeds:.0f} without"
    )
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the plans on the scenarios and write results, report and figures."""
    config = evaluation_config(args.store)
    seeds, scenarios = evaluation_plan(config, args.reduced, args.seeds)
    print(
        f"evaluating {len(seeds)} seeds, {len(scenarios)} scenarios, store {args.store}, "
        f"{config.evaluation.replications} repetitions each"
    )
    frame = run_evaluation(config, seeds, scenarios, jobs=args.jobs, progress=True)
    path = results_path(config, args.reduced)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    for written in write_outputs(frame, config, args.store, args.reduced):
        print(f"wrote {written}")
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Rebuild the report and figures from the saved results."""
    config = evaluation_config(args.store)
    path = results_path(config, args.reduced)
    if not path.exists():
        print(f"no results at {path}; run `planner evaluate` first")
        return 1
    frame = pd.read_csv(path)
    for written in write_outputs(frame, config, args.store, args.reduced):
        print(f"wrote {written}")
    return 0


def run_queue_check(args: argparse.Namespace) -> int:
    """Compare the queue simulator with Erlang C; the exit code says whether they agree."""
    table = queue_validation()
    for row in table.itertuples(index=False):
        print(
            f"{row.servers} registers, load {row.load:.2f}, {row.discipline}: simulated wait "
            f"{row.simulated_wait_min:.3f} min, Erlang C {row.erlang_wait_min:.3f} min "
            f"({row.relative_gap:+.1%})"
        )
    shared = table[table["discipline"] == "shared"]
    worst = float(shared["relative_gap"].abs().max())
    print(f"largest gap for the shared queue: {worst:.1%}")
    return 0 if worst <= QUEUE_CHECK_TOLERANCE else 1


def run_roster_gate(args: argparse.Namespace) -> int:
    """Run the roster gate and write its JSON report; the exit code says whether it passed."""
    summary = run_gate(args.scenarios, args.first_seed, args.processes)
    write_gate_report(summary, Path(args.out))
    print(
        f"{summary['legal_rosters']} of {summary['scenarios']} rosters pass the checker with no "
        f"errors; mean uncovered {summary['mean_shortfall_minutes']} staff-minutes; largest "
        f"hours deviation {summary['max_hours_deviation']} h; largest tier gap "
        f"{summary['max_tier_gap']}; {summary['wall_seconds']} s"
    )
    passed = gate_passed(summary)
    print(f"roster gate: {'pass' if passed else 'FAIL'}; report in {args.out}")
    return 0 if passed else 1


HANDLERS = {
    "daily": run_daily,
    "api": run_api,
    "audit": run_audit_command,
    "dump-config": run_dump_config,
    "evaluate": run_evaluate,
    "stability": run_stability_command,
    "export-openapi": run_export_openapi,
    "queue-check": run_queue_check,
    "report": run_report,
    "labour-rules": run_labour_rules,
    "requirement": run_requirement,
    "roster": run_roster,
    "roster-gate": run_roster_gate,
    "smoke": run_smoke,
    "ui": run_ui,
}


def main(argv: list[str] | None = None) -> int:
    """Run the chosen planner command and return its exit code."""
    args = build_parser().parse_args(argv)
    return int(HANDLERS[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
