"""Roster solving: the two-stage CP-SAT solver, its fallbacks and the entry points."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from ortools.sat.python import cp_model

from ree.planner.checker.checks import check_roster
from ree.planner.checker.report import Report
from ree.planner.config import PlannerConfig
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.patterns import PatternSet, PatternSpec, solver_pattern_set
from ree.planner.domain.plan import RosterPlan, WorkerRecord
from ree.planner.roster.fairness import FairnessMetrics, fairness_metrics
from ree.planner.roster.model import (
    COVERAGE_TERMS,
    RELAXATION_ORDER,
    CoverageStart,
    RosterProblem,
    ScoreInputs,
    build_model,
    chosen_assignments,
    extract_plan,
    fix_weekly_hours,
    hint_assignments,
    neutral_scores,
    objective_expression,
    plan_from_start,
    roster_scores,
    set_objective,
    term_values,
)
from ree.planner.roster.refine import refine_scores, repair_tiers

SOLVED = (cp_model.OPTIMAL, cp_model.FEASIBLE)


@dataclass
class RosterSolution:
    """A solved roster with its status, relaxed rule groups, checks and fairness measures."""

    plan: RosterPlan
    status: str
    objective: float | None
    best_bound: float | None
    seconds: float
    relaxed: list[str]
    terms: dict[str, float]
    backend: str
    report: Report | None = None
    metrics: FairnessMetrics | None = None
    notes: list[str] = field(default_factory=list)
    swaps: int = 0


class Solver(Protocol):
    """Interface of a roster solver."""

    def solve(self, problem: RosterProblem) -> RosterSolution:
        """Solve a roster problem."""
        ...


class CpSatSolver:
    """OR-Tools CP-SAT roster solver with a fixed random seed."""

    def __init__(self, config: PlannerConfig):
        self.config = config

    def _parameters(self, share: float = 1.0) -> cp_model.CpSolver:
        settings = self.config.solver
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = settings.random_seed
        solver.parameters.num_workers = settings.num_workers
        solver.parameters.relative_gap_limit = settings.relative_gap
        solver.parameters.log_search_progress = settings.log_search
        if settings.mode == "reproducible":
            budget = settings.deterministic_time * share
            solver.parameters.interleave_search = True
            solver.parameters.max_deterministic_time = budget
            solver.parameters.max_time_in_seconds = budget * 60
        else:
            solver.parameters.max_time_in_seconds = settings.time_limit_seconds * share
        return solver

    def coverage_start(
        self, problem: RosterProblem, relaxed: frozenset[str] = frozenset()
    ) -> CoverageStart | None:
        """Solve for coverage alone, ignoring scores, to get the first-stage solution."""
        neutral = dataclasses.replace(
            problem,
            scores=neutral_scores([w.worker_id for w in problem.workers]),
            burden_history={},
            last_roster={},
            coverage_start=None,
        )
        built = build_model(neutral, relaxed)
        set_objective(built, COVERAGE_TERMS)
        solver = self._parameters(self.config.solver.coverage_share)
        status = solver.solve(built.model)
        if status not in SOLVED:
            return None
        return chosen_assignments(built, solver)

    def solve(self, problem: RosterProblem) -> RosterSolution:
        """Solve the roster: coverage first, then the full objective.

        If no legal plan exists, rule groups give way in a fixed order and the result says
        which ones. If that fails too, the constructive heuristic is used.
        """
        started = time.perf_counter()
        settings = self.config.solver
        attempts = [frozenset()] + [
            frozenset(RELAXATION_ORDER[: count + 1]) for count in range(len(RELAXATION_ORDER))
        ]
        notes: list[str] = []
        for relaxed in attempts:
            start = problem.coverage_start if not relaxed else None
            if settings.coverage_first and start is None:
                start = self.coverage_start(problem, relaxed)
                if start is None:
                    notes.append(
                        f"no coverage solution with relaxed {sorted(relaxed) or 'nothing'}"
                    )
                    continue
            built = build_model(problem, relaxed)
            share = 1.0
            if start is not None:
                hint_assignments(built, start)
                fix_weekly_hours(built, start)
                built.model.add(objective_expression(built, COVERAGE_TERMS) <= start.cost)
                share = 1.0 - settings.coverage_share
            solver = self._parameters(share)
            status = solver.solve(built.model)
            ordered = [name for name in RELAXATION_ORDER if name in relaxed]
            if status not in SOLVED and start is None:
                notes.append(
                    f"{solver.status_name(status)} with relaxed {sorted(relaxed) or 'nothing'}"
                )
                continue
            if ordered:
                notes.append("rule groups that had to give way: " + ", ".join(ordered))
            if status in SOLVED:
                plan = extract_plan(built, solver)
                name = solver.status_name(status)
                objective: float | None = solver.objective_value
                bound: float | None = solver.best_objective_bound
                terms = term_values(built, solver)
            else:
                assert start is not None
                plan = plan_from_start(problem, start)
                name, objective, bound, terms = "COVERAGE_ONLY", None, None, {}
                notes.append(
                    "the coverage solution was kept because the second stage found nothing"
                )
            if start is not None:
                notes.append(f"coverage first: stage one cost {start.cost}")
            repaired, repairs = repair_tiers(problem, plan)
            if repairs:
                notes.append(f"tier repair swapped {len(repairs)} pairs of shifts")
            refined, swaps = refine_scores(problem, repaired)
            if swaps:
                notes.append(f"score refinement swapped {len(swaps)} pairs of shifts")
            return RosterSolution(
                plan=refined,
                status=name,
                objective=objective,
                best_bound=bound,
                seconds=time.perf_counter() - started,
                relaxed=ordered,
                terms=terms,
                backend="cp_sat",
                notes=notes,
                swaps=len(swaps),
            )
        from ree.planner.roster.fallback import HeuristicSolver

        fallback = HeuristicSolver(self.config).solve(problem)
        fallback.notes = [*notes, "fell back to the constructive heuristic", *fallback.notes]
        fallback.seconds = time.perf_counter() - started
        return fallback


def make_solver(config: PlannerConfig) -> Solver:
    """Solver chosen by the configuration."""
    if config.solver.backend == "heuristic":
        from ree.planner.roster.fallback import HeuristicSolver

        return HeuristicSolver(config)
    return CpSatSolver(config)


def plan_roster(
    workers: Sequence[WorkerRecord],
    requirement: RequirementTable,
    config: PlannerConfig,
    scores: ScoreInputs | None = None,
    patterns: PatternSet | None = None,
    start: date | None = None,
    weeks: int | None = None,
    burden_history: Mapping[str, float] | None = None,
    last_roster: Mapping[tuple[str, date], str] | None = None,
    solver: Solver | None = None,
    coverage_start: CoverageStart | None = None,
) -> RosterSolution:
    """Solve a roster for the workers and requirement, check it and measure its fairness."""
    horizon = weeks if weeks is not None else config.rules.planning_weeks
    first = start if start is not None else date.fromisoformat(config.start_date)
    used_scores = (
        roster_scores(scores, config)
        if scores is not None
        else neutral_scores([w.worker_id for w in workers])
    )
    problem = RosterProblem(
        start_date=first,
        weeks=horizon,
        workers=list(workers),
        patterns=patterns if patterns is not None else solver_pattern_set(config, PatternSpec()),
        requirement=requirement,
        scores=used_scores,
        config=config,
        burden_history=dict(burden_history or {}),
        last_roster=dict(last_roster or {}),
        coverage_start=coverage_start,
    )
    used = solver if solver is not None else make_solver(config)
    solution = used.solve(problem)
    solution.report = check_roster(solution.plan, config.rules, requirement)
    solution.metrics = fairness_metrics(
        solution.plan,
        config,
        used_scores.tiers,
        used_scores.z,
        burden_history,
        last_roster,
    )
    return solution


def solve_coverage(
    workers: Sequence[WorkerRecord],
    requirement: RequirementTable,
    config: PlannerConfig,
    patterns: PatternSet | None = None,
    start: date | None = None,
    weeks: int | None = None,
) -> CoverageStart | None:
    """Coverage-only solution of a roster problem, ignoring scores."""
    horizon = weeks if weeks is not None else config.rules.planning_weeks
    first = start if start is not None else date.fromisoformat(config.start_date)
    problem = RosterProblem(
        start_date=first,
        weeks=horizon,
        workers=list(workers),
        patterns=patterns if patterns is not None else solver_pattern_set(config, PatternSpec()),
        requirement=requirement,
        scores=neutral_scores([w.worker_id for w in workers]),
        config=config,
    )
    return CpSatSolver(config).coverage_start(problem)
