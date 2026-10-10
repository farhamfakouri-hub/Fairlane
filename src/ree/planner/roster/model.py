"""CP-SAT model of the roster: variables, constraints and objective terms."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from ortools.sat.python import cp_model

from ree.api.provider import ScoreProvider
from ree.planner.config import PlannerConfig
from ree.planner.domain.demand_types import PlannerCalendar, RequirementTable
from ree.planner.domain.patterns import PatternSet
from ree.planner.domain.plan import Assignment, RosterPlan, WorkerRecord
from ree.planner.domain.shifts import DAY_MINUTES, ShiftPattern
from ree.planner.roster.fairness import BURDEN_SCALE, day_burden, tier_groups

NO_SCORES_TIER = "all"
UNRATED_TIER = "unrated"
FAR_START = 3000
RELAXATION_ORDER: tuple[str, ...] = ("floor", "weekly_rest", "contract_hours")
COVERAGE_TERMS: tuple[str, ...] = (
    "understaff",
    "overstaff",
    "register_short",
    "sco_short",
    "floor",
)


@dataclass(frozen=True)
class ScoreInputs:
    """Scores as the roster sees them: clipped z scores, tiers and workers who need exploration."""

    z: dict[str, float]
    tiers: dict[str, str]
    exploration: frozenset[str]


def neutral_scores(worker_ids: Sequence[str]) -> ScoreInputs:
    """Score inputs with every worker average and in one tier, so scores play no part."""
    return ScoreInputs(
        z={worker_id: 0.0 for worker_id in worker_ids},
        tiers={worker_id: NO_SCORES_TIER for worker_id in worker_ids},
        exploration=frozenset(),
    )


def clipped_z(score: float, clip: float) -> float:
    """Standard score of a 50 + 15 z score, clipped to a symmetric limit."""
    return max(-clip, min(clip, (score - 50.0) / 15.0))


def roster_scores(scores: ScoreInputs, config: PlannerConfig) -> ScoreInputs:
    """Score inputs the roster may use: neutral when the configuration turns scores off."""
    if config.objective.scores_in_roster:
        return scores
    return neutral_scores(sorted(scores.z))


def score_inputs(
    provider: ScoreProvider, worker_ids: Sequence[str], config: PlannerConfig
) -> ScoreInputs:
    """Score inputs from a score provider.

    Cashiers who are unrated or provisional get a neutral score and join the exploration
    group; the others get a z score scaled by their confidence.
    """
    z: dict[str, float] = {}
    tiers: dict[str, str] = {}
    exploration: set[str] = set()
    for worker_id in worker_ids:
        record = provider.get_score(worker_id)
        if record is None:
            z[worker_id] = 0.0
            tiers[worker_id] = UNRATED_TIER
            exploration.add(worker_id)
            continue
        tier = record.tier_for("peak")
        unrated = tier == UNRATED_TIER or record.provisional
        tiers[worker_id] = UNRATED_TIER if unrated else tier
        if unrated:
            exploration.add(worker_id)
            z[worker_id] = 0.0
        else:
            confidence = max(0.0, min(1.0, record.confidence_for("peak")))
            z[worker_id] = (
                clipped_z(record.score_for("peak"), config.objective.score_clip) * confidence
            )
    return ScoreInputs(z, tiers, frozenset(exploration))


@dataclass(frozen=True)
class CoverageStart:
    """A coverage-only solution: the chosen assignments and their cost."""

    assignments: frozenset[tuple[str, date, str]]
    cost: int


@dataclass
class RosterProblem:
    """Everything a roster solver needs: workers, patterns, requirement, scores and history."""

    start_date: date
    weeks: int
    workers: list[WorkerRecord]
    patterns: PatternSet
    requirement: RequirementTable
    scores: ScoreInputs
    config: PlannerConfig
    burden_history: dict[str, float] = field(default_factory=dict)
    last_roster: dict[tuple[str, date], str] = field(default_factory=dict)
    coverage_start: CoverageStart | None = None

    def dates(self) -> list[date]:
        """Every date of the cycle."""
        return [self.start_date + timedelta(days=offset) for offset in range(7 * self.weeks)]


def rush_weights(requirement: RequirementTable) -> dict[tuple[date, int], float]:
    """Weight from 0 to 1 of each slot by its offered load, so that busy slots count more."""
    loads = [row.offered_load for day in requirement.dates() for _, row in requirement.slots(day)]
    if not loads:
        return {}
    low, high = min(loads), max(loads)
    span = high - low
    weights: dict[tuple[date, int], float] = {}
    for day in requirement.dates():
        for start, row in requirement.slots(day):
            weights[(day, start)] = (row.offered_load - low) / span if span > 0 else 0.0
    return weights


@dataclass
class BuiltModel:
    """A CP-SAT model with its variables and objective terms."""

    model: cp_model.CpModel
    problem: RosterProblem
    days: list[date]
    x: dict[tuple[int, int, int], cp_model.IntVar]
    patterns_by_day: list[tuple[ShiftPattern, ...]]
    terms: dict[str, tuple[list[cp_model.IntVar], list[int]]]
    relaxed: frozenset[str]
    shortage: list[cp_model.IntVar] = field(default_factory=list)


def _blocked_for(worker: WorkerRecord, day: date, pattern: ShiftPattern) -> bool:
    return any(worker.is_blocked(day, block.paid_start, block.paid_end) for block in pattern.blocks)


def _softly_blocked(worker: WorkerRecord, day: date, pattern: ShiftPattern) -> bool:
    return any(worker.is_softly_blocked(day, b.paid_start, b.paid_end) for b in pattern.blocks)


def _add_rest_cliques(
    model: cp_model.CpModel,
    x: dict[tuple[int, int, int], cp_model.IntVar],
    by_worker_day: dict[tuple[int, int], list[int]],
    patterns_by_day: list[tuple[ShiftPattern, ...]],
    wi: int,
    di: int,
    rest_limit: int,
) -> None:
    first = by_worker_day.get((wi, di), [])
    second = by_worker_day.get((wi, di + 1), [])
    if not first or not second:
        return
    for end in sorted({patterns_by_day[di][pi].paid_end for pi in first}):
        late = [x[(wi, di, pi)] for pi in first if patterns_by_day[di][pi].paid_end >= end]
        early = [
            x[(wi, di + 1, qi)]
            for qi in second
            if patterns_by_day[di + 1][qi].paid_start < end - rest_limit
        ]
        if early:
            model.add(sum(late) + sum(early) <= 1)


def build_model(problem: RosterProblem, relaxed: frozenset[str] = frozenset()) -> BuiltModel:
    """Build the CP-SAT model of a roster problem.

    There is one choice of pattern for each worker and day, with the labour rules as
    constraints and coverage shortfalls as cost terms. Rule groups named in relaxed are
    left out, in the fixed order of RELAXATION_ORDER.
    """
    config = problem.config
    rules = config.rules
    objective = config.objective
    model = cp_model.CpModel()
    days = problem.dates()
    total_days = len(days)
    workers = problem.workers
    slot_minutes = problem.requirement.slot_minutes
    patterns_by_day = [problem.patterns.for_weekday(day.weekday()) for day in days]
    weights = rush_weights(problem.requirement)

    x: dict[tuple[int, int, int], cp_model.IntVar] = {}
    by_worker_day: dict[tuple[int, int], list[int]] = defaultdict(list)
    for wi, worker in enumerate(workers):
        for di, day in enumerate(days):
            for pi, pattern in enumerate(patterns_by_day[di]):
                if _blocked_for(worker, day, pattern):
                    continue
                x[(wi, di, pi)] = model.new_bool_var(f"x_{wi}_{di}_{pi}")
                by_worker_day[(wi, di)].append(pi)

    works: dict[tuple[int, int], cp_model.IntVar] = {}
    end_expr: dict[tuple[int, int], cp_model.LinearExprT] = {}
    start_expr: dict[tuple[int, int], cp_model.LinearExprT] = {}
    for wi in range(len(workers)):
        for di in range(total_days):
            choices = by_worker_day.get((wi, di), [])
            flag = model.new_bool_var(f"works_{wi}_{di}")
            works[(wi, di)] = flag
            variables = [x[(wi, di, pi)] for pi in choices]
            model.add(sum(variables) == flag)
            pats = patterns_by_day[di]
            end_expr[(wi, di)] = cp_model.LinearExpr.weighted_sum(
                variables, [pats[pi].paid_end for pi in choices]
            )
            start_expr[(wi, di)] = cp_model.LinearExpr.weighted_sum(
                variables, [pats[pi].paid_start for pi in choices]
            ) + FAR_START * (1 - flag)

    rest_limit = DAY_MINUTES - int(rules.min_rest_hours * 60)
    for wi in range(len(workers)):
        for di in range(total_days - 1):
            _add_rest_cliques(model, x, by_worker_day, patterns_by_day, wi, di, rest_limit)
        window = rules.max_consecutive_days + 1
        for di in range(total_days - window + 1):
            model.add(sum(works[(wi, d)] for d in range(di, di + window)) <= window - 1)

    if "weekly_rest" not in relaxed:
        weekly_rest_limit = 2 * DAY_MINUTES - int(rules.weekly_rest_hours * 60)
        for wi in range(len(workers)):
            rest_days: list[cp_model.IntVar] = []
            for di in range(total_days):
                flag = model.new_bool_var(f"rest_{wi}_{di}")
                rest_days.append(flag)
                model.add(flag + works[(wi, di)] <= 1)
                if 1 <= di <= total_days - 2:
                    model.add(
                        end_expr[(wi, di - 1)] - start_expr[(wi, di + 1)] <= weekly_rest_limit
                    ).only_enforce_if(flag)
            for week in range(problem.weeks):
                model.add(sum(rest_days[7 * week : 7 * week + 7]) >= 1)

    widen = 3.0 if "contract_hours" in relaxed else 1.0
    for wi, worker in enumerate(workers):
        tolerance = (
            worker.hours_tolerance
            if worker.hours_tolerance is not None
            else rules.hours_tolerance_hours
        ) * widen
        low = round((worker.contract_hours - tolerance) * 60)
        high = round((worker.contract_hours + tolerance) * 60)
        for week in range(problem.weeks):
            terms, coefficients = [], []
            for di in range(7 * week, 7 * week + 7):
                for pi in by_worker_day.get((wi, di), []):
                    terms.append(x[(wi, di, pi)])
                    coefficients.append(patterns_by_day[di][pi].paid_minutes)
            paid = cp_model.LinearExpr.weighted_sum(terms, coefficients)
            model.add(paid >= max(low, 0))
            model.add(paid <= high)

    register_capable = [wi for wi, w in enumerate(workers) if not w.self_checkout_only]
    cleared = [
        wi for wi, w in enumerate(workers) if w.cleared_self_checkout or w.self_checkout_only
    ]
    register_set, cleared_set = set(register_capable), set(cleared)

    aggregate: dict[tuple[str, int, int], cp_model.IntVar] = {}
    for di in range(total_days):
        for pi in range(len(patterns_by_day[di])):
            members = [wi for wi in range(len(workers)) if (wi, di, pi) in x]
            for kind, allowed in (("all", None), ("reg", register_set), ("sco", cleared_set)):
                chosen = [x[(wi, di, pi)] for wi in members if allowed is None or wi in allowed]
                var = model.new_int_var(0, max(len(chosen), 0), f"y_{kind}_{di}_{pi}")
                model.add(sum(chosen) == var)
                aggregate[(kind, di, pi)] = var

    terms: dict[str, tuple[list[cp_model.IntVar], list[int]]] = {
        name: ([], [])
        for name in (
            "understaff",
            "overstaff",
            "register_short",
            "sco_short",
            "score_reward",
            "burden_spread",
            "stability",
            "exploration",
            "soft_availability",
            "floor",
            "tier",
        )
    }
    shortage: list[cp_model.IntVar] = []

    def add_term(name: str, variable: cp_model.IntVar, coefficient: int) -> None:
        if coefficient != 0:
            terms[name][0].append(variable)
            terms[name][1].append(coefficient)

    cache: dict[tuple[str, int], int] = {}

    def coverage(pattern: ShiftPattern, slot_start: int) -> int:
        key = (pattern.pattern_id, slot_start)
        if key not in cache:
            cache[key] = pattern.coverage_minutes(slot_start, slot_start + slot_minutes)
        return cache[key]

    for di, day in enumerate(days):
        for slot_start, row in problem.requirement.slots(day):
            weight = weights.get((day, slot_start), 0.0)
            overlapping = [
                (pi, coverage(pattern, slot_start))
                for pi, pattern in enumerate(patterns_by_day[di])
                if coverage(pattern, slot_start) > 0
            ]
            total_need = slot_minutes * row.headcount
            register_need = slot_minutes * (row.registers + row.floaters)
            sco_need = slot_minutes * row.k_sc
            total_cover = cp_model.LinearExpr.weighted_sum(
                [aggregate[("all", di, pi)] for pi, _ in overlapping], [m for _, m in overlapping]
            )
            register_cover = cp_model.LinearExpr.weighted_sum(
                [aggregate[("reg", di, pi)] for pi, _ in overlapping], [m for _, m in overlapping]
            )
            sco_cover = cp_model.LinearExpr.weighted_sum(
                [aggregate[("sco", di, pi)] for pi, _ in overlapping], [m for _, m in overlapping]
            )
            under = model.new_int_var(0, total_need, f"u_{di}_{slot_start}")
            over = model.new_int_var(
                0, total_need + slot_minutes * len(workers), f"o_{di}_{slot_start}"
            )
            model.add(total_cover + under - over == total_need)
            register_short = model.new_int_var(0, register_need, f"ur_{di}_{slot_start}")
            model.add(register_cover + register_short >= register_need)
            sco_short = model.new_int_var(0, sco_need, f"us_{di}_{slot_start}")
            model.add(sco_cover + sco_short >= sco_need)
            if "floor" in relaxed:
                floor_short = model.new_int_var(0, slot_minutes * 2, f"uf_{di}_{slot_start}")
                model.add(register_cover + floor_short >= slot_minutes * 2)
                add_term("floor", floor_short, objective.floor_penalty)
            else:
                model.add(register_cover >= slot_minutes * 2)
            under_weight = (
                objective.under_penalty
                * (100 + objective.rush_under_extra * round(100 * weight))
                // 100
            )
            add_term("understaff", under, under_weight)
            add_term("overstaff", over, objective.over_penalty)
            add_term("register_short", register_short, objective.register_under_penalty)
            add_term("sco_short", sco_short, objective.sco_under_penalty)
            shortage.append(under)

    z = problem.scores.z
    for wi, worker in enumerate(workers):
        value = z.get(worker.worker_id, 0.0)
        for di, day in enumerate(days):
            for pi in by_worker_day.get((wi, di), []):
                pattern = patterns_by_day[di][pi]
                variable = x[(wi, di, pi)]
                if value != 0.0 and objective.score_weight != 0:
                    pressure = (
                        sum(
                            weights.get((day, slot_start), 0.0) * coverage(pattern, slot_start)
                            for slot_start, _ in problem.requirement.slots(day)
                        )
                        / slot_minutes
                    )
                    reward = round(objective.score_scale * value * pressure)
                    add_term("score_reward", variable, -objective.score_weight * reward)
                if _softly_blocked(worker, day, pattern):
                    add_term("soft_availability", variable, objective.availability_soft_penalty)

    history_weeks = rules.history_weeks if problem.burden_history else 0
    horizon_weeks = problem.weeks + history_weeks
    rate_vars: list[cp_model.IntVar] = []
    rate_workers: list[int] = []
    rate_sum_terms: dict[int, tuple[list[cp_model.IntVar], list[int], int]] = {}
    for wi, worker in enumerate(workers):
        if worker.contract_hours <= 0:
            continue
        scale = BURDEN_SCALE / (worker.contract_hours * horizon_weeks)
        variables, coefficients = [], []
        for di, day in enumerate(days):
            for pi in by_worker_day.get((wi, di), []):
                burden = day_burden(patterns_by_day[di][pi], day, config)
                if burden:
                    variables.append(x[(wi, di, pi)])
                    coefficients.append(round(burden * scale))
        constant = round(problem.burden_history.get(worker.worker_id, 0.0) * scale)
        upper = sum(coefficients) + constant
        rate = model.new_int_var(0, max(upper, 1), f"rate_{wi}")
        model.add(cp_model.LinearExpr.weighted_sum(variables, coefficients) + constant == rate)
        rate_vars.append(rate)
        rate_workers.append(wi)
        rate_sum_terms[wi] = (variables, coefficients, constant)
    if rate_vars:
        high = model.new_int_var(0, BURDEN_SCALE * 10, "rate_high")
        low = model.new_int_var(0, BURDEN_SCALE * 10, "rate_low")
        for rate in rate_vars:
            model.add(high >= rate)
            model.add(low <= rate)
        add_term("burden_spread", high, objective.burden_spread_weight)
        add_term("burden_spread", low, -objective.burden_spread_weight)
        tiers = problem.scores.tiers
        groups = tier_groups(
            {
                workers[wi].worker_id: tiers.get(workers[wi].worker_id, NO_SCORES_TIER)
                for wi in rate_workers
            }
        )
        index_of = {
            workers[wi].worker_id: rate for wi, rate in zip(rate_workers, rate_vars, strict=True)
        }
        count = len(rate_vars)
        total_rate = sum(rate_vars)
        epsilon = round(100 * objective.tier_epsilon)
        for tier_name, members in groups.items():
            if len(members) < 2 or len(members) >= count:
                continue
            group_rate = sum(index_of[member] for member in members)
            excess = model.new_int_var(0, BURDEN_SCALE * 100, f"tier_{tier_name}")
            model.add(
                100 * count * group_rate - (100 + epsilon) * len(members) * total_rate
                <= 100 * count * excess
            )
            model.add(
                (100 - epsilon) * len(members) * total_rate - 100 * count * group_rate
                <= 100 * count * excess
            )
            add_term("tier", excess, objective.tier_penalty)

    quota = round(objective.exploration_hours_per_week * 60)
    if quota > 0:
        for wi, worker in enumerate(workers):
            if worker.worker_id not in problem.scores.exploration:
                continue
            for week in range(problem.weeks):
                variables, coefficients = [], []
                for di in range(7 * week, 7 * week + 7):
                    day = days[di]
                    for pi in by_worker_day.get((wi, di), []):
                        rush_minutes = sum(
                            coverage(patterns_by_day[di][pi], slot_start)
                            for slot_start, row in problem.requirement.slots(day)
                            if row.rush
                        )
                        if rush_minutes:
                            variables.append(x[(wi, di, pi)])
                            coefficients.append(rush_minutes)
                missing = model.new_int_var(0, quota, f"explore_{wi}_{week}")
                model.add(
                    cp_model.LinearExpr.weighted_sum(variables, coefficients) + missing >= quota
                )
                add_term("exploration", missing, objective.exploration_penalty)

    if problem.last_roster:
        index_of_worker = {worker.worker_id: wi for wi, worker in enumerate(workers)}
        for (worker_id, day), pattern_id in problem.last_roster.items():
            wi = index_of_worker.get(worker_id)
            if wi is None or day not in days:
                continue
            di = days.index(day)
            kept = next(
                (
                    pi
                    for pi, pattern in enumerate(patterns_by_day[di])
                    if pattern.pattern_id == pattern_id and (wi, di, pi) in x
                ),
                None,
            )
            changed = model.new_bool_var(f"changed_{wi}_{di}")
            if kept is None:
                model.add(changed == works[(wi, di)])
            else:
                model.add(changed + x[(wi, di, kept)] == 1)
            add_term("stability", changed, objective.stability_weight)

    built = BuiltModel(model, problem, days, x, patterns_by_day, terms, relaxed, shortage)
    set_objective(built)
    return built


def objective_expression(
    built: BuiltModel, names: Sequence[str] | None = None
) -> cp_model.LinearExprT:
    """Weighted sum of the chosen objective terms, all of them by default."""
    variables: list[cp_model.IntVar] = []
    coefficients: list[int] = []
    for name, (term_variables, term_coefficients) in built.terms.items():
        if names is None or name in names:
            variables.extend(term_variables)
            coefficients.extend(term_coefficients)
    return cp_model.LinearExpr.weighted_sum(variables, coefficients)


def set_objective(built: BuiltModel, names: Sequence[str] | None = None) -> None:
    """Minimise the chosen objective terms."""
    built.model.minimize(objective_expression(built, names))


def term_cost(values: Mapping[str, float], names: Sequence[str]) -> int:
    """Rounded sum of the named term values."""
    return round(sum(values.get(name, 0.0) for name in names))


def chosen_assignments(built: BuiltModel, solver: cp_model.CpSolver) -> CoverageStart:
    """Assignments chosen by a solved model, with their coverage cost."""
    problem = built.problem
    chosen = frozenset(
        (problem.workers[wi].worker_id, built.days[di], built.patterns_by_day[di][pi].pattern_id)
        for (wi, di, pi), variable in built.x.items()
        if solver.boolean_value(variable)
    )
    return CoverageStart(chosen, term_cost(term_values(built, solver), COVERAGE_TERMS))


def hint_assignments(built: BuiltModel, start: CoverageStart) -> None:
    """Give the solver a known solution as a starting hint."""
    problem = built.problem
    for (wi, di, pi), variable in built.x.items():
        key = (
            problem.workers[wi].worker_id,
            built.days[di],
            built.patterns_by_day[di][pi].pattern_id,
        )
        built.model.add_hint(variable, 1 if key in start.assignments else 0)


def fix_weekly_hours(built: BuiltModel, start: CoverageStart) -> None:
    """Fix the weekly paid minutes of every worker to those of a coverage solution."""
    problem = built.problem
    weekly: dict[tuple[int, int], tuple[list[cp_model.IntVar], list[int]]] = {}
    fixed: dict[tuple[int, int], int] = {}
    for (wi, di, pi), variable in built.x.items():
        pattern = built.patterns_by_day[di][pi]
        key = (wi, di // 7)
        variables, coefficients = weekly.setdefault(key, ([], []))
        variables.append(variable)
        coefficients.append(pattern.paid_minutes)
        if (problem.workers[wi].worker_id, built.days[di], pattern.pattern_id) in start.assignments:
            fixed[key] = fixed.get(key, 0) + pattern.paid_minutes
    for key, (variables, coefficients) in weekly.items():
        built.model.add(
            cp_model.LinearExpr.weighted_sum(variables, coefficients) == fixed.get(key, 0)
        )


def extract_plan(built: BuiltModel, solver: cp_model.CpSolver) -> RosterPlan:
    """Roster plan from a solved model."""
    problem = built.problem
    assignments: list[Assignment] = []
    for (wi, di, pi), variable in sorted(built.x.items()):
        if solver.boolean_value(variable):
            assignments.append(
                Assignment(
                    problem.workers[wi].worker_id, built.days[di], built.patterns_by_day[di][pi]
                )
            )
    workers: Mapping[str, WorkerRecord] = {w.worker_id: w for w in problem.workers}
    return RosterPlan(problem.start_date, problem.weeks, dict(workers), assignments)


def plan_from_start(problem: RosterProblem, start: CoverageStart) -> RosterPlan:
    """Roster plan from a coverage solution."""
    assignments: list[Assignment] = []
    ordered = sorted(start.assignments, key=lambda item: (item[0], item[1]))
    for worker_id, day, pattern_id in ordered:
        for pattern in problem.patterns.for_weekday(day.weekday()):
            if pattern.pattern_id == pattern_id:
                assignments.append(Assignment(worker_id, day, pattern))
                break
    workers: Mapping[str, WorkerRecord] = {w.worker_id: w for w in problem.workers}
    return RosterPlan(problem.start_date, problem.weeks, dict(workers), assignments)


def term_values(built: BuiltModel, solver: cp_model.CpSolver) -> dict[str, float]:
    """Value of each objective term in a solved model."""
    values: dict[str, float] = {}
    for name, (variables, coefficients) in built.terms.items():
        values[name] = float(
            sum(solver.value(v) * c for v, c in zip(variables, coefficients, strict=True))
        )
    return values


def build_calendar(config: PlannerConfig) -> PlannerCalendar:
    """Planner calendar for a configuration."""
    return PlannerCalendar(config)
