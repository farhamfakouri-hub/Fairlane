"""Constructive heuristic used when CP-SAT cannot find a legal roster."""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from ree.config import make_rng
from ree.planner.checker.checks import check_worker_rules
from ree.planner.domain.plan import Assignment, RosterPlan, WorkerRecord
from ree.planner.domain.shifts import ShiftPattern
from ree.planner.roster.model import RosterProblem, rush_weights

ATTEMPTS_PER_WEEK = 60
LEGAL_CODES = frozenset({"MIN_REST", "MAX_CONSECUTIVE_DAYS", "WEEKLY_REST", "AVAILABILITY"})


class HeuristicSolver:
    """Builds a roster week by week for each worker, then tries to improve it."""

    def __init__(self, config):
        self.config = config
        self.improvement_passes = 2

    def solve(self, problem: RosterProblem):
        """Build and improve a roster; the result is marked as coming from the heuristic."""
        from ree.planner.roster.solver import RosterSolution

        started = time.perf_counter()
        rng = make_rng(self.config.solver.random_seed, "heuristic-roster")
        state = _State(problem)
        order = sorted(
            range(len(problem.workers)),
            key=lambda i: (-problem.workers[i].contract_hours, problem.workers[i].worker_id),
        )
        notes: list[str] = []
        for week in range(problem.weeks):
            for wi in order:
                if not self._plan_week(state, wi, week, rng):
                    notes.append(f"{problem.workers[wi].worker_id}: no legal week {week + 1}")
        for _ in range(self.improvement_passes):
            for wi in [int(i) for i in rng.permutation(len(problem.workers))]:
                for week in range(problem.weeks):
                    before = state.cost()
                    saved = state.clear_week(wi, week)
                    if not self._plan_week(state, wi, week, rng) or state.cost() > before:
                        state.clear_week(wi, week)
                        state.restore(wi, saved)
        plan = state.plan()
        return RosterSolution(
            plan=plan,
            status="HEURISTIC",
            objective=float(state.cost()),
            best_bound=None,
            seconds=time.perf_counter() - started,
            relaxed=[],
            terms={},
            backend="heuristic",
            notes=notes,
        )

    def _plan_week(self, state: _State, wi: int, week: int, rng: np.random.Generator) -> bool:
        problem = state.problem
        worker = problem.workers[wi]
        rules = self.config.rules
        tolerance = (
            worker.hours_tolerance
            if worker.hours_tolerance is not None
            else rules.hours_tolerance_hours
        )
        target = worker.contract_hours * 60
        low, high = target - tolerance * 60, target + tolerance * 60
        pool = [*problem.patterns.weekday, *problem.patterns.sunday]
        paid = [p.paid_minutes for p in pool]
        smallest, largest = min(paid), max(paid)
        least = max(1, int(np.ceil(low / largest)))
        most = min(6, int(high // smallest))
        mean_paid = float(np.mean([p.paid_minutes for p in problem.patterns.weekday]))
        best: list[tuple[int, ShiftPattern]] | None = None
        best_cost: float | None = None
        successes = 0
        for _ in range(ATTEMPTS_PER_WEEK):
            guess = round(target / mean_paid) + int(rng.integers(-1, 2))
            count = min(max(guess, least), most)
            chosen = [pool[int(rng.integers(len(pool)))] for _ in range(count - 1)]
            partial = sum(p.paid_minutes for p in chosen)
            fitting = [p for p in pool if low <= partial + p.paid_minutes <= high]
            if not fitting:
                continue
            chosen.append(fitting[int(rng.integers(len(fitting)))])
            placed = self._place(state, wi, week, chosen, rng)
            if placed is None:
                continue
            cost = state.trial_cost(wi, placed)
            if best_cost is None or cost < best_cost:
                best, best_cost = placed, cost
            successes += 1
            if successes >= 3:
                break
        if best is None:
            return False
        for di, pattern in best:
            state.assign(wi, di, pattern)
        return True

    def _place(
        self,
        state: _State,
        wi: int,
        week: int,
        patterns: list[ShiftPattern],
        rng: np.random.Generator,
    ) -> list[tuple[int, ShiftPattern]] | None:
        problem = state.problem
        worker = problem.workers[wi]
        days = problem.dates()
        used: set[int] = set()
        placement: list[tuple[int, ShiftPattern]] = []
        for pattern in sorted(patterns, key=lambda p: -p.paid_minutes):
            options: list[tuple[float, int]] = []
            for di in range(7 * week, 7 * week + 7):
                if di in used or state.schedule[wi].get(di) is not None:
                    continue
                if pattern not in problem.patterns.for_weekday(days[di].weekday()):
                    continue
                if any(
                    worker.is_blocked(days[di], b.paid_start, b.paid_end) for b in pattern.blocks
                ):
                    continue
                options.append((state.gain(wi, di, pattern) + float(rng.random()), di))
            options.sort(reverse=True)
            chosen_day = None
            for _, di in options:
                trial = {**state.schedule[wi], **{d: p for d, p in placement}, di: pattern}
                if state.legal(worker, trial):
                    chosen_day = di
                    break
            if chosen_day is None:
                return None
            used.add(chosen_day)
            placement.append((chosen_day, pattern))
        return placement


class _State:
    def __init__(self, problem: RosterProblem):
        self.problem = problem
        self.days = problem.dates()
        self.schedule: list[dict[int, ShiftPattern | None]] = [dict() for _ in problem.workers]
        self.slot_minutes = problem.requirement.slot_minutes
        self.need: dict[tuple[int, int], int] = {}
        self.have: dict[tuple[int, int], int] = defaultdict(int)
        weights = rush_weights(problem.requirement)
        self.weight: dict[tuple[int, int], float] = {}
        for di, day in enumerate(self.days):
            for start, row in problem.requirement.slots(day):
                self.need[(di, start)] = row.headcount * self.slot_minutes
                self.weight[(di, start)] = 1.0 + 3.0 * weights.get((day, start), 0.0)
        self.z = problem.scores.z

    def cost(self) -> float:
        total = 0.0
        for key, need in self.need.items():
            have = self.have.get(key, 0)
            total += self.weight[key] * max(0, need - have) * 4 + max(0, have - need)
        return total

    def _coverage(self, di: int, pattern: ShiftPattern) -> list[tuple[tuple[int, int], int]]:
        found = []
        for (day_index, start), _ in self.need.items():
            if day_index != di:
                continue
            minutes = pattern.coverage_minutes(start, start + self.slot_minutes)
            if minutes:
                found.append(((di, start), minutes))
        return found

    def gain(self, wi: int, di: int, pattern: ShiftPattern) -> float:
        total = 0.0
        value = self.z.get(self.problem.workers[wi].worker_id, 0.0)
        for key, minutes in self._coverage(di, pattern):
            short = max(0, self.need[key] - self.have.get(key, 0))
            total += self.weight[key] * min(short, minutes) * 4
            total -= max(0, minutes - short)
            total += 5.0 * value * (self.weight[key] - 1.0) * minutes / self.slot_minutes
        return total

    def trial_cost(self, wi: int, placement: list[tuple[int, ShiftPattern]]) -> float:
        for di, pattern in placement:
            self.assign(wi, di, pattern)
        cost = self.cost()
        for di, pattern in placement:
            self.unassign(wi, di, pattern)
        return cost

    def assign(self, wi: int, di: int, pattern: ShiftPattern) -> None:
        self.schedule[wi][di] = pattern
        for key, minutes in self._coverage(di, pattern):
            self.have[key] += minutes

    def unassign(self, wi: int, di: int, pattern: ShiftPattern) -> None:
        self.schedule[wi].pop(di, None)
        for key, minutes in self._coverage(di, pattern):
            self.have[key] -= minutes

    def clear_week(self, wi: int, week: int) -> list[tuple[int, ShiftPattern]]:
        removed: list[tuple[int, ShiftPattern]] = []
        for di in range(7 * week, 7 * week + 7):
            pattern = self.schedule[wi].get(di)
            if pattern is not None:
                self.unassign(wi, di, pattern)
                removed.append((di, pattern))
        return removed

    def restore(self, wi: int, saved: list[tuple[int, ShiftPattern]]) -> None:
        for di, pattern in saved:
            self.assign(wi, di, pattern)

    def legal(self, worker: WorkerRecord, schedule: dict[int, ShiftPattern | None]) -> bool:
        problem = self.problem
        assignments = [
            Assignment(worker.worker_id, self.days[di], pattern)
            for di, pattern in schedule.items()
            if pattern is not None
        ]
        plan = RosterPlan(
            problem.start_date, problem.weeks, {worker.worker_id: worker}, assignments
        )
        found = check_worker_rules(plan, self.problem.config.rules)
        return not any(item.code in LEGAL_CODES for item in found)

    def plan(self) -> RosterPlan:
        problem = self.problem
        assignments = [
            Assignment(problem.workers[wi].worker_id, self.days[di], pattern)
            for wi, schedule in enumerate(self.schedule)
            for di, pattern in sorted(schedule.items())
            if pattern is not None
        ]
        workers = {w.worker_id: w for w in problem.workers}
        return RosterPlan(problem.start_date, problem.weeks, workers, assignments)
