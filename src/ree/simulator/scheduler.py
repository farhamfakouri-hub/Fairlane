"""Shift scheduler of the simulator: who works when, at which register, with optional bias."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ree.config import Config, SchedulerConfig, make_rng, replicate_stream
from ree.domain.calendar import StoreCalendar
from ree.simulator.store import RegisterProfiles

SHIFT_TEMPLATES: dict[str, tuple[tuple[int, int], ...]] = {
    "early": ((510, 600), (620, 840)),
    "mid": ((660, 855), (875, 1020)),
    "late": ((840, 1020), (1040, 1200)),
    "split": ((570, 810), (900, 1110)),
    "sunday": ((510, 750),),
}
WAVE_CAPACITY: dict[str, int] = {"early": 8, "mid": 4, "late": 8, "split": 3, "sunday": 12}
WEEKDAY_WAVES: tuple[str, ...] = ("early", "mid", "late")
ANCHOR_WAVES_WEEKDAY: tuple[str, ...] = ("early", "late")
ANCHOR_WAVES_SUNDAY: tuple[str, ...] = ("sunday",)
SUNDAY_INDEX = 6
ZONE_A_INDEX = 0
ZONE_B_INDEX = 1
PRIORITY_INDEX = 2
CARD_ONLY_INDEX = 3


@dataclass
class _Shift:
    worker_index: int
    wave: str
    day_index: int
    anchor_register: int | None = None


@dataclass
class ScheduleReport:
    """Counts of sessions created and of shifts or segments that found no place."""

    dropped_shifts: int = 0
    dropped_segments: int = 0
    sessions: int = 0
    extra: dict[str, int] = field(default_factory=dict)


def worker_classes(ground_truth: pd.DataFrame, scheduler: SchedulerConfig) -> np.ndarray:
    """Label each cashier "strong", "weak" or "neutral".

    Only the biased mode labels the fastest and slowest groups; otherwise all are neutral.
    """
    count = len(ground_truth)
    classes = np.full(count, "neutral", dtype=object)
    if scheduler.mode != "biased":
        return classes
    group_size = max(1, round(scheduler.strong_share * count))
    order = np.argsort(-ground_truth["true_speed"].to_numpy(), kind="stable")
    classes[order[:group_size]] = "strong"
    classes[order[-group_size:]] = "weak"
    return classes


def _class_multipliers(worker_class: str, scheduler: SchedulerConfig) -> np.ndarray:
    if scheduler.mode != "biased" or worker_class == "neutral":
        return np.ones(4)
    raw = np.array(
        scheduler.biased_strong_zone_weights
        if worker_class == "strong"
        else scheduler.biased_weak_zone_weights
    )
    return (1.0 - scheduler.bias_strength) + scheduler.bias_strength * raw


def _wave_probabilities(worker_class: str, scheduler: SchedulerConfig) -> np.ndarray:
    base = np.array(scheduler.wave_probabilities_early_mid_late)
    if scheduler.mode != "biased" or worker_class == "neutral":
        return base
    target = np.array(
        scheduler.biased_strong_wave_probabilities
        if worker_class == "strong"
        else scheduler.biased_weak_wave_probabilities
    )
    return (1.0 - scheduler.bias_strength) * base + scheduler.bias_strength * target


def _choose_shift_days(
    rng: np.random.Generator,
    config: Config,
    workers: pd.DataFrame,
    days: list[date],
) -> dict[int, list[int]]:
    scheduler = config.simulator.scheduler
    day_weights = np.array(scheduler.day_weights, dtype=np.float64)
    hire_dates = [timestamp.date() for timestamp in pd.to_datetime(workers["hire_date"])]
    contracts = workers["contract_type"].to_numpy()
    by_day: dict[int, list[int]] = {index: [] for index in range(len(days))}
    for week in range(config.simulator.n_weeks):
        for worker_index in range(len(workers)):
            available = [
                offset for offset in range(7) if days[7 * week + offset] >= hire_dates[worker_index]
            ]
            if not available:
                continue
            low, high = (
                scheduler.shifts_per_week_student
                if contracts[worker_index] == "student"
                else scheduler.shifts_per_week_full_time
            )
            shift_count = min(int(rng.integers(low, high + 1)), len(available))
            weights = day_weights[available]
            chosen = rng.choice(
                available, size=shift_count, replace=False, p=weights / weights.sum()
            )
            for offset in chosen:
                by_day[7 * week + int(offset)].append(worker_index)
    return by_day


def _assign_waves(
    rng: np.random.Generator,
    config: Config,
    day_index: int,
    weekday: int,
    worker_indices: list[int],
    classes: np.ndarray,
    contracts: np.ndarray,
    report: ScheduleReport,
) -> list[_Shift]:
    scheduler = config.simulator.scheduler
    load = {wave: 0 for wave in WAVE_CAPACITY}
    shifts: list[_Shift] = []
    for worker_index in rng.permutation(worker_indices):
        worker_index = int(worker_index)
        if weekday == SUNDAY_INDEX:
            order = ["sunday"]
        else:
            probabilities = _wave_probabilities(str(classes[worker_index]), scheduler)
            wave_order = list(rng.choice(WEEKDAY_WAVES, size=3, replace=False, p=probabilities))
            wants_split = (
                contracts[worker_index] == "full_time"
                and rng.random() < scheduler.split_shift_probability
            )
            order = (["split"] if wants_split else []) + wave_order + ["split"]
        for wave in order:
            if load[wave] < WAVE_CAPACITY[wave]:
                load[wave] += 1
                shifts.append(_Shift(worker_index=worker_index, wave=wave, day_index=day_index))
                break
        else:
            report.dropped_shifts += 1
    return shifts


def _pick_weighted(rng: np.random.Generator, weights: np.ndarray) -> int:
    total = weights.sum()
    if total <= 0:
        return int(rng.integers(0, len(weights)))
    return int(rng.choice(len(weights), p=weights / total))


def _assign_anchor_registers(
    rng: np.random.Generator,
    config: Config,
    weekday: int,
    shifts: list[_Shift],
    classes: np.ndarray,
) -> None:
    scheduler = config.simulator.scheduler
    store = config.store
    anchor_waves = ANCHOR_WAVES_SUNDAY if weekday == SUNDAY_INDEX else ANCHOR_WAVES_WEEKDAY
    for wave in anchor_waves:
        candidates = [shift for shift in shifts if shift.wave == wave]
        for register, multiplier_index in (
            (store.card_only_register, CARD_ONLY_INDEX),
            (store.priority_register, PRIORITY_INDEX),
        ):
            open_candidates = [shift for shift in candidates if shift.anchor_register is None]
            if not open_candidates:
                continue
            weights = np.array(
                [
                    _class_multipliers(str(classes[shift.worker_index]), scheduler)[
                        multiplier_index
                    ]
                    for shift in open_candidates
                ]
            )
            open_candidates[_pick_weighted(rng, weights)].anchor_register = register


def _is_free(occupied: list[tuple[int, int]], start: int, end: int) -> bool:
    return all(end <= other_start or start >= other_end for other_start, other_end in occupied)


def _assign_registers_for_day(
    rng: np.random.Generator,
    config: Config,
    calendar: StoreCalendar,
    profiles: RegisterProfiles,
    shifts: list[_Shift],
    classes: np.ndarray,
    report: ScheduleReport,
) -> list[dict[str, object]]:
    scheduler = config.simulator.scheduler
    store = config.store
    special = {store.priority_register, store.card_only_register}
    ordinary = [int(r) for r in profiles.registers if int(r) not in special]
    ordinary_zone = {r: calendar.lane_group(r) for r in ordinary}
    occupied: dict[int, list[tuple[int, int]]] = {int(r): [] for r in profiles.registers}
    rows: list[dict[str, object]] = []

    for shift in shifts:
        if shift.anchor_register is None:
            continue
        for start, end in SHIFT_TEMPLATES[shift.wave]:
            occupied[shift.anchor_register].append((start, end))
            rows.append(_segment_row(shift, shift.anchor_register, start, end))

    for shift in shifts:
        if shift.anchor_register is not None:
            continue
        multipliers = _class_multipliers(str(classes[shift.worker_index]), scheduler)
        previous_register: int | None = None
        for start, end in SHIFT_TEMPLATES[shift.wave]:
            free_registers = [r for r in ordinary if _is_free(occupied[r], start, end)]
            if not free_registers:
                report.dropped_segments += 1
                continue
            keep_previous = (
                previous_register is not None
                and previous_register in free_registers
                and rng.random() >= config.simulator.scheduler.register_change_probability
            )
            if keep_previous:
                register = previous_register
            else:
                weights = np.array(
                    [
                        (
                            scheduler.register_weight_zone_a * multipliers[ZONE_A_INDEX]
                            if ordinary_zone[r] == "zone_a"
                            else scheduler.register_weight_zone_b * multipliers[ZONE_B_INDEX]
                        )
                        for r in free_registers
                    ]
                )
                register = free_registers[_pick_weighted(rng, weights)]
            occupied[register].append((start, end))
            rows.append(_segment_row(shift, int(register), start, end))
            previous_register = int(register)
    return rows


def _segment_row(shift: _Shift, register: int, start: int, end: int) -> dict[str, object]:
    return {
        "worker_index": shift.worker_index,
        "register": register,
        "start_minute": start,
        "end_minute": end,
        "wave": shift.wave,
    }


def generate_sessions(
    config: Config,
    workers: pd.DataFrame,
    ground_truth: pd.DataFrame,
    calendar: StoreCalendar,
    profiles: RegisterProfiles,
) -> tuple[pd.DataFrame, ScheduleReport]:
    """Create the login sessions of every cashier for all simulated weeks.

    Shift days, waves, registers and drawers are drawn in turn. Returns the sessions
    table and a report of what could not be placed.
    """
    simulator = config.simulator
    scheduler = simulator.scheduler
    rng = make_rng(config.seed, replicate_stream(config, "scheduler"))
    start_day = simulator.start_date
    days = [start_day + timedelta(days=offset) for offset in range(7 * simulator.n_weeks)]
    classes = worker_classes(ground_truth, scheduler)
    contracts = workers["contract_type"].to_numpy()
    worker_ids = workers["worker_id"].to_numpy()
    by_day = _choose_shift_days(rng, config, workers, days)
    report = ScheduleReport()
    rows: list[dict[str, object]] = []

    for day_index, day in enumerate(days):
        weekday = day.weekday()
        shifts = _assign_waves(
            rng, config, day_index, weekday, by_day[day_index], classes, contracts, report
        )
        _assign_anchor_registers(rng, config, weekday, shifts, classes)
        day_rows = _assign_registers_for_day(
            rng, config, calendar, profiles, shifts, classes, report
        )
        drawer_ids = _drawer_ids(rng, scheduler, day, day_rows, worker_ids)
        for row, drawer_id in zip(day_rows, drawer_ids, strict=True):
            midnight = pd.Timestamp(day)
            rows.append(
                {
                    "worker_id": worker_ids[int(row["worker_index"])],
                    "register": row["register"],
                    "login_ts": midnight + pd.Timedelta(minutes=int(row["start_minute"])),
                    "logout_ts": midnight + pd.Timedelta(minutes=int(row["end_minute"])),
                    "date": midnight,
                    "weekday": weekday,
                    "drawer_id": drawer_id,
                    "wave": row["wave"],
                }
            )

    sessions = pd.DataFrame(rows)
    sessions = sessions.sort_values(["login_ts", "register"], kind="stable").reset_index(drop=True)
    sessions.insert(0, "session_id", np.arange(1, len(sessions) + 1, dtype=np.int64))
    sessions["register"] = sessions["register"].astype(np.int64)
    sessions["login_ts"] = sessions["login_ts"].astype("datetime64[ns]")
    sessions["logout_ts"] = sessions["logout_ts"].astype("datetime64[ns]")
    sessions["date"] = sessions["date"].astype("datetime64[ns]")
    report.sessions = len(sessions)
    return sessions, report


def _drawer_ids(
    rng: np.random.Generator,
    scheduler: SchedulerConfig,
    day: date,
    day_rows: list[dict[str, object]],
    worker_ids: np.ndarray,
) -> list[str]:
    by_register: dict[int, set[int]] = {}
    for row in day_rows:
        by_register.setdefault(int(row["register"]), set()).add(int(row["worker_index"]))
    shared_registers = {
        register
        for register, worker_set in sorted(by_register.items())
        if len(worker_set) >= 2 and rng.random() < scheduler.shared_till_probability
    }
    stamp = day.isoformat()
    drawer_ids: list[str] = []
    for row in day_rows:
        register = int(row["register"])
        if register in shared_registers:
            drawer_ids.append(f"{stamp}-R{register:02d}-shared")
        else:
            drawer_ids.append(f"{stamp}-R{register:02d}-{worker_ids[int(row['worker_index'])]}")
    return drawer_ids
