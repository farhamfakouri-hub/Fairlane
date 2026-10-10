"""Configuration of the shift planner: hours, demand, staffing, labour rules, solver, evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from ree.config import (
    CLOCK_PATTERN,
    MONTH_DAY_PATTERN,
    StrictModel,
    Weekday,
    clock_to_minutes,
    config_hash,
)

DEFAULT_PLANNER_CONFIG_ENV = "REE_PLANNER_CONFIG"
OPENING_ORDER: tuple[int, ...] = (1, 4, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
Clock = str
WEEKDAYS_IN_ORDER: tuple[Weekday, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _clock(value: str) -> str:
    if not CLOCK_PATTERN.match(value):
        raise ValueError(f"expected HH:MM, got {value!r}")
    return value


def _clock_pair(value: tuple[str, str]) -> tuple[str, str]:
    first, second = _clock(value[0]), _clock(value[1])
    if clock_to_minutes(first) >= clock_to_minutes(second):
        raise ValueError("start must be before end")
    return first, second


def _month_day_pair(value: tuple[str, str]) -> tuple[str, str]:
    for part in value:
        if not MONTH_DAY_PATTERN.match(part):
            raise ValueError(f"expected MM-DD, got {part!r}")
    return value


class OpeningConfig(StrictModel):
    """Opening and closing clock times for Monday to Saturday and for Sunday."""

    monday_to_saturday: tuple[Clock, Clock] = ("08:30", "21:30")
    sunday: tuple[Clock, Clock] = ("08:30", "12:30")

    @field_validator("monday_to_saturday", "sunday")
    @classmethod
    def valid_pair(cls, value: tuple[str, str]) -> tuple[str, str]:
        """Opening must be a valid clock time before closing."""
        return _clock_pair(value)


class RushConfig(StrictModel):
    """Rush windows: weekday windows, Friday and Saturday changes, Sunday and month start."""

    weekday_windows: tuple[tuple[Clock, Clock], ...] = (("10:30", "14:00"), ("18:00", "20:00"))
    friday_saturday_evening_end: Clock = "21:00"
    friday_saturday_morning_window: tuple[Clock, Clock] = ("08:00", "09:00")
    sunday_window: tuple[Clock, Clock] = ("11:00", "12:30")
    quiet_last_minutes: int = 30
    month_start_first_day: int = 28
    month_start_last_day: int = 4
    month_start_extends_evening: bool = True

    @field_validator("weekday_windows")
    @classmethod
    def windows_valid(cls, value: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
        """Each window must be a pair of valid clock times, start before end."""
        return tuple(_clock_pair(pair) for pair in value)

    @field_validator("friday_saturday_morning_window", "sunday_window")
    @classmethod
    def pair_valid(cls, value: tuple[str, str]) -> tuple[str, str]:
        """The window must be a pair of valid clock times, start before end."""
        return _clock_pair(value)

    @field_validator("friday_saturday_evening_end")
    @classmethod
    def clock_valid(cls, value: str) -> str:
        """The evening end must be a valid HH:MM clock time."""
        return _clock(value)


class SeasonConfig(StrictModel):
    """Seasonal multipliers for customer arrivals and service time, with manager overrides."""

    month_start_arrival: float = 1.10
    month_start_service: float = 1.00
    christmas_window: tuple[str, str] = ("12-20", "01-03")
    christmas_arrival: float = 1.35
    christmas_service: float = 1.15
    late_summer_window: tuple[str, str] = ("08-25", "09-07")
    late_summer_arrival: float = 1.15
    late_summer_service: float = 1.05
    manager_arrival_override: dict[str, float] = Field(default_factory=dict)

    @field_validator("christmas_window", "late_summer_window")
    @classmethod
    def month_days_valid(cls, value: tuple[str, str]) -> tuple[str, str]:
        """Season windows must be dates written MM-DD."""
        return _month_day_pair(value)


class DemandConfig(StrictModel):
    """Demand model settings: slot length, daily customers, rush profile and quantiles."""

    slot_minutes: int = 30
    daily_customers: dict[Weekday, float] = Field(
        default_factory=lambda: {
            "mon": 1300.0,
            "tue": 1250.0,
            "wed": 1400.0,
            "thu": 1450.0,
            "fri": 1800.0,
            "sat": 2000.0,
            "sun": 460.0,
        }
    )
    rush_intensity: float = 2.2
    edge_taper: float = 0.6
    level_ewma_alpha: float = 0.35
    history_weeks: int = 8
    dispersion_k: float = 60.0
    day_noise_sd: float = 0.06
    quantile_rush: float = 0.8
    quantile_calm: float = 0.5
    profile_prior_customers: float = 30.0
    min_history_weeks: int = 3

    @model_validator(mode="after")
    def slots_divide_the_hour(self) -> DemandConfig:
        """Slots must divide the hour; the calm quantile may not exceed the rush one."""
        if 60 % self.slot_minutes != 0 or self.slot_minutes % 5 != 0:
            raise ValueError("slot_minutes must divide 60 and be a multiple of 5")
        if not 0.0 < self.quantile_calm <= self.quantile_rush < 1.0:
            raise ValueError("quantiles must satisfy 0 < calm <= rush < 1")
        return self


class StaffingConfig(StrictModel):
    """Staffing model settings: service time, wait target, registers, self-checkout and relief."""

    service_minutes: float = 2.0
    t_target_minutes: float = 3.0
    alpha_rush: float = 0.20
    alpha_calm: float = 0.35
    queue_limit_customers: int = 5
    max_registers: int = 15
    minimum_open_registers: int = 2
    opening_order: tuple[int, ...] = OPENING_ORDER
    k_sc_calm: int = 3
    k_sc_rush: int = 5
    sco_zone1_calm: int = 1
    sco_zone1_rush: int = 2
    sco_zone2_calm: int = 2
    sco_zone2_rush: int = 3
    floaters_morning: int = 2
    floaters_afternoon: int = 2
    floater_switch_time: Clock = "14:00"
    break_share: float = 0.06
    safety_multiplier: float = 1.0

    @model_validator(mode="after")
    def counts_consistent(self) -> StaffingConfig:
        """Zone counts must add up and the opening order must list every register."""
        if self.sco_zone1_calm + self.sco_zone2_calm != self.k_sc_calm:
            raise ValueError("self-checkout zone counts must add up to k_sc_calm")
        if self.sco_zone1_rush + self.sco_zone2_rush != self.k_sc_rush:
            raise ValueError("self-checkout zone counts must add up to k_sc_rush")
        if sorted(self.opening_order) != list(range(1, self.max_registers + 1)):
            raise ValueError("opening_order must list every register once")
        if self.opening_order[:2] != (1, 4):
            raise ValueError("registers 1 and 4 must open first")
        return self


class LabourRules(StrictModel):
    """Labour rules as parameters with defaults; each is to be confirmed with HR."""

    max_daily_hours: float = 10.0
    min_rest_hours: float = 11.0
    weekly_rest_hours: float = 35.0
    max_weekly_hours: float = 48.0
    average_weekly_hours: float = 44.0
    average_weeks: int = 12
    max_consecutive_days: int = 6
    max_cuts_per_day: int = 1
    cut_minutes: int = 60
    cut_max_minutes: int = 90
    hours_tolerance_hours: float = 1.0
    min_block_hours: float = 3.0
    break_minutes_per_hour: float = 3.0
    break_rounding_minutes: int = 5
    break_min_block_hours: float = 3.0
    break_window_tolerance_minutes: int = 30
    max_simultaneous_breaks: int = 2
    limit_case_simultaneous_breaks: int = 3
    max_break_shift_minutes: int = 90
    clock_in_margin_minutes: int = 10
    clock_out_margin_minutes: int = 5
    margin_tolerance_minutes: int = 0
    opening_slack_minutes: int = 60
    closing_slack_minutes: int = 0
    planning_weeks: int = 3
    sunday_mandatory: bool = True
    history_weeks: int = 8


class BurdenConfig(StrictModel):
    """Burden points of unpopular shifts: closing, Sunday, Saturday, split, early start."""

    closing: int = 2
    sunday: int = 3
    saturday: int = 2
    split: int = 2
    early: int = 1
    friday_evening: int = 1
    early_before: Clock = "09:00"


class SolverConfig(StrictModel):
    """Solver settings: backend, mode, random seed, time budget and coverage-first solving."""

    backend: Literal["cp_sat", "heuristic"] = "cp_sat"
    mode: Literal["reproducible", "fast"] = "reproducible"
    random_seed: int = 7
    num_workers: int = 8
    deterministic_time: float = 60.0
    time_limit_seconds: float = 60.0
    relative_gap: float = 0.0
    log_search: bool = False
    coverage_first: bool = True
    coverage_share: float = 0.6


class ObjectiveConfig(StrictModel):
    """Weights of the roster objective and the limit of the fairness guard."""

    under_penalty: int = 400
    rush_under_extra: int = 3
    over_penalty: int = 2
    scores_in_roster: bool = True
    neutral_swaps: bool = True
    score_weight: int = 0
    score_clip: float = 2.0
    score_scale: int = 100
    burden_spread_weight: int = 30
    stability_weight: int = 15
    exploration_hours_per_week: float = 1.5
    exploration_penalty: int = 400
    tier_epsilon: float = 0.35
    availability_soft_penalty: int = 40
    sco_under_penalty: int = 250
    register_under_penalty: int = 500
    floor_penalty: int = 3000
    tier_penalty: int = 40


class DailyConfig(StrictModel):
    """Weights of the daily plan: post importance, relief handovers and break scheduling."""

    theta: float = 6.0
    calm_weight: float = 0.25
    handover_cost: float = 2.0
    importance_priority: float = 1.0
    importance_main: float = 0.8
    importance_other: float = 0.4
    importance_sco: float = 0.6
    main_registers: int = 7
    preferred_sco_bonus: float = 1.0
    break_capacity_weight: int = 100
    break_limit_weight: int = 40
    break_over_limit_weight: int = 1000
    break_rush_weight: int = 3
    break_same_start_weight: int = 2
    break_deviation_weight: int = 1
    break_solver_seconds: float = 10.0


class WorkforceConfig(StrictModel):
    """Synthetic workforce of the simulation: size, contracts and self-checkout clearance."""

    n_workers: int = 40
    student_share: float = 0.45
    student_hours: tuple[float, ...] = (15.0, 21.0)
    full_time_hours: tuple[float, ...] = (35.0, 35.0, 35.0, 30.0)
    self_checkout_only_share: float = 0.075
    preferred_self_checkout_share: float = 0.10
    cleared_self_checkout_share: float = 0.60
    blocked_days_per_week_students: float = 1.6
    blocked_days_per_week_full_time: float = 0.2
    part1_weeks: int = 8
    part1_bootstrap: int = 60


class ServiceConfig(StrictModel):
    """Customer service model of the queue simulator."""

    overhead_seconds: float = 40.0
    work_seconds: float = 80.0
    service_cv: float = 0.6
    distribution: Literal["lognormal", "exponential"] = "lognormal"
    queue_discipline: Literal["shortest_line", "shared"] = "shortest_line"
    patience_minutes: float = 8.0
    position_penalty: float = 0.15
    sco_share: float = 0.0


class LiveConfig(StrictModel):
    """Thresholds of live re-planning: busyness levels, windows and freeze period."""

    rho_open: float = 0.90
    rho_close: float = 0.55
    open_window_minutes: int = 10
    close_window_minutes: int = 20
    same_proposal_gap_minutes: int = 10
    freeze_minutes: int = 10
    stability_penalty: float = 5.0
    lateness_minutes: int = 10
    live_blend: float = 0.5


class EvaluationConfig(StrictModel):
    """Settings of the evaluation: seeds, scenarios and bootstrap replicates."""

    n_seeds: int = 8
    bootstrap_replicates: int = 2000
    wait_target_minutes: float = 3.0
    scenarios: tuple[str, ...] = (
        "normal_week",
        "saturday_surge",
        "absences",
        "season_peak",
        "forecast_error",
        "small_team",
    )
    reduced_seeds: int = 2
    replications: int = 5
    layout_penalties: tuple[float, ...] = (0.0, 0.3)


class PathsConfig(StrictModel):
    """Where the planner writes its data and outputs."""

    data_dir: str = "data"
    outputs_dir: str = "outputs/planner"
    store_name: str = "planner"


class PlannerConfig(StrictModel):
    """Complete planner configuration; the defaults match config/planner.yaml."""

    seed: int = 20261004
    start_date: str = "2026-11-01"
    opening: OpeningConfig = Field(default_factory=OpeningConfig)
    rush: RushConfig = Field(default_factory=RushConfig)
    season: SeasonConfig = Field(default_factory=SeasonConfig)
    demand: DemandConfig = Field(default_factory=DemandConfig)
    staffing: StaffingConfig = Field(default_factory=StaffingConfig)
    rules: LabourRules = Field(default_factory=LabourRules)
    burden: BurdenConfig = Field(default_factory=BurdenConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    daily: DailyConfig = Field(default_factory=DailyConfig)
    workforce: WorkforceConfig = Field(default_factory=WorkforceConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("start_date")
    @classmethod
    def start_is_a_sunday(cls, value: str) -> str:
        """The roster cycle starts on a Sunday because weeks run Sunday to Saturday."""
        from datetime import date

        parsed = date.fromisoformat(value)
        if parsed.weekday() != 6:
            raise ValueError("the roster cycle starts on a Sunday (weeks run Sunday to Saturday)")
        return value

    def with_overrides(self, overrides: Mapping[str, Any]) -> PlannerConfig:
        """Copy of the configuration with dotted-path overrides applied."""
        payload = self.model_dump(mode="python")
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            cursor = payload
            for part in parts[:-1]:
                if part not in cursor:
                    raise KeyError(f"unknown planner config section {part!r} in {dotted_key!r}")
                cursor = cursor[part]
            if parts[-1] not in cursor:
                raise KeyError(f"unknown planner config key {dotted_key!r}")
            cursor[parts[-1]] = value
        return PlannerConfig.model_validate(payload)


def planner_config_hash(config: PlannerConfig) -> str:
    """SHA-256 of the canonical JSON of a planner configuration."""
    return config_hash(config)


def load_planner_config(path: str | Path | None = None) -> PlannerConfig:
    """Load a planner configuration from a YAML file.

    Without a path it uses the file named by the environment variable, or the defaults.
    """
    if path is None:
        import os

        env_path = os.environ.get(DEFAULT_PLANNER_CONFIG_ENV)
        if env_path:
            path = env_path
    if path is None:
        return PlannerConfig()
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return PlannerConfig.model_validate(payload or {})


def dump_planner_config(config: PlannerConfig, path: str | Path) -> None:
    """Write a planner configuration to a YAML file."""
    text = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, default_flow_style=False)
    Path(path).write_bytes(text.encode("utf-8"))
