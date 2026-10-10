"""Typed configuration of the whole project, with defaults, overrides and hashing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAY_INDEX: dict[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
CLOCK_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
MONTH_DAY_PATTERN = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
DEFAULT_CONFIG_ENV = "REE_CONFIG"


class StrictModel(BaseModel):
    """Base class of the configuration models: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def clock_to_minutes(clock: str) -> int:
    """Convert an "HH:MM" clock string to minutes after midnight."""
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


def _check_clock(value: str) -> str:
    if not CLOCK_PATTERN.match(value):
        raise ValueError(f"expected HH:MM, got {value!r}")
    return value


def _check_probabilities(values: tuple[float, ...], name: str) -> None:
    if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"{name} must be non-negative and sum to 1, got {values}")


class OpeningHours(StrictModel):
    """Opening and closing clock times for Monday to Saturday and for Sunday."""

    monday_to_saturday: tuple[str, str] = ("08:30", "20:00")
    sunday: tuple[str, str] = ("08:30", "12:30")

    @field_validator("monday_to_saturday", "sunday")
    @classmethod
    def clocks_valid(cls, value: tuple[str, str]) -> tuple[str, str]:
        """Both times must be valid clock times and opening must be before closing."""
        _check_clock(value[0])
        _check_clock(value[1])
        if clock_to_minutes(value[0]) >= clock_to_minutes(value[1]):
            raise ValueError("opening must be before closing")
        return value


class RushWindow(StrictModel):
    """A daily rush window: the days it applies to and its start and end times."""

    days: list[Weekday]
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def clock_valid(cls, value: str) -> str:
        """Start and end must be valid HH:MM clock times."""
        return _check_clock(value)


class SeasonalWindow(StrictModel):
    """A yearly period with multipliers for customer traffic and basket size."""

    name: str
    start: str
    end: str
    traffic_multiplier: float
    basket_multiplier: float

    @field_validator("start", "end")
    @classmethod
    def month_day_valid(cls, value: str) -> str:
        """Start and end must be dates written MM-DD."""
        if not MONTH_DAY_PATTERN.match(value):
            raise ValueError(f"expected MM-DD, got {value!r}")
        return value


def _default_rush_windows() -> list[RushWindow]:
    working_days: list[Weekday] = ["mon", "tue", "wed", "thu", "fri", "sat"]
    return [
        RushWindow(days=working_days, start="10:30", end="14:00"),
        RushWindow(days=working_days, start="18:00", end="20:00"),
        RushWindow(days=["fri", "sat"], start="08:00", end="09:00"),
        RushWindow(days=["sun"], start="11:00", end="12:30"),
    ]


def _default_seasonal_windows() -> list[SeasonalWindow]:
    return [
        SeasonalWindow(
            name="christmas",
            start="12-28",
            end="01-04",
            traffic_multiplier=1.35,
            basket_multiplier=1.20,
        ),
        SeasonalWindow(
            name="late_summer",
            start="08-25",
            end="09-07",
            traffic_multiplier=1.20,
            basket_multiplier=1.10,
        ),
    ]


class StoreConfig(StrictModel):
    """Layout and calendar of the simulated store: registers, hours, rushes, seasons."""

    n_registers: int = 15
    opening: OpeningHours = OpeningHours()
    rush_windows: list[RushWindow] = _default_rush_windows()
    busy_days: list[Weekday] = ["fri", "sat", "sun"]
    priority_register: int = 1
    card_only_register: int = 4
    busy_zone_last_register: int = 7
    seasonal: list[SeasonalWindow] = _default_seasonal_windows()

    @model_validator(mode="after")
    def registers_consistent(self) -> StoreConfig:
        """The special registers and the busy zone must refer to registers that exist."""
        for register in (self.priority_register, self.card_only_register):
            if not 1 <= register <= self.n_registers:
                raise ValueError("special registers must exist")
        if not 1 <= self.busy_zone_last_register <= self.n_registers:
            raise ValueError("busy zone must end at an existing register")
        return self


class PopulationConfig(StrictModel):
    """Spread of cashier skill, contracts and newcomers in the simulated staff."""

    n_cashiers: int = 25
    student_share: float = 0.40
    student_weekly_hours: float = 20.0
    full_time_weekly_hours: float = 35.0
    true_speed_sd: float = 0.20
    peak_effect_mean: float = 0.0
    peak_effect_sd: float = 0.08
    speed_error_correlation: float = -0.30
    newcomer_share: float = 0.12
    newcomer_deficit_low: float = 0.15
    newcomer_deficit_high: float = 0.40
    learning_tau_weeks: float = 4.0
    veteran_min_tenure_weeks: int = 26
    veteran_max_tenure_weeks: int = 260
    session_effect_sd: float = 0.03

    @field_validator("speed_error_correlation")
    @classmethod
    def correlation_valid(cls, value: float) -> float:
        """The speed-error correlation must lie strictly between -1 and 1."""
        if not -1.0 < value < 1.0:
            raise ValueError("correlation must be inside (-1, 1)")
        return value


class TimingConfig(StrictModel):
    """Parameters of the transaction timing model: item gaps, fixed costs, extras."""

    alpha_seconds: float = 25.0
    alpha_split: tuple[float, float, float] = (0.32, 0.28, 0.40)
    fixed_noise_sigma: float = 0.25
    beta_seconds: float = 2.9
    gap_sigma: float = 0.35
    gamma_seconds: float = 0.7
    hand_keyed_share: float = 0.15
    tag_seconds_low: float = 5.0
    tag_seconds_high: float = 7.0
    tagged_share: float = 0.07
    loyalty_seconds: float = 8.0
    loyalty_probability: float = 0.60
    loyalty_at_end_probability: float = 0.70
    voucher_seconds: float = 4.0
    voucher_count_probabilities: tuple[float, float, float, float] = (0.80, 0.14, 0.045, 0.015)
    customer_delay_probability_per_transaction: float = 0.04
    customer_delay_low: float = 10.0
    customer_delay_high: float = 60.0
    interruption_probability_per_transaction: float = 0.015
    interruption_low: float = 60.0
    interruption_high: float = 240.0
    interruption_mix_supervisor_cheque_price: tuple[float, float, float] = (0.5, 0.3, 0.2)
    emit_exception_events: bool = False
    unload_low: float = 3.0
    unload_high: float = 12.0
    post_payment_low: float = 10.0
    post_payment_high: float = 40.0
    priority_lane_slowness: float = 1.25
    manual_entry_extra_low: float = 3.0
    manual_entry_extra_high: float = 8.0
    void_extra_low: float = 5.0
    void_extra_high: float = 15.0

    @model_validator(mode="after")
    def probabilities_valid(self) -> TimingConfig:
        """Each probability vector must be non-negative and sum to one."""
        _check_probabilities(self.alpha_split, "alpha_split")
        _check_probabilities(self.voucher_count_probabilities, "voucher_count_probabilities")
        _check_probabilities(
            self.interruption_mix_supervisor_cheque_price,
            "interruption_mix_supervisor_cheque_price",
        )
        return self


class BasketConfig(StrictModel):
    """Parameters of the basket generator: class mixture per lane group, item counts."""

    mixture_zone_a: tuple[float, float, float] = (0.45, 0.40, 0.15)
    mixture_zone_b: tuple[float, float, float] = (0.62, 0.32, 0.06)
    mixture_priority: tuple[float, float, float] = (0.50, 0.37, 0.13)
    mixture_card_only: tuple[float, float, float] = (0.80, 0.19, 0.01)
    small_median_items: float = 7.0
    small_log_sigma: float = 0.60
    medium_decay_items: float = 12.0
    big_excess_median_items: float = 18.0
    big_excess_log_sigma: float = 0.60
    big_max_items: int = 150
    tagged_register_log_sigma: float = 0.35
    tagged_zone_multiplier_zone_a: float = 1.0
    tagged_zone_multiplier_zone_b: float = 1.2
    tagged_zone_multiplier_priority: float = 0.7
    tagged_zone_multiplier_card_only: float = 0.6

    @model_validator(mode="after")
    def mixtures_valid(self) -> BasketConfig:
        """Each class mixture must be non-negative and sum to one."""
        for name in ("mixture_zone_a", "mixture_zone_b", "mixture_priority", "mixture_card_only"):
            _check_probabilities(getattr(self, name), name)
        return self


class TrafficConfig(StrictModel):
    """Parameters of customer arrivals: lane utilisation and period multipliers."""

    utilization_zone_a: float = 0.80
    utilization_zone_b: float = 0.62
    utilization_priority: float = 0.50
    utilization_card_only: float = 0.62
    peak_multiplier: float = 1.20
    off_peak_multiplier: float = 0.80
    busy_day_multiplier: float = 1.10
    max_utilization: float = 0.95
    min_utilization: float = 0.15
    nominal_cycle_seconds: float = 140.0
    candidate_margin: float = 1.20
    stop_before_logout_seconds: float = 90.0


class ErrorConfig(StrictModel):
    """Parameters of voids, manual entries and cash discrepancies."""

    cashier_void_rate: float = 0.015
    cashier_void_slope: float = 0.45
    customer_void_rate: float = 0.010
    supervisor_void_rate: float = 0.004
    void_reason_known_probability: float = 0.70
    manual_entry_rate: float = 0.012
    manual_entry_slope: float = 0.35
    cash_share: float = 0.25
    price_per_item_mean: float = 2.4
    price_log_sigma: float = 0.30
    mistake_rate_per_cash_transaction: float = 0.008
    mistake_slope: float = 0.60
    mistake_amount_median: float = 3.0
    mistake_amount_log_sigma: float = 0.80
    natural_noise_sd: float = 0.15
    unknown_owner_probability: float = 0.015


class SchedulerConfig(StrictModel):
    """Parameters of the shift scheduler, including the biased assignment mode."""

    mode: Literal["random", "biased"] = "random"
    bias_strength: float = 0.80
    strong_share: float = 0.40
    shifts_per_week_student: tuple[int, int] = (3, 4)
    shifts_per_week_full_time: tuple[int, int] = (5, 6)
    split_shift_probability: float = 0.20
    day_weights: tuple[float, float, float, float, float, float, float] = (
        0.85,
        0.85,
        0.90,
        0.90,
        1.20,
        1.30,
        0.50,
    )
    wave_probabilities_early_mid_late: tuple[float, float, float] = (0.40, 0.25, 0.35)
    biased_strong_wave_probabilities: tuple[float, float, float] = (0.50, 0.40, 0.10)
    biased_weak_wave_probabilities: tuple[float, float, float] = (0.15, 0.20, 0.65)
    register_weight_zone_a: float = 1.3
    register_weight_zone_b: float = 0.8
    biased_strong_zone_weights: tuple[float, float, float, float] = (0.4, 1.6, 0.5, 4.0)
    biased_weak_zone_weights: tuple[float, float, float, float] = (1.8, 0.7, 1.5, 0.2)
    register_change_probability: float = 0.30
    shared_till_probability: float = 0.25

    @model_validator(mode="after")
    def probabilities_valid(self) -> SchedulerConfig:
        """Each wave probability vector must be non-negative and sum to one."""
        _check_probabilities(self.wave_probabilities_early_mid_late, "wave probabilities")
        _check_probabilities(self.biased_strong_wave_probabilities, "strong wave probabilities")
        _check_probabilities(self.biased_weak_wave_probabilities, "weak wave probabilities")
        return self


class SimulatorConfig(StrictModel):
    """Everything needed to simulate a store: start date, length and sub-models."""

    start_date: date = date(2025, 11, 24)
    n_weeks: int = 12
    replicate: int = 0
    population: PopulationConfig = PopulationConfig()
    timing: TimingConfig = TimingConfig()
    baskets: BasketConfig = BasketConfig()
    traffic: TrafficConfig = TrafficConfig()
    errors: ErrorConfig = ErrorConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    @field_validator("start_date")
    @classmethod
    def starts_on_monday(cls, value: date) -> date:
        """The simulation must start on a Monday so that weeks line up."""
        if value.weekday() != 0:
            raise ValueError("start_date must be a Monday")
        return value


class PauseRuleConfig(StrictModel):
    """Thresholds of the pause rule that separates interruptions from scanning."""

    t_idle_seconds: float = 15.0
    t_stop_seconds: float = 60.0
    tune_from_data: bool = False
    tune_quantile: float = 0.995
    tune_stop_multiplier: float = 3.0
    hard_stop_seconds: float = 5.0

    @model_validator(mode="after")
    def thresholds_ordered(self) -> PauseRuleConfig:
        """The idle threshold must be positive and below the stop threshold."""
        if not 0 < self.t_idle_seconds < self.t_stop_seconds:
            raise ValueError("t_idle must be positive and below t_stop")
        return self


class SpeedConfig(StrictModel):
    """Settings of the speed score: method, bins, regression and shrinkage."""

    method: Literal["v1", "v2"] = "v2"
    regression: Literal["ols", "huber"] = "ols"
    bin_width: int = 5
    max_binned_items: int = 80
    min_cashier_bin_transactions: int = 5
    min_store_bin_transactions: int = 30
    shrinkage_k: float = 150.0
    mad_threshold: float = 3.0
    delta_expected_range: tuple[float, float] = (5.0, 7.0)
    gamma_expected_max: float = 1.0


class AccuracyConfig(StrictModel):
    """Settings of the accuracy score: tolerance, window, weights and shrinkage."""

    tolerance_euros: float = 1.0
    window_weeks: int = 12
    half_life_weeks: float = 4.0
    error_weight: float = 0.6
    void_weight: float = 0.4
    shrinkage_k_weeks: float = 4.0
    shrinkage_k_transactions: float = 150.0
    min_cash_euros: float = 500.0

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> AccuracyConfig:
        """The error weight and the void weight must sum to one."""
        if abs(self.error_weight + self.void_weight - 1.0) > 1e-9:
            raise ValueError("error and void weights must sum to 1")
        return self


class CombineConfig(StrictModel):
    """Weights and scale that combine speed and accuracy into one score."""

    peak_speed_weight: float = 0.65
    off_peak_speed_weight: float = 0.50
    score_center: float = 50.0
    score_scale: float = 15.0
    z_cap: float = 3.0


class IntervalConfig(StrictModel):
    """Settings of the bootstrap intervals around a score."""

    level: float = 0.90
    n_bootstrap: int = 200
    seed: int = 7
    width_limit: float = 20.0
    min_weeks_for_bootstrap: int = 2


class TierConfig(StrictModel):
    """Settings of the tier assignment (A, B, C)."""

    top_share: float = 0.20
    bottom_share: float = 0.20
    min_group_size: int = 3
    min_weeks_of_data: int = 4
    conservative: bool = True


class ReviewConfig(StrictModel):
    """Thresholds of the review flag."""

    speed_z_threshold: float = 1.0
    rate_z_threshold: float = 1.5


class EngineConfig(StrictModel):
    """All settings of the scoring engine."""

    pause: PauseRuleConfig = PauseRuleConfig()
    speed: SpeedConfig = SpeedConfig()
    accuracy: AccuracyConfig = AccuracyConfig()
    combine: CombineConfig = CombineConfig()
    intervals: IntervalConfig = IntervalConfig()
    tiers: TierConfig = TierConfig()
    review: ReviewConfig = ReviewConfig()
    provisional_weeks: int = 4
    min_transactions_rated: int = 100


class ExperimentsConfig(StrictModel):
    """Settings of the experiments: seeds, parallel jobs and report parameters."""

    n_seeds: int = 10
    reduced_seeds: int = 2
    base_seed: int = 1000
    jobs: int = 4
    top_k: int = 5
    calibration_replicates: int = 12
    learning_curve_weeks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 12)
    reduced_cashiers: int = 12
    reduced_weeks: int = 3
    confidence_level: float = 0.95


class FairnessConfig(StrictModel):
    """Limits and groups used by the bias audit."""

    max_adjusted_gap_points: float = 5.0
    max_slope_gap: float = 0.35
    tenure_band_edges_weeks: tuple[int, ...] = (4, 12, 26)
    min_group_size: int = 5
    audit_seeds: int = 10


class PathsConfig(StrictModel):
    """Where simulated stores and generated outputs are written."""

    data_dir: str = "data"
    store_name: str = "default"
    outputs_dir: str = "outputs"


class Config(StrictModel):
    """Complete configuration of the project (defaults mirror config/default.yaml)."""

    seed: int = 20261004
    store: StoreConfig = StoreConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    engine: EngineConfig = EngineConfig()
    experiments: ExperimentsConfig = ExperimentsConfig()
    fairness: FairnessConfig = FairnessConfig()
    paths: PathsConfig = PathsConfig()

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        """Copy of the configuration with dotted-path overrides applied."""
        return apply_overrides(self, overrides)


def make_rng(seed: int, stream: str) -> np.random.Generator:
    """Return a random generator seeded from a base seed and the name of a stream."""
    sequence = np.random.SeedSequence([seed & 0xFFFFFFFF, zlib.crc32(stream.encode("utf-8"))])
    return np.random.default_rng(sequence)


def replicate_stream(config: Config, name: str) -> str:
    """Name of a random stream, made distinct for each simulation replicate."""
    replicate = config.simulator.replicate
    return name if replicate == 0 else f"{name}:replicate{replicate}"


def apply_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """Copy a configuration with dotted-path overrides, e.g. "engine.speed.method"."""
    payload = config.model_dump(mode="python")
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor = payload
        for part in parts[:-1]:
            if part not in cursor:
                raise KeyError(f"unknown config section {part!r} in {dotted_key!r}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise KeyError(f"unknown config key {dotted_key!r}")
        cursor[parts[-1]] = value
    return Config.model_validate(payload)


def canonical_json(model: BaseModel) -> str:
    """Serialise a model to JSON with sorted keys, so equal models give equal text."""
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def config_hash(model: BaseModel) -> str:
    """SHA-256 of the canonical JSON of a model."""
    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> Config:
    """Load a configuration from YAML, or return the defaults when there is no file."""
    if path is None:
        env_path = os.environ.get(DEFAULT_CONFIG_ENV)
        if env_path:
            path = env_path
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Config.model_validate(payload or {})


def dump_config(config: Config, path: str | Path) -> None:
    """Write a configuration to a YAML file."""
    text = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, default_flow_style=False)
    Path(path).write_text(text, encoding="utf-8")


def project_root() -> Path:
    """Path of the repository root."""
    return Path(__file__).resolve().parents[2]
