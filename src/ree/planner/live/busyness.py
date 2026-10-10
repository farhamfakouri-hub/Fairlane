"""Measures of how busy the registers are, kept over a short window."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ree.planner.config import LiveConfig
from ree.planner.staffing.erlang import mean_wait


@dataclass(frozen=True)
class RegisterUsage:
    """Open and busy seconds of one register."""

    register: int
    open_seconds: float
    busy_seconds: float


@dataclass(frozen=True)
class Reading:
    """A reading at a minute: busyness, open registers and customers waiting."""

    minute: int
    busyness: float
    open_registers: int
    waiting: int


def busyness(usage: Sequence[RegisterUsage]) -> float:
    """Share of the open time that registers were busy, between zero and one."""
    opened = sum(item.open_seconds for item in usage)
    if opened <= 0.0:
        return 0.0
    return min(1.0, sum(item.busy_seconds for item in usage) / opened)


@dataclass
class BusynessTracker:
    """Keeps recent readings and tells whether busyness stayed above or below a level."""

    step_minutes: int = 5
    readings: list[Reading] = field(default_factory=list)

    def add(self, reading: Reading) -> None:
        """Store a reading."""
        self.readings.append(reading)

    def recent(self, now: int, window_minutes: int) -> list[Reading]:
        """Readings in the window that ends now."""
        return [r for r in self.readings if now - window_minutes < r.minute <= now]

    def covers(self, now: int, window_minutes: int) -> bool:
        """Whether there are enough readings to span the window."""
        needed = max(1, window_minutes // self.step_minutes)
        return len(self.recent(now, window_minutes)) >= needed

    def sustained_above(self, now: int, window_minutes: int, level: float) -> bool:
        """Whether every reading in the window is above the level."""
        return self.covers(now, window_minutes) and all(
            r.busyness > level and r.open_registers > 0 for r in self.recent(now, window_minutes)
        )

    def sustained_below(self, now: int, window_minutes: int, level: float) -> bool:
        """Whether every reading in the window is below the level."""
        return self.covers(now, window_minutes) and all(
            r.busyness < level for r in self.recent(now, window_minutes)
        )


def blended_rate(observed: float, forecast: float, config: LiveConfig) -> float:
    """Arrival rate that mixes the observed and the forecast rate."""
    return config.live_blend * observed + (1.0 - config.live_blend) * forecast


def expected_wait_minutes(
    arrivals_per_minute: float, service_minutes: float, open_registers: int
) -> float:
    """Expected wait for an arrival rate, service time and number of open registers."""
    if open_registers <= 0:
        return float("inf")
    return mean_wait(open_registers, arrivals_per_minute * service_minutes, service_minutes)
