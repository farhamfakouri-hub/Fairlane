from __future__ import annotations

import pytest

from ree.planner.config import PlannerConfig
from ree.planner.sim.roster_gate import GATE_SCENARIOS, gate_passed, run_gate

pytestmark = pytest.mark.gate


def test_fifty_seeded_scenarios_all_give_a_legal_roster_with_exact_hours() -> None:
    summary = run_gate()
    config = PlannerConfig()
    assert summary["scenarios"] == GATE_SCENARIOS
    assert summary["failed_seeds"] == []
    assert summary["legal_rosters"] == GATE_SCENARIOS
    assert summary["all_solved"] is True
    assert float(summary["max_hours_deviation"]) <= config.rules.hours_tolerance_hours
    assert 0.0 <= float(summary["max_tier_gap"]) < 1.0
    assert float(summary["mean_shortfall_minutes"]) < 120.0
    assert gate_passed(summary)
