from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from ree.planner.config import (
    OPENING_ORDER,
    PlannerConfig,
    dump_planner_config,
    load_planner_config,
    planner_config_hash,
)
from ree.planner.domain.patterns import PatternSpec, dump_pattern_spec, load_pattern_spec

ROOT = Path(__file__).resolve().parents[2]


def test_the_planner_yaml_files_equal_the_code_defaults() -> None:
    assert load_planner_config(ROOT / "config" / "planner.yaml") == PlannerConfig()
    assert load_pattern_spec(ROOT / "config" / "shift_patterns.yaml") == PatternSpec()


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    changed = PlannerConfig().with_overrides(
        {"rules.min_rest_hours": 12.0, "workforce.n_workers": 31, "solver.num_workers": 4}
    )
    path = tmp_path / "planner.yaml"
    dump_planner_config(changed, path)
    assert load_planner_config(path) == changed
    spec = PatternSpec()
    pattern_path = tmp_path / "patterns.yaml"
    dump_pattern_spec(spec, pattern_path)
    assert load_pattern_spec(pattern_path) == spec


def test_the_hash_changes_with_every_parameter() -> None:
    base = PlannerConfig()
    assert planner_config_hash(base) == planner_config_hash(PlannerConfig())
    assert planner_config_hash(base) != planner_config_hash(
        base.with_overrides({"objective.score_weight": 2})
    )


def test_overrides_reject_unknown_keys() -> None:
    with pytest.raises(KeyError):
        PlannerConfig().with_overrides({"rules.no_such_rule": 1})
    with pytest.raises(KeyError):
        PlannerConfig().with_overrides({"nowhere.x": 1})


def test_the_cycle_must_start_on_a_sunday() -> None:
    assert date.fromisoformat(PlannerConfig().start_date).weekday() == 6
    with pytest.raises(ValidationError):
        PlannerConfig(start_date="2026-11-02")


def test_store_facts_match_the_owner_answers() -> None:
    config = PlannerConfig()
    assert config.opening.monday_to_saturday == ("08:30", "21:30")
    assert config.opening.sunday == ("08:30", "12:30")
    assert config.staffing.opening_order[:7] == (1, 4, 2, 3, 5, 6, 7)
    assert config.staffing.opening_order == OPENING_ORDER
    assert (config.staffing.k_sc_calm, config.staffing.k_sc_rush) == (3, 5)
    assert config.staffing.sco_zone1_calm + config.staffing.sco_zone2_calm == 3
    assert config.staffing.sco_zone1_rush + config.staffing.sco_zone2_rush == 5
    assert (config.staffing.floaters_morning, config.staffing.floaters_afternoon) == (2, 2)
    assert config.rules.planning_weeks == 3
    assert config.rules.max_simultaneous_breaks == 2
    assert config.rules.limit_case_simultaneous_breaks == 3
    assert config.rules.max_break_shift_minutes == 90
    assert (config.rules.clock_in_margin_minutes, config.rules.clock_out_margin_minutes) == (10, 5)
    assert config.rules.max_cuts_per_day == 1 and config.rules.cut_minutes == 60


def test_inconsistent_staffing_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannerConfig().with_overrides({"staffing.k_sc_rush": 6})
    with pytest.raises(ValidationError):
        PlannerConfig().with_overrides({"staffing.opening_order": (2, 1, *OPENING_ORDER[2:])})
    with pytest.raises(ValidationError):
        PlannerConfig().with_overrides({"demand.slot_minutes": 25})
    with pytest.raises(ValidationError):
        PlannerConfig().with_overrides({"opening.sunday": ("12:30", "08:30")})


def test_the_labour_rules_document_is_generated_from_the_code_defaults() -> None:
    from ree.planner.config import LabourRules
    from ree.planner.domain.rules import RULE_NOTES, labour_rules_markdown

    assert set(RULE_NOTES) == set(LabourRules.model_fields)
    text = (ROOT / "docs" / "labour_rules.md").read_text(encoding="utf-8")
    assert labour_rules_markdown(PlannerConfig().rules) in text
    assert text.count("confirm") >= len(RULE_NOTES)
