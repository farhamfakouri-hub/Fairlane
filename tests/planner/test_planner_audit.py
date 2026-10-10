from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ree.planner.cli import main
from ree.planner.eval.fairness_audit import (
    CONTROL,
    NOISE_LEVELS,
    ORDER,
    _noisy,
    audit_passed,
    audit_reading,
    audit_report,
    audit_seed,
    write_audit,
)
from ree.planner.roster.model import ScoreInputs
from ree.planner.sim.scenarios import store_config


def fast_config(**extra: object):
    return store_config("tiny", **{"solver.backend": "heuristic", **extra})


@pytest.fixture(scope="module")
def audited() -> pd.DataFrame:
    rows = audit_seed(fast_config(), 1)
    rows += audit_seed(fast_config(), 2)
    return pd.DataFrame(rows)


def test_the_audit_compares_the_plans_with_noisy_scores_and_without_the_quota(audited) -> None:
    assert set(audited["variant"]) == set(ORDER)
    assert len(audited) == 2 * len(ORDER)
    assert len(NOISE_LEVELS) == 3


def test_a_plan_overlaps_completely_with_itself_and_pays_the_same_hours_as_plan_b(audited) -> None:
    own = audited[audited["variant"] == "C"]
    assert (own["overlap_with_C"] == 1.0).all()
    base = audited[audited["variant"] == "B"]
    assert (base["paid_difference_vs_B"] == 0.0).all()
    assert (audited["max_hours_deviation"] <= 1.0 + 1e-9).all()


def test_the_audit_reports_every_measure_for_every_variant(audited) -> None:
    for column in (
        "max_tier_gap",
        "burden_spread",
        "corr_weekend",
        "corr_late",
        "corr_split",
        "corr_burden",
        "corr_availability_burden",
        "exploration_rush_minutes",
        "true_alignment",
        "days_changed_vs_B",
        "patterns_changed_vs_B",
        "corr_hours",
        "tier_hours_gap",
    ):
        assert audited[column].notna().all()
    assert (audited["overlap_with_C"].between(0.0, 1.0)).all()


def test_plan_b_differs_from_itself_in_nothing_and_the_shares_are_shares(audited) -> None:
    base = audited[audited["variant"] == "B"]
    assert (base["days_changed_vs_B"] == 0.0).all()
    assert (base["patterns_changed_vs_B"] == 0.0).all()
    assert (audited["patterns_changed_vs_B"].between(0.0, 1.0)).all()
    assert (audited["days_changed_vs_B"] >= 0.0).all()


def crafted_audit(quota_binds: bool = False, identical_shares: bool = True) -> pd.DataFrame:
    rows = []
    for seed in range(1, 5):
        for variant in ORDER:
            guarded = variant.startswith("C")
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "max_tier_gap": 0.30 if guarded else 0.20,
                    "corr_burden": -0.4 if variant == "C" else 0.0,
                    "corr_hours": 0.5 if variant == CONTROL else 0.05 if variant == "C" else 0.0,
                    "tier_hours_gap": 1.2 if variant == CONTROL else 0.2 if variant == "C" else 0.0,
                    "max_hours_deviation": 1.0,
                    "paid_difference_vs_B": 1.5 if variant == CONTROL else 0.0,
                    "corr_late": -0.3 if variant == "C" else 0.01,
                    "corr_weekend": 0.1 + (0.0 if identical_shares else 0.05 * len(variant)),
                    "corr_split": -0.1,
                    "days_changed_vs_B": 0.0 if variant == "B" else 6.0,
                    "patterns_changed_vs_B": 0.0 if variant == "B" else 0.25,
                    "exploration_rush_minutes": 600.0,
                    "overlap_with_C": (
                        0.8
                        if variant == "C without learning quota" and quota_binds
                        else 1.0
                        if variant in ("C", "C without learning quota")
                        else 0.6
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_the_reading_reports_the_tier_gap_the_correlations_and_what_changes() -> None:
    text = " ".join(audit_reading(crafted_audit(), fast_config()))
    assert "above plan B's in 4 of 4 seeds (mean difference +0.100)" in text
    assert "does not bring it down to the level of a plan that ignores the scores" in text
    assert "Score against burden per paid hour: plan C -0.40 (below zero in 4 of 4 seeds)" in text
    assert "plan B +0.00, placebo E +0.00" in text
    assert "moves 6 worker-days" in text and "25% of its shifts" in text
    assert "signed deviation of paid hours from the contract" in text
    assert "+0.05 in plan C, +0.00 in plan B and +0.00 in the placebo E" in text
    assert "tier A minus those of tier C: +0.20 h a week" in text
    assert "Negative control" in text
    assert "+0.50 and tier A minus tier C is +1.20 h a week" in text
    assert "the same in every variant" in text


def test_the_reading_says_the_quota_does_not_bind_only_when_the_roster_is_identical() -> None:
    silent = " ".join(audit_reading(crafted_audit(), fast_config()))
    assert "does not bind here" in silent
    binding = " ".join(audit_reading(crafted_audit(quota_binds=True), fast_config()))
    assert "does not bind here" not in binding
    assert "Learning quota" in binding


def test_the_reading_does_not_claim_equal_shares_when_the_correlations_differ() -> None:
    text = " ".join(audit_reading(crafted_audit(identical_shares=False), fast_config()))
    assert "the same in every variant" not in text


def test_the_audit_passes_only_when_the_score_blind_stage_fixes_the_hours() -> None:
    config = fast_config()
    assert config.solver.coverage_first is False
    frame = crafted_audit()
    assert audit_passed(frame, config) is True
    frame.loc[frame["variant"] == "C", "paid_difference_vs_B"] = 3.0
    assert audit_passed(frame, config) is False
    frozen = fast_config(**{"solver.coverage_first": True})
    clean = crafted_audit()
    assert audit_passed(clean, frozen) is True
    clean.loc[clean["variant"] == "C", "paid_difference_vs_B"] = 0.5
    assert audit_passed(clean, frozen) is False


def test_the_audit_ignores_the_negative_control_in_its_limits() -> None:
    frame = crafted_audit()
    frame.loc[frame["variant"] == CONTROL, "max_tier_gap"] = 0.9
    assert audit_passed(frame, fast_config()) is True


def test_the_reading_says_when_the_roster_does_not_respond_to_noisy_scores() -> None:
    frame = crafted_audit()
    quiet = frame[frame["variant"].str.startswith("C with noise")]
    assert (quiet["overlap_with_C"] == 0.6).all()
    text = " ".join(audit_reading(frame, fast_config()))
    assert "60%, 60%, 60% of plan C's shifts survive." in text
    frame.loc[frame["variant"].str.startswith("C with noise"), "overlap_with_C"] = 1.0
    text = " ".join(audit_reading(frame, fast_config()))
    assert "the roster does not respond to the scores at all" in text


def test_the_reading_is_empty_without_the_three_plans() -> None:
    frame = crafted_audit()
    assert audit_reading(frame[frame["variant"] != "E"], fast_config()) == []


def test_noise_is_seeded_clipped_and_leaves_unrated_workers_alone() -> None:
    scores = ScoreInputs(
        z={"a": 1.9, "b": -1.9, "c": 0.0, "d": 0.4},
        tiers={"a": "A", "b": "C", "c": "unrated", "d": "B"},
        exploration=frozenset({"c"}),
    )
    first = _noisy(scores, 1.0, 3, 2.0)
    again = _noisy(scores, 1.0, 3, 2.0)
    assert first.z == again.z
    assert first.z["c"] == 0.0
    assert all(abs(v) <= 2.0 for v in first.z.values())
    assert first.tiers == scores.tiers and first.exploration == scores.exploration
    assert first.z != _noisy(scores, 1.0, 4, 2.0).z
    assert np.allclose(list(_noisy(scores, 0.0, 3, 2.0).z.values()), list(scores.z.values()))


def test_the_report_states_the_checks_and_the_table(audited) -> None:
    text = audit_report(audited, fast_config())
    assert "# Fairness audit of the roster" in text
    assert "## Checks" in text and "## Reading" in text
    assert "## Burden, hours and pay" in text
    assert "## What the scores change and how it correlates with them" in text
    assert "Largest score-tier burden gap" in text
    assert "Learning quota" in text
    for variant in ORDER:
        assert variant in text


def test_the_audit_files_are_written(audited, tmp_path) -> None:
    config = fast_config(**{"paths.outputs_dir": str(tmp_path)})
    paths = write_audit(audited, config, reduced=False)
    assert all(path.exists() for path in paths)
    assert (tmp_path / "reports" / "fairness_audit.md").exists()


def test_the_audit_command_runs_and_writes_its_report(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ree.planner.cli as cli

    config = fast_config(**{"paths.outputs_dir": str(tmp_path)})
    monkeypatch.setattr(cli, "evaluation_config", lambda store: config)
    code = main(["audit", "--reduced", "--store", "tiny", "--seeds", "1", "--jobs", "1"])
    assert code in (0, 1)
    assert (tmp_path / "reduced" / "fairness_audit.md").exists()
    assert "largest tier gap" in capsys.readouterr().out


def test_the_report_only_flag_rebuilds_the_report_from_the_saved_table(
    tmp_path, monkeypatch, audited, capsys: pytest.CaptureFixture[str]
) -> None:
    import ree.planner.cli as cli

    config = fast_config(**{"paths.outputs_dir": str(tmp_path)})
    monkeypatch.setattr(cli, "evaluation_config", lambda store: config)
    assert main(["audit", "--reduced", "--store", "tiny", "--report-only"]) == 1
    assert "run `planner audit` first" in capsys.readouterr().out
    write_audit(audited, config, reduced=True)
    table = tmp_path / "reduced" / "fairness_audit.csv"
    report = tmp_path / "reduced" / "fairness_audit.md"
    saved = table.read_bytes()
    report.unlink()
    code = main(["audit", "--reduced", "--store", "tiny", "--report-only"])
    assert code in (0, 1)
    assert report.exists()
    assert table.read_bytes() == saved


def test_the_audit_is_reproducible(audited) -> None:
    again = pd.DataFrame(audit_seed(fast_config(), 1))
    first = audited[audited["seed"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, again.reset_index(drop=True))
