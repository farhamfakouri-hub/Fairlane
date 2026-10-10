from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pytest
from planner_factories import CONFIG, TUESDAY, requirement_from_duty, staggered_day

from ree.config import make_rng
from ree.planner.daily.daily_plan import plan_day_from_shifts
from ree.planner.eval.live_runner import simulate_plan_live
from ree.planner.eval.plans import build_inputs, build_plan
from ree.planner.eval.world import SCENARIOS, build_world, simulate_plan
from ree.planner.live.busyness import (
    BusynessTracker,
    Reading,
    RegisterUsage,
    blended_rate,
    busyness,
    expected_wait_minutes,
)
from ree.planner.live.proposals import (
    ACCEPTED,
    CLOSE_REGISTER,
    COVER_ABSENCE,
    OPEN_REGISTER,
    REJECTED,
    REPLAN,
    STALE,
    ProposalBook,
)
from ree.planner.live.replan import (
    close_register,
    cover_absence,
    open_register,
    replan_rest_of_day,
    timeline_of,
)
from ree.planner.live.triggers import LiveController
from ree.planner.sim.queue_sim import (
    DayCustomers,
    Observation,
    QueueSettings,
    StaffInterval,
    simulate_day,
)
from ree.planner.sim.scenarios import store_config

NOW = 610


@pytest.fixture(scope="module")
def day():
    _, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    return result.plan, requirement, scores.z


def fresh_book(day) -> ProposalBook:
    plan, requirement, _ = day
    return ProposalBook(plan, requirement, CONFIG.rules)


def test_busyness_is_busy_time_over_open_time_of_the_open_registers() -> None:
    usage = [RegisterUsage(1, 300.0, 270.0), RegisterUsage(4, 300.0, 150.0)]
    assert busyness(usage) == pytest.approx(0.7)
    assert busyness([RegisterUsage(1, 0.0, 0.0)]) == 0.0
    assert busyness([RegisterUsage(1, 100.0, 140.0)]) == 1.0


def test_the_live_rate_blends_the_observed_and_forecast_arrivals() -> None:
    assert blended_rate(4.0, 2.0, CONFIG.live) == pytest.approx(3.0)
    assert expected_wait_minutes(1.5, 2.0, 4) == pytest.approx(1.0185, abs=1e-3)
    assert expected_wait_minutes(1.5, 2.0, 0) == float("inf")
    assert expected_wait_minutes(2.5, 2.0, 4) == float("inf")


def test_a_threshold_needs_the_whole_window_before_it_counts() -> None:
    tracker = BusynessTracker()
    tracker.add(Reading(600, 0.95, 4, 0))
    assert not tracker.sustained_above(600, 10, 0.9)
    tracker.add(Reading(605, 0.95, 4, 0))
    assert tracker.sustained_above(605, 10, 0.9)
    tracker.add(Reading(610, 0.85, 4, 0))
    assert not tracker.sustained_above(610, 10, 0.9)
    assert not tracker.sustained_below(610, 20, 0.55)


def test_opening_a_register_uses_the_best_floater_and_the_next_register_in_order(day) -> None:
    plan, requirement, z = day
    draft = open_register(plan, requirement, z, NOW, CONFIG, rate=1.2)
    assert draft is not None and draft.kind == OPEN_REGISTER
    assert {c.worker_id for c in draft.changes} == {"W07"}
    assert all(c.before == "FLOAT" and c.after == "R07" for c in draft.changes)
    assert min(c.start for c in draft.changes) == 620
    assert draft.wait_after is not None and draft.wait_after < draft.wait_before
    assert plan.post_at("W07", 620) == "FLOAT"
    assert draft.result.post_at("W07", 620) == "R07"
    assert draft.result.post_at("W07", 610) == plan.post_at("W07", 610)


def test_no_register_opens_when_nobody_is_free() -> None:
    _, workers, shifts, requirement, scores = staggered_day(pairs=2)
    plan = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG).plan
    assert open_register(plan, requirement, scores.z, 510, CONFIG) is None


def test_closing_a_register_releases_the_last_one_opened_and_keeps_the_minimum(day) -> None:
    plan, requirement, z = day
    opened = open_register(plan, requirement, z, NOW, CONFIG)
    assert opened is not None
    draft = close_register(opened.result, requirement, 650, CONFIG, rate=0.5)
    assert draft is not None and draft.kind == CLOSE_REGISTER
    assert {c.after for c in draft.changes} == {"FLOAT"}
    assert {c.before for c in draft.changes} == {"R07"}
    early = close_register(plan, requirement, 515, CONFIG)
    assert early is None


def test_an_absence_is_covered_by_floaters_and_the_worker_leaves_the_plan(day) -> None:
    plan, _, z = day
    draft = cover_absence(plan, "W01", z, 540, CONFIG)
    assert draft is not None and draft.kind == COVER_ABSENCE
    assert "W01" not in draft.result.shifts
    assert "W01" in draft.reason
    held = timeline_of(plan)["W01"][545]
    assert held in draft.result.occupants(545).values()
    assert cover_absence(plan, "nobody", z, 540, CONFIG) is None


def test_replanning_the_rest_of_the_day_keeps_the_past_and_the_next_ten_minutes(day) -> None:
    plan, requirement, z = day
    rows = []
    for start, row in requirement.slots(TUESDAY):
        rows.append(
            (start, dataclasses.replace(row, registers=row.registers + 1) if start >= 720 else row)
        )
    bigger = dataclasses.replace(requirement, days={TUESDAY: tuple(rows)})
    draft = replan_rest_of_day(plan, bigger, z, 660, CONFIG, "promotion")
    assert draft.kind == REPLAN and draft.changes
    assert min(c.start for c in draft.changes) >= 670
    assert draft.reason == "promotion"


def test_a_heavy_stability_penalty_keeps_the_plan_as_it_is(day) -> None:
    plan, requirement, z = day
    heavy = CONFIG.with_overrides({"live.stability_penalty": 100000.0})
    draft = replan_rest_of_day(plan, requirement, z, 660, heavy, "same requirement")
    assert draft.changes == ()


def test_a_proposal_changes_nothing_until_a_manager_accepts_it(day) -> None:
    plan, requirement, z = day
    book = fresh_book(day)
    draft = open_register(plan, requirement, z, NOW, CONFIG)
    proposal = book.propose(draft, NOW)
    assert book.plan is plan and book.version == 0
    assert proposal.state == "pending" and proposal.new_errors == ()
    assert book.audit[-1].action == "proposed"
    updated = book.accept(proposal.proposal_id, NOW + 5, "busy at the door")
    assert updated is proposal.result and book.plan is proposal.result
    assert proposal.state == ACCEPTED and proposal.note == "busy at the door"
    assert book.version == 1 and book.audit[-1].action == "accepted"


def test_a_rejected_proposal_is_stored_with_its_reason_and_changes_nothing(day) -> None:
    plan, requirement, z = day
    book = fresh_book(day)
    proposal = book.propose(open_register(plan, requirement, z, NOW, CONFIG), NOW)
    book.reject(proposal.proposal_id, NOW + 5, "the queue is already moving")
    assert proposal.state == REJECTED and proposal.note == "the queue is already moving"
    assert book.plan is plan
    with pytest.raises(ValueError):
        book.accept(proposal.proposal_id, NOW + 6)
    assert book.acceptance_rate() == 0.0


def test_accepting_one_proposal_makes_the_others_stale(day) -> None:
    plan, requirement, z = day
    book = fresh_book(day)
    first = book.propose(open_register(plan, requirement, z, NOW, CONFIG), NOW)
    second = book.propose(cover_absence(plan, "W01", z, 540, CONFIG), NOW)
    book.accept(first.proposal_id, NOW + 1)
    assert second.state == STALE
    with pytest.raises(ValueError):
        book.accept(second.proposal_id, NOW + 2)
    assert [e.action for e in book.audit].count("stale") >= 1


def test_a_proposal_that_would_break_a_rule_cannot_be_accepted(day) -> None:
    plan, requirement, z = day
    book = fresh_book(day)
    draft = open_register(plan, requirement, z, NOW, CONFIG)
    broken = dataclasses.replace(
        draft, result=dataclasses.replace(draft.result, breaks=draft.result.breaks[1:])
    )
    proposal = book.propose(broken, NOW)
    assert "BREAK_MISSING" in proposal.new_errors
    with pytest.raises(ValueError):
        book.accept(proposal.proposal_id, NOW + 1)
    assert book.plan is plan
    assert book.audit[-1].action == "blocked"


def test_a_noisy_but_flat_load_never_triggers_a_proposal(day) -> None:
    book = fresh_book(day)
    controller = LiveController(book, day[2], CONFIG)
    rng = make_rng(11, "flat-load")
    for minute in range(600, 900, 5):
        reading = Reading(minute, float(rng.uniform(0.62, 0.86)), 6, 0)
        assert controller.observe(reading, 1.0) == []
    assert book.proposals == {}


def test_a_sustained_rush_gives_one_proposal_and_then_waits_ten_minutes(day) -> None:
    book = fresh_book(day)
    controller = LiveController(book, day[2], CONFIG)
    made = []
    for minute in range(600, 650, 5):
        made.extend(controller.observe(Reading(minute, 0.97, 6, 3), 1.5))
    opens = [p for p in made if p.kind == OPEN_REGISTER]
    assert len(opens) == 1 and opens[0].minute == 605
    for proposal in opens:
        book.reject(proposal.proposal_id, 606, "not now")
    later = []
    for minute in range(650, 700, 5):
        later.extend(controller.observe(Reading(minute, 0.97, 6, 3), 1.5))
    assert all(p.minute - opens[0].minute >= CONFIG.live.same_proposal_gap_minutes for p in later)


def test_a_register_is_closed_only_after_a_long_quiet_spell(day) -> None:
    book = fresh_book(day)
    controller = LiveController(book, day[2], CONFIG)
    made = []
    for minute in range(600, 640, 5):
        made.extend(controller.observe(Reading(minute, 0.40, 6, 0), 0.4))
    closes = [p for p in made if p.kind == CLOSE_REGISTER]
    assert len(closes) == 1
    assert closes[0].minute == 615
    assert all(c.before.startswith("R") and c.after == "FLOAT" for c in closes[0].changes)


def test_a_register_is_not_closed_soon_after_one_was_opened(day) -> None:
    book = fresh_book(day)
    controller = LiveController(book, day[2], CONFIG)
    controller.last_minute[OPEN_REGISTER] = 612
    made = []
    for minute in range(600, 640, 5):
        made.extend(controller.observe(Reading(minute, 0.40, 7, 0), 0.4))
    assert [p.minute for p in made if p.kind == CLOSE_REGISTER] == [635]


def test_a_load_between_the_two_thresholds_changes_nothing(day) -> None:
    controller = LiveController(fresh_book(day), day[2], CONFIG)
    for minute in range(600, 700, 5):
        assert controller.observe(Reading(minute, 0.75, 6, 0), 1.0) == []
    assert controller.book.proposals == {}


def test_a_missing_worker_is_noticed_ten_minutes_after_the_shift_starts(day) -> None:
    book = fresh_book(day)
    controller = LiveController(book, day[2], CONFIG)
    starts = {"W01": 510}
    assert controller.check_logins(515, starts, set()) == []
    made = controller.check_logins(520, starts, set())
    assert len(made) == 1 and made[0].kind == COVER_ABSENCE
    assert controller.check_logins(525, starts, set()) == []
    assert controller.check_logins(520, starts, {"W01"}) == []


def single_register_customers(count: int, gap: float, service: float) -> DayCustomers:
    return DayCustomers(
        np.arange(count, dtype=float) * gap,
        np.zeros(count),
        np.full(count, service),
        np.full(count, np.inf),
    )


def test_the_checkpoint_observation_reports_busy_time_open_time_and_arrivals() -> None:
    seen: list[Observation] = []

    def watch(observation: Observation):
        seen.append(observation)
        return None

    stream = DayCustomers(
        np.array([0.0, 100.0]), np.zeros(2), np.array([400.0, 400.0]), np.full(2, np.inf)
    )
    settings = QueueSettings(
        close_second=1.0e9, checkpoint_seconds=300.0, checkpoint_start=300.0, checkpoint_end=300.0
    )
    simulate_day(stream, [StaffInterval(1, 0.0, 1000.0, "A")], lambda w, t: 1.0, settings, watch)
    assert len(seen) == 1
    observation = seen[0]
    assert observation.arrivals == 1 and observation.waiting == 1
    register, opened, busy = observation.registers[0]
    assert (register, opened, busy) == (1, 300.0, 300.0)


def test_a_controller_can_open_a_register_in_the_middle_of_the_day() -> None:
    stream = single_register_customers(60, 20.0, 40.0)
    base = [StaffInterval(1, 0.0, 1.0e5, "A")]
    settings = QueueSettings(
        close_second=1.0e9,
        checkpoint_seconds=100.0,
        checkpoint_start=100.0,
        checkpoint_end=100.0,
    )
    quiet = simulate_day(stream, base, lambda w, t: 1.0, settings, lambda o: None)

    def opener(observation: Observation):
        return {2: [StaffInterval(2, observation.second, 1.0e5, "B")]}

    helped = simulate_day(stream, base, lambda w, t: 1.0, settings, opener)
    assert helped.wait.mean() < 0.7 * quiet.wait.mean()
    assert set(helped.register[helped.served]) == {1, 2}


def test_a_controller_can_close_a_register_and_its_queue_moves() -> None:
    stream = single_register_customers(30, 100.0, 30.0)
    base = [StaffInterval(1, 0.0, 1.0e5, "A"), StaffInterval(2, 0.0, 1.0e5, "B")]
    settings = QueueSettings(
        close_second=1.0e9,
        checkpoint_seconds=500.0,
        checkpoint_start=500.0,
        checkpoint_end=500.0,
    )

    def closer(observation: Observation):
        return {2: []}

    outcome = simulate_day(stream, base, lambda w, t: 1.0, settings, closer)
    assert outcome.served.all()
    late = outcome.arrival > 600.0
    assert set(outcome.register[late]) == {1}


@pytest.fixture(scope="module")
def live_world():
    config = store_config(
        "small",
        **{
            "solver.backend": "heuristic",
            "evaluation.replications": 1,
            "live.rho_open": 0.2,
            "live.same_proposal_gap_minutes": 5,
        },
    )
    inputs = build_inputs(config, 1)
    bundle = build_plan(inputs, "C")
    world = build_world(
        config, SCENARIOS["saturday_surge"], 1, [w.worker_id for w in inputs.workers]
    )
    return bundle, world, inputs.workforce.truth


def test_a_rejecting_manager_leaves_the_simulated_day_exactly_as_planned(live_world) -> None:
    bundle, world, truth = live_world
    static = simulate_plan(bundle, world, truth)
    runs, stats = simulate_plan_live(bundle, world, truth, policy="reject_all")
    assert stats.proposed > 0 and stats.accepted == 0
    for planned, live in zip(static, runs, strict=True):
        assert np.array_equal(planned.outcome.wait, live.outcome.wait)
        assert np.array_equal(planned.outcome.register, live.outcome.register)


def test_an_accepting_manager_changes_the_staffing_of_the_day(live_world) -> None:
    bundle, world, truth = live_world
    static = simulate_plan(bundle, world, truth)
    runs, stats = simulate_plan_live(bundle, world, truth, policy="accept_all")
    assert stats.accepted > 0 and stats.opened > 0
    assert any(
        not np.array_equal(a.outcome.register, b.outcome.register)
        for a, b in zip(static, runs, strict=True)
    )


def test_the_requirement_helper_matches_the_duty_profile() -> None:
    _, _, shifts, requirement, _ = staggered_day()
    assert requirement.opening_hours(TUESDAY) == (510, 1020)
    assert requirement_from_duty(shifts).days == requirement.days


def test_even_a_wildly_noisy_load_cannot_make_the_proposals_flap(day) -> None:
    book = fresh_book(day)
    controller = LiveController(book, day[2], CONFIG)
    rng = make_rng(21, "wild-load")
    made = []
    for minute in range(520, 1000, 5):
        reading = Reading(minute, float(rng.uniform(0.0, 1.0)), 6, 0)
        for proposal in controller.observe(reading, 1.0):
            made.append(proposal)
            book.reject(proposal.proposal_id, minute, "ignored")
    live = CONFIG.live
    by_kind: dict[str, list[int]] = {}
    for proposal in made:
        by_kind.setdefault(proposal.kind, []).append(proposal.minute)
    for minutes in by_kind.values():
        assert all(b - a >= live.same_proposal_gap_minutes for a, b in pairwise(minutes))
    opens = by_kind.get(OPEN_REGISTER, [])
    closes = by_kind.get(CLOSE_REGISTER, [])
    for opened in opens:
        assert all(not 0 <= later - opened < live.close_window_minutes for later in closes)
    for closed in closes:
        assert all(not 0 <= later - closed < live.open_window_minutes for later in opens)


def test_a_load_that_hovers_around_the_opening_threshold_gives_few_proposals(day) -> None:
    book = fresh_book(day)
    controller = LiveController(book, day[2], CONFIG)
    rng = make_rng(5, "threshold-load")
    made = []
    for minute in range(600, 900, 5):
        reading = Reading(minute, float(rng.uniform(0.84, 0.96)), 6, 0)
        made.extend(controller.observe(reading, 1.0))
        for proposal in made:
            if proposal.state == "pending":
                book.reject(proposal.proposal_id, minute, "wait and see")
    assert all(p.kind == OPEN_REGISTER for p in made)
    assert len(made) <= (900 - 600) // CONFIG.live.same_proposal_gap_minutes


def test_a_manager_event_proposes_a_replan_of_the_rest_of_the_day_that_waits_for_acceptance(
    day,
) -> None:
    plan, requirement, z = day
    rows = []
    for start, row in requirement.slots(TUESDAY):
        bigger = dataclasses.replace(row, registers=row.registers + 1) if start >= 720 else row
        rows.append((start, bigger))
    promotion = dataclasses.replace(requirement, days={TUESDAY: tuple(rows)})
    book = fresh_book(day)
    controller = LiveController(book, z, CONFIG)
    proposal = controller.manager_event(promotion, 660, "promotion at the door")
    assert proposal.kind == REPLAN and proposal.reason == "promotion at the door"
    assert proposal.state == "pending" and book.plan is plan
    assert min(c.start for c in proposal.changes) >= 670
    book.accept(proposal.proposal_id, 662, "agreed")
    assert book.plan is proposal.result
    assert book.acceptance_rate() == 1.0


def test_an_absence_that_leaves_posts_uncovered_says_so_in_the_reason() -> None:
    _, workers, shifts, requirement, scores = staggered_day(pairs=2)
    plan = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG).plan
    draft = cover_absence(plan, "W01", scores.z, 520, CONFIG)
    assert draft is not None
    assert "W01 has not logged in" in draft.reason
    assert "post-minutes stay uncovered" in draft.reason
