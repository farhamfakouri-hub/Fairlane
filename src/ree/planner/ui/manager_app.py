"""Streamlit screen for the shift manager: roster, day, edits, live proposals, fairness, audit."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from ree.planner.api.service import Edit, PlannerService
from ree.planner.domain.shifts import minutes_to_clock
from ree.planner.sim.scenarios import STORE_SIZES

VIEWS = ("Roster", "Day", "Edit and check", "Live proposals", "Fairness", "Audit log")
BACKENDS = ("cp_sat", "heuristic")
DEFAULT_STORE = "small"
DEFAULT_SEED = 3


def service() -> PlannerService:
    """Planner service of the session, created on first use."""
    current = st.session_state.get("service")
    if current is None:
        current = PlannerService()
        st.session_state["service"] = current
    return current


def ensure_roster(planner: PlannerService) -> None:
    """Generate a roster for the default store when there is none."""
    if planner.version is None:
        with st.spinner("No roster yet: solving one for a simulated store (about 15 seconds)"):
            planner.generate(DEFAULT_STORE, DEFAULT_SEED, 1, BACKENDS[0])


def sidebar(planner: PlannerService) -> str:
    """Draw the sidebar with the store and solver controls; returns the chosen view."""
    st.sidebar.title("Shift planner")
    view = st.sidebar.radio("View", VIEWS, key="view")
    st.sidebar.divider()
    st.sidebar.subheader("Generate a roster")
    store = st.sidebar.selectbox("Store size", STORE_SIZES, index=1, key="store")
    seed = int(st.sidebar.number_input("Seed", value=DEFAULT_SEED, step=1, key="seed"))
    backend = st.sidebar.selectbox("Solver", BACKENDS, key="backend")
    posts_only = st.sidebar.checkbox(
        "Scores in the posts only (the roster ignores them)", key="posts_only"
    )
    if st.sidebar.button("Generate", key="generate"):
        with st.spinner("Solving the roster"):
            planner.generate(store, seed, 1, backend, posts_only)
        st.rerun()
    st.sidebar.caption(
        "Every plan is a proposal. Nothing changes until a manager accepts it, and every "
        "override keeps a short reason."
    )
    return view


def week_days(planner: PlannerService) -> list[date]:
    """Dates of the current roster."""
    return planner.current().solution.plan.dates()


def day_label(day: date) -> str:
    """Weekday and date as text."""
    return f"{day.strftime('%a')} {day.isoformat()}"


def shift_text(planner: PlannerService, worker_id: str, day: date) -> str:
    """A worker's shift on a day as text, or "-" when off."""
    pattern = planner.current().solution.plan.pattern_of(worker_id, day)
    if pattern is None:
        return "-"
    return " + ".join(
        f"{minutes_to_clock(b.register_start)}-{minutes_to_clock(b.register_end)}"
        for b in pattern.blocks
    )


def report_table(report) -> pd.DataFrame:
    """Table of the violations of a check report."""
    return pd.DataFrame(
        [
            {
                "severity": item.severity,
                "code": item.code,
                "message": item.message,
                "worker": item.worker_id or "",
                "day": item.day.isoformat() if item.day else "",
            }
            for item in [*report.errors, *report.warnings]
        ],
        columns=["severity", "code", "message", "worker", "day"],
    )


def roster_view(planner: PlannerService) -> None:
    """Roster grid with the reasons for each shift."""
    version = planner.current()
    solution = version.solution
    report = solution.report
    columns = st.columns(4)
    columns[0].metric("Shifts", len(solution.plan.assignments))
    columns[1].metric("Rule errors", len(report.errors) if report else 0)
    columns[2].metric("Warnings", len(report.warnings) if report else 0)
    columns[3].metric(
        "Uncovered staff hours", f"{(report.shortfall_minutes() if report else 0.0) / 60.0:.1f}"
    )
    st.caption(
        f"Roster version {version.number}, solved with {solution.backend} ({solution.status})."
    )
    for note in version.notes:
        st.info(note)
    days = week_days(planner)
    frame = pd.DataFrame(
        {
            day_label(day): [shift_text(planner, w, day) for w in sorted(solution.plan.workers)]
            for day in days
        },
        index=sorted(solution.plan.workers),
    )
    st.dataframe(frame, width="stretch", key="roster_grid")
    st.subheader("Why a shift is where it is")
    worker = st.selectbox("Worker", sorted(solution.plan.workers), key="reason_worker")
    day = st.selectbox("Day", days, format_func=day_label, key="reason_day")
    reasons = version.explanations.get((worker, day), [])
    if reasons:
        for reason in reasons:
            st.markdown(f"- **{reason.code}**: {reason.text}")
    else:
        st.write("This worker does not work on that day.")


def day_view(planner: PlannerService) -> None:
    """Posts, breaks, handovers and the relief chain of one day."""
    days = [
        d
        for d in week_days(planner)
        if any(a.day == d for a in planner.current().solution.plan.assignments)
    ]
    day = st.selectbox("Day", days, format_func=day_label, key="day")
    result = planner.daily(day)
    plan = result.plan
    columns = st.columns(4)
    columns[0].metric("Breaks", result.metrics.breaks)
    columns[1].metric("Most breaks at once", result.metrics.peak_simultaneous_breaks)
    columns[2].metric("Relief handovers", result.metrics.relief_handovers)
    columns[3].metric("Rule errors", len(result.report.errors))
    for alert in plan.alerts:
        st.warning(alert)
    board = pd.DataFrame(
        [
            {
                "worker": worker_id,
                "timeline": ", ".join(
                    f"{minutes_to_clock(s.start)} {s.post}"
                    for s in sorted(plan.segments, key=lambda s: s.start)
                    if s.worker_id == worker_id
                ),
            }
            for worker_id in sorted(plan.shifts)
        ]
    )
    st.dataframe(board, width="stretch", key="board")
    st.subheader("Relief chain")
    if result.relief_lines:
        st.code("\n".join(result.relief_lines))
    else:
        st.write("Nobody is relieved on this day.")
    breaks = pd.DataFrame(
        [
            {
                "worker": b.worker_id,
                "block": b.block_index + 1,
                "starts": minutes_to_clock(b.start),
                "minutes": b.minutes,
            }
            for b in plan.breaks
        ]
    )
    st.dataframe(breaks, width="stretch", key="breaks")


def show_outcome(outcome) -> None:
    """Show the result of a check or an edit."""
    if outcome.new_errors:
        st.error(outcome.message)
    else:
        st.success(outcome.message)
    table = report_table(outcome.report)
    if len(table):
        st.dataframe(table, width="stretch", key="violations")


def edit_view(planner: PlannerService) -> None:
    """Check or apply a change of one shift, or a move of a break, with a reason."""
    plan = planner.current().solution.plan
    st.subheader("Change one shift")
    worker = st.selectbox("Worker", sorted(plan.workers), key="edit_worker")
    day = st.selectbox("Day", week_days(planner), format_func=day_label, key="edit_day")
    st.caption(f"Today's shift: {shift_text(planner, worker, day)}")
    code = st.text_input("New shift code (for example C0830-1530)", key="edit_code")
    off = st.checkbox("Take the day off", key="edit_off")
    reason = st.text_input("Short reason (kept in the audit log)", key="edit_reason")
    edit = Edit(worker, day, None if off else (code.strip() or None))
    if st.button("Check the edit", key="check_edit"):
        if not off and not code.strip():
            st.warning("Type a shift code or tick the day off.")
        else:
            try:
                show_outcome(planner.check_edit(edit))
            except ValueError as error:
                st.error(str(error))
    if st.button("Apply as an override", key="apply_override"):
        try:
            outcome = planner.override(edit, reason)
        except ValueError as error:
            st.warning(str(error))
        else:
            show_outcome(outcome)
            if outcome.applied:
                st.rerun()
    st.divider()
    st.subheader("Move a break")
    days = [d for d in week_days(planner) if any(a.day == d for a in plan.assignments)]
    break_day = st.selectbox("Day", days, format_func=day_label, key="break_day")
    entries = planner.daily(break_day).plan.breaks
    labels = {
        f"{e.worker_id} block {e.block_index + 1} at {minutes_to_clock(e.start)}": e
        for e in entries
    }
    if labels:
        choice = labels[st.selectbox("Break", list(labels), key="break_choice")]
        new_start = int(
            st.number_input(
                "New start (minutes after midnight)", value=choice.start, step=5, key="break_start"
            )
        )
        if st.button("Check the move", key="check_break"):
            show_outcome(
                planner.move_break(break_day, choice.worker_id, choice.block_index, new_start)
            )


def live_view(planner: PlannerService) -> None:
    """Send live readings and decide on proposals."""
    days = [
        d
        for d in week_days(planner)
        if any(a.day == d for a in planner.current().solution.plan.assignments)
    ]
    day = st.selectbox("Day", days, format_func=day_label, key="live_day")
    st.caption(
        "Feed the busyness of the open registers (from the register events) and the planner "
        "proposes changes. Nothing is applied until you accept."
    )
    columns = st.columns(3)
    minute = int(
        columns[0].number_input("Minute of the day", value=600, step=5, key="reading_minute")
    )
    busy = float(columns[1].slider("Busyness", 0.0, 1.0, 0.95, key="reading_busyness"))
    opened = int(columns[2].number_input("Open registers", value=3, step=1, key="reading_open"))
    if st.button("Send reading", key="send_reading"):
        planner.observe(day, minute, busy, opened)
    for proposal in planner.proposals(day):
        with st.container(border=True):
            st.markdown(
                f"**#{proposal.proposal_id} {proposal.kind.replace('_', ' ')}** ({proposal.state})"
            )
            st.write(proposal.reason)
            if proposal.wait_before is not None and proposal.wait_after is not None:
                st.caption(
                    f"Expected wait {proposal.wait_before:.1f} min now, "
                    f"{proposal.wait_after:.1f} min with the change; "
                    f"{proposal.cost} people affected."
                )
            if proposal.state == "pending":
                note = st.text_input("Short note", key=f"note_{proposal.proposal_id}")
                left, right = st.columns(2)
                if left.button("Accept", key=f"accept_{proposal.proposal_id}"):
                    try:
                        planner.decide(day, proposal.proposal_id, True, note)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
                if right.button("Reject", key=f"reject_{proposal.proposal_id}"):
                    planner.decide(day, proposal.proposal_id, False, note)
                    st.rerun()


def fairness_view(planner: PlannerService) -> None:
    """Fairness measures of the roster, including whether scores leak into shift quality."""
    solution = planner.current().solution
    metrics = solution.metrics
    if metrics is None:
        st.write("No fairness metrics for this roster.")
        return
    columns = st.columns(3)
    columns[0].metric("Burden spread", f"{metrics.burden_spread:.3f}")
    columns[1].metric("Largest tier gap", f"{metrics.max_tier_gap:.1%}")
    columns[2].metric("Largest hours deviation", f"{metrics.max_hours_deviation:.2f} h")
    gaps = pd.DataFrame(
        [{"tier": tier, "gap against the average": gap} for tier, gap in metrics.tier_gaps.items()]
    )
    st.dataframe(gaps, width="stretch", key="tier_gaps")
    st.subheader("Do scores leak into shift quality?")
    correlations = pd.DataFrame(
        [
            {"measure": name, "correlation with score": value}
            for name, value in metrics.correlations.items()
        ]
    )
    st.dataframe(correlations, width="stretch", key="correlations")


def audit_view(planner: PlannerService) -> None:
    """The audit log."""
    records = planner.audit_log()
    st.dataframe(
        pd.DataFrame(
            [
                {"at": r.at, "kind": r.kind, "subject": r.subject, "detail": str(r.detail)}
                for r in records
            ]
        ),
        width="stretch",
        key="audit",
    )


def main() -> None:
    """Entry point of the screen."""
    st.set_page_config(page_title="Shift planner", layout="wide")
    planner = service()
    ensure_roster(planner)
    view = sidebar(planner)
    if view == "Roster":
        roster_view(planner)
    elif view == "Day":
        day_view(planner)
    elif view == "Edit and check":
        edit_view(planner)
    elif view == "Live proposals":
        live_view(planner)
    elif view == "Fairness":
        fairness_view(planner)
    else:
        audit_view(planner)


if __name__ == "__main__":
    main()
