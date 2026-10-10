"""Streamlit screen for a worker: own shifts, breaks and reasons, and a swap check."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from ree.planner.api.service import PlannerService
from ree.planner.ui.manager_app import BACKENDS, DEFAULT_SEED, DEFAULT_STORE, show_outcome

CONTRACT_NOTE = (
    "Your shifts add up to your contract hours. The planner decides where and when you work "
    "inside the contract, never how much."
)


def service() -> PlannerService:
    """Planner service of the session, created on first use."""
    current = st.session_state.get("service")
    if current is None:
        current = PlannerService()
        st.session_state["service"] = current
    return current


def main() -> None:
    """Entry point of the screen."""
    st.set_page_config(page_title="My plan", layout="centered")
    planner = service()
    if planner.version is None:
        with st.spinner("No roster yet: solving one for a simulated store (about 15 seconds)"):
            planner.generate(DEFAULT_STORE, DEFAULT_SEED, 1, BACKENDS[0])
    plan = planner.current().solution.plan
    st.title("My plan")
    worker_id = st.selectbox("Who are you?", sorted(plan.workers), key="worker")
    view = planner.worker_view(worker_id)
    columns = st.columns(2)
    columns[0].metric("Contract hours", f"{view['contract_hours']:g}")
    columns[1].metric("Paid hours this week", f"{view['paid_hours']:.2f}")
    st.caption(CONTRACT_NOTE)
    shifts = view["shifts"]
    if not shifts:
        st.write("You have no shifts in this roster.")
    for shift in shifts:
        with st.container(border=True):
            st.markdown(
                f"**{shift['weekday']} {shift['day']}**: {' and '.join(shift['blocks'])} "
                f"({shift['paid_hours']:.2f} paid hours)"
            )
            for item in shift["breaks"]:
                st.write(f"Break at {item['start']} for {item['minutes']} minutes")
            for line in shift["why"]:
                st.caption(line)
    st.divider()
    st.subheader("Ask for a swap")
    days = [date.fromisoformat(s["day"]) for s in shifts]
    if days:
        day = st.selectbox("Day", days, key="swap_day")
        colleagues = sorted(
            w for w in plan.workers if w != worker_id and plan.pattern_of(w, day) is not None
        )
        if colleagues:
            other = st.selectbox("Swap with", colleagues, key="swap_with")
            if st.button("Check the swap", key="check_swap"):
                show_outcome(planner.check_swap(worker_id, other, day))
                st.caption("A legal swap still needs the manager's approval.")
        else:
            st.write("Nobody else works that day.")
    if isinstance(shifts, list) and shifts:
        st.dataframe(pd.DataFrame(shifts)[["weekday", "day", "paid_hours"]], key="shift_table")


if __name__ == "__main__":
    main()
