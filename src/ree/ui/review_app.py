"""Streamlit review screen: manager table, worker view and method explainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ree.config import Config, load_config
from ree.domain.calendar import StoreCalendar
from ree.domain.models import AppealRequest, CashierScoreRecord
from ree.engine.pause_rule import (
    PauseThresholds,
    extract_gaps,
    thresholds_from_config,
    transaction_timing,
)
from ree.engine.pipeline import score_directory
from ree.engine.speed_v2 import PRICE_TERMS
from ree.simulator.run import simulate_to_directory
from ree.storage import db

VIEWS = ("Manager table", "Worker view", "Method explainer")
SORT_OPTIONS = {
    "Peak score": "score_peak",
    "Off-peak score": "score_off",
    "Worker": "worker",
}
TIER_ORDER = ("A", "B", "C", "unrated")
MIN_EXAMPLE_ITEMS = 8
EXAMPLE_POOL = 400


def active_config() -> Config:
    """Configuration of the session, loaded on first use."""
    config = st.session_state.get("config")
    if config is None:
        config = load_config()
        st.session_state["config"] = config
    return config


def ensure_scores(config: Config) -> None:
    """Simulate and score a store when the store has no scores yet."""
    directory = db.store_directory(config)
    connection = db.connect(directory)
    try:
        db.init_schema(connection)
        has_scores = db.latest_run_id(connection) is not None
    finally:
        connection.close()
    if not has_scores:
        with st.spinner("No scores yet: simulating a store and scoring it"):
            simulate_to_directory(config)
            score_directory(config)


@st.cache_data(show_spinner=False)
def load_scores(directory: str, version: int) -> list[CashierScoreRecord]:
    """Score records of the store, cached until the version changes."""
    connection = db.connect(directory)
    try:
        return db.load_scores(connection)
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def load_price_list(directory: str, version: int) -> pd.DataFrame:
    """Price list of the latest run, cached until the version changes."""
    connection = db.connect(directory)
    try:
        return db.load_price_list(connection)
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def load_transactions(directory: str, version: int) -> pd.DataFrame:
    """Transactions table of the store, cached until the version changes."""
    return pd.read_parquet(f"{directory}/{db.TRANSACTIONS_FILE}")


def load_events(directory: str, transaction_id: int) -> pd.DataFrame:
    """Events of one transaction."""
    return pd.read_parquet(
        f"{directory}/{db.EVENTS_FILE}", filters=[("transaction_id", "==", transaction_id)]
    )


def interval_text(lower: float, upper: float) -> str:
    """An interval as text, rounded to whole points."""
    return f"{lower:.0f} to {upper:.0f}"


def records_frame(records: list[CashierScoreRecord]) -> pd.DataFrame:
    """Table of score records for the manager view."""
    return pd.DataFrame(
        {
            "worker": [record.worker_id for record in records],
            "score_peak": [record.score_peak for record in records],
            "interval_peak": [
                interval_text(record.interval_peak.lower, record.interval_peak.upper)
                for record in records
            ],
            "tier_peak": [record.tier_peak for record in records],
            "score_off": [record.score_off for record in records],
            "interval_off": [
                interval_text(record.interval_off.lower, record.interval_off.upper)
                for record in records
            ],
            "tier_off": [record.tier_off for record in records],
            "transactions": [record.n_transactions for record in records],
            "weeks": [record.weeks_of_data for record in records],
            "provisional": [record.provisional for record in records],
            "review": [record.review_flag or "" for record in records],
            "drivers": ["; ".join(record.drivers) for record in records],
        }
    )


def sidebar(config: Config) -> str:
    """Draw the sidebar with the view choice and the simulator controls.

    When asked it regenerates the store and re-scores. Returns the chosen view.
    """
    simulator = config.simulator
    with st.sidebar:
        st.title("Cashier scores")
        view = st.radio("View", VIEWS, key="view")
        st.divider()
        st.subheader("Simulator control")
        seed = st.number_input("Seed", value=int(config.seed), step=1, key="control_seed")
        cashiers = st.number_input(
            "Cashiers",
            min_value=6,
            max_value=40,
            value=simulator.population.n_cashiers,
            key="control_cashiers",
        )
        weeks = st.number_input(
            "Weeks", min_value=3, max_value=20, value=simulator.n_weeks, key="control_weeks"
        )
        mode = st.selectbox(
            "Assignment",
            ("random", "biased"),
            index=0 if simulator.scheduler.mode == "random" else 1,
            key="control_mode",
        )
        submitted = st.button("Regenerate store and re-score", key="regenerate")
        st.caption(
            "This is simulated data. Scores are decision support, not decisions: a manager "
            "validates every plan and any worker can ask for a review."
        )
    if submitted:
        updated = config.with_overrides(
            {
                "seed": int(seed),
                "simulator.population.n_cashiers": int(cashiers),
                "simulator.n_weeks": int(weeks),
                "simulator.scheduler.mode": mode,
            }
        )
        with st.spinner("Simulating and scoring"):
            simulate_to_directory(updated)
            score_directory(updated)
        st.session_state["config"] = updated
        st.session_state["version"] = st.session_state.get("version", 0) + 1
        st.rerun()
    return str(view)


def manager_view(records: list[CashierScoreRecord]) -> None:
    """Manager table: counts, filters and the scores with their main reasons."""
    st.header("Manager table")
    frame = records_frame(records)
    columns = st.columns(4)
    columns[0].metric("Cashiers", len(frame))
    unrated = int((frame["tier_peak"] == "unrated").sum())
    columns[1].metric("Unrated at peak", unrated)
    columns[2].metric("To review (flag)", int((frame["review"] != "").sum()))
    columns[3].metric("Provisional", int(frame["provisional"].sum()))
    controls = st.columns([2, 1, 2])
    label = controls[0].selectbox("Sort by", list(SORT_OPTIONS), key="sort_by")
    ascending = controls[1].toggle("Ascending", value=False, key="ascending")
    tiers = controls[2].multiselect("Tiers", TIER_ORDER, default=list(TIER_ORDER), key="tiers")
    shown = frame[frame["tier_peak"].isin(tiers) | frame["tier_off"].isin(tiers)]
    shown = shown.sort_values(SORT_OPTIONS[label], ascending=ascending, kind="stable")
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            "score_peak": st.column_config.ProgressColumn(
                "Peak score", min_value=0, max_value=100, format="%.1f"
            ),
            "score_off": st.column_config.ProgressColumn(
                "Off-peak score", min_value=0, max_value=100, format="%.1f"
            ),
            "interval_peak": "Peak 90% range",
            "interval_off": "Off-peak 90% range",
            "tier_peak": "Peak tier",
            "tier_off": "Off-peak tier",
            "transactions": "Transactions",
            "weeks": "Weeks",
            "review": "Review flag",
            "drivers": st.column_config.TextColumn("Main reasons", width="large"),
        },
    )
    st.caption(
        "Unrated means the range is too wide or the history too short; the planner treats "
        "unrated workers as average. A review flag asks a human to look and never lowers a score."
    )


def explain_worker(record: CashierScoreRecord) -> None:
    """Explain in plain words how one cashier's score was computed."""
    parts = record.components
    peak_range = interval_text(record.interval_peak.lower, record.interval_peak.upper)
    off_range = interval_text(record.interval_off.lower, record.interval_off.upper)
    st.subheader("How this score was computed")
    st.markdown(
        f"1. **Timing.** {parts.n_valid_peak} of your peak transactions and {parts.n_valid_off} "
        "of your off-peak transactions were timed from the first scan to the payment press. "
        "Customers searching for a card, supervisor calls and voids are not counted as your "
        "speed, and gaps are capped so a long wait cannot hide or inflate your pace.\n"
        f"2. **Speed score** ({parts.speed_score_peak:.0f} at peak, {parts.speed_score_off:.0f} "
        "off-peak). Each transaction is compared with how long the store as a whole takes for the "
        "same basket (items, hand-keyed items, tags, loyalty card, vouchers) in the same lane and "
        "period. 50 is the store average; 15 points is one standard deviation.\n"
        f"3. **Accuracy score** ({parts.accuracy_score:.0f}). It combines weekly cash differences "
        "beyond the tolerance, weighted toward recent weeks, with voids per 100 transactions"
        + (
            f" (yours: {parts.void_rate_per_100:.1f})."
            if parts.void_rate_per_100 is not None
            else "."
        )
        + "\n4. **Combined.** At peak the speed score counts 65% and accuracy 35%; off-peak both "
        f"count 50%. That gives {record.score_peak:.1f} at peak and "
        f"{record.score_off:.1f} off-peak.\n"
        f"5. **Range and tier.** The range ({peak_range} at peak, {off_range} off-peak) "
        "shows how sure the engine is; a wide range means little data. Tiers are shown only when "
        "the range is narrow enough."
    )


def appeal_section(config: Config, record: CashierScoreRecord) -> None:
    """Let a cashier ask for a review, and list the requests already sent."""
    st.subheader("Ask for a review")
    directory = db.store_directory(config)
    connection = db.connect(directory)
    try:
        previous = db.list_appeals(connection, record.worker_id)
    finally:
        connection.close()
    reason = st.text_area("What should a manager look at?", max_chars=2000, key="appeal_reason")
    sent = st.button("Send review request", key="send_appeal")
    if sent:
        if len(reason.strip()) < 3:
            st.warning("Please write a few words so the reviewer knows what to check.")
        else:
            connection = db.connect(directory)
            try:
                created = db.create_appeal(
                    connection,
                    AppealRequest(worker_id=record.worker_id, reason=reason.strip()),
                    record.model_version,
                )
            finally:
                connection.close()
            st.success(f"Review request {created.appeal_id} sent. A manager will look at it.")
            previous = [*previous, created]
    if previous:
        st.dataframe(
            pd.DataFrame(
                {
                    "request": [item.appeal_id for item in previous],
                    "sent": [item.requested_at for item in previous],
                    "status": [item.status for item in previous],
                    "reason": [item.reason for item in previous],
                }
            ),
            hide_index=True,
        )


def worker_view(config: Config, records: list[CashierScoreRecord]) -> None:
    """Worker view: one cashier's scores, reasons, explanation and review requests."""
    st.header("Worker view")
    by_id = {record.worker_id: record for record in records}
    worker_id = st.selectbox("Worker", sorted(by_id), key="worker")
    record = by_id[worker_id]
    columns = st.columns(4)
    columns[0].metric("Peak score", f"{record.score_peak:.1f}", record.tier_peak)
    columns[1].metric("Off-peak score", f"{record.score_off:.1f}", record.tier_off)
    columns[2].metric("Transactions", f"{record.n_transactions:,}")
    columns[3].metric("Weeks of data", record.weeks_of_data)
    if record.provisional:
        st.info(
            "Provisional: fewer than 4 weeks of tenure, so you are compared with other newcomers."
        )
    if record.review_flag:
        st.warning(f"Flagged for a human look (not a penalty): {record.review_flag}")
    st.markdown("**Main reasons**")
    for reason in record.drivers:
        st.markdown(f"- {reason}")
    explain_worker(record)
    appeal_section(config, record)
    st.caption(f"Model version {record.model_version}, data up to {record.computed_at:%d %b %Y}.")


def pick_example(directory: str, version: int) -> int:
    """Choose the transaction to explain; a button steps through a sample of them."""
    transactions = load_transactions(directory, version)
    pool = transactions.iloc[:: max(1, len(transactions) // EXAMPLE_POOL)][
        "transaction_id"
    ].tolist()
    if "example_index" not in st.session_state:
        st.session_state["example_index"] = 0
    if st.button("Show another transaction", key="next_example"):
        st.session_state["example_index"] += 1
    return int(pool[st.session_state["example_index"] % len(pool)])


def example_transaction(config: Config, version: int) -> None:
    """Walk through one transaction: its events, its timing under the pause rule and
    the expected duration from the price list.
    """
    directory = str(db.store_directory(config))
    transaction_id = pick_example(directory, version)
    events = load_events(directory, transaction_id).sort_values("offset_s", kind="stable")
    transactions = load_transactions(directory, version)
    header = transactions[transactions["transaction_id"] == transaction_id].iloc[0]
    calendar = StoreCalendar(config.store)
    started = pd.Timestamp(header["started_at"])
    lane = calendar.lane_group(int(header["register"]))
    period = str(calendar.period_labels(pd.DatetimeIndex([started]))[0])
    thresholds: PauseThresholds = thresholds_from_config(config.engine.pause)
    timing = transaction_timing(
        extract_gaps(events), thresholds, config.engine.pause.hard_stop_seconds
    ).iloc[0]

    st.markdown(
        f"Transaction {transaction_id}: register {int(header['register'])} "
        f"({lane.replace('_', ' ')}), "
        f"{period.replace('off', 'off-peak')} period, started {started:%a %H:%M}."
    )
    gap_events = events[~events["event_type"].astype(str).isin(["transaction_started"])]
    first = float(
        events.loc[
            events["event_type"].astype(str).isin(["item_scanned", "item_hand_keyed"]), "offset_s"
        ].min()
    )
    shown = gap_events.assign(seconds_since_first_scan=(gap_events["offset_s"] - first).round(1))[
        ["event_type", "seconds_since_first_scan", "detail"]
    ]
    st.dataframe(shown.astype({"event_type": str}), hide_index=True, height=220)

    counts = events["event_type"].astype(str).value_counts()
    items = int(counts.get("item_scanned", 0) + counts.get("item_hand_keyed", 0))
    keyed = int(counts.get("item_hand_keyed", 0))
    tags = int(counts.get("tag_removed", 0))
    loyalty = int(counts.get("loyalty_scanned", 0) > 0)
    vouchers = int(counts.get("voucher_applied", 0))
    columns = st.columns(3)
    columns[0].metric("Raw time", f"{timing['duration_raw']:.1f} s")
    columns[1].metric(
        f"Capped at {thresholds.t_idle_seconds:.0f} s per gap", f"{timing['duration_capped']:.1f} s"
    )
    columns[2].metric(
        "Interrupted?",
        "yes" if timing["interrupted_by_gap"] else "no",
        f"longest gap {timing['max_gap']:.0f} s",
    )
    prices = load_price_list(directory, version)
    context = prices[(prices["lane_group"] == lane) & (prices["period"] == period)]
    if context.empty:
        st.info("The price list is only available when the speed method is v2.")
        return
    values = context.set_index("term")["estimate"]
    design = [1, items, keyed, tags, loyalty, vouchers]
    expected = float(
        sum(values[term] * count for term, count in zip(PRICE_TERMS, design, strict=True))
    )
    ratio = float(np.log(expected / timing["duration_capped"]))
    table = pd.DataFrame(
        {
            "term": PRICE_TERMS,
            "seconds per unit": [float(values[term]) for term in PRICE_TERMS],
            "units in this transaction": design,
        }
    )
    table["seconds"] = table["seconds per unit"] * table["units in this transaction"]
    st.dataframe(table.round(2), hide_index=True)
    verdict = "faster" if ratio > 0 else "slower"
    st.markdown(
        f"Expected time {expected:.1f} s against {timing['duration_capped']:.1f} s measured: "
        f"s = ln({expected:.1f} / {timing['duration_capped']:.1f}) = **{ratio:+.3f}**, "
        f"{abs(np.expm1(ratio)) * 100:.0f}% {verdict} than the store expects for this basket."
    )


def explainer_view(config: Config, version: int) -> None:
    """Method explainer: the steps of the engine, with a live example."""
    st.header("Method explainer")
    st.markdown(
        "The engine turns register events into a fair score in five steps. The formulas are the "
        "ones in the method document; the live example below applies them to one simulated "
        "transaction."
    )
    st.subheader("1. Pause rule")
    st.latex(r"d_i = \sum_j \min(g_{i,j},\; T_{\mathrm{idle}})")
    st.markdown(
        "Gaps between consecutive events are capped at "
        f"T_idle = {config.engine.pause.t_idle_seconds:.0f} s. A transaction with a gap above "
        f"T_stop = {config.engine.pause.t_stop_seconds:.0f} s is marked interrupted: "
        "it is left out "
        "of speed and kept for accuracy."
    )
    st.subheader("2. Speed against an expected duration")
    st.latex(r"E_i = \alpha + \beta n_i + \gamma m_i + \delta a_i + \lambda l_i + \mu k_i")
    st.latex(r"s_i = \ln\!\left(\frac{E_i}{d_i}\right)")
    st.markdown(
        "n items, m hand-keyed items, a tagged items, l loyalty card, k vouchers. A price list "
        "is fitted per lane group and period from the whole store; the log ratio makes "
        "1.25 times faster and 1.25 times slower cancel."
    )
    st.subheader("3. Shrinkage and scale")
    st.latex(
        r"S_w = \frac{n_w}{n_w + k}\,\operatorname{median}(s_i), \qquad "
        r"\mathrm{SpeedScore}_w = 50 + 15 Z_w"
    )
    st.subheader("4. Accuracy")
    st.latex(r"\mathrm{ErrRate}_w = \frac{\sum_t \max(0, |\Delta_{w,t}| - \tau)}{\sum_t C_{w,t}}")
    st.latex(
        r"\mathrm{AccuracyScore}_w = 50 - 15\left(0.6\,Z(\mathrm{ErrRate}_w) + 0.4\,Z(V_w)\right)"
    )
    st.subheader("5. Combined score")
    st.latex(
        r"\mathrm{Score}^{(c)}_w = \omega_S^{(c)}\,\mathrm{SpeedScore}^{(c)}_w "
        r"+ \omega_A\,\mathrm{AccuracyScore}_w"
    )
    st.markdown("Weights: speed 0.65 and accuracy 0.35 at peak, 0.5 and 0.5 off-peak.")
    st.subheader("Live example")
    example_transaction(config, version)


def main() -> None:
    """Entry point of the screen."""
    st.set_page_config(page_title="Cashier scores", layout="wide")
    config = active_config()
    ensure_scores(config)
    version = int(st.session_state.get("version", 0))
    view = sidebar(config)
    config = active_config()
    directory = str(db.store_directory(config))
    records = load_scores(directory, version)
    if view == "Manager table":
        manager_view(records)
    elif view == "Worker view":
        worker_view(config, records)
    else:
        explainer_view(config, version)


if __name__ == "__main__":
    main()
