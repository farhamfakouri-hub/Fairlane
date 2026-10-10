"""Transaction simulator: customers served in each session, with timings and register events."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from ree.config import Config, make_rng, replicate_stream
from ree.domain.calendar import StoreCalendar
from ree.domain.events import EVENT_CODES, VoidReason, make_events_frame
from ree.simulator.baskets import sample_item_counts
from ree.simulator.store import RegisterProfiles, utilization_tables

CHUNK_SESSIONS = 400
MILLISECONDS = 1000.0
EXCEPTION_OFFSET_EPSILON = 1e-3
CASH_EXTRA_LOW = 10.0
CASH_EXTRA_HIGH = 25.0
MAX_EVENT_PROBABILITY = 0.5
MIN_CANDIDATES_PER_ROUND = 2

STARTED = EVENT_CODES["transaction_started"]
SCANNED = EVENT_CODES["item_scanned"]
KEYED = EVENT_CODES["item_hand_keyed"]
TAG = EVENT_CODES["tag_removed"]
VOUCHER = EVENT_CODES["voucher_applied"]
LOYALTY = EVENT_CODES["loyalty_scanned"]
PAYMENT = EVENT_CODES["payment_pressed"]
SUPERVISOR = EVENT_CODES["supervisor_call"]
CHEQUE = EVENT_CODES["cheque_payment"]
PRICE_CHECK = EVENT_CODES["price_check"]
VOID = EVENT_CODES["void"]
MANUAL = EVENT_CODES["manual_entry"]


@dataclass
class _Context:
    config: Config
    profiles: RegisterProfiles
    peak_table: np.ndarray
    rho_tables: np.ndarray
    session_worker: np.ndarray
    session_register_index: np.ndarray
    session_lane: np.ndarray
    session_day: np.ndarray
    session_weekday: np.ndarray
    session_login_s: np.ndarray
    session_logout_s: np.ndarray
    session_base_speed: np.ndarray
    session_peak_effect: np.ndarray
    session_basket_multiplier: np.ndarray
    error_propensity: np.ndarray


@dataclass
class _Batch:
    count: int
    session: np.ndarray
    n_items: np.ndarray
    n_keyed: np.ndarray
    n_tagged: np.ndarray
    has_loyalty: np.ndarray
    n_vouchers: np.ndarray
    unload: np.ndarray
    post: np.ndarray
    idle_unit: np.ndarray
    fixed_part: np.ndarray
    speed_part: np.ndarray
    row_candidate: np.ndarray
    row_type: np.ndarray
    row_unscaled: np.ndarray
    row_scaled: np.ndarray
    row_count: np.ndarray
    exception_candidate: np.ndarray
    exception_type: np.ndarray
    exception_anchor: np.ndarray
    exception_detail: np.ndarray
    customer_delay: np.ndarray
    interruption: np.ndarray
    cash_amount: np.ndarray
    mistake_amount: np.ndarray


@dataclass
class TransactionBundle:
    """Result of the transaction simulation: tables, hidden values and session end times."""

    transactions: pd.DataFrame
    events: pd.DataFrame
    latent: pd.DataFrame
    session_logout: np.ndarray
    rounds: int = 0


def _lognormal_unit(rng: np.random.Generator, sigma: float, size: int) -> np.ndarray:
    return np.exp(sigma * rng.standard_normal(size) - sigma**2 / 2.0)


def _rank_within_candidate(
    values: np.ndarray, candidate: np.ndarray, start: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    order = np.lexsort((values, candidate))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values)) - np.repeat(start, counts)
    return ranks


def _random_gap_row(
    rng: np.random.Generator, row_start: np.ndarray, gap_rows: np.ndarray, subset: np.ndarray
) -> np.ndarray:
    draws = rng.random(len(subset))
    return row_start[subset] + 1 + np.floor(draws * gap_rows[subset]).astype(np.int64)


def _generate_candidates(rng: np.random.Generator, ctx: _Context, session: np.ndarray) -> _Batch:
    simulator = ctx.config.simulator
    timing = simulator.timing
    errors = simulator.errors
    count = len(session)
    register_index = ctx.session_register_index[session]
    worker = ctx.session_worker[session]
    lane_slowness = ctx.profiles.lane_slowness[register_index]
    error_propensity = ctx.error_propensity[worker]

    n_items = sample_item_counts(
        rng,
        ctx.profiles.mixtures[register_index],
        ctx.session_basket_multiplier[session],
        simulator.baskets,
    )
    n_keyed = rng.binomial(n_items, timing.hand_keyed_share)
    n_tagged = rng.binomial(n_items - n_keyed, ctx.profiles.tagged_share[register_index])
    has_loyalty = rng.random(count) < timing.loyalty_probability
    n_vouchers = rng.choice(4, size=count, p=np.array(timing.voucher_count_probabilities))

    total_items = int(n_items.sum())
    item_candidate = np.repeat(np.arange(count), n_items)
    item_start = np.cumsum(n_items) - n_items
    item_position = np.arange(total_items) - np.repeat(item_start, n_items)
    keyed_rank = _rank_within_candidate(
        rng.random(total_items), item_candidate, item_start, n_items
    )
    is_keyed = keyed_rank < np.repeat(n_keyed, n_items)
    tag_rank = _rank_within_candidate(
        rng.random(total_items) + 2.0 * is_keyed, item_candidate, item_start, n_items
    )
    is_tagged = tag_rank < np.repeat(n_tagged, n_items)

    sigma = timing.gap_sigma
    item_scaled = (
        timing.beta_seconds
        * lane_slowness[item_candidate]
        * _lognormal_unit(rng, sigma, total_items)
    )
    keyed_scaled = (
        timing.gamma_seconds
        * lane_slowness[item_candidate]
        * _lognormal_unit(rng, sigma, total_items)
    )
    item_scaled = item_scaled + np.where(is_keyed, keyed_scaled, 0.0)
    item_type = np.where(is_keyed, KEYED, SCANNED)

    tag_candidate = item_candidate[is_tagged]
    tag_key = item_position[is_tagged] + 0.5
    tag_unscaled = rng.uniform(timing.tag_seconds_low, timing.tag_seconds_high, len(tag_candidate))

    loyalty_candidate = np.flatnonzero(has_loyalty)
    loyalty_items = n_items[loyalty_candidate]
    at_end = (rng.random(len(loyalty_candidate)) < timing.loyalty_at_end_probability) | (
        loyalty_items == 1
    )
    middle_position = np.floor(
        rng.random(len(loyalty_candidate)) * np.maximum(loyalty_items - 1, 1)
    )
    loyalty_key = np.where(at_end, loyalty_items + 0.1, middle_position + 0.75)
    loyalty_unscaled = timing.loyalty_seconds * _lognormal_unit(
        rng, timing.fixed_noise_sigma, len(loyalty_candidate)
    )

    voucher_candidate = np.repeat(np.arange(count), n_vouchers)
    voucher_start = np.cumsum(n_vouchers) - n_vouchers
    voucher_index = np.arange(len(voucher_candidate)) - np.repeat(voucher_start, n_vouchers)
    voucher_key = n_items[voucher_candidate] + 0.2 + 0.01 * voucher_index
    voucher_unscaled = timing.voucher_seconds * _lognormal_unit(
        rng, timing.fixed_noise_sigma, len(voucher_candidate)
    )

    alpha_first, alpha_middle, alpha_close = (
        timing.alpha_seconds * share for share in timing.alpha_split
    )
    payment_scaled = timing.beta_seconds * lane_slowness * _lognormal_unit(rng, sigma, count)
    payment_unscaled = alpha_close * _lognormal_unit(rng, timing.fixed_noise_sigma, count)

    row_candidate = np.concatenate(
        [item_candidate, tag_candidate, loyalty_candidate, voucher_candidate, np.arange(count)]
    )
    row_key = np.concatenate(
        [
            item_position.astype(np.float64),
            tag_key,
            loyalty_key,
            voucher_key,
            n_items + 1.0,
        ]
    )
    row_type = np.concatenate(
        [
            item_type,
            np.full(len(tag_candidate), TAG),
            np.full(len(loyalty_candidate), LOYALTY),
            np.full(len(voucher_candidate), VOUCHER),
            np.full(count, PAYMENT),
        ]
    )
    row_unscaled = np.concatenate(
        [np.zeros(total_items), tag_unscaled, loyalty_unscaled, voucher_unscaled, payment_unscaled]
    )
    row_scaled = np.concatenate(
        [
            item_scaled,
            np.zeros(len(tag_candidate) + len(loyalty_candidate) + len(voucher_candidate)),
            payment_scaled,
        ]
    )
    order = np.lexsort((row_key, row_candidate))
    row_candidate = row_candidate[order]
    row_type = row_type[order]
    row_unscaled = row_unscaled[order]
    row_scaled = row_scaled[order]

    row_count = n_items + n_tagged + has_loyalty.astype(np.int64) + n_vouchers + 1
    row_start = np.cumsum(row_count) - row_count
    gap_rows = row_count - 1
    row_unscaled[row_start] = 0.0
    row_scaled[row_start] = 0.0

    everyone = np.arange(count)
    np.add.at(
        row_unscaled,
        row_start + 1,
        alpha_first * _lognormal_unit(rng, timing.fixed_noise_sigma, count),
    )
    np.add.at(
        row_unscaled,
        _random_gap_row(rng, row_start, gap_rows, everyone),
        alpha_middle * _lognormal_unit(rng, timing.fixed_noise_sigma, count),
    )

    customer_delay = rng.random(count) < timing.customer_delay_probability_per_transaction
    delayed = np.flatnonzero(customer_delay)
    np.add.at(
        row_unscaled,
        _random_gap_row(rng, row_start, gap_rows, delayed),
        rng.uniform(timing.customer_delay_low, timing.customer_delay_high, len(delayed)),
    )
    interruption = rng.random(count) < timing.interruption_probability_per_transaction
    interrupted = np.flatnonzero(interruption)
    interruption_row = _random_gap_row(rng, row_start, gap_rows, interrupted)
    np.add.at(
        row_unscaled,
        interruption_row,
        rng.uniform(timing.interruption_low, timing.interruption_high, len(interrupted)),
    )

    exception_candidate: list[np.ndarray] = []
    exception_type: list[np.ndarray] = []
    exception_anchor: list[np.ndarray] = []
    exception_detail: list[np.ndarray] = []

    if timing.emit_exception_events and len(interrupted):
        kinds = rng.choice(
            [SUPERVISOR, CHEQUE, PRICE_CHECK],
            size=len(interrupted),
            p=np.array(timing.interruption_mix_supervisor_cheque_price),
        )
        exception_candidate.append(interrupted)
        exception_type.append(kinds)
        exception_anchor.append(interruption_row - 1)
        exception_detail.append(np.full(len(interrupted), None, dtype=object))

    def add_extra_with_event(
        mask: np.ndarray, event_code: int, low: float, high: float, details: np.ndarray | None
    ) -> None:
        chosen = np.flatnonzero(mask)
        if len(chosen) == 0:
            return
        rows = _random_gap_row(rng, row_start, gap_rows, chosen)
        np.add.at(row_unscaled, rows, rng.uniform(low, high, len(chosen)))
        exception_candidate.append(chosen)
        exception_type.append(np.full(len(chosen), event_code))
        exception_anchor.append(rows - 1)
        exception_detail.append(
            np.full(len(chosen), None, dtype=object) if details is None else details[chosen]
        )

    def void_reasons(reason: VoidReason) -> np.ndarray:
        known = rng.random(count) < errors.void_reason_known_probability
        return np.where(known, reason.value, VoidReason.UNKNOWN.value).astype(object)

    cashier_void_probability = np.minimum(
        MAX_EVENT_PROBABILITY,
        errors.cashier_void_rate * np.exp(errors.cashier_void_slope * error_propensity),
    )
    for mask, reason in (
        (rng.random(count) < cashier_void_probability, VoidReason.CASHIER_ERROR),
        (rng.random(count) < errors.customer_void_rate, VoidReason.CUSTOMER_CHANGE),
        (rng.random(count) < errors.supervisor_void_rate, VoidReason.SUPERVISOR_REQUESTED),
    ):
        add_extra_with_event(
            mask, VOID, timing.void_extra_low, timing.void_extra_high, void_reasons(reason)
        )
    manual_probability = np.minimum(
        MAX_EVENT_PROBABILITY,
        errors.manual_entry_rate * np.exp(errors.manual_entry_slope * error_propensity),
    )
    add_extra_with_event(
        rng.random(count) < manual_probability,
        MANUAL,
        timing.manual_entry_extra_low,
        timing.manual_entry_extra_high,
        None,
    )

    is_cash = rng.random(count) < ctx.profiles.cash_share[register_index]
    basket_value = (
        n_items * errors.price_per_item_mean * _lognormal_unit(rng, errors.price_log_sigma, count)
    )
    cash_amount = np.where(is_cash, basket_value, 0.0)
    mistake_probability = np.minimum(
        MAX_EVENT_PROBABILITY,
        errors.mistake_rate_per_cash_transaction * np.exp(errors.mistake_slope * error_propensity),
    )
    mistake = is_cash & (rng.random(count) < mistake_probability)
    sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    mistake_amount = np.where(
        mistake,
        sign
        * errors.mistake_amount_median
        * np.exp(errors.mistake_amount_log_sigma * rng.standard_normal(count)),
        0.0,
    )

    unload = rng.uniform(timing.unload_low, timing.unload_high, count)
    post = rng.uniform(timing.post_payment_low, timing.post_payment_high, count) + np.where(
        is_cash, rng.uniform(CASH_EXTRA_LOW, CASH_EXTRA_HIGH, count), 0.0
    )
    idle_unit = rng.exponential(1.0, count)

    fixed_part = np.bincount(row_candidate, weights=row_unscaled, minlength=count)
    speed_part = np.bincount(row_candidate, weights=row_scaled, minlength=count)

    if exception_candidate:
        exc_candidate = np.concatenate(exception_candidate)
        exc_type = np.concatenate(exception_type)
        exc_anchor = np.concatenate(exception_anchor)
        exc_detail = np.concatenate(exception_detail)
    else:
        exc_candidate = np.zeros(0, dtype=np.int64)
        exc_type = np.zeros(0, dtype=np.int64)
        exc_anchor = np.zeros(0, dtype=np.int64)
        exc_detail = np.zeros(0, dtype=object)

    return _Batch(
        count=count,
        session=session,
        n_items=n_items,
        n_keyed=n_keyed,
        n_tagged=n_tagged,
        has_loyalty=has_loyalty,
        n_vouchers=n_vouchers,
        unload=unload,
        post=post,
        idle_unit=idle_unit,
        fixed_part=fixed_part,
        speed_part=speed_part,
        row_candidate=row_candidate,
        row_type=row_type,
        row_unscaled=row_unscaled,
        row_scaled=row_scaled,
        row_count=row_count,
        exception_candidate=exc_candidate,
        exception_type=exc_type,
        exception_anchor=exc_anchor,
        exception_detail=exc_detail,
        customer_delay=customer_delay,
        interruption=interruption,
        cash_amount=cash_amount,
        mistake_amount=mistake_amount,
    )


def _run_timeline(
    batch: _Batch,
    ctx: _Context,
    active_sessions: np.ndarray,
    candidate_offsets: np.ndarray,
    candidate_counts: np.ndarray,
    now: np.ndarray,
    last_end: np.ndarray,
    still_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traffic = ctx.config.simulator.traffic
    nominal_cycle = traffic.nominal_cycle_seconds
    stop_margin = traffic.stop_before_logout_seconds
    fixed_part = batch.fixed_part.tolist()
    speed_part = batch.speed_part.tolist()
    unload = batch.unload.tolist()
    post = batch.post.tolist()
    idle_unit = batch.idle_unit.tolist()
    served = [False] * batch.count
    start_seconds = [0.0] * batch.count
    effective_speed = [0.0] * batch.count
    is_peak_flags = [False] * batch.count

    for position in range(len(active_sessions)):
        session = int(active_sessions[position])
        first = int(candidate_offsets[position])
        last = first + int(candidate_counts[position])
        clock = float(now[session])
        logout = float(ctx.session_logout_s[session])
        base_speed = float(ctx.session_base_speed[session])
        peak_effect = float(ctx.session_peak_effect[session])
        utilization_row = ctx.rho_tables[ctx.session_day[session], ctx.session_lane[session]]
        peak_row = ctx.peak_table[ctx.session_weekday[session]]
        finished = False
        for candidate in range(first, last):
            minute = min(int(clock // 60.0), len(utilization_row) - 1)
            utilization = float(utilization_row[minute])
            idle = idle_unit[candidate] * nominal_cycle * (1.0 - utilization) / utilization
            start = round(clock + idle, 3)
            if start + stop_margin > logout:
                finished = True
                break
            peak = bool(peak_row[min(int(start // 60.0), len(peak_row) - 1)])
            speed = base_speed + (peak_effect if peak else 0.0)
            duration = fixed_part[candidate] + speed_part[candidate] * math.exp(-speed)
            clock = start + unload[candidate] + duration + post[candidate]
            served[candidate] = True
            start_seconds[candidate] = start
            effective_speed[candidate] = speed
            is_peak_flags[candidate] = peak
            last_end[session] = clock
        now[session] = clock
        still_active[session] = (not finished) and (logout - clock > stop_margin)

    return (
        np.array(served, dtype=bool),
        np.array(start_seconds),
        np.array(effective_speed),
        np.array(is_peak_flags, dtype=bool),
    )


@dataclass
class _Accumulator:
    transactions: list[pd.DataFrame] = field(default_factory=list)
    latent: list[pd.DataFrame] = field(default_factory=list)
    event_tx: list[np.ndarray] = field(default_factory=list)
    event_type: list[np.ndarray] = field(default_factory=list)
    event_offset: list[np.ndarray] = field(default_factory=list)
    event_detail: list[np.ndarray] = field(default_factory=list)
    next_index: int = 0


def _materialize(
    batch: _Batch,
    ctx: _Context,
    served: np.ndarray,
    start_seconds: np.ndarray,
    effective_speed: np.ndarray,
    is_peak: np.ndarray,
    sessions: pd.DataFrame,
    accumulator: _Accumulator,
) -> None:
    served_index = np.flatnonzero(served)
    served_count = len(served_index)
    if served_count == 0:
        return
    row_served = served[batch.row_candidate]
    rows = np.flatnonzero(row_served)
    row_candidate = batch.row_candidate[rows]
    gap = batch.row_unscaled[rows] + batch.row_scaled[rows] * np.exp(
        -effective_speed[row_candidate]
    )
    served_row_counts = batch.row_count[served_index]
    new_start = np.cumsum(served_row_counts) - served_row_counts
    cumulative = np.cumsum(gap)
    base = cumulative[new_start]
    local = np.repeat(np.arange(served_count), served_row_counts)
    offsets = batch.unload[row_candidate] + cumulative - base[local]

    new_row_index = np.cumsum(row_served) - 1
    exception_served = served[batch.exception_candidate]
    exception_anchor = new_row_index[batch.exception_anchor[exception_served]]
    exception_local = np.searchsorted(served_index, batch.exception_candidate[exception_served])
    exception_offsets = offsets[exception_anchor] + EXCEPTION_OFFSET_EPSILON

    global_base = accumulator.next_index
    accumulator.event_tx.append(
        np.concatenate(
            [
                np.arange(served_count) + global_base,
                local + global_base,
                exception_local + global_base,
            ]
        )
    )
    accumulator.event_type.append(
        np.concatenate(
            [
                np.full(served_count, STARTED),
                batch.row_type[rows],
                batch.exception_type[exception_served],
            ]
        )
    )
    accumulator.event_offset.append(
        np.concatenate([np.zeros(served_count), offsets, exception_offsets])
    )
    detail = np.concatenate(
        [
            np.full(served_count + len(rows), None, dtype=object),
            batch.exception_detail[exception_served],
        ]
    )
    accumulator.event_detail.append(detail)

    session = batch.session[served_index]
    midnight = sessions["date"].to_numpy(dtype="datetime64[ns]")[session]
    started_ns = midnight + (
        np.rint(start_seconds[served_index] * MILLISECONDS).astype(np.int64) * 1_000_000
    ).astype("timedelta64[ns]")
    register = ctx.profiles.registers[ctx.session_register_index[session]]
    session_ids = sessions["session_id"].to_numpy()[session]
    accumulator.transactions.append(
        pd.DataFrame(
            {
                "transaction_id": np.arange(served_count, dtype=np.int64) + global_base,
                "session_id": session_ids,
                "register": register.astype(np.int64),
                "started_at": started_ns,
            }
        )
    )
    duration = batch.fixed_part[served_index] + batch.speed_part[served_index] * np.exp(
        -effective_speed[served_index]
    )
    accumulator.latent.append(
        pd.DataFrame(
            {
                "transaction_id": np.arange(served_count, dtype=np.int64) + global_base,
                "session_id": session_ids,
                "worker_index": ctx.session_worker[session],
                "effective_log_speed": effective_speed[served_index],
                "is_peak": is_peak[served_index],
                "n_items": batch.n_items[served_index],
                "n_hand_keyed": batch.n_keyed[served_index],
                "n_tagged": batch.n_tagged[served_index],
                "has_loyalty": batch.has_loyalty[served_index],
                "n_vouchers": batch.n_vouchers[served_index],
                "true_duration_seconds": duration,
                "customer_delay": batch.customer_delay[served_index],
                "interruption": batch.interruption[served_index],
                "cash_amount": batch.cash_amount[served_index],
                "mistake_amount": batch.mistake_amount[served_index],
            }
        )
    )
    accumulator.next_index += served_count


def simulate_transactions(
    config: Config,
    sessions: pd.DataFrame,
    workers: pd.DataFrame,
    ground_truth: pd.DataFrame,
    profiles: RegisterProfiles,
    calendar: StoreCalendar,
) -> TransactionBundle:
    """Simulate the customers served in every session.

    Each transaction has a duration that depends on the cashier's hidden speed, the basket,
    the period and the lane. Returns the transactions, their events, the hidden latent
    values and the real end of each session.
    """
    simulator = config.simulator
    rng = make_rng(config.seed, replicate_stream(config, "transactions"))
    start_day = simulator.start_date
    days = [start_day + timedelta(days=offset) for offset in range(7 * simulator.n_weeks)]
    rho_tables = utilization_tables(config, calendar, days)
    season_basket = np.array([calendar.season(day)[1] for day in days])

    worker_index_of = {worker_id: index for index, worker_id in enumerate(workers["worker_id"])}
    session_worker = sessions["worker_id"].map(worker_index_of).to_numpy(dtype=np.int64)
    session_register_index = sessions["register"].to_numpy(dtype=np.int64) - 1
    session_day = ((sessions["date"] - pd.Timestamp(start_day)).dt.days).to_numpy(dtype=np.int64)
    login_s = (
        (sessions["login_ts"] - sessions["date"]).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    logout_s = (
        (sessions["logout_ts"] - sessions["date"]).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    population = simulator.population
    true_speed = ground_truth["true_speed"].to_numpy()
    deficit = ground_truth["learning_deficit"].to_numpy()
    hire = pd.to_datetime(workers["hire_date"]).to_numpy(dtype="datetime64[ns]")
    weeks_since_hire = np.maximum(
        0.0,
        (sessions["date"].to_numpy(dtype="datetime64[ns]") - hire[session_worker])
        / np.timedelta64(1, "D")
        / 7.0,
    )
    session_effect = population.session_effect_sd * rng.standard_normal(len(sessions))
    base_speed = (
        true_speed[session_worker]
        - deficit[session_worker] * np.exp(-weeks_since_hire / population.learning_tau_weeks)
        + session_effect
    )
    ctx = _Context(
        config=config,
        profiles=profiles,
        peak_table=calendar.peak_table,
        rho_tables=rho_tables,
        session_worker=session_worker,
        session_register_index=session_register_index,
        session_lane=profiles.lane_codes[session_register_index],
        session_day=session_day,
        session_weekday=sessions["weekday"].to_numpy(dtype=np.int64),
        session_login_s=login_s,
        session_logout_s=logout_s,
        session_base_speed=base_speed,
        session_peak_effect=ground_truth["peak_effect"].to_numpy()[session_worker],
        session_basket_multiplier=season_basket[session_day],
        error_propensity=ground_truth["error_propensity"].to_numpy(),
    )

    traffic = simulator.traffic
    session_count = len(sessions)
    now = login_s.copy()
    last_end = login_s.copy()
    still_active = np.ones(session_count, dtype=bool)
    accumulator = _Accumulator()
    rounds = 0

    for chunk_start in range(0, session_count, CHUNK_SESSIONS):
        chunk = np.arange(chunk_start, min(chunk_start + CHUNK_SESSIONS, session_count))
        while True:
            active = chunk[still_active[chunk]]
            if len(active) == 0:
                break
            rounds += 1
            remaining = ctx.session_logout_s[active] - now[active]
            mean_utilization = np.array(
                [
                    ctx.rho_tables[ctx.session_day[s], ctx.session_lane[s]][
                        int(now[s] // 60) : max(
                            int(ctx.session_logout_s[s] // 60), int(now[s] // 60) + 1
                        )
                    ].mean()
                    for s in active
                ]
            )
            expected = remaining * mean_utilization / traffic.nominal_cycle_seconds
            candidate_counts = np.maximum(
                MIN_CANDIDATES_PER_ROUND,
                np.ceil(expected * traffic.candidate_margin).astype(np.int64),
            )
            candidate_offsets = np.cumsum(candidate_counts) - candidate_counts
            candidate_session = np.repeat(active, candidate_counts)
            batch = _generate_candidates(rng, ctx, candidate_session)
            served, start_seconds, effective_speed, is_peak = _run_timeline(
                batch, ctx, active, candidate_offsets, candidate_counts, now, last_end, still_active
            )
            _materialize(
                batch, ctx, served, start_seconds, effective_speed, is_peak, sessions, accumulator
            )

    transactions = pd.concat(accumulator.transactions, ignore_index=True)
    latent = pd.concat(accumulator.latent, ignore_index=True)
    started_ns = transactions["started_at"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    order = np.lexsort((transactions["session_id"].to_numpy(), started_ns))
    new_id = np.empty(len(transactions), dtype=np.int64)
    new_id[order] = np.arange(1, len(transactions) + 1)
    transactions = transactions.iloc[order].reset_index(drop=True)
    transactions["transaction_id"] = np.arange(1, len(transactions) + 1, dtype=np.int64)
    latent = latent.iloc[order].reset_index(drop=True)
    latent["transaction_id"] = transactions["transaction_id"].to_numpy()

    event_tx = new_id[np.concatenate(accumulator.event_tx)]
    event_type = np.concatenate(accumulator.event_type)
    event_offset = np.concatenate(accumulator.event_offset)
    event_detail = np.concatenate(accumulator.event_detail)
    event_order = np.lexsort((event_offset, event_tx))
    events = make_events_frame(
        event_tx[event_order],
        event_type[event_order],
        event_offset[event_order],
        event_detail[event_order],
    )

    return TransactionBundle(
        transactions=transactions,
        events=events,
        latent=latent,
        session_logout=np.maximum(logout_s, last_end),
        rounds=rounds,
    )
