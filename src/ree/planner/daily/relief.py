"""Segments, handovers and the relief chain derived from a minute-by-minute timeline."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from ree.planner.daily.breaks import STEP
from ree.planner.daily.posts import Timeline
from ree.planner.domain.plan import (
    BREAK,
    FLOATER,
    BreakEntry,
    Handover,
    PostSegment,
    is_register_post,
    is_self_checkout_post,
)
from ree.planner.domain.shifts import ShiftPattern, minutes_to_clock

RELIEF = "relief"
RETURN = "return"
OPENING = "opening"
RELEASE = "release"
ROTATION = "rotation"


def is_required_post(post: str) -> bool:
    """Whether a post is a register or a self-checkout place that has to be staffed."""
    return is_register_post(post) or is_self_checkout_post(post)


def build_segments(
    day: date, timeline: Timeline, shifts: Mapping[str, ShiftPattern]
) -> list[PostSegment]:
    """Merge each worker's timeline into stretches at one post."""
    segments: list[PostSegment] = []
    for worker_id in sorted(timeline):
        cells = timeline[worker_id]
        for block in shifts[worker_id].blocks:
            run_post: str | None = None
            run_start = block.register_start
            for minute in range(block.register_start, block.register_end, STEP):
                post = cells.get(minute)
                if post is None:
                    raise ValueError(f"no post for {worker_id} at minute {minute}")
                if post != run_post:
                    if run_post is not None:
                        segments.append(PostSegment(worker_id, day, run_start, minute, run_post))
                    run_post, run_start = post, minute
            if run_post is not None:
                segments.append(
                    PostSegment(worker_id, day, run_start, block.register_end, run_post)
                )
    return segments


def derive_handovers(day: date, timeline: Timeline) -> list[Handover]:
    """Find the changes of post and give each a reason.

    The reasons are relief, return, opening, release and rotation.
    """
    leaving: dict[tuple[int, str], str] = {}
    held_before: dict[tuple[int, str], str] = {}
    for worker_id, cells in timeline.items():
        for minute, post in cells.items():
            before = cells.get(minute - STEP)
            if before is None:
                continue
            if is_required_post(before):
                held_before[(minute, before)] = worker_id
                if post == BREAK:
                    leaving[(minute, before)] = worker_id
    handovers: list[Handover] = []
    for worker_id in sorted(timeline):
        cells = timeline[worker_id]
        for minute in sorted(cells):
            before = cells.get(minute - STEP)
            after = cells[minute]
            if before is None or before == after or after == BREAK:
                continue
            replaced = leaving.get((minute, after)) if is_required_post(after) else None
            if replaced is not None and replaced != worker_id:
                reason = RELIEF
            elif before == BREAK:
                reason = RETURN
            elif is_required_post(after):
                reason = ROTATION if (minute, after) in held_before else OPENING
            elif before != FLOATER and after == FLOATER:
                reason = RELEASE
            else:
                reason = ROTATION
            handovers.append(
                Handover(
                    day,
                    minute,
                    worker_id,
                    before,
                    after,
                    reason,
                    replaced if reason == RELIEF else None,
                )
            )
    handovers.sort(key=lambda item: (item.minute, item.worker_id))
    return handovers


def describe_relief(handovers: list[Handover], breaks: list[BreakEntry]) -> list[str]:
    """One line for each relief handover, saying who takes over from whom."""
    lengths = {(item.worker_id, item.start): item.minutes for item in breaks}
    lines: list[str] = []
    for item in handovers:
        if item.reason != RELIEF or item.replaces is None:
            continue
        minutes = lengths.get((item.replaces, item.minute))
        suffix = f" for a {minutes}-minute break" if minutes else ""
        lines.append(
            f"{minutes_to_clock(item.minute)}  {item.worker_id} takes {item.to_post} "
            f"from {item.replaces}{suffix}"
        )
    return lines
