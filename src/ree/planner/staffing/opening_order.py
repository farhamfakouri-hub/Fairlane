"""The order in which registers are opened and closed."""

from __future__ import annotations

from collections.abc import Sequence

from ree.planner.config import OPENING_ORDER


def registers_open(count: int, order: Sequence[int] = OPENING_ORDER) -> tuple[int, ...]:
    """Registers in service when a number of registers are open."""
    return tuple(order[: max(0, min(count, len(order)))])


def next_register_to_open(
    open_registers: Sequence[int], order: Sequence[int] = OPENING_ORDER
) -> int | None:
    """Next register to open in the order, or None when all are open."""
    for register in order:
        if register not in open_registers:
            return register
    return None


def last_register_to_close(
    open_registers: Sequence[int], order: Sequence[int] = OPENING_ORDER
) -> int | None:
    """Open register that closes first (the last one opened), or None."""
    for register in reversed(order):
        if register in open_registers:
            return register
    return None
