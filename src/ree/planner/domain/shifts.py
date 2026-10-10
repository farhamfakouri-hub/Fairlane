"""Shift building blocks: paid and register time, shift patterns and their burden."""

from __future__ import annotations

from dataclasses import dataclass

from ree.planner.config import BurdenConfig, OpeningConfig, clock_to_minutes

DAY_MINUTES = 1440
SUNDAY = 6
SATURDAY = 5
FRIDAY = 4


def minutes_to_clock(minutes: int) -> str:
    """Format minutes after midnight as "HH:MM"."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def clock_compact(minutes: int) -> str:
    """Format minutes after midnight as "HHMM"."""
    return f"{minutes // 60:02d}{minutes % 60:02d}"


@dataclass(frozen=True)
class Block:
    """One stretch of work: the paid time and the time on a register inside it."""

    paid_start: int
    paid_end: int
    register_start: int
    register_end: int

    @property
    def register_minutes(self) -> int:
        """Minutes on a register."""
        return self.register_end - self.register_start

    @property
    def paid_minutes(self) -> int:
        """Paid minutes, including the clock-in and clock-out margins."""
        return self.paid_end - self.paid_start

    def overlap_with_register(self, start: int, end: int) -> int:
        """Minutes of register time that fall inside an interval."""
        return max(0, min(end, self.register_end) - max(start, self.register_start))

    def on_register_at(self, minute: int) -> bool:
        """Whether the worker is on a register at a minute."""
        return self.register_start <= minute < self.register_end


def make_block(
    register_start: int, register_end: int, clock_in_margin: int, clock_out_margin: int
) -> Block:
    """Build a block from its register times, adding the paid margins before and after."""
    return Block(
        paid_start=register_start - clock_in_margin,
        paid_end=register_end + clock_out_margin,
        register_start=register_start,
        register_end=register_end,
    )


@dataclass(frozen=True)
class ShiftPattern:
    """A day's work as one block, or two blocks separated by a cut for a split shift."""

    pattern_id: str
    kind: str
    blocks: tuple[Block, ...]

    @property
    def paid_start(self) -> int:
        """Start of the paid time."""
        return self.blocks[0].paid_start

    @property
    def paid_end(self) -> int:
        """End of the paid time."""
        return self.blocks[-1].paid_end

    @property
    def register_start(self) -> int:
        """Start of the first register block."""
        return self.blocks[0].register_start

    @property
    def register_end(self) -> int:
        """End of the last register block."""
        return self.blocks[-1].register_end

    @property
    def register_minutes(self) -> int:
        """Total minutes on a register."""
        return sum(block.register_minutes for block in self.blocks)

    @property
    def paid_minutes(self) -> int:
        """Total paid minutes."""
        return sum(block.paid_minutes for block in self.blocks)

    @property
    def cuts(self) -> tuple[int, ...]:
        """Lengths of the cuts between blocks."""
        return tuple(
            later.register_start - earlier.register_end
            for earlier, later in zip(self.blocks, self.blocks[1:], strict=False)
        )

    def coverage_minutes(self, slot_start: int, slot_end: int) -> int:
        """Minutes on a register inside a slot."""
        return sum(block.overlap_with_register(slot_start, slot_end) for block in self.blocks)

    def on_register_at(self, minute: int) -> bool:
        """Whether the worker is on a register at a minute."""
        return any(block.on_register_at(minute) for block in self.blocks)


def pattern_id_for(kind: str, blocks: tuple[Block, ...]) -> str:
    """Identifier made from the kind and the register times of the blocks."""
    parts = "+".join(
        f"{clock_compact(block.register_start)}-{clock_compact(block.register_end)}"
        for block in blocks
    )
    return f"{kind[0].upper()}{parts}"


def pattern_burden(
    pattern: ShiftPattern, weekday: int, burden: BurdenConfig, opening: OpeningConfig
) -> int:
    """Burden points of a pattern on a weekday.

    Points are added for closing, Friday evening, Sunday, Saturday, a split shift and an
    early start.
    """
    total = 0
    close_minute = clock_to_minutes(
        opening.sunday[1] if weekday == SUNDAY else opening.monday_to_saturday[1]
    )
    if weekday == SUNDAY:
        total += burden.sunday
    elif pattern.register_end >= close_minute:
        total += burden.closing
        if weekday == FRIDAY:
            total += burden.friday_evening
    if weekday == SATURDAY:
        total += burden.saturday
    if len(pattern.blocks) > 1:
        total += burden.split
    if pattern.register_start < clock_to_minutes(burden.early_before):
        total += burden.early
    return total
