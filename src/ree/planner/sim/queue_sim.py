"""Discrete-event queue simulator of one store day: customers, registers and their staff."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

STAFF = 0
COMPLETE = 1
ARRIVE = 2
ABANDON = 3
CHECK = 4

WAITING = 1
IN_SERVICE = 2
DONE = 3
LEFT = 4
CLOSED = -1


@dataclass(frozen=True)
class DayCustomers:
    """Customers of a day: arrival time, fixed and variable service time, and patience."""

    arrival: np.ndarray
    fixed: np.ndarray
    variable: np.ndarray
    patience: np.ndarray

    def __len__(self) -> int:
        return int(self.arrival.shape[0])


@dataclass(frozen=True)
class StaffInterval:
    """A worker staffing a register over a time span."""

    register: int
    start: float
    end: float
    worker_id: str


@dataclass(frozen=True)
class QueueSettings:
    """How customers choose a line and how the day is observed."""

    discipline: str = "shortest_line"
    position_penalty: float = 0.0
    queue_limit: int = 5
    close_second: float = float("inf")
    register_positions: tuple[tuple[int, int], ...] = ()
    checkpoint_seconds: float = 0.0
    checkpoint_start: float = 0.0
    checkpoint_end: float = 0.0


@dataclass(frozen=True)
class Observation:
    """What a live controller sees at a checkpoint: register usage, arrivals and queue."""

    second: float
    registers: tuple[tuple[int, float, float], ...]
    arrivals: int
    window_seconds: float
    waiting: int


@dataclass
class DayOutcome:
    """Waits, service and abandonment of every customer, with register and queue totals."""

    arrival: np.ndarray
    wait: np.ndarray
    served: np.ndarray
    abandoned: np.ndarray
    register: np.ndarray
    service: np.ndarray
    busy_seconds: float
    open_seconds: float
    queue_area: float
    over_limit_seconds: float


Multiplier = Callable[[str, float], float]
Controller = Callable[[Observation], Mapping[int, Sequence[StaffInterval]] | None]


class _Day:
    def __init__(
        self,
        customers: DayCustomers,
        intervals: Sequence[StaffInterval],
        multiplier: Multiplier,
        settings: QueueSettings,
        controller: Controller | None = None,
    ) -> None:
        self.customers = customers
        self.controller = controller
        self.multiplier = multiplier
        self.settings = settings
        count = len(customers)
        self.status = np.zeros(count, dtype=np.int8)
        self.wait = np.zeros(count)
        self.register = np.full(count, -1, dtype=np.int64)
        self.service = np.zeros(count)
        self.where = np.full(count, -1, dtype=np.int64)
        names = sorted({item.worker_id for item in intervals})
        self.names = list(names)
        self.name_index = {name: index for index, name in enumerate(names)}
        self.registers = sorted({item.register for item in intervals})
        positions = dict(settings.register_positions)
        self.penalty = {
            register: settings.position_penalty * positions.get(register, register)
            for register in self.registers
        }
        self.worker_at = {register: CLOSED for register in self.registers}
        self.busy = {register: False for register in self.registers}
        self.queues: dict[int, deque[int]] = {register: deque() for register in self.registers}
        self.qlen = {register: 0 for register in self.registers}
        self.limbo: deque[int] = deque()
        self.heap: list[tuple[float, int, int, int, int]] = []
        self.sequence = 0
        self.waiting_now = 0
        self.busy_seconds = 0.0
        self.open_seconds = 0.0
        self.open_since: dict[int, float] = {}
        self.queue_area = 0.0
        self.over_limit_seconds = 0.0
        self.over = False
        self.clock = 0.0
        self.generation = {register: 0 for register in self.registers}
        self.event_generation: dict[int, tuple[int, int]] = {}
        self.service_log: list[tuple[int, float, float]] = []
        self.open_log: dict[int, list[list[float | None]]] = {r: [] for r in self.registers}
        self._schedule(intervals)

    def _push(self, time: float, kind: int, first: int, second: int = 0) -> None:
        heapq.heappush(self.heap, (time, kind, self.sequence, first, second))
        self.sequence += 1

    def _index(self, name: str) -> int:
        if name not in self.name_index:
            self.name_index[name] = len(self.names)
            self.names.append(name)
        return self.name_index[name]

    def _ensure_register(self, register: int) -> None:
        if register in self.generation:
            return
        self.registers = sorted([*self.registers, register])
        self.generation[register] = 0
        self.open_log[register] = []
        self.penalty[register] = self.settings.position_penalty * dict(
            self.settings.register_positions
        ).get(register, register)
        self.worker_at[register] = CLOSED
        self.busy[register] = False
        self.queues[register] = deque()
        self.qlen[register] = 0

    def _push_staff(self, time: float, register: int, worker: int) -> None:
        self.event_generation[self.sequence] = (register, self.generation[register])
        self._push(time, STAFF, register, worker)

    def _plan_register(
        self, register: int, own: Sequence[StaffInterval], not_before: float | None = None
    ) -> None:
        ordered = sorted(own, key=lambda i: i.start)
        for position, item in enumerate(ordered):
            if not_before is not None and item.end <= not_before:
                continue
            when = item.start if not_before is None else max(item.start, not_before)
            self._push_staff(when, register, self._index(item.worker_id))
            following = ordered[position + 1].start if position + 1 < len(ordered) else None
            continues = following is not None and following <= item.end
            if not continues and item.end < self.settings.close_second:
                self._push_staff(item.end, register, CLOSED)

    def _schedule(self, intervals: Sequence[StaffInterval]) -> None:
        for register in self.registers:
            self._plan_register(register, [i for i in intervals if i.register == register])
        settings = self.settings
        if self.controller is not None and settings.checkpoint_seconds > 0.0:
            moment = settings.checkpoint_start
            while moment <= settings.checkpoint_end:
                self._push(moment, CHECK, 0)
                moment += settings.checkpoint_seconds
        for index in range(len(self.customers)):
            self._push(float(self.customers.arrival[index]), ARRIVE, index)

    def _open_registers(self) -> list[int]:
        return [r for r in self.registers if self.worker_at[r] != CLOSED]

    def _advance(self, time: float) -> None:
        elapsed = time - self.clock
        if elapsed > 0.0:
            self.queue_area += self.waiting_now * elapsed
            if self.over:
                self.over_limit_seconds += elapsed
        self.clock = time

    def _refresh_over(self) -> None:
        limit = self.settings.queue_limit
        self.over = any(self.qlen[r] > limit for r in self.registers if self.worker_at[r] != CLOSED)

    def _start(self, register: int, customer: int, time: float) -> None:
        if self.status[customer] == WAITING:
            self.waiting_now -= 1
        self.status[customer] = IN_SERVICE
        self.wait[customer] = time - self.customers.arrival[customer]
        self.register[customer] = register
        self.where[customer] = -1
        worker = self.names[self.worker_at[register]]
        duration = float(
            self.customers.fixed[customer]
            + self.customers.variable[customer] * self.multiplier(worker, time)
        )
        self.service[customer] = duration
        self.busy[register] = True
        self.busy_seconds += duration
        self.service_log.append((register, time, time + duration))
        self._push(time + duration, COMPLETE, register, customer)

    def _pull(self, register: int, time: float) -> None:
        if self.busy[register] or self.worker_at[register] == CLOSED:
            return
        if self.settings.discipline != "shared":
            queue = self.queues[register]
            while queue:
                customer = queue.popleft()
                if self.status[customer] == WAITING and self.where[customer] == register:
                    self.qlen[register] -= 1
                    self._start(register, customer, time)
                    return
        while self.limbo:
            customer = self.limbo.popleft()
            if self.status[customer] == WAITING and self.where[customer] == CLOSED:
                self._start(register, customer, time)
                return

    def _place(self, customer: int, time: float, fresh: bool) -> None:
        if fresh:
            self.status[customer] = WAITING
            self.waiting_now += 1
        open_registers = self._open_registers()
        if not open_registers:
            self.where[customer] = CLOSED
            self.limbo.append(customer)
            return
        if self.settings.discipline == "shared":
            idle = [r for r in open_registers if not self.busy[r]]
            if idle:
                self._start(idle[0], customer, time)
            else:
                self.where[customer] = CLOSED
                self.limbo.append(customer)
            return
        chosen = min(
            open_registers,
            key=lambda r: (self.qlen[r] + (1 if self.busy[r] else 0) + self.penalty[r], r),
        )
        if not self.busy[chosen] and self.qlen[chosen] == 0:
            self._start(chosen, customer, time)
            return
        self.queues[chosen].append(customer)
        self.qlen[chosen] += 1
        self.where[customer] = chosen

    def _staff(self, register: int, worker: int, time: float) -> None:
        previous = self.worker_at[register]
        self.worker_at[register] = worker
        if worker == CLOSED:
            if previous != CLOSED:
                self.open_seconds += time - self.open_since.pop(register, time)
                if self.open_log[register] and self.open_log[register][-1][1] is None:
                    self.open_log[register][-1][1] = time
            stranded = [c for c in self.queues[register] if self.where[c] == register]
            self.queues[register].clear()
            self.qlen[register] = 0
            for customer in stranded:
                self.where[customer] = -1
                self._place(customer, time, fresh=False)
            return
        if previous == CLOSED:
            self.open_since[register] = time
            self.open_log[register].append([time, None])
            pending = [c for c in self.limbo if self.status[c] == WAITING]
            self.limbo.clear()
            for customer in pending:
                self.where[customer] = -1
                self._place(customer, time, fresh=False)
        self._pull(register, time)

    def _window(self, time: float) -> Observation:
        window = self.settings.checkpoint_seconds
        low = time - window
        usage: list[tuple[int, float, float]] = []
        for register in self.registers:
            opened = 0.0
            for start, end in self.open_log[register]:
                finish = time if end is None else min(end, time)
                opened += max(0.0, finish - max(start, low))
            busy = sum(
                max(0.0, min(end, time) - max(start, low))
                for owner, start, end in self.service_log
                if owner == register and end > low
            )
            usage.append((register, opened, busy))
        arrivals = int(
            np.searchsorted(self.customers.arrival, time, side="right")
            - np.searchsorted(self.customers.arrival, low, side="right")
        )
        return Observation(time, tuple(usage), arrivals, window, self.waiting_now)

    def _check(self, time: float) -> None:
        if self.controller is None:
            return
        changes = self.controller(self._window(time))
        if not changes:
            return
        for register, own in changes.items():
            self._ensure_register(register)
            self.generation[register] += 1
            current = [i for i in own if i.end > time]
            self._plan_register(register, current, not_before=time)
            covers_now = any(i.start <= time < i.end for i in current)
            if not covers_now and self.worker_at[register] != CLOSED:
                self._push_staff(time, register, CLOSED)

    def run(self) -> DayOutcome:
        patience = self.customers.patience
        while self.heap:
            time, kind, sequence, first, second = heapq.heappop(self.heap)
            self._advance(time)
            if kind == STAFF:
                tag = self.event_generation.pop(sequence, None)
                if tag is None or self.generation.get(tag[0]) == tag[1]:
                    self._staff(first, second, time)
            elif kind == CHECK:
                self._check(time)
            elif kind == COMPLETE:
                self.status[second] = DONE
                self.busy[first] = False
                self._pull(first, time)
            elif kind == ARRIVE:
                self._place(first, time, fresh=True)
                if np.isfinite(patience[first]):
                    self._push(time + float(patience[first]), ABANDON, first)
            else:
                if self.status[first] == WAITING:
                    self.status[first] = LEFT
                    self.wait[first] = time - self.customers.arrival[first]
                    self.waiting_now -= 1
                    if self.where[first] >= 0:
                        self.qlen[self.where[first]] -= 1
                    self.where[first] = -1
            self._refresh_over()
        for since in self.open_since.values():
            self.open_seconds += self.clock - since
        served = self.status == DONE
        return DayOutcome(
            arrival=self.customers.arrival,
            wait=self.wait,
            served=served,
            abandoned=self.status == LEFT,
            register=self.register,
            service=self.service,
            busy_seconds=self.busy_seconds,
            open_seconds=self.open_seconds,
            queue_area=self.queue_area,
            over_limit_seconds=self.over_limit_seconds,
        )


def simulate_day(
    customers: DayCustomers,
    intervals: Sequence[StaffInterval],
    multiplier: Multiplier,
    settings: QueueSettings | None = None,
    controller: Controller | None = None,
) -> DayOutcome:
    """Simulate a day; a controller may change the staffing at each checkpoint."""
    return _Day(customers, intervals, multiplier, settings or QueueSettings(), controller).run()
