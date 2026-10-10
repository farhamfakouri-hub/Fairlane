"""Erlang queueing formulas used to size the number of open registers."""

from __future__ import annotations

from math import exp, inf


def erlang_b(servers: int, load: float) -> float:
    """Erlang B blocking probability for a number of servers and an offered load."""
    blocking = 1.0
    for count in range(1, servers + 1):
        blocking = load * blocking / (count + load * blocking)
    return blocking


def erlang_c(servers: int, load: float) -> float:
    """Erlang C probability that an arriving customer has to wait."""
    if load <= 0.0:
        return 0.0
    if servers <= load:
        return 1.0
    blocking = erlang_b(servers, load)
    return blocking / (1.0 - (load / servers) * (1.0 - blocking))


def mean_wait(servers: int, load: float, service_minutes: float) -> float:
    """Mean waiting time in minutes, infinite when the load reaches the number of servers."""
    if load <= 0.0:
        return 0.0
    if servers <= load:
        return inf
    return erlang_c(servers, load) * service_minutes / (servers - load)


def wait_exceeds(servers: int, load: float, service_minutes: float, target_minutes: float) -> float:
    """Probability that a customer waits longer than a target time."""
    if load <= 0.0:
        return 0.0
    if servers <= load:
        return 1.0
    rate = (servers - load) * target_minutes / service_minutes
    return erlang_c(servers, load) * exp(-rate)


def required_servers(
    load: float,
    service_minutes: float,
    target_minutes: float,
    alpha: float,
    minimum: int,
    maximum: int,
) -> tuple[int, bool]:
    """Fewest registers that keep the chance of exceeding the wait target below alpha.

    Returns the count and whether the maximum had to be used.
    """
    start = max(minimum, int(load) + 1)
    for servers in range(start, maximum + 1):
        if servers > load and wait_exceeds(servers, load, service_minutes, target_minutes) <= alpha:
            return servers, False
    return maximum, True


def queue_limit_target_minutes(
    queue_limit: int, service_minutes: float, open_registers: int
) -> float:
    """Waiting time that corresponds to a limit on the number of customers in line."""
    return queue_limit * service_minutes / max(open_registers, 1)
