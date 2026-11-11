"""Certificate for retaining a fixed, executable continuation policy."""

from math import ceil
from time import perf_counter

from .continuation_confidence import continuation_lower
from .temporal_scan import sample_clocks, template_trip_map


def check_continuation(bounds, state, plan, corridor, rng, *, tolerance, samples, alpha,
                       allow_middle, sample_sizes=None):
    """Check a frozen continuation, optionally stopping at sample prefixes.

    ``sample_sizes`` are cumulative counts ending at ``samples``. Each prefix
    adds fresh independent draws, never redraws earlier worlds, and uses the
    same e-process and alpha. The default retains the original single batch.
    """
    graph, goal, deadline = bounds.graph, bounds.destination, bounds.deadline
    if (plan is None or not plan.primary or state.current_trip is None
            or state.time != state.last_arrival or graph.connections[plan.primary[-1]].arr_stop != goal):
        return None
    if plan.primary[0] in state.unavailable_ids:
        return None
    terminal = all(graph.connections[cid].trip_id == state.current_trip for cid in plan.primary)
    if not terminal and not allow_middle:
        return None

    started = perf_counter()
    upper = bounds.evaluate(state, plan).upper
    details = dict(upper=upper, witness_seconds=perf_counter() - started)
    terminal_lower = 0.
    if terminal:
        started = perf_counter()
        value = bounds.transfer_gate(state, plan.primary)
        terminal_lower = value.lower if value is not None else 0.
        accepted = upper-terminal_lower <= tolerance
        details.update(terminal_lower=terminal_lower, terminal_seconds=perf_counter()-started)
        if accepted or not allow_middle or len(plan.primary) == 1:
            return dict(details, kind='terminal', lower=terminal_lower, accepted=accepted, alpha=0., worlds=0)

    scan = corridor.scan
    transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
    # A deterministic clipped oracle is still a lower bound for any admissible policy.
    cutoff = min(deadline, state.time + ceil(upper + 3 * tolerance))
    started = perf_counter()
    connections, floor = corridor.connections_and_floor(
        bounds.belief, bounds.parameters, state, goal, cutoff, transfer=transfers)
    trips = {bounds.belief._trip_by_cid[cid] for cid in connections}
    corridor_seconds = perf_counter() - started
    lower = max(terminal_lower, floor)
    details.update(support_floor=floor, cutoff=cutoff, trips=len(trips),
                   connections=len(connections), corridor_seconds=corridor_seconds)
    if upper-lower <= tolerance or not trips:
        return dict(details, kind='middle', lower=lower, accepted=upper-lower <= tolerance,
                    alpha=0., worlds=0)
    started = perf_counter()
    templates = template_trip_map(bounds.belief, bounds.parameters)
    template_seconds = perf_counter() - started
    sizes = (samples,) if sample_sizes is None else tuple(sample_sizes)
    if not sizes or sizes[-1] != samples or any(a >= b for a, b in zip((0, *sizes), sizes)):
        raise ValueError('Sample prefixes must increase from positive counts to samples')
    losses, checkpoints = [], []
    draw_seconds = scan_seconds = 0.
    for size in sizes:
        started = perf_counter()
        clocks = sample_clocks(bounds.belief, bounds.parameters, rng, size-len(losses),
                               trips=trips, templates=templates)
        draw_seconds += perf_counter() - started
        started = perf_counter()
        losses.extend((scan.arrivals(*clocks, state, goal, cutoff, transfer=transfers) - state.time).tolist())
        scan_seconds += perf_counter() - started
        lower = max(terminal_lower, continuation_lower(losses, cutoff-state.time, alpha, floor=floor))
        oracle_mean = float(sum(losses) / len(losses))
        accepted = upper-lower <= tolerance
        checkpoints.append(dict(worlds=len(losses), lower=lower, oracle_mean=oracle_mean, accepted=accepted))
        if accepted:
            break
    return dict(details, kind='middle', lower=lower, accepted=upper-lower <= tolerance, alpha=alpha,
                worlds=len(losses), oracle_mean=oracle_mean, sample_checkpoints=checkpoints,
                template_seconds=template_seconds, draw_seconds=draw_seconds, scan_seconds=scan_seconds)
