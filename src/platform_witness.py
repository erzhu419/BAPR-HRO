"""Check the primary-only witness bridge to local-event platform execution."""


def platform_witness_conditions(bounds, state, plan, *, footpaths=(), transfer_time=None):
    """Return unmet conditions; an empty tuple authorizes the witness bridge.

    The executor must use the V10 primary-only adapter without ``wait_until``:
    board the next target when visible and immediately alight for a new trip.
    It uses the graph's station transfer times, integer local event clocks and
    the same frozen posterior, root and deadline as ``bounds``. This structural
    check evaluates no posterior laws and draws no worlds.
    """
    issues = []
    if footpaths or transfer_time is not None:
        issues.append('The bridge requires station transfer times and no footpaths')
    if bounds.belief.public_occurrences:
        issues.append('The witness posterior must have no public occurrence feed')
    if plan is None or not plan.primary:
        return tuple(issues + ['A nonempty primary itinerary is required'])
    if plan.rescue:
        issues.append('Only the primary itinerary is executed by this adapter')
    if not state.current_trip or state.time != state.last_arrival:
        issues.append('The root must be a fresh onboard arrival')

    graph = bounds.graph
    used = set(state.unavailable_ids)
    stop, previous_trip, previous_slot = state.stop, state.current_trip, None
    seen_trips = set()
    block_trip = None
    for cid in plan.primary:
        c = graph.connections[cid]
        if not c.trip_id:
            issues.append('Every primary connection must have a named trip')
            break
        if c.dep_stop != stop or cid in used or stop == bounds.destination:
            issues.append('The primary must be contiguous, unused and end at its first goal visit')
            break
        trip, slot = bounds.belief._positions[cid, 'departure']
        if trip != block_trip:
            if trip in seen_trips:
                issues.append('Each trip must occupy one consecutive itinerary block')
                break
            seen_trips.add(trip)
            block_trip = trip
        elif slot != previous_slot + 2:
            issues.append('Same-trip blocks must use consecutive modeled connection slots')
            break
        if c.trip_id == previous_trip:
            if any(other.id != cid and other.id not in used and other.trip_id == previous_trip
                   for other in graph.get_connections_from(stop)):
                issues.append('An unused same-trip departure could divert platform WAIT inertia')
                break
        elif not c.pickup_allowed:
            issues.append('Every transfer target must permit boarding')
            break
        used.add(cid)
        stop, previous_trip, previous_slot = c.arr_stop, c.trip_id, slot
    if stop != bounds.destination:
        issues.append('The primary must reach the declared destination')
    return tuple(issues)
