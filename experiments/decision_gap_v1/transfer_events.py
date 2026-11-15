"""Public registration and separate P0/archived-label transfer event ledgers.

The event is physical feasibility of the same outgoing trip at two stations,
not the controller's use of that trip. Registration never receives labels.
"""
from collections import Counter

import numpy as np

from experiments.reconstruction_v41.physical_posterior import posterior_transition


PARTITIONS = ('00', '01', '10', '11')  # upstream catch, downstream catch


def register_opportunities(graph, source_ride, current_cid, clock, expanded_ride=None):
    """Enumerate current candidates x later declared, nonabsorbing exits.

    Retain unsuccessful structural pairs as well as roots without opportunities.
    The expanded source ride supplies the menus; the original source ride
    supplies the physical continuation. Neither realized labels nor scores are
    inputs. Each through departure occurrence receives a distinct event row.
    """
    expanded_ride = source_ride if expanded_ride is None else expanded_ride
    platform = expanded_ride.exits.get(current_cid)
    current = graph.connections[current_cid]
    base = dict(current_cid=current_cid, clock=float(clock),
                incoming_trip=current.trip_id, upstream_stop=current.arr_stop,
                upstream_mct=float(graph.stops[current.arr_stop].min_transfer_time))
    if platform is None:
        return dict(**base, candidates=0, candidate_pairs=[], rows=[],
                    reason='no_current_platform', no_opportunity=True)
    source_position = source_ride.path.index(current_cid)
    source_suffix = list(source_ride.path[source_position+1:])
    pairs, rows = [], []
    for source_cid in expanded_ride.path[expanded_ride.path.index(current_cid)+1:]:
        if source_cid not in expanded_ride.exits:
            continue
        downstream = expanded_ride.exits[source_cid]
        if downstream is None:  # Destination is physically absorbing.
            break
        arrival = graph.connections[source_cid]
        if not getattr(arrival, 'dropoff_allowed', True):
            continue
        suffix = (source_suffix[:source_suffix.index(source_cid)+1]
                  if source_cid in source_suffix else None)
        for up_cid, ride in sorted(platform.rides.items()):
            up = graph.connections[up_cid]
            candidates = [cid for cid in ride.path[ride.path.index(up_cid)+1:]
                          if graph.connections[cid].dep_stop == arrival.arr_stop
                          and graph.connections[cid].pickup_allowed
                          and graph.connections[cid].trip_id == up.trip_id]
            pair = dict(upstream_cid=up_cid, source_arrival_cid=source_cid,
                        downstream_stop=arrival.arr_stop, outgoing_trip=up.trip_id,
                        hub_cids=candidates,
                        status='through_departure' if candidates else 'no_hub_departure')
            pairs.append(pair)
            for hub_cid in candidates:
                rows.append(dict(**base, event_index=len(rows),
                    upstream_cid=up_cid, hub_cid=hub_cid,
                    source_arrival_cid=source_cid, outgoing_trip=up.trip_id,
                    downstream_stop=arrival.arr_stop,
                    downstream_mct=float(graph.stops[arrival.arr_stop].min_transfer_time),
                    source_path=suffix,
                    outgoing_to_hub_path=list(ride.path[:ride.path.index(hub_cid)]),
                    hub_in_action_menu=hub_cid in downstream.rides,
                    same_incoming_outgoing_trip=current.trip_id == up.trip_id,
                    structure_status='registered' if suffix is not None else 'source_path_mismatch'))
    return dict(**base, candidates=len(platform.rides), candidate_pairs=pairs,
                pair_counts=dict(Counter(p['status'] for p in pairs)), rows=rows,
                no_opportunity=not rows, reason=None if rows else 'no_through_departure_pair')


def public_event_features(graph, event, public_history):
    """Prespecified descriptive groups and continuous, root-visible covariates."""
    forecasts = [row for row in public_history if row.get('trip') == event['outgoing_trip']
                 and row.get('evidence') == 'forecast']
    selected = [row for row in forecasts if row.get('emission_selected')]
    age = (event['clock']-max(row['receipt_seconds'] for row in selected)) if selected else None
    mct_difference = event['downstream_mct']-event['upstream_mct']
    source_delay = event['clock']-graph.connections[event['current_cid']].arr_time
    def edges(path):
        return {(graph.connections[cid].dep_stop, graph.connections[cid].arr_stop) for cid in path}
    shared = bool(edges(event['source_path'] or []) & edges(event['outgoing_to_hub_path']))
    return dict(upstream_ready=event['clock']+event['upstream_mct'],
        mct_source='gtfs_station_max', outgoing_selected_forecasts=selected,
        endpoint_forecasts=[row for row in forecasts if row['cid'] in (event['upstream_cid'], event['hub_cid'])],
        selected_forecast_age_seconds=age, mct_difference_seconds=mct_difference,
        source_observed_delay_seconds=source_delay, shared_directed_station_edge=shared,
        groups=dict(forecast_age='none' if age is None else 'at_most_300s' if age <= 300 else 'over_300s',
            mct_difference='nonpositive' if mct_difference <= 0 else 'at_most_180s' if mct_difference <= 180 else 'over_180s',
            source_delay='nonpositive' if source_delay <= 0 else 'at_most_60s' if source_delay <= 60 else 'over_60s',
            shared_corridor='shared' if shared else 'not_shared'))


def _kernel(law, left, right):
    kernel = posterior_transition(law, left)
    for slot in range(left+1, right):
        kernel = kernel @ posterior_transition(law, slot)
    return kernel


def source_arrival_mass(model, history, event, deadline):
    """Subprobability of reaching the downstream exit on the current ride.

    Only executed path endpoints impose service/order restrictions. Hidden
    chain slots transmit posterior probability, including future evidence.
    """
    positions = [model._positions[cid, kind] for cid in event['source_path']
                 for kind in ('departure', 'arrival')]
    trip = event['incoming_trip']
    slots = [slot for _, slot in positions]
    if any(other != trip for other, _ in positions) or any(
            right <= left for left, right in zip(slots, slots[1:])):
        raise ValueError('Source continuation is not an ordered single-trip path')
    law = history.law(trip)
    first = slots[0]
    mass = law.marginal(first).copy()
    canceled, late = law.canceled[first], law.times[first] >= deadline
    earlier = law.times[first] < event['clock']
    failed = dict(canceled=float(mass[canceled].sum()),
                  deadline=float(mass[~canceled & late].sum()),
                  time_order=float(mass[~canceled & ~late & earlier].sum()))
    mass *= ~canceled & ~late & ~earlier
    for left, right in zip(slots, slots[1:]):
        joint = mass[:, None]*_kernel(law, left, right)
        canceled, late = law.canceled[right], law.times[right] >= deadline
        earlier = law.times[right][None, :] < law.times[left][:, None]
        failed['canceled'] += float(joint[:, canceled].sum())
        failed['deadline'] += float(joint[:, ~canceled & late].sum())
        failed['time_order'] += float((joint*(~canceled & ~late)[None, :]*earlier).sum())
        mass = (joint*(~canceled & ~late)[None, :]*~earlier).sum(axis=0)
    return law.times[slots[-1]], mass, failed


def event_probabilities(model, history, event, deadline, cache=None):
    """Exact finite-chain P0 partition given the complete root public history.

    No future remote message is advanced. The outgoing endpoints stay joint;
    distinct incoming/outgoing trips factor under the frozen V41 model.
    """
    if event['structure_status'] != 'registered':
        return dict(status='probability_unresolved', reason=event['structure_status'])
    if event['same_incoming_outgoing_trip']:
        return dict(status='probability_unresolved', reason='incoming_outgoing_same_chain')
    cache = {} if cache is None else cache
    source_key = ('source', tuple(event['source_path']), event['clock'], deadline)
    if source_key not in cache:
        cache[source_key] = source_arrival_mass(model, history, event, deadline)
    source_times, source_mass, source_failures = cache[source_key]
    trip, first = model._positions[event['upstream_cid'], 'departure']
    other, last = model._positions[event['hub_cid'], 'departure']
    if trip != other or first >= last:
        return dict(status='probability_unresolved', reason='outgoing_endpoint_order')
    law = history.law(trip)
    joint_key = ('joint', trip, first, last)
    if joint_key not in cache:
        cache[joint_key] = law.marginal(first)[:, None]*_kernel(law, first, last)
    joint = cache[joint_key]
    up = (~law.canceled[first] & (law.times[first] >= event['clock']+event['upstream_mct'])
          & (law.times[first] < deadline))
    hub_active = ~law.canceled[last] & (law.times[last] < deadline)
    hub_given_time = np.array([source_mass[source_times <= when-event['downstream_mct']].sum()
                               for when in law.times[last]])*hub_active
    p_up = float(joint[up, :].sum())
    p_hub = float((joint @ hub_given_time).sum())
    p11 = float((joint[up, :] @ hub_given_time).sum())
    values = np.array([1-p_up-p_hub+p11, p_hub-p11, p_up-p11, p11])
    if values.min() < -1e-9 or abs(values.sum()-1) > 1e-9:
        raise ValueError('Invalid catch-event probability partition')
    return dict(status='resolved', partitions=dict(zip(PARTITIONS, np.maximum(values, 0.).tolist())),
                upstream_catch=p_up, hub_catch=p_hub, source_reach_probability=float(source_mass.sum()),
                source_first_failure_probabilities=source_failures,
                upstream_canceled=float(joint[law.canceled[first], :].sum()),
                hub_canceled=float(joint[:, law.canceled[last]].sum()))


def _endpoint(metadata, cid, kind):
    value = metadata.get(cid, metadata.get(str(cid), {})).get(kind, {})
    return dict(cid=cid, kind=kind, status=value.get('status', 'MISSING'),
                time=value.get('time_seconds'), journey=value.get('official_journey'))


def label_event(metadata, event, deadline):
    """Retrospective archived-endpoint labels; unknowns retain indicator bounds."""
    up = _endpoint(metadata, event['upstream_cid'], 'departure')
    hub = _endpoint(metadata, event['hub_cid'], 'departure')
    current = _endpoint(metadata, event['current_cid'], 'arrival')
    source = [_endpoint(metadata, cid, kind) for cid in event['source_path'] or []
              for kind in ('departure', 'arrival')]

    def known(endpoint):
        return endpoint['status'] == 'REAL' and endpoint['time'] is not None

    def active_after(endpoint, ready):
        if endpoint['status'] == 'CANCELED':
            return False
        if known(endpoint):
            return bool(ready <= endpoint['time'] < deadline)
        return None

    up_label = active_after(up, event['clock']+event['upstream_mct'])
    source_reach, source_arrival = True, event['clock']
    source_clock_inconsistent = False
    identity_bad = (current['journey'] is None or any(
        endpoint['journey'] is not None and endpoint['journey'] != current['journey']
        for endpoint in source))
    if not source or identity_bad:
        source_reach = None
    else:
        unknown = False
        for endpoint in source:
            if endpoint['status'] == 'CANCELED':
                source_reach = False
                break
            if not known(endpoint) or endpoint['journey'] is None:
                unknown = True
                continue
            if endpoint['time'] < source_arrival:
                source_clock_inconsistent = True
                source_reach = None
                break
            if endpoint['time'] >= deadline:
                source_reach = False
                break
            source_arrival = endpoint['time']
        else:
            if unknown:
                source_reach = None
    outgoing_identity_bad = (up['journey'] is not None and hub['journey'] is not None
                             and up['journey'] != hub['journey'])
    hub_label = (False if source_reach is False or hub['status'] == 'CANCELED' else
                 active_after(hub, source_arrival+event['downstream_mct']) if source_reach else None)
    if outgoing_identity_bad:
        # A malformed identity cannot be interpreted as a named through trip.
        up_label = hub_label = None
    possible = [str(u)+str(v) for u in (0, 1) for v in (0, 1)
                if (up_label is None or u == up_label) and (hub_label is None or v == hub_label)]
    return dict(status='resolved' if len(possible) == 1 else 'label_unresolved',
                upstream_catch=up_label, hub_catch=hub_label, source_reaches=source_reach,
                source_arrival=source_arrival if source_reach else None,
                upstream_ready_margin_seconds=(up['time']-event['clock']-event['upstream_mct']) if known(up) else None,
                hub_ready_margin_seconds=(hub['time']-source_arrival-event['downstream_mct'])
                    if known(hub) and source_reach else None,
                actual_partition=possible[0] if len(possible) == 1 else None,
                possible_partitions=possible,
                partition_bounds={key: [int(possible == [key]), int(key in possible)] for key in PARTITIONS},
                identity_issues=dict(incoming=identity_bad, outgoing=outgoing_identity_bad,
                    outgoing_missing=up['journey'] is None or hub['journey'] is None),
                source_clock_inconsistent=source_clock_inconsistent,
                endpoints=dict(upstream=up, hub=hub, current=current, source=source))


def archived_policy_usage(receipt, graph, event):
    """Recorded actions, separate from catch feasibility and model-world arms."""
    outcome = receipt['outcome']
    complete = outcome['loss'] is not None
    actions = receipt['actions']
    boards = [dict(cid=action['choice'], trip=graph.connections[action['choice']].trip_id,
                   stop=action['stop'], clock=action['clock'])
              for action in actions if action['kind'] == 'platform' and action['choice'] is not None]
    visits = [action for action in outcome.get('trace', [])
              if action['kind'] in ('platform', 'observation')]
    def occurred(condition):
        return True if condition else False if complete else None
    return dict(loss=outcome['loss'], status=outcome['status'],
        loss_bounds=[outcome['loss'], outcome['loss']] if complete else [0., receipt['loss_cap']],
        boards=boards,
        boarded_named_upstream=occurred(any(row['cid'] == event['upstream_cid'] for row in boards)),
        boarded_named_at_hub=occurred(any(row['cid'] == event['hub_cid'] for row in boards)),
        visited_hub_as_platform=occurred(any(row['stop'] == event['downstream_stop'] for row in visits)),
        source='archived endpoint replay actions, not observed passenger travel')


def sampled_events(events, event, deadline):
    """Evaluate the same two physical events on an existing indexed world bank."""
    up, canceled_up = events[event['upstream_cid'], 'departure']
    hub, canceled_hub = events[event['hub_cid'], 'departure']
    count = len(up)
    reaches = np.ones(count, dtype=bool)
    previous = np.full(count, event['clock'])
    for cid in event['source_path']:
        for kind in ('departure', 'arrival'):
            when, canceled = events[cid, kind]
            reaches &= ~canceled & (when >= previous) & (when < deadline)
            previous = when
    upstream = ~canceled_up & (up >= event['clock']+event['upstream_mct']) & (up < deadline)
    downstream = reaches & ~canceled_hub & (hub >= previous+event['downstream_mct']) & (hub < deadline)
    return upstream, downstream, reaches


def partition_saved_bank(events, event, gate, deadline):
    """Decompose saved paired losses without rerunning either control policy.

    Unfinished arm losses use their existing [0,H-clock] bounds. Physical catch
    feasibility is not substituted for actual boarding, which requires traces.
    """
    up, hub, reaches = sampled_events(events, event, deadline)
    count = len(up)
    span = deadline-event['clock']
    rows = gate['rows']
    indexed = {arm: {r['world_index']: r for r in rows[arm]} for arm in ('alight', 'continue')}
    lower, upper = [], []
    for index in range(count):
        bounds = {}
        for arm in indexed:
            row = indexed[arm][index]
            if row.get('loss') is not None:
                bounds[arm] = (row['loss'], row['loss'])
            elif 'value' in row:
                bounds[arm] = (row['value']['lower'], row['value']['upper'])
            else:
                bounds[arm] = (0., span)
        lower.append(bounds['alight'][0]-bounds['continue'][1])
        upper.append(bounds['alight'][1]-bounds['continue'][0])
    lower, upper = np.array(lower), np.array(upper)
    overall = [float(lower.mean()), float(upper.mean())]
    saved = gate.get('paired', {}).get('full_bank_mean_gain_interval')
    if saved is not None and not np.allclose(overall, saved, atol=1e-9, rtol=0):
        raise ValueError('Partition input losses do not recover the saved full-bank difference bounds')
    partitions = {}
    for key in PARTITIONS:
        mask = (up == bool(int(key[0]))) & (hub == bool(int(key[1])))
        n = int(mask.sum())
        partitions[key] = dict(count=n, contribution_bounds=[float(lower[mask].sum()/count),
            float(upper[mask].sum()/count)],
            conditional_difference_bounds=[float(lower[mask].mean()), float(upper[mask].mean())] if n else None)
    return dict(count=count, partitions=partitions, source_reach_count=int(reaches.sum()),
                overall_difference_bounds=overall, saved_full_bank_difference_bounds=saved,
                unresolved_pairs=int(np.sum(lower != upper)),
                world_rows=[dict(world_index=index, upstream_catch=bool(up[index]), hub_catch=bool(hub[index]),
                    source_reaches=bool(reaches[index]), difference_bounds=[float(lower[index]), float(upper[index])])
                    for index in range(count)],
                actual_policy_usage='not inferred from physical catch; consult saved policy traces',
                orientation='ALIGHT loss minus CONTINUE loss')
