"""Public-history selection and retrospective fixed-action labels for V30.

Future archive records are read only to reconstruct public events and label a
fixed action. Neither current-vehicle nor target choice is made here.
"""

from experiments.reconstruction_v24.run import options_from
from experiments.reconstruction_v25.run import departure, first_tail, physical_path, public_offers


def build_day_histories(graph, metadata, roots, v29_cases=()):
    """All roots' first legal offers, plus the six previously frozen splits.

    Coverage retains empty menus and unresolved roots. An overlapping V29
    split receives a second tag rather than a second calibration observation.
    """
    histories, coverage, by_key, offers_by_root = [], [], {}, {}
    root_by_id = {root['root_id']: root for root in roots}

    def add(root, entry, ready, public, tag):
        key = root['root_id'], public['clock']
        if key not in by_key:
            row = dict(root=root, day=root['day'], root_id=root['root_id'],
                history_id=f"{root['root_id']}@{public['clock']}",
                entry=entry, ready=ready, clock=public['clock'],
                visible=list(public['visible']), seen=list(public['seen']), tags=[])
            by_key[key] = row
            histories.append(row)
        row = by_key[key]
        row['tags'].append(tag)
        return row

    for root in roots:
        entry, status = root['start'], 'arrived'
        if root['prefix']:
            entry, status = physical_path(metadata, root['prefix'], root['start']+30, root['deadline'])
        row = dict(root_id=root['root_id'], day=root['day'], root_kind=root['root_kind'],
                   menu_size=len(root['full_options']), entry=entry, ready=None, clock=None)
        if status != 'arrived':
            row['status'] = 'prefix_'+status
        else:
            ready = entry+(max(30, graph.stops[root['stop']].min_transfer_time) if root['prefix'] else 30)
            row['ready'] = ready
            cids = [option['first_cid'] for option in root['full_options']]
            offers = public_offers(metadata, cids, entry, ready, root['deadline'])
            offers_by_root[root['root_id']] = entry, ready, offers
            row['status'] = ('empty_menu' if not cids else 'unresolved_menu_departure'
                             if offers is None else 'no_legal_offer' if not offers else 'offer')
            if row['status'] == 'offer':
                row['clock'] = offers[0]['clock']
                add(root, entry, ready, offers[0], 'first_offer')
        coverage.append(row)

    for case in v29_cases:
        saved = case['root']
        if saved['root_id'] not in root_by_id:
            continue
        root = root_by_id[saved['root_id']]
        if root != saved:
            raise AssertionError('V29 root differs from its frozen V25 menu')
        entry, ready, offers = offers_by_root[root['root_id']]
        public = next((offer for offer in offers or [] if offer['clock'] == case['clock']), None)
        if (public is None or entry != case['entry'] or ready != case['ready']
                or set(public['visible']) != set(case['visible']) or set(public['seen']) != set(case['seen'])):
            raise AssertionError('V29 public history no longer reproduces')
        row = add(root, entry, ready, public, 'v29_disagreement')
        row.update(frozen_current_cid=case['current_cid'], frozen_target_cid=case['target_cid'])
    return histories, coverage


def _observed_path(metadata, path, ready, deadline):
    """Physical same-vehicle arrival, preserving identified clocks after H."""
    now, vehicle, deadline_cost_known = ready, None, ready >= deadline
    for cid in path:
        row = metadata[cid]
        event = departure(row)
        if event is None:
            return None, 'unresolved_departure', deadline_cost_known
        clock, canceled = event
        if canceled or clock < now:
            return None, 'path_not_operating' if canceled else 'missed_connection', True
        deadline_cost_known |= clock >= deadline
        dep, arr = row['departure'], row['arrival']
        if arr['status'] != 'REAL':
            return None, 'unresolved_arrival', deadline_cost_known
        identity = dep['official_journey']
        if identity is None or identity != arr['official_journey'] or (vehicle is not None and identity != vehicle):
            return None, 'ambiguous_vehicle', deadline_cost_known
        vehicle, now = identity, arr['time_seconds']
        if now < clock:
            return None, 'inconsistent_clock', deadline_cost_known
        deadline_cost_known |= now >= deadline
    return now, 'completed', deadline_cost_known


def actual_option(graph, metadata, root, cid, entry, ready, clock, mode='board'):
    """Label now-BOARD or exclusive target-WAIT with the same frozen rescue.

    Cost is the absolute truncated arrival clock. Physical arrival labels are
    never filled with the deadline. Missing tails cannot become known failures.
    """
    if mode not in ('board', 'target'):
        raise ValueError(mode)
    option, deadline = options_from(root['full_options'])[cid], root['deadline']
    out = dict(cid=cid, mode=mode, status=None, missing_reason=None, cost=None,
        root_departure=None, remaining_wait=None, first_arrival=None,
        first_path_completed=None, tail_cid=None, tail_departure=None,
        tail_wait=None, tail_offer_known=None, tail_offer_observed=None,
        final_arrival=None, arrived=False)

    def finish(status, cost=None, missing=None):
        out.update(status=status, cost=cost, missing_reason=missing)
        if option.tails and cost == deadline and out['tail_offer_known'] is None:
            # Known nonboarding or an already reached deadline precludes a
            # legal tail even when later physical endpoints remain unknown.
            out.update(tail_offer_known=True, tail_offer_observed=False)
        return out

    event = departure(metadata[cid])
    if event is None:
        return finish('unresolved_root_departure', missing='unresolved_root_departure')
    time, canceled = event
    if canceled:
        out['first_path_completed'] = False
        return finish('target_canceled' if mode == 'target' else 'current_canceled', deadline)
    if mode == 'board' and not (time == clock and ready <= time < deadline):
        raise ValueError('BOARD label requires the currently visible legal departure')
    if mode == 'target' and time <= clock:
        if time >= entry:
            raise ValueError('Target already appeared in the public history')
        out['first_path_completed'] = False
        return finish('target_departed_before_entry', deadline)
    if time >= deadline:
        out['first_path_completed'] = False
        return finish('target_after_deadline', deadline)
    if time < ready:
        out['first_path_completed'] = False
        return finish('target_before_ready', deadline)
    out.update(root_departure=time, remaining_wait=time-clock)
    arrival, status, deadline_cost_known = _observed_path(metadata, option.first_path, time, deadline)
    if status != 'completed':
        known_failure = status in ('path_not_operating', 'missed_connection')
        out['first_path_completed'] = False if known_failure else None
        return finish('first_path_'+status, deadline if known_failure or deadline_cost_known else None,
                      None if known_failure else status)
    out.update(first_arrival=arrival, first_path_completed=True)
    if not option.tails:
        out.update(final_arrival=arrival, arrived=arrival <= deadline)
        return finish('arrived' if arrival <= deadline else 'deadline', min(arrival, deadline))
    if arrival >= deadline:
        out.update(tail_offer_known=True, tail_offer_observed=False)
        return finish('deadline', deadline)
    stop = graph.connections[option.first_path[-1]].arr_stop
    tail, status = first_tail(graph, metadata, option, arrival+graph.stops[stop].min_transfer_time, deadline)
    if status == 'unresolved_tail_departure':
        out.update(tail_offer_known=False)
        return finish(status, missing=status)
    out.update(tail_offer_known=True, tail_offer_observed=tail is not None)
    if tail is None:
        return finish(status, deadline)
    tail_time = departure(metadata[tail])[0]
    out.update(tail_cid=tail, tail_departure=tail_time, tail_wait=tail_time-arrival)
    final, status, deadline_cost_known = _observed_path(metadata, option.tails[tail], tail_time, deadline)
    if status != 'completed':
        known_failure = status in ('path_not_operating', 'missed_connection')
        return finish('tail_path_'+status, deadline if known_failure or deadline_cost_known else None,
                      None if known_failure else status)
    out.update(final_arrival=final, arrived=final <= deadline)
    return finish('arrived' if final <= deadline else 'deadline', min(final, deadline))
