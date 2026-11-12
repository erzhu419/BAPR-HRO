"""Outcome-blind screening and bounded actual-clock mechanism replays."""

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
import json

import numpy as np

from experiments.reconstruction_v4.run_city_replay import OD_NAMES, load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v6.run_city_new_od import PAIRS
from src.evolving_trip_belief import EvolvingTripBelief
from src.offgrid_conditional import ActualClockScores
from src.platform_nominal_index import PlatformNominalIndex
from src.platform_transit import PlatformState
from src.public_rescue_candidates import PublicItinerary
from src.public_wait_recourse import compile_through_scores
from src.recourse_itinerary_router import build_trip_blocks, trip_key


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = ('experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz',
           'experiments/reconstruction_v6/city_confirmation_source_completed.json.gz',
           'experiments/reconstruction_data_audit/city_past_actual.json.gz')
READY_TIMES = tuple(range(7*3600+40*60, 9*3600, 15*60))
CAP = 40*60
PUBLICATION = 30
DIRECT_WINDOW, LATER_WINDOW, MIN_RUN_GAIN, CRITICAL_MARGIN = 600, 180, 15, 45
MIN_WAIT_GAIN = 10
TRANSFER_RISK = (.1, .9)
EARLY_SENSITIVITY = 120


def direct_paths(graph, blocks, stop, goal, ready, deadline):
    trips, positions = blocks
    options = []
    for c in graph.get_connections_from(stop):
        if not c.pickup_allowed or not ready <= c.dep_time <= ready+DIRECT_WINDOW:
            continue
        path, at = [], stop
        for cid in trips[trip_key(c)][positions[c.id]:]:
            leg = graph.connections[cid]
            if leg.dep_stop != at or leg.arr_time > deadline:
                break
            path.append(cid)
            at = leg.arr_stop
            if at == goal:
                options.append((c.dep_time, leg.arr_time-c.dep_time, c.id, tuple(path)))
                break
    return sorted(options)


def direct_pair(graph, options):
    if not options:
        return None, 'no_direct_template'
    first = options[0]
    later = min((other for other in options
                 if 0 < other[0]-first[0] <= LATER_WINDOW and first[1]-other[1] >= MIN_RUN_GAIN
                 and trip_key(graph.connections[first[2]]) != trip_key(graph.connections[other[2]])),
                key=lambda option: (option[0]+option[1], option[0], option[2]), default=None)
    if later is None:
        return None, 'normal_without_later_faster'
    margin = later[0]+later[1]-first[0]-first[1]
    return (first, later), 'near_critical' if abs(margin) <= CRITICAL_MARGIN else 'normal_with_later_faster'


def downstream_catalog(graph, blocks, stop, ready, deadline):
    """All scheduled direct destinations at the fixed origins and transfer stops."""
    trips, positions = blocks
    catalog = defaultdict(list)
    for c in graph.get_connections_from(stop):
        if not c.pickup_allowed or not ready-EARLY_SENSITIVITY <= c.dep_time <= ready+DIRECT_WINDOW:
            continue
        path, at, seen = [], stop, {stop}
        for cid in trips[trip_key(c)][positions[c.id]:]:
            leg = graph.connections[cid]
            if leg.dep_stop != at or leg.arr_time > deadline:
                break
            path.append(cid)
            at = leg.arr_stop
            if at not in seen:
                catalog[at].append((c.dep_time, leg.arr_time-c.dep_time, c.id, tuple(path)))
                seen.add(at)
    return catalog


def sensitivity_pair(graph, options):
    """Choose a public nominal trade-off before reading either realized trip."""
    pairs = ((slow,fast) for slow in options for fast in options
             if abs(fast[0]-slow[0]) <= LATER_WINDOW and slow[1]-fast[1] >= MIN_RUN_GAIN
             and trip_key(graph.connections[slow[2]]) != trip_key(graph.connections[fast[2]]))
    return min(pairs, key=lambda p: (abs(sum(p[0][:2])-sum(p[1][:2])),
                                    max(p[0][0],p[1][0]),p[0][2],p[1][2]), default=None)


def supported_departure(distribution, ready, deadline):
    clocks, mass, cancel = distribution
    return (1-cancel)*mass[(ready <= clocks) & (clocks < deadline)].sum() > 0


def scan_reordering(graph, blocks, belief, params, actual, boundaries):
    """Sensitivity to schedule-early offers and static downstream destination choice."""
    rows, public_windows = [], Counter()
    for (stop,clock,mct,incoming), (label,prefix) in boundaries.items():
        ready = clock+max(mct,PUBLICATION)
        catalog = downstream_catalog(graph,blocks,stop,ready,label['deadline'])
        public_windows['platform_boundaries'] += 1
        entry, status = fixed_arrival(graph,actual,prefix,label['start']+PUBLICATION,label['deadline']) if prefix else (clock,'arrived')
        distributions = {}
        for goal, options in catalog.items():
            public_windows['direct_destination_windows'] += 1
            if len({trip_key(graph.connections[option[2]]) for option in options}) < 2:
                continue
            public_windows['multi_trip_destination_windows'] += 1
            if sensitivity_pair(graph,options) is None:
                continue
            for option in options:
                cid = option[2]
                if cid not in distributions:
                    distributions[cid] = belief.event_distribution(cid,'departure',params)
            pair = sensitivity_pair(graph,[option for option in options
                                            if supported_departure(distributions[option[2]],ready,label['deadline'])])
            if pair is None:
                continue
            margin = sum(pair[0][:2])-sum(pair[1][:2])
            category = 'near_critical' if abs(margin) <= CRITICAL_MARGIN else 'normal_with_tradeoff'
            replay = (replay_direct(graph,actual,distributions[pair[1][2]],pair,entry,mct,label['deadline'])
                      if status=='arrived' else dict(status='prefix_'+status))
            shadow = (exact_shadow(graph,belief,params,actual,pair,entry,mct,prefix,label['deadline'],replay)
                      if category=='near_critical' else None)
            rows.append(dict(**label, mechanism='direct', stop=stop, sensitivity_goal=graph.stops[goal].name,
                             incoming=incoming, prefix=prefix, mct=mct, **{'class':category}, scheduled_ready=ready,
                             scheduled_margin=margin, pair=pair, replay=replay, exact_shadow=shadow))
    return rows, dict(public_windows)


def fixed_arrival(graph, actual, path, ready, deadline):
    """Replay one fixed path; absent endpoints stay unresolved."""
    at, previous = ready, None
    for cid in path:
        c = graph.connections[cid]
        event = actual.events.get(cid)
        if event is None:
            return None, 'unresolved'
        earliest = at + (graph.stops[c.dep_stop].min_transfer_time
                         if previous is not None and previous != c.trip_id else 0)
        if event.canceled or event.departure < earliest or event.departure >= deadline or event.arrival > deadline:
            return deadline, 'deadline'
        at, previous = event.arrival, c.trip_id
    return at, 'arrived'


def wait_score(distribution, run, entry, observed, deadline):
    """Forecast conditional on no departure observed during this platform visit."""
    clocks, weights, cancel = distribution
    weights = weights*(1-cancel)
    future = clocks > observed
    missed = clocks < entry
    mass = float(weights[future | missed].sum()+cancel)
    if mass <= 0.:
        return deadline
    loss = np.minimum(deadline, clocks[future]+run)
    return float((weights[future]@loss+deadline*(weights[missed].sum()+cancel))/mass)


def replay_direct(graph, actual, distribution, pair, entry, mct, deadline):
    first, later = pair
    a, b = (actual.events.get(option[2]) for option in pair)
    if a is None or b is None:
        return dict(status='unresolved_departure')
    ready = entry+max(mct, PUBLICATION)
    available = sorted((event.departure, option[1], option[2], option)
                       for event, option in ((a, first), (b, later))
                       if not event.canceled and ready <= event.departure < deadline)
    if not available:
        return dict(status='paired', first=None, comparison=None, gain=0., action_changed=False,
                    gain_vs_schedule_wait=0., arrival_first=False, arrival_comparison=False)
    tick, _, cid, chosen = available[0]
    choice, scheduled_choice, forecast = chosen, chosen, None
    if cid == first[2] and not (not b.canceled and entry <= b.departure <= tick):
        # The action sees the first offer and earlier local nonoccurrence, not
        # B's future departure clock, eventual cancellation, or measured arrival.
        forecast = wait_score(distribution, later[1], entry, tick, deadline)
        if tick+first[1]-forecast >= MIN_WAIT_GAIN:
            choice = later
        if later[0] > tick and tick+first[1]-sum(later[:2]) >= MIN_WAIT_GAIN:
            scheduled_choice = later
    simple, simple_status = fixed_arrival(graph, actual, chosen[-1], tick, deadline)
    if chosen == choice:
        advanced, advanced_status = simple, simple_status
    else:
        advanced, advanced_status = fixed_arrival(graph, actual, choice[-1], tick+1, deadline)
    if scheduled_choice == choice:
        scheduled = advanced
    elif scheduled_choice == chosen:
        scheduled = simple
    else:
        scheduled, _ = fixed_arrival(graph, actual, scheduled_choice[-1], tick+1, deadline)
    if simple is None or advanced is None:
        return dict(status='unresolved_tail', decision_clock=tick, first=cid, comparison=choice[2],
                    action_changed=chosen != choice, predicted_wait_clock=forecast)
    return dict(status='paired', decision_clock=tick, first=cid, comparison=choice[2],
                action_changed=chosen != choice, schedule_comparison=scheduled_choice[2],
                predicted_wait_clock=forecast, gain=simple-advanced,
                gain_vs_schedule_wait=None if scheduled is None else scheduled-advanced,
                arrival_first=simple_status=='arrived', arrival_comparison=advanced_status=='arrived')


def exact_shadow(graph, belief, params, actual, pair, entry, mct, prefix, deadline, nominal):
    """Change only the offered and waiting marks at the witnessed decision."""
    tick, first = nominal.get('decision_clock'), nominal.get('first')
    if tick is None or first is None:
        return dict(status=nominal['status'], wall_seconds=0., changed=False,
                    gain=0. if nominal['status']=='paired' else None)
    started = perf_counter()
    slow, fast = pair
    a, b = (actual.events[option[2]] for option in pair)
    both_visible = not a.canceled and not b.canceled and a.departure==b.departure==tick
    may_wait = first == slow[2] and not (not b.canceled and entry <= b.departure <= tick)
    if not both_visible and not may_wait:
        return dict(status=nominal['status'], wall_seconds=perf_counter()-started, changed=False,
                    gain=0. if nominal['status']=='paired' else None)
    options = {option[2]:PublicItinerary(option[-1],sum(option[:2])) for option in pair}
    trips = {trip_key(graph.connections[option[2]]) for option in pair}
    if trips.intersection(trip_key(graph.connections[cid]) for cid in prefix):
        return dict(status='overlapping_prefix_trip', wall_seconds=perf_counter()-started, changed=False, gain=None)
    stop = graph.connections[first].dep_stop
    goal = graph.connections[slow[-1][-1]].arr_stop
    station_mct = graph.stops[stop].min_transfer_time
    root = PlatformState(stop,entry,last_arrival=entry+mct-station_mct,used_connections=frozenset(prefix))
    scores, _ = compile_through_scores(belief,params,graph,root,goal,deadline,{},options)
    visible = tuple(options) if both_visible else (first,)
    observation = PlatformState(stop,tick,last_arrival=root.last_arrival,used_connections=root.used_connections,
                                visible_connections=visible,
                                public_log=tuple((tick,cid,'departure') for cid in visible),initialized=True)
    clock_scores = ActualClockScores(belief,params,scores,goal)
    marks = {cid:clock_scores.at_visible(cid,observation,'projection') for cid in visible}
    if any(mark is None for mark in marks.values()) or fast[2] not in scores.rows:
        return dict(status='unscored_observed_clock', wall_seconds=perf_counter()-started,changed=False,gain=None)
    offer = marks[first].expected_arrival_or_deadline
    exact_choice, wait = first, None
    if both_visible:
        exact_choice = min(marks,key=lambda cid:(marks[cid].expected_arrival_or_deadline,cid))
    else:
        times, mass, cancel = belief.event_distribution(fast[2],'departure',params)
        survives = (1-cancel)*mass[(times < entry) | (times > tick)].sum()+cancel
        benefit = sum(row.probability*(deadline-row.expected_arrival_or_deadline)
                      for t,row in scores.rows[fast[2]].items() if tick < t < deadline)
        wait = float(deadline-benefit/survives) if survives > 0 else deadline
        if offer-wait >= MIN_WAIT_GAIN:
            exact_choice = fast[2]
    charged = perf_counter()-started
    original_choice = nominal['comparison']
    changed = exact_choice != original_choice
    known, _ = fixed_arrival(graph,actual,options[original_choice].path,
                             tick+int(original_choice not in visible),deadline)
    new, _ = (fixed_arrival(graph,actual,options[exact_choice].path,tick+int(exact_choice not in visible),deadline)
              if changed else (known,None))
    return dict(status='paired' if known is not None and new is not None else 'unresolved_tail',
                changed=changed, nominal_first=original_choice, exact_first=exact_choice,
                scored_clock_mode=marks[first].method, score_board=offer, score_wait=wait, wall_seconds=charged,
                gain=None if known is None or new is None else known-new)


def catch_probability(departure, arrival, mct):
    d, pd, cancel = departure
    a, pa, _ = arrival
    return float((1-cancel)*(pd @ (d[:, None] >= a+mct) @ pa))


def replay_transfer(graph, actual, dep, arrival_distribution, incoming, target_path, backup,
                    start, prefix, mct, deadline):
    clock, prefix_status = fixed_arrival(graph, actual, prefix, start+PUBLICATION, deadline)
    if prefix_status != 'arrived':
        return dict(status='prefix_'+prefix_status)
    target = target_path[0]
    clocks, mass, canceled = dep
    ready = clock+max(mct, PUBLICATION)
    prior = catch_probability(dep, arrival_distribution, mct)
    posterior = float((1-canceled)*mass[clocks >= ready].sum())
    # When ready, previously observed departing vehicles may have been ruled
    # out. This witness issues a route at arrival, before those future receipts.
    before = backup if backup is not None and prior < .5 else target_path
    chosen = backup if backup is not None and posterior < .5 else target_path
    original, original_status = fixed_arrival(graph, actual, before, ready, deadline)
    updated, updated_status = ((original, original_status) if chosen == before else
                               fixed_arrival(graph, actual, chosen, ready, deadline))
    event = actual.events.get(target)
    slack = None if event is None or event.canceled else event.departure-ready
    return dict(status='paired' if original is not None and updated is not None else 'unresolved_tail',
                observed_incoming=incoming, observed_arrival=clock, public_ready=ready, actual_slack=slack,
                prior_catch=prior, updated_catch=posterior,
                threshold_crossed=(prior < .5) != (posterior < .5), first=before[0],
                comparison=chosen[0], action_changed=chosen != before,
                gain=None if original is None or updated is None else original-updated,
                arrival_first=original_status=='arrived', arrival_comparison=updated_status=='arrived')


def summarize(rows):
    result = {}
    for kind in ('direct', 'transfer'):
        cohort = [r for r in rows if r['mechanism']==kind]
        counts = dict(Counter(r['class'] for r in cohort))
        groups = {}
        for label in ('near_critical', 'normal'):
            records = [r['replay'] for r in cohort if (r['class']=='near_critical') == (label=='near_critical')]
            paired = [r for r in records if r['status']=='paired']
            gain = [r['gain'] for r in paired]
            schedule_gain = [r['gain_vs_schedule_wait'] for r in paired if r.get('gain_vs_schedule_wait') is not None]
            groups[label] = dict(windows=len(records), replay_status=dict(Counter(r['status'] for r in records)),
                                 paired=len(paired), changed=sum(r['action_changed'] for r in paired),
                                 threshold_crossed=sum(r.get('threshold_crossed',False) for r in records),
                                 improved=sum(g > 0 for g in gain), harmed=sum(g < 0 for g in gain),
                                 mean_gain=mean(gain) if gain else None,
                                 paired_vs_schedule_wait=len(schedule_gain),
                                 mean_gain_vs_schedule_wait=mean(schedule_gain) if schedule_gain else None,
                                 arrival_first=sum(r['arrival_first'] for r in paired),
                                 arrival_comparison=sum(r['arrival_comparison'] for r in paired))
        result[kind] = dict(classes=counts, groups=groups)
    return result


def unique_changed_events(rows):
    def key(row):
        replay = row['replay']
        return row['day'], row['stop'], replay['decision_clock'], replay['first'], replay['comparison']

    paired = [row for row in rows if row['replay']['status']=='paired' and row['replay']['action_changed']]
    return dict(changed=len({key(row) for row in paired}),
                improved=len({key(row) for row in paired if row['replay']['gain'] > 0}),
                harmed=len({key(row) for row in paired if row['replay']['gain'] < 0}))


def summarize_shadow(rows):
    results = [r['exact_shadow'] for r in rows if r['exact_shadow'] is not None]
    paired = [r for r in results if r['status']=='paired']
    return dict(windows=len(results),statuses=dict(Counter(r['status'] for r in results)),
                paired=len(paired), changed=sum(r['changed'] for r in results),
                paired_changed=sum(r['changed'] for r in paired),
                improved=sum(r['gain']>0 for r in paired),harmed=sum(r['gain']<0 for r in paired),
                mean_gain=mean(r['gain'] for r in paired) if paired else None,
                scored=sum('score_board' in r for r in results),total_wall_seconds=sum(r['wall_seconds'] for r in results))


def run_day(data, day, fit):
    graph, actual, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    blocks = build_trip_blocks(graph)
    by_name = {row['name']: stops[key] for key,row in sorted(data['stops'].items())}
    rows, issues, unique_transfers, sensitivity_boundaries = [], Counter(), set(), {}
    for origin_name, goal_name in (*OD_NAMES, *PAIRS):
        origin, goal = by_name[origin_name], by_name[goal_name]
        for start in READY_TIMES:
            deadline = start+CAP
            label = dict(day=day, origin=origin_name, goal=goal_name, start=start, deadline=deadline)
            index = PlatformNominalIndex(graph, goal, start, deadline, trip_blocks=blocks)
            nominal = index.path(origin, start+PUBLICATION)
            issues['no_nominal_path' if not nominal else 'nominal_path'] += 1
            boundaries = [(origin, start, 0, (), None)]
            for i, (incoming,target) in enumerate(zip(nominal or (), (nominal or ())[1:]), start=1):
                a,b = graph.connections[incoming], graph.connections[target]
                if a.trip_id == b.trip_id:
                    continue
                key = origin, goal, incoming, target
                if key in unique_transfers:
                    continue
                unique_transfers.add(key)
                mct = graph.stops[a.arr_stop].min_transfer_time
                prefix, target_path = nominal[:i], nominal[i:]
                boundaries.append((a.arr_stop,a.arr_time,mct,prefix,incoming))
                departure = belief.event_distribution(target, 'departure', params)
                arrival = belief.event_distribution(incoming, 'arrival', params)
                p = catch_probability(departure, arrival, mct)
                backup = index.path(a.arr_stop, a.arr_time+mct, excluded_trip=b.trip_id)
                category = 'near_critical' if TRANSFER_RISK[0] <= p <= TRANSFER_RISK[1] else 'normal'
                rows.append(dict(**label, mechanism='transfer', stop=a.arr_stop, **{'class':category},
                                 scheduled_ready=a.arr_time+mct, mct=mct, nominal_slack=b.dep_time-a.arr_time-mct,
                                 prefix=prefix, target=target_path, backup=backup, catch_before_prefix=p,
                                 replay=replay_transfer(graph,actual,departure,arrival,incoming,target_path,backup,
                                                        start,prefix,mct,deadline)))
            for stop,clock,mct,prefix,incoming in boundaries:
                sensitivity_boundaries.setdefault((stop,clock,mct,incoming),(label,prefix))
                nominal_ready = clock+max(mct,PUBLICATION)
                options = direct_paths(graph,blocks,stop,goal,nominal_ready,deadline)
                pair, category = direct_pair(graph,options)
                replay = dict(status='no_pair')
                if pair is not None:
                    entry, status = fixed_arrival(graph,actual,prefix,start+PUBLICATION,deadline) if prefix else (start,'arrived')
                    if status == 'arrived':
                        distribution = belief.event_distribution(pair[1][2],'departure',params)
                        replay = replay_direct(graph,actual,distribution,pair,entry,mct,deadline)
                    else:
                        replay = dict(status='prefix_'+status)
                rows.append(dict(**label, mechanism='direct', stop=stop, **{'class':category},
                                 scheduled_ready=nominal_ready, incoming=incoming, options=len(options),
                                 pair=pair, replay=replay))
    sensitivity, public_windows = scan_reordering(graph,blocks,belief,params,actual,sensitivity_boundaries)
    return rows, sensitivity, dict(od_windows=dict(issues), source_status=dict(Counter(r['outcome'] for r in metadata.values())),
                                   publications=len(actual.publications or {}), summary=summarize(rows),
                                   reordering_counts=public_windows, reordering=summarize(sensitivity)['direct'])


def main():
    rows, sensitivity, by_day, fit = [], [], {}, None
    for name in SOURCES:
        data, _ = load_city(ROOT/name)
        if fit is None:
            fit = fit_training(data,retain_no_pickup=True)
        for day in data['holdout_dates']:
            day_rows, sensitivity_rows, audit = run_day(data,day,fit)
            rows.extend(day_rows)
            sensitivity.extend(sensitivity_rows)
            by_day[day] = audit
            print(day, audit, flush=True)
    record = dict(status='complete', scope='Fixed OD/time mechanism screen on previously exposed development dates',
                  inputs=SOURCES, train_dates=fit['train_dates'], od_pairs=(*OD_NAMES,*PAIRS), ready_times=READY_TIMES,
                  deadline_seconds=CAP, common_publication_delay=PUBLICATION,
                  direct_rule=dict(offer_window=DIRECT_WINDOW,later_departure_window=LATER_WINDOW,
                                   minimum_nominal_running_gain=MIN_RUN_GAIN,critical_abs_arrival_margin=CRITICAL_MARGIN,
                                   minimum_forecast_wait_gain=MIN_WAIT_GAIN),
                  transfer_rule=dict(near_critical_catch=TRANSFER_RISK,switch_below=.5,
                                     transfer_time='GTFS station max, default 120s'),
                  limitations=['The screen uses static OD, schedule, July25 fit and public prefix receipts only.',
                               'Each bounded replay compares two fixed paths under an ideal local-departure event clock.',
                               'The forecast waiting witness uses nominal running times and fitted departure survival. '
                               'The schedule-only waiting witness uses published future departure and arrival clocks.',
                               'Missing endpoints censor evaluation. No observed door-close, passenger reaction, or cancellation publication time.',
                               'Repeated windows within a date are dependent. This is developmental mechanism evidence.'],
                  summary=summarize(rows),by_day=by_day,rows=rows,
                  reordering_sensitivity=dict(
                      scope='After the primary fixed OD screen, broaden the destination set using public direct-trip paths only. '
                            'At the same platform/time boundaries, admit late-running scheduled-early offers and either schedule order. '
                            'One prospectively ranked pair per reachable destination and boundary. Cases are dependent development diagnostics.',
                      earlier_schedule_seconds=EARLY_SENSITIVITY,
                      counts=dict(sum((Counter(row['reordering_counts']) for row in by_day.values()),Counter())),
                      summary=summarize(sensitivity)['direct'],
                      exact_vs_fitted_nominal=summarize_shadow(sensitivity),
                      unique_changed_events=unique_changed_events(sensitivity),rows=sensitivity))
    (HERE/'natural_screen.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
