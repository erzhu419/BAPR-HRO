"""Frozen full-menu controls on independently selected archive dates."""

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from experiments.reconstruction_v24.run import options_from, ranks_from
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.evolving_trip_belief import EvolvingTripBelief
from src.full_menu_opportunity import FullMenuOpportunity
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
METHODS = ('full_first_offer', 'original_full_nominal', 'empirical_full_nominal')


def departure(row):
    value = row['departure']
    if value['status'] == 'REAL':
        return value['time_seconds'], False
    if value['status'] == 'CANCELED':
        return row['scheduled_departure_seconds'], True
    return None


def physical_path(metadata, path, ready, deadline):
    """Read only the chosen path's actual outcome; retain partial endpoints."""
    now, vehicle = ready, None
    for cid in path:
        row = metadata[cid]
        event = departure(row)
        if event is None:
            return None, 'unresolved_departure'
        clock, canceled = event
        if canceled or clock < now or clock >= deadline:
            return deadline, 'deadline'
        dep, arr = row['departure'], row['arrival']
        if arr['status'] != 'REAL':
            return None, 'unresolved_arrival'
        identity = dep['official_journey']
        if identity is None or identity != arr['official_journey'] or (vehicle is not None and identity != vehicle):
            return None, 'ambiguous_vehicle'
        vehicle = identity
        now = arr['time_seconds']
        if now < clock:
            return None, 'inconsistent_clock'
        if now > deadline:
            return deadline, 'deadline'
    return now, 'arrived'


def first_tail(graph, metadata, option, ready, deadline):
    events = {cid: departure(metadata[cid]) for cid in option.tails}
    if any(event is None for event in events.values()):
        return None, 'unresolved_tail_departure'
    offers = [(clock, graph.connections[path[-1]].arr_time-graph.connections[cid].dep_time, cid)
              for cid, path in option.tails.items() for clock, canceled in (events[cid],)
              if not canceled and ready <= clock < deadline]
    return (min(offers)[2], 'offer') if offers else (None, 'no_tail_offer')


def route_outcome(graph, metadata, option, clock, deadline):
    arrival, status = physical_path(metadata, option.first_path, clock, deadline)
    if status != 'arrived' or not option.tails:
        return arrival, status, None
    stop = graph.connections[option.first_path[-1]].arr_stop
    ready = arrival+graph.stops[stop].min_transfer_time
    tail, status = first_tail(graph, metadata, option, ready, deadline)
    if tail is None:
        return (None if status.startswith('unresolved') else deadline), status, None
    arrival, status = physical_path(metadata, option.tails[tail], ready, deadline)
    return arrival, status, tail


def public_offers(metadata, cids, entry, ready, deadline):
    events = {cid: departure(metadata[cid]) for cid in cids}
    if any(event is None for event in events.values()):
        return None
    clocks = sorted({clock for clock, canceled in events.values() if not canceled and ready <= clock < deadline})
    return [dict(clock=clock,
                 visible=tuple(cid for cid, (time, canceled) in events.items() if not canceled and time == clock),
                 seen=tuple(cid for cid, (time, canceled) in events.items() if not canceled and entry <= time <= clock))
            for clock in clocks]


def execute(graph, metadata, root, options, ranks, offers, controller=None):
    """Expose successive public offers only; a refusal never locks a future trip."""
    deadline, started = root['deadline'], perf_counter()
    trace, online, selected, tail = [], 0., None, None
    if not options:
        arrival, status = deadline, 'no_menu'
    elif offers is None:
        arrival, status = None, 'unresolved_root_departure'
    else:
        arrival, status = deadline, 'no_offer'
        for public in offers:
            clock, visible, seen = public['clock'], public['visible'], public['seen']
            tick = perf_counter()
            if controller is None:
                chosen, receipt = min(visible, key=ranks.__getitem__), {}
            else:
                chosen, receipt = controller.choose(visible, seen, clock)
            elapsed = perf_counter()-tick
            online += elapsed
            trace.append(dict(**public, choice=chosen, receipt=receipt, wall=elapsed))
            if chosen is not None:
                selected = chosen
                arrival, status, tail = route_outcome(graph, metadata, options[chosen], clock, deadline)
                break
        if trace and selected is None:
            status = 'declined_all_offers'
    return dict(action=selected, tail=tail, loss=None if arrival is None else arrival-root['start'],
                arrived=status == 'arrived', status=status, trace=trace,
                online_wall=online, physical_replay_wall=perf_counter()-started-online)


def old_wait_attribution(metadata, options, outcome, deadline):
    """Retrospective diagnosis only, never an input to execute or choose."""
    empty = 0
    for event in outcome['trace']:
        if event['choice'] is not None:
            continue
        unseen = set(options).difference(event['seen'])
        future = [cid for cid in unseen if departure(metadata[cid]) is not None
                  and not departure(metadata[cid])[1]
                  and event['clock'] < departure(metadata[cid])[0] < deadline]
        empty += not future
    return dict(refusals=sum(e['choice'] is None for e in outcome['trace']),
                refusals_with_no_actual_future_menu_offer=empty)


def run_root(graph, metadata, models, params, root):
    options = options_from(root['full_options'])
    ranks = ranks_from(graph, root['full_options'])
    entry, status = (root['start'], 'arrived')
    if root['prefix']:
        entry, status = physical_path(metadata, root['prefix'], root['start']+30, root['deadline'])
    if status != 'arrived':
        loss = None if entry is None else root['deadline']-root['start']
        return dict(root_id=root['root_id'], day=root['day'], root_kind=root['root_kind'],
            outcomes={m: dict(action=None, loss=loss, arrived=False, status='prefix_'+status,
                             trace=[], online_wall=0., physical_replay_wall=0.) for m in METHODS},
            compile_wall={m: 0. for m in METHODS}, menu_size=len(options))
    ready = entry+(max(30, graph.stops[root['stop']].min_transfer_time) if root['prefix'] else 30)
    offers = public_offers(metadata, tuple(options), entry, ready, root['deadline'])
    outcomes, compilation = {}, {}
    for method in METHODS:
        tick = perf_counter()
        if method == 'full_first_offer' or not options:
            controller = None
        else:
            compiler = TransferOpportunityScores(graph, models[method], params, root['deadline'], options)
            controller = FullMenuOpportunity(compiler, ranks, entry, ready, root['deadline'])
        compilation[method] = perf_counter()-tick
        outcomes[method] = execute(graph, metadata, root, options, ranks, offers, controller)
        outcomes[method].update(old_wait_attribution(metadata, options, outcomes[method], root['deadline']))
    return dict(root_id=root['root_id'], day=root['day'], root_kind=root['root_kind'], entry=entry, ready=ready,
                outcomes=outcomes, compile_wall=compilation, menu_size=len(options))


def summarize(rows):
    common = [r for r in rows if all(r['outcomes'][m]['loss'] is not None for m in METHODS)]
    means = {m: mean(r['outcomes'][m]['loss'] for r in common) if common else None for m in METHODS}
    comparisons = {}
    for left, right in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
        paired = [r for r in rows if all(r['outcomes'][m]['loss'] is not None for m in (left, right))]
        delta = [r['outcomes'][left]['loss']-r['outcomes'][right]['loss'] for r in paired]
        comparisons[f'{right}_vs_{left}'] = dict(pairs=len(paired),
            mean_gain=mean(delta) if delta else None, wins=sum(d > 0 for d in delta), harms=sum(d < 0 for d in delta),
            changed=sum(r['outcomes'][left]['action'] != r['outcomes'][right]['action'] for r in paired))
    return dict(roots=len(rows), common_complete=len(common), mean_loss=means,
        arrivals={m: sum(r['outcomes'][m]['arrived'] for r in common) for m in METHODS},
        statuses={m: dict(Counter(r['outcomes'][m]['status'] for r in rows)) for m in METHODS},
        comparisons=comparisons,
        refusals={m: sum(r['outcomes'][m].get('refusals', 0) for r in rows) for m in METHODS},
        no_future_refusals={m: sum(r['outcomes'][m].get('refusals_with_no_actual_future_menu_offer', 0)
                                  for r in rows) for m in METHODS},
        compilation_wall={m: sum(r['compile_wall'][m] for r in rows) for m in METHODS},
        online_wall={m: sum(r['outcomes'][m]['online_wall'] for r in rows) for m in METHODS},
        physical_replay_wall={m: sum(r['outcomes'][m]['physical_replay_wall'] for r in rows) for m in METHODS})


def main():
    started = perf_counter()
    selection = json.loads((HERE/'selection.json').read_text())
    fit = json.loads((HERE/'model_fit.json').read_text())
    data, _ = load_city(HERE/'city_transfer_endpoints.json.gz')
    loading = perf_counter()-started
    results, day_preparation, duration_support = [], {}, {}
    for day, group in selection['by_day'].items():
        tick = perf_counter()
        graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        with gzip.open(HERE/f'city_schedule_{day}.json.gz', 'rt') as handle:
            schedule = json.load(handle)
        static = [row for row in schedule['connections'] if row['dropoff_allowed']]
        if len(metadata) != len(static) or any(row != {k: metadata[cid][k] for k in row}
                                             for cid, row in enumerate(static)):
            raise ValueError('Actual source changed the frozen static connection identities')
        params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        models = dict(original_full_nominal=EvolvingTripBelief(graph, public_occurrences=False,
                          grid_step_seconds=20, **fit['increments']),
                      empirical_full_nominal=EmpiricalDurationBelief(graph, duration_fit=fit['duration_fit'],
                          public_occurrences=False, grid_step_seconds=20, **fit['increments']))
        day_preparation[day] = perf_counter()-tick
        for root in group['roots']:
            result = run_root(graph, metadata, models, params, root)
            results.append(result)
            print(root['root_id'], {m: (r['loss'], r['status']) for m, r in result['outcomes'].items()}, flush=True)
        repaired = models['empirical_full_nominal']
        duration_support[day] = dict(used_groups=sorted(repaired.used_duration_groups),
            pooled_groups=sorted(repaired.pooled_duration_groups),
            residual_groups=sorted(repaired.residual_duration_groups))
    summary = dict(loading_wall=loading, day_preparation_wall=day_preparation,
        duration_support=duration_support,
        by_type={kind: summarize([r for r in results if r['root_kind'] == kind]) for kind in ('origin', 'first_transfer')},
        by_day={day: {kind: summarize([r for r in results if r['day'] == day and r['root_kind'] == kind])
                      for kind in ('origin', 'first_transfer')} for day in selection['by_day']},
        total_wall=perf_counter()-started)
    with gzip.open(HERE/'results.json.gz', 'wt') as handle:
        json.dump(dict(complete=True, results=results, summary=summary), handle, separators=(',', ':'))
    (HERE/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
