"""Frozen, schedule-selected departure calibration on retrospective archive dates."""

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import same_trip_replay
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.evolving_trip_belief import EvolvingTripBelief, _TripMemo


LEVELS = (.50, .80, .90, .95)
LAWS = ('original20', 'route_empirical20')


@dataclass(frozen=True)
class DepartureForecast:
    clocks: np.ndarray
    operating_mass: np.ndarray
    cancel: float

    def moments(self):
        total = float(self.operating_mass.sum())
        if total <= 0:
            return None, {}
        conditional = self.operating_mass / total
        cumulative = np.cumsum(conditional)
        intervals = {}
        for level in LEVELS:
            indices = np.searchsorted(cumulative, [(1-level)/2, (1+level)/2])
            indices = np.minimum(indices, len(self.clocks)-1)
            intervals[str(level)] = [int(self.clocks[i]) for i in indices]
        return float(self.clocks @ conditional), intervals

    def future_probability(self, entry, clock, observed):
        """Condition only on locally observed departure or its nonoccurrence."""
        if observed:
            return 0.
        unseen = (self.clocks < entry) | (self.clocks > clock)
        denominator = float(self.cancel + self.operating_mass[unseen].sum())
        if denominator <= 0:
            return None
        return float(self.operating_mass[self.clocks > clock].sum()/denominator)


def endpoint_record(forecast, metadata, identity, **labels):
    endpoint = metadata['departure']
    status = endpoint['status']
    valid_identity = identity != 'ambiguous' and bool(endpoint.get('official_journey'))
    actual = int(endpoint['time_seconds']) if status == 'REAL' and valid_identity else None
    predicted, intervals = forecast.moments()
    service = (int(status == 'CANCELED')
               if status == 'CANCELED' or actual is not None else None)
    return dict(**labels, status=status, trip_identity=identity,
        actual_clock=actual, predicted_clock=predicted,
        error=None if actual is None or predicted is None else predicted-actual,
        predicted_cancel=forecast.cancel, actual_cancel=service,
        intervals={level: dict(lower=lo, upper=hi, width=hi-lo,
            covered=None if actual is None else lo <= actual <= hi,
            below=None if actual is None else actual < lo,
            above=None if actual is None else actual > hi)
            for level, (lo, hi) in intervals.items()})


def summarize_endpoints(rows):
    errors = [r['error'] for r in rows if r['error'] is not None]
    service = [r for r in rows if r['actual_cancel'] is not None]
    intervals = {}
    for level in map(str, LEVELS):
        values = [r['intervals'][level] for r in rows if level in r['intervals']
                  and r['intervals'][level]['covered'] is not None]
        intervals[level] = dict(n=len(values), **{
            key: mean(v[key] for v in values) if values else None
            for key in ('covered', 'width', 'below', 'above')})
    return dict(eligible=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
        identities=dict(Counter(r['trip_identity'] for r in rows)),
        measured_departures=sum(r['actual_clock'] is not None for r in rows),
        scored_departures=len(errors), mean_error=mean(errors) if errors else None,
        mean_absolute_error=mean(abs(x) for x in errors) if errors else None,
        intervals=intervals, service=dict(n=len(service),
            predicted_rate=mean(r['predicted_cancel'] for r in service) if service else None,
            actual_rate=mean(r['actual_cancel'] for r in service) if service else None,
            brier=mean((r['predicted_cancel']-r['actual_cancel'])**2 for r in service) if service else None))


def summarize_future(rows):
    def group(values):
        scored = [r for r in values if r['future_probability'] is not None]
        return dict(n=len(values), root_candidate_occurrences=sum(len(r.get('root_ids', [None])) for r in values),
            scored=len(scored),
            unsupported_history=len(values)-len(scored),
            predicted_rate=mean(r['future_probability'] for r in scored) if scored else None,
            actual_rate=mean(r['actual_future'] for r in scored) if scored else None,
            brier=mean((r['future_probability']-r['actual_future'])**2 for r in scored) if scored else None,
            actually_departed_before_entry=sum(r['actual_before_entry'] for r in scored))

    unseen = [r for r in rows if r['public_status'] == 'unseen']
    return dict(all=group(rows), unseen=group(unseen), unseen_probability_bins={
        f'{index/5:.1f}-{(index+1)/5:.1f}': group([r for r in unseen
            if r['future_probability'] is not None
            and min(4, int(r['future_probability']*5)) == index])
        for index in range(5)})


def selected_trips(belief, roots):
    cids = set()
    for root in roots:
        cids.update(root['prefix'])
        for option in root['full_options']:
            cids.update(option['first_path'])
            for path in option['tail_paths']:
                cids.update(path)
    return {belief._trip_by_cid[cid] for cid in cids}


def trip_identity(metadata, slots):
    identities = {endpoint.get('official_journey')
        for slot in slots[::2] for endpoint in (metadata[slot.cid]['departure'], metadata[slot.cid]['arrival'])
        if endpoint['status'] == 'REAL' and endpoint.get('official_journey')}
    return 'ambiguous' if len(identities) > 1 else 'identified' if identities else 'missing'


def root_context(graph, actual, metadata, root):
    entry, status = root['start'], 'arrived'
    if root['prefix']:
        entry, status = same_trip_replay(graph, actual.events, root['prefix'], root['start']+30,
                                        root['deadline'], metadata)
    if status != 'arrived':
        return dict(status='prefix_'+status)
    ready = entry+(max(30, graph.stops[root['stop']].min_transfer_time) if root['prefix'] else 30)
    cids = [o['first_cid'] for o in root['full_options']]
    if not cids:
        return dict(status='empty_menu')
    departures = {}
    for cid in cids:
        endpoint = metadata[cid]['departure']
        if endpoint['status'] == 'REAL' and endpoint.get('official_journey'):
            departures[cid] = int(endpoint['time_seconds']), False
        elif endpoint['status'] == 'CANCELED':
            departures[cid] = None, True
        else:
            return dict(status='unresolved_menu_departure')
    offers = [clock for clock, canceled in departures.values()
              if not canceled and ready <= clock < root['deadline']]
    if not offers:
        return dict(status='no_legal_offer')
    return dict(status='offer', entry=entry, ready=ready, clock=min(offers), departures=departures)


def evaluate_day(day, selected, data, fit):
    started = perf_counter()
    graph, actual, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
              for c in graph.connections}
    roots = selected['roots']
    base = EvolvingTripBelief(graph, minute=60, public_occurrences=False,
                             grid_step_seconds=20, **fit['increments'])
    trips = selected_trips(base, roots)
    identities = {trip: trip_identity(metadata, base._trips[trip]) for trip in trips}
    menu_cids = {o['first_cid'] for r in roots for o in r['full_options']}
    endpoint_rows, forecasts, wall = [], {}, {}
    for name in LAWS:
        tick = perf_counter()
        belief = (base if name == 'original20' else EmpiricalDurationBelief(
            graph, duration_fit=fit['duration_fit'], minute=60, public_occurrences=False,
            grid_step_seconds=20, **fit['increments']))
        forecasts[name] = {}
        for trip in sorted(trips, key=str):
            law = belief._law(trip, params)
            for slot_index, slot in enumerate(belief._trips[trip]):
                if slot.kind != 'departure':
                    continue
                mass = law.marginal(slot_index)
                operating = ~law.canceled[slot_index]
                forecast = DepartureForecast(law.times[slot_index][operating], mass[operating],
                                              float(mass[~operating].sum()))
                if slot.cid in menu_cids:
                    forecasts[name][slot.cid] = forecast
                phase = slot_index//10
                endpoint_rows.append(endpoint_record(forecast, metadata[slot.cid], identities[trip],
                    day=day, law=name, cid=slot.cid, trip=str(trip), route=graph.connections[slot.cid].route,
                    scheduled_clock=slot.scheduled, departure_position=slot_index//2,
                    phase=f'{phase*5}-{phase*5+4}'))
            # Keep just small first-departure marginals, not dense full-trip kernels.
            belief._cache[trip] = _TripMemo()
            belief._transition_cache.clear()
            if name == 'route_empirical20':
                belief._empirical_transition_cache.clear()
        wall[name] = perf_counter()-tick
    root_receipts, records = [], {}
    for root in roots:
        ambiguous = any(identities[base._trip_by_cid[o['first_cid']]] == 'ambiguous'
                        for o in root['full_options'])
        context = (dict(status='ambiguous_menu_vehicle') if ambiguous else
                   root_context(graph, actual, metadata, root))
        root_receipts.append(dict(day=day, root_id=root['root_id'], root_kind=root['root_kind'],
                                  status=context['status']))
        if context['status'] != 'offer':
            continue
        entry, ready, clock = (context[key] for key in ('entry', 'ready', 'clock'))
        for cid, (departure, canceled) in context['departures'].items():
            trip = base._trip_by_cid[cid]
            if identities[trip] == 'ambiguous':
                continue
            observed = not canceled and entry <= departure <= clock
            public_status = 'visible' if observed and departure == clock else 'observed_departed' if observed else 'unseen'
            for name in LAWS:
                key = name, root['root_kind'], root['stop'], entry, ready, clock, cid
                if key in records:
                    records[key]['root_ids'].append(root['root_id'])
                    continue
                probability = forecasts[name][cid].future_probability(entry, clock, observed)
                records[key] = dict(day=day, law=name, root_kind=root['root_kind'],
                    root_ids=[root['root_id']], cid=cid, trip=str(trip), route=graph.connections[cid].route,
                    stop=root['stop'], entry=entry, ready=ready, clock=clock, public_status=public_status,
                    future_probability=probability, actual_future=not canceled and departure > clock,
                    actual_before_entry=not canceled and departure < entry)
    future_rows = list(records.values())
    summary = dict(selected_trips=len(trips), selected_routes=len({graph.connections[base._trips[t][0].cid].route for t in trips}),
        root_coverage={kind: dict(Counter(r['status'] for r in root_receipts if r['root_kind'] == kind))
                       for kind in sorted({r['root_kind'] for r in roots})},
        endpoint_calibration={name: dict(all=summarize_endpoints([r for r in endpoint_rows if r['law'] == name]),
            by_phase={phase: summarize_endpoints([r for r in endpoint_rows if r['law'] == name and r['phase'] == phase])
                      for phase in sorted({r['phase'] for r in endpoint_rows}, key=lambda p: int(p.split('-')[0]))},
            by_route={route: summarize_endpoints([r for r in endpoint_rows if r['law'] == name and r['route'] == route])
                      for route in sorted({r['route'] for r in endpoint_rows})}) for name in LAWS},
        future_calibration={name: {kind: summarize_future([r for r in future_rows if r['law'] == name and r['root_kind'] == kind])
                            for kind in sorted({r['root_kind'] for r in roots})} for name in LAWS},
        wall=wall, total_wall=perf_counter()-started)
    return summary, endpoint_rows, future_rows, root_receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection, fit = (json.loads(path.read_text()) for path in (args.selection, args.fit))
    data, _ = load_city(args.source)
    summaries, endpoints, futures, contexts = {}, [], [], []
    for day, selected in selection['by_day'].items():
        summary, rows, future, roots = evaluate_day(day, selected, data, fit)
        summaries[day] = summary
        endpoints.extend(rows)
        futures.extend(future)
        contexts.extend(roots)
        print(json.dumps(dict(day=day, trips=summary['selected_trips'], wall=summary['wall'])), flush=True)
    records_path = args.output.with_name(args.output.stem+'_records.json.gz')
    with gzip.open(records_path, 'wt') as handle:
        json.dump(dict(endpoints=endpoints, future_histories=futures, root_contexts=contexts), handle, separators=(',', ':'))
    output = dict(scope='Frozen-model retrospective dates; schedule-selected endpoints, not an independent-endpoint confidence experiment.',
        source=str(args.source), selection=str(args.selection), fit=str(args.fit), grid_seconds=20,
        central_interval_levels=LEVELS, future_bins=[i/5 for i in range(6)],
        records=str(records_path), by_day=summaries)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
