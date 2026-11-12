"""Select transfer histories from schedules, then open and replay fresh endpoints."""

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import cache
import gzip
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import OD_NAMES, load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v6.run_city_new_od import PAIRS
from experiments.reconstruction_v22.run_natural_screen import (
    CAP, PUBLICATION, READY_TIMES, ROOT, downstream_catalog, fixed_arrival,
)
from src.evolving_trip_belief import EvolvingTripBelief
from src.offgrid_conditional import ActualClockScores
from src.platform_nominal_index import PlatformNominalIndex
from src.platform_transit import PlatformState
from src.public_rescue_candidates import PublicItinerary
from src.public_wait_recourse import compile_through_scores
from src.recourse_itinerary_router import build_trip_blocks, trip_key


HERE = Path(__file__).resolve().parent
TRAINING = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
DATES = ('2023-07-23', '2023-07-24')
STEPS = (60, 20, 10)
METHODS = ('first_offer', 'nominal', 'conditional')


def read_gz(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def model(graph, fit):
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    return belief, parameters


@dataclass(frozen=True)
class DepartureMarks:
    clocks: np.ndarray
    mass: np.ndarray
    cancel: float
    duration: int

    def saving_at(self, ready, deadline):
        eligible = (self.clocks >= np.asarray(ready)[..., None]) & (self.clocks < deadline)
        gain = np.maximum(deadline-self.clocks-self.duration, 0)
        return eligible @ (self.mass*gain), eligible @ self.mass

    def board(self, clock, deadline):
        return min(clock+self.duration, deadline)

    def wait(self, entry, clock, deadline, exact_rows=None):
        future = (self.clocks > clock) & (self.clocks < deadline)
        survival = self.mass[(self.clocks < entry) | (self.clocks > clock)].sum()+self.cancel
        if survival <= 0:
            return float(deadline)
        if exact_rows is None:
            gain = np.maximum(deadline-self.clocks[future]-self.duration, 0)
            saving = self.mass[future] @ gain
        else:
            saving = sum(row.probability*(deadline-row.expected_arrival_or_deadline)
                         for time, row in exact_rows.items() if clock < time < deadline)
        return float(deadline-saving/survival)


def departure_marks(belief, params):
    nominal = belief.mean_events(params)

    @cache
    def marks(path):
        clocks, mass, cancel = belief.event_distribution(path[0], 'departure', params)
        first, last = nominal[path[0]].departure, nominal[path[-1]].arrival
        duration = max(0, round(last)-round(first)) if first is not None and last is not None else 0
        return DepartureMarks(clocks, mass*(1-cancel), cancel, duration)

    return marks


def scheduled_input(schedule, transfer_bounds):
    # The standard dated builder keeps the same connection IDs at replay.
    # There are no measured fields in the static schedule input.
    connections = [dict(row, outcome='UNRESOLVED') for row in schedule['connections']]
    day = schedule['date']
    return dict(stops=schedule['stops'], dates={day: {'connections': connections}},
                station_transfer_bounds=transfer_bounds)


def screen_day(schedule, transfer_bounds, fit):
    day = schedule['date']
    graph, *_ = dated_graph(scheduled_input(schedule, transfer_bounds), day, 'gtfs_station_max', True)
    base, params = model(graph, fit)
    belief = base.with_grid_step(STEPS[0])
    marks = departure_marks(belief, params)
    blocks = build_trip_blocks(graph)
    by_name = {stop.name: stop.id for stop in graph.stops.values()}
    counts, selected, seen = Counter(), [], set()
    for origin_name, goal_name in (*OD_NAMES, *PAIRS):
        origin, original_goal = by_name[origin_name], by_name[goal_name]
        for start in READY_TIMES:
            deadline = start+CAP
            counts['od_windows'] += 1
            nominal = PlatformNominalIndex(graph, original_goal, start, deadline, trip_blocks=blocks)
            route = nominal.path(origin, start+PUBLICATION)
            if not route:
                counts['no_scheduled_path'] += 1
                continue
            first_trip = trip_key(graph.connections[route[0]])
            transfer = next((i for i, cid in enumerate(route) if trip_key(graph.connections[cid]) != first_trip), None)
            if transfer is None:
                counts['scheduled_without_transfer'] += 1
                continue
            counts['scheduled_transfer'] += 1
            prefix = route[:transfer]
            inbound = graph.connections[prefix[-1]]
            stop = inbound.arr_stop
            delay = max(graph.stops[stop].min_transfer_time, PUBLICATION)
            clocks, weights, _ = belief.event_distribution(inbound.id, 'arrival', params)
            if not weights.size:
                counts['no_model_incoming_support'] += 1
                continue
            support = weights > 0
            ready = clocks[support]+delay
            weight = weights[support]
            catalog = downstream_catalog(graph, blocks, stop, inbound.arr_time+delay, deadline)
            for goal, options in sorted(catalog.items()):
                if goal == stop:
                    continue
                candidates = [o for o in options if trip_key(graph.connections[o[2]]) != first_trip]
                pairs = [(a, b) for a, b in combinations(sorted(candidates, key=lambda o: o[2]), 2)
                         if trip_key(graph.connections[a[2]]) != trip_key(graph.connections[b[2]])]
                if not pairs:
                    continue
                counts['destination_windows_with_two_scheduled_trips'] += 1
                ordered = sorted(pairs, key=lambda p: (abs(sum(p[0][:2])-sum(p[1][:2])),
                                                       max(p[0][0], p[1][0]), p[0][2], p[1][2]))
                values = {o[2]: marks(o[-1]).saving_at(ready, deadline) for o in candidates}
                normal, winner = None, None
                for a, b in ordered:
                    ga, pa = values[a[2]]
                    gb, pb = values[b[2]]
                    if not np.any((pa > 0) & (pb > 0)):
                        continue
                    ranking = ga-gb
                    # Count an actual trade-off only on support where both
                    # proposed vehicles could still be legally boarded.
                    shared = (pa > 0) & (pb > 0)
                    flips = bool(np.any(ranking[shared] > 1e-9) and np.any(ranking[shared] < -1e-9))
                    record = (a, b, float(weight @ ranking))
                    if flips:
                        winner = record
                        break
                    if normal is None:
                        normal = record
                choice = winner or normal
                if choice is None:
                    counts['no_shared_legal_model_clock'] += 1
                    continue
                a, b, prior = choice
                key = (prefix, goal, a[2], b[2])
                if key in seen:
                    counts['duplicate_windows'] += 1
                    continue
                seen.add(key)
                cohort = 'rank_reversal' if winner else 'normal'
                counts[cohort] += 1
                counts['named_'+cohort] += int(goal == original_goal)
                selected.append(dict(day=day, start=start, deadline=deadline, origin=origin_name,
                                     named_goal=goal_name, primary=goal == original_goal,
                                     goal=goal, destination=graph.stops[goal].name, stop=stop,
                                     delay=delay, prefix=prefix, options=(a, b), cohort=cohort,
                                     prior_nominal_saving_difference=prior))
    return dict(counts=dict(counts), selected=selected)


def same_vehicle(metadata, path):
    identities = {metadata[cid]['departure']['official_journey'] for cid in path
                  if metadata[cid]['outcome'] == 'MEASURED'}
    return len(identities) <= 1


def replay_outcome(graph, actual, metadata, path, clock, deadline):
    if not same_vehicle(metadata, path):
        return None, 'ambiguous_vehicle'
    return fixed_arrival(graph, actual, path, clock, deadline)


def choose_nominal(visible, marks, entry, clock, deadline, available_later):
    cid = min(visible, key=lambda c: (marks[c].board(clock, deadline), c))
    other = available_later
    if other is None:
        return cid, None
    gap = marks[other].wait(entry, clock, deadline)-marks[cid].board(clock, deadline)
    return (other if gap < 0 else cid), gap


def choose_conditional(graph, belief, params, row, root, visible, marks, options, clock, available_later):
    scores, _ = compile_through_scores(belief, params, graph, root, row['goal'], row['deadline'], {}, options)
    if set(scores.rows) != set(options):
        return None, None, 'unscored_path'
    observation = PlatformState(root.stop, clock, last_arrival=root.last_arrival,
                                used_connections=root.used_connections, visible_connections=visible,
                                public_log=tuple((clock, cid, 'departure') for cid in visible),
                                initialized=True, local_since=root.local_since)
    clock_scores = ActualClockScores(belief, params, scores, row['goal'])
    observed = {cid: clock_scores.at_visible(cid, observation, 'projection') for cid in visible}
    if any(score is None for score in observed.values()):
        return None, None, 'unscored_observed_clock'
    cid = min(visible, key=lambda c: (observed[c].expected_arrival_or_deadline, c))
    other = available_later
    if other is None:
        return cid, None, None
    wait = marks[other].wait(root.local_since, clock, row['deadline'], scores.rows[other])
    gap = wait-observed[cid].expected_arrival_or_deadline
    return (other if gap < 0 else cid), gap, None


def transfer_case(graph, actual, metadata, row):
    start, deadline, prefix = row['start'], row['deadline'], row['prefix']
    entry, status = replay_outcome(graph, actual, metadata, prefix, start+PUBLICATION, deadline)
    if status != 'arrived':
        outcomes = ({method: dict(action=None, loss=deadline-start, arrived=False) for method in METHODS}
                    if status == 'deadline' else {})
        return dict(status='prefix_'+status, outcomes=outcomes)
    ready = entry+row['delay']
    options = {o[2]: PublicItinerary(tuple(o[-1]), sum(o[:2])) for o in row['options']}
    durations = {o[2]: o[1] for o in row['options']}
    events = {cid: actual.events.get(cid) for cid in options}
    if any(event is None for event in events.values()):
        return dict(status='unresolved_departure', entry=entry)
    offers = {cid: event.departure for cid, event in events.items()
              if not event.canceled and ready <= event.departure < deadline}
    if not offers:
        return dict(status='no_legal_offer', entry=entry, ready=ready, actual_legal_options=0,
                    outcomes={method: dict(action=None, loss=deadline-start, arrived=False)
                              for method in METHODS})
    clock = min(offers.values())
    visible = tuple(sorted(cid for cid, time in offers.items() if time == clock))
    # A past local departure is public. A departure before entry, or future
    # absence, does not appear as a receipt at this decision.
    other = next((cid for cid, event in events.items() if cid not in visible
                  and not (not event.canceled and entry <= event.departure <= clock)), None)
    station_mct = graph.stops[row['stop']].min_transfer_time
    root = PlatformState(row['stop'], entry, last_arrival=ready-station_mct, used_connections=frozenset(prefix),
                         local_since=entry)
    return dict(status='offers', entry=entry, ready=ready, clock=clock, visible=visible, other=other,
                root=root, options=options, durations=durations, actual_legal_options=len(offers))


def receipt_ranking(row, ready, marks, belief, params):
    clocks, weights, _ = belief.event_distribution(row['prefix'][-1], 'arrival', params)
    a, b = (o[2] for o in row['options'])
    model_ready = clocks+row['delay']
    ga = marks[a].saving_at(model_ready, row['deadline'])[0]
    gb = marks[b].saving_at(model_ready, row['deadline'])[0]
    prior = float(weights @ (ga-gb))
    ga = float(marks[a].saving_at(ready, row['deadline'])[0])
    gb = float(marks[b].saving_at(ready, row['deadline'])[0])
    updated = ga-gb
    return dict(prior=a if prior >= 0 else b, updated=a if updated >= 0 else b,
                reversed=updated*prior < -1e-9)


def replay_case(graph, actual, metadata, belief, params, marks_for, row, case):
    if case['status'] != 'offers':
        return case
    entry, clock, visible, other, options = (case[name] for name in ('entry', 'clock', 'visible', 'other', 'options'))
    deadline = row['deadline']
    started = perf_counter()
    first = min(visible, key=lambda cid: (case['durations'][cid], cid))
    first_cost = perf_counter()-started
    started = perf_counter()
    marks = {cid: marks_for(option.path) for cid, option in options.items()}
    prep = perf_counter()-started
    started = perf_counter()
    nominal, nominal_gap = choose_nominal(visible, marks, entry, clock, deadline, other)
    nominal_cost = perf_counter()-started
    started = perf_counter()
    if len(visible) == 1 and other is None:
        conditional, conditional_gap, error = first, None, None
    else:
        conditional, conditional_gap, error = choose_conditional(
            graph, belief, params, row, case['root'], visible, marks, options, clock, other)
        if conditional is None:
            conditional = nominal
    conditional_cost = perf_counter()-started
    # Evaluate both fixed continuations after the choices. The unrealized
    # alternative is used only for retrospective attribution.
    observed = {}
    for cid, option in options.items():
        arrival, status = replay_outcome(graph, actual, metadata, option.path, case['ready'], deadline)
        observed[cid] = dict(action=cid, loss=None if arrival is None else arrival-row['start'],
                             arrived=status == 'arrived', status=status)
    outcomes = {method: observed[cid] for method, cid in zip(METHODS, (first, nominal, conditional), strict=True)}
    rank = receipt_ranking(row, case['ready'], marks, belief, params)
    a, b = options
    return dict(status='offers', entry=entry, ready=case['ready'], clock=clock, visible=visible,
                actual_legal_options=case['actual_legal_options'],
                receipt_reverses_rank=rank['reversed'], conditional_fallback=error,
                same_line=graph.connections[a].route == graph.connections[b].route,
                wait_minus_board=dict(nominal=nominal_gap, conditional=conditional_gap),
                alternative_outcomes=observed, committed_target=rank,
                outcomes=outcomes, wall_seconds=dict(first_offer=first_cost, shared_mark_prep=prep,
                                                      nominal=nominal_cost, conditional=conditional_cost))


def summary(rows):
    def comparison(rows, left, right):
        known = [r for r in rows if all(r.get('outcomes', {}).get(m, {}).get('loss') is not None
                                        for m in (left, right))]
        gain = [r['outcomes'][left]['loss']-r['outcomes'][right]['loss'] for r in known]
        return dict(pairs=len(known), mean_gain=mean(gain) if gain else None,
                    wins=sum(g > 0 for g in gain), harms=sum(g < 0 for g in gain),
                    changed=sum(r['outcomes'][left]['action'] != r['outcomes'][right]['action'] for r in known),
                    mean_loss={m: mean(r['outcomes'][m]['loss'] for r in known) if known else None
                               for m in (left, right)},
                    arrivals={m: sum(r['outcomes'][m]['arrived'] for r in known) for m in (left, right)})

    def group(rows):
        return dict(windows=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                    rank_reversals=sum(r.get('receipt_reverses_rank', False) for r in rows),
                    conditional_fallbacks=dict(Counter(r['conditional_fallback'] for r in rows
                                                       if r.get('conditional_fallback'))),
                    comparisons={f'{right}_vs_{left}': comparison(rows, left, right)
                                 for left, right in (('first_offer', 'nominal'), ('first_offer', 'conditional'),
                                                     ('nominal', 'conditional'))},
                    wall_seconds={m: sum(r.get('wall_seconds', {}).get(m, 0.) for r in rows)
                                  for m in ('first_offer', 'shared_mark_prep', 'nominal', 'conditional')})

    return {cohort: {label: group([r['replay'] for r in rows if r['cohort'] == cohort and keep(r)])
                     for label, keep in (
                         ('all', lambda r: True), ('named_od', lambda r: r['primary']),
                         ('two_actual_options', lambda r: r['replay'].get('actual_legal_options') == 2),
                         ('receipt_reversed_two_options', lambda r: r['replay'].get('actual_legal_options') == 2
                          and r['replay'].get('receipt_reverses_rank', False)))}
            for cohort in ('rank_reversal', 'normal')}


def competition_attribution(rows):
    competing = [r for r in rows if r['replay'].get('actual_legal_options') == 2]
    reversed_rows = [r for r in competing if r['replay']['receipt_reverses_rank']]
    episodes = {(r['day'], r['prefix'][-1], r['stop'], r['replay']['clock'], tuple(o[2] for o in r['options']))
                for r in reversed_rows}
    all_complete = [r['replay'] for r in competing
                    if all(a['loss'] is not None for a in r['replay']['alternative_outcomes'].values())]
    overtakes = sum(min(a['loss'] for a in r['alternative_outcomes'].values())
                    < r['outcomes']['first_offer']['loss'] for r in all_complete)
    known = [r['replay'] for r in reversed_rows
             if all(a['loss'] is not None for a in r['replay']['alternative_outcomes'].values())]
    oracle_gains, target_gains, target_vs_first = [], [], []
    for row in known:
        option = row['alternative_outcomes']
        first = row['outcomes']['first_offer']['loss']
        oracle_gains.append(first-min(o['loss'] for o in option.values()))
        prior, updated = (option[row['committed_target'][phase]]['loss'] for phase in ('prior', 'updated'))
        target_gains.append(prior-updated)
        target_vs_first.append(first-updated)
    gaps = [r['replay']['wait_minus_board']['conditional'] for r in reversed_rows
            if r['replay']['wait_minus_board']['conditional'] is not None]
    return dict(two_offers=len(competing), two_offers_same_line=sum(r['replay']['same_line'] for r in competing),
                complete_two_offer_pairs=len(all_complete), overtakes_in_complete_pairs=overtakes,
                receipt_reversal_windows=len(reversed_rows), receipt_reversal_episodes=len(episodes),
                both_tails_known=len(known), hindsight_faster=sum(g > 0 for g in oracle_gains),
                hindsight_mean_gain=mean(oracle_gains) if oracle_gains else None,
                hindsight_max_gain=max(oracle_gains, default=None),
                updated_vs_prior_target_mean=mean(target_gains) if target_gains else None,
                updated_vs_first_mean=mean(target_vs_first) if target_vs_first else None,
                conditional_wait_comparisons=len(gaps),
                min_conditional_wait_minus_board=min(gaps, default=None),
                mean_conditional_wait_minus_board=mean(gaps) if gaps else None)


def run_replay(schedules, source, selection, fit):
    data, _ = load_city(source)
    output = {}
    for day in DATES:
        graph, actual, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        planned = [row for row in schedules[day]['connections'] if row['dropoff_allowed']]
        if (len(metadata) != len(planned)
                or any(schedule != {k: metadata[cid][k] for k in schedule}
                       for cid, schedule in enumerate(planned))):
            raise ValueError(f'Joined data do not preserve the {day} schedule')
        base, params = model(graph, fit)
        candidates = selection[day]['selected']
        trips = {trip_key(graph.connections[o[2]]) for row in candidates for o in row['options']}
        trips.update(trip_key(graph.connections[row['prefix'][-1]]) for row in candidates)
        cases = [transfer_case(graph, actual, metadata, row) for row in candidates]
        output[day] = {}
        for step in STEPS:
            started = perf_counter()
            belief = base.with_grid_step(step, trips=trips)
            marks = departure_marks(belief, params)
            rows = [dict(**row, replay=replay_case(graph, actual, metadata, belief, params, marks, row, case))
                    for row, case in zip(candidates, cases, strict=True)]
            output[day][str(step)] = dict(summary=summary(rows), attribution=competition_attribution(rows),
                                          rows=rows, wall_seconds=perf_counter()-started)
            print(day, step, {k: v['all'] for k, v in output[day][str(step)]['summary'].items()}, flush=True)
    combined, attribution = {}, {}
    for step in map(str, STEPS):
        rows = [r for day in DATES for r in output[day][step]['rows']]
        combined[step] = summary(rows)
        attribution[step] = competition_attribution(rows)
    with gzip.open(HERE / 'transfer_results.json.gz', 'wt') as handle:
        json.dump(dict(by_day=output, combined=combined, attribution=attribution),
                  handle, ensure_ascii=False, separators=(',', ':'))
    by_day = {day: {step: {k: v for k, v in run.items() if k != 'rows'}
                   for step, run in grids.items()} for day, grids in output.items()}
    write_json('transfer_summary.json', dict(combined=combined, attribution=attribution, by_day=by_day))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'replay'))
    parser.add_argument('schedules', nargs=2, type=Path)
    parser.add_argument('--endpoints', type=Path, default=HERE/'city_transfer_endpoints.json.gz')
    args = parser.parse_args()
    schedules = {schedule['date']: schedule for schedule in map(read_gz, args.schedules)}
    if set(schedules) != set(DATES):
        raise ValueError('Expected both dates in protocol.json')
    training, _ = load_city(TRAINING)
    fit = fit_training(training, retain_no_pickup=True)
    if args.stage == 'screen':
        selection = {day: screen_day(schedules[day], training['station_transfer_bounds'], fit) for day in DATES}
        write_json('transfer_selection.json', selection)
        print({day: result['counts'] for day, result in selection.items()}, flush=True)
    else:
        selection = json.loads((HERE/'transfer_selection.json').read_text())
        run_replay(schedules, args.endpoints, selection, fit)


if __name__ == '__main__':
    main()
