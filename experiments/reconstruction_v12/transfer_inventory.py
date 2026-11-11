"""Inventory three frozen transfer roots from paid receipts; no policy searches."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v9.search_reference import ARCHIVE, HISTORIES
from src.committed_transit import build_committed_model
from src.risk_bounds import CommitmentBounds
from src.simulator import LazyTripScenario


CASES = (('2023-07-26', 0, 1), ('2023-07-26', 0, 2), ('2023-07-27', 0, 1))
V9 = ROOT / 'experiments/reconstruction_v9/witness_reference_results.json'
V10 = ROOT / 'experiments/reconstruction_v10/tail_rescue_results.json'
V11 = ROOT / 'experiments/reconstruction_v11/sequential_platform_results.json'


def key(row):
    return row['date'], row['query'], row['decision']


def read(path):
    return json.loads(path.read_text())


def state_spec(state):
    return dict(stop=state.stop, clock=state.time, current_trip=state.current_trip,
        last_arrival=state.last_arrival, unavailable_ids=sorted(state.unavailable_ids),
        learned_trips=sorted(state.learned_trips), initialized=state.initialized)


def connection(graph, cid):
    c = graph.connections[cid]
    return dict(cid=cid, trip=c.trip_id, route=c.route, dep_stop=c.dep_stop,
        dep_name=graph.stops[c.dep_stop].name, arr_stop=c.arr_stop,
        arr_name=graph.stops[c.arr_stop].name, scheduled_departure=c.dep_time,
        scheduled_arrival=c.arr_time, pickup_allowed=c.pickup_allowed)


def menu(graph, state, goal, deadline):
    # Only actions() is called. It reads public timetable/state, never this empty scene.
    model = build_committed_model(graph, [(1., LazyTripScenario({}, minute=60, publications={}))],
        state.stop, goal, state.time, deadline, initial_state=state, lookahead=1500)
    return [list(action) for action in model.actions(state)]


def actual_leg(graph, actual, cid):
    c, event = graph.connections[cid], actual.events.get(cid)
    return dict(cid=cid, canceled=None if event is None else event.canceled,
        departure=None if event is None else event.departure,
        arrival=None if event is None else event.arrival,
        dep_stop=c.dep_stop, arr_stop=c.arr_stop)


def boundary(graph, actual, belief, parameters, state, path, goal, deadline, prefix_length, rides):
    """Condition only on already-recorded incumbent rides, never scan future outcomes."""
    updated = belief.copy()
    reached = state
    prefix = path[:prefix_length]
    for cid, receipt in zip(prefix, rides):
        assert cid == receipt['ridden_connection'], 'Boundary prefix must have an existing ride receipt'
        c = graph.connections[cid]
        updated.observe_connection_event(cid, 'departure', receipt['time'] - c.dep_time)
        updated.observe_connection_event(cid, 'arrival', receipt['next_time'] - c.arr_time)
        reached = replace(reached, stop=receipt['next_stop'], time=receipt['next_time'],
            last_arrival=receipt['next_time'], current_trip=c.trip_id,
            unavailable_ids=reached.unavailable_ids | {cid},
            learned_trips=reached.learned_trips | {c.trip_id})
    assert len(rides) >= prefix_length
    target = path[prefix_length] if prefix_length < len(path) else None
    result = dict(state=state_spec(reached), prefix=list(prefix),
        source='Existing V11 incumbent_references.rides; this is an observed replay branch, not a root input',
        candidate_actions=menu(graph, reached, goal, deadline), next_incumbent_cid=target)
    if target is not None:
        risk = CommitmentBounds(updated, parameters, graph, goal, deadline).commit(reached, target)
        event = actual.events.get(target)
        result.update(next_incumbent_risk=asdict(risk),
            next_incumbent_actual=actual_leg(graph, actual, target),
            actual_departure_minus_transfer_ready=(None if event is None or event.canceled
                                                   else event.departure - risk.ready))
    return result


def main():
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    histories = {key(r): r for r in read(HISTORIES)['rows']}
    journeys = {(r['date'], r['query']): r for r in read(ARCHIVE)['rows']
                if r['method'] == 'evolving_rollout'}
    v9, v10, v11 = {key(r): r for r in read(V9)['rows']}, {key(r): r for r in read(V10)['rows']}, read(V11)
    arms = {(*key(r), r['method']): r for r in v11['rows']}
    references = {key(r): r for r in v11['incumbent_references']}
    graphs, rows = {}, []
    for identity in CASES:
        day, query, decision = identity
        if day not in graphs:
            graphs[day] = dated_graph(data, day, 'gtfs_station_max', True)
        graph, actual, stops, _, metadata, _ = graphs[day]
        history, journey = histories[identity], journeys[day, query]
        old, tail = v9[identity], v10[identity]
        off = arms[identity + ('off64',)]
        checked, reference = arms[identity + ('middle64',)], references[identity]
        belief, state = context(graph, metadata, journey, decision, fit)
        goal, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
        path = tuple(history['witness'])
        assert (state.stop, state.time, goal) == (history['stop'], history['clock'], history['destination'])
        assert list(path) == off['incumbent_path'] == old['same_primary']['primary']
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                      for c in graph.connections}
        bounds = CommitmentBounds(belief, parameters, graph, goal, deadline)
        root_actions = menu(graph, state, goal, deadline)
        alternative = old['uct128_alternative']
        candidates = sorted({path[0], *([alternative['connection']] if alternative else []),
                             *([off['work']['proposed_first']] if 'proposed_first' in off['work'] else [])})
        root_risks = {str(cid): asdict(bounds.commit(state, cid)) for cid in candidates}
        transfer_index = next(i for i in range(1, len(path))
                              if graph.connections[path[i-1]].trip_id != graph.connections[path[i]].trip_id)
        next_boundary = boundary(graph, actual, belief, parameters, state, path, goal, deadline,
                                 1, reference['rides'])
        transfer_boundary = (next_boundary if transfer_index == 1 else boundary(
            graph, actual, belief, parameters, state, path, goal, deadline,
            transfer_index, reference['rides']))
        gate = checked['work']['gate']
        paths = dict(incumbent=list(path), v9_root_failure_rescue=old['same_primary']['rescue'],
            v9_alternative=alternative, v10_fixed_tail_rescue=tail['fixed_primary_tail_rescue']['plan'],
            v10_alternative_tail_rescue=[r['plan'] for r in tail['alternative_tail_rescue']],
            v11_replacement=off['replacement_path'])
        all_cids = {a[1] for actions in (root_actions, next_boundary['candidate_actions'],
                    transfer_boundary['candidate_actions']) for a in actions if a[0] == 'select'}
        for candidate_path in (path, old['same_primary']['rescue'],
                alternative['primary'] if alternative else (), alternative['rescue'] if alternative else (),
                tail['fixed_primary_tail_rescue']['plan']['rescue'], off['replacement_path']):
            all_cids.update(candidate_path)
        upper = dict(incumbent=old['incumbent_upper_seconds'], v9_best=old['best_upper_seconds'],
            v10_fixed_tail_rescue=tail['fixed_primary_tail_rescue']['upper'], v10_best=tail['upper'])
        rows.append(dict(date=day, query=query, decision=decision,
            root=dict(**state_spec(state), destination=goal, destination_name=graph.stops[goal].name,
                deadline=deadline, incoming_cid=journey['decisions'][decision-1]['action'],
                archive_prefix_decisions=journey['decisions'][:decision]),
            paths=paths, root_candidate_actions=root_actions, root_commitment_risks=root_risks,
            connections={str(cid): connection(graph, cid) for cid in sorted(all_cids)},
            incumbent_actual_path=[actual_leg(graph, actual, cid) for cid in path],
            next_action_boundary=next_boundary, next_transfer_boundary=transfer_boundary,
            next_transfer_index=transfer_index, upper_seconds=upper,
            incumbent_completion_probability=old['same_primary']['primary_completion_probability'],
            fixed_tail_rescue_failure_probability=tail['fixed_primary_tail_rescue']['failure_probability'],
            fixed_tail_rescue_gain_seconds=tail['fixed_primary_tail_rescue']['rescue_gain'],
            recorded_gate=dict(lower=gate['lower'], oracle_mean=gate['oracle_mean'], worlds=gate['worlds'],
                alpha=gate['alpha'], accepted=gate['accepted'],
                residuals_to_point={name: value-gate['oracle_mean'] for name,value in upper.items()},
                residuals_to_lower={name: value-gate['lower'] for name,value in upper.items()}),
            recorded_worker=dict(first_action=['wait'] if off['deferred'] else ['select',off['work']['proposed_first']],
                retained=off['retained'], published=off['published'], status=off['status'],
                realized_loss=off['loss_upper'], marginal_loss=off['marginal_loss_upper']),
            recorded_incumbent=reference))
    result = dict(protocol=dict(cases=[list(k) for k in CASES],
        source=str(SOURCE.relative_to(ROOT)), archive=str(ARCHIVE.relative_to(ROOT)),
        histories=str(HISTORIES.relative_to(ROOT)),
        receipts=[str(p.relative_to(ROOT)) for p in (V9,V10,V11)],
        computation='Local public graph/context, analytic commitment risk, and action inventory only; no search, sampling, or performance replay.',
        menu='Full common committed public menu, lookahead 1500 seconds, including WAIT; no outcome-based filtering.',
        boundary='Both next action return and first downstream transfer, conditioned only on existing incumbent ride receipts.',
        interpretation='U minus clairvoyant point/lower is certificate slack; it is not true causal regret. Improved witness gains are established separately.'), rows=rows)
    write_json(HERE / 'transfer_inventory.json', result)
    for r in rows:
        b = r['next_transfer_boundary']
        print(key(r), 'U', r['upper_seconds'], 'boundary', b['state']['stop'], b['state']['clock'],
              'target', b['next_incumbent_cid'], 'actual transfer slack', b['actual_departure_minus_transfer_ready'],
              'root/menu', len(r['root_candidate_actions']), len(b['candidate_actions']), flush=True)


if __name__ == '__main__':
    main()
