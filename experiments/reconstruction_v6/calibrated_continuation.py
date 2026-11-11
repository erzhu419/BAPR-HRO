"""Matched, already-exposed rail contexts: common calibrated law to policy loss."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v5.calibration_run import data_inputs
from experiments.reconstruction_v6.itinerary_mc_evaluator import FixedWitness, execute_fixed_witness
from src.committed_transit import CommittedState, WAIT, build_committed_model
from src.evolving_trip_belief import EvolvingTripBelief
from src.recourse_itinerary_router import RecourseItineraryRouter, nominal_itinerary
from src.simulator import LazyTripScenario
from src.trip_marginal_tilt import TripMarginalTilt


class FrozenParameterRouter(RecourseItineraryRouter):
    def _parameters(self):
        return self.priors


def read(path):
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
        return json.load(handle)


def main():
    fit = read(ROOT / 'experiments/reconstruction_v5/calibration_frozen_fit.json')['evolving_fit']
    development = read(HERE / 'calibration_development_results.json.gz')['rows_by_date']
    with open(HERE / 'calibration_frozen_model.pkl', 'rb') as handle:
        calibration = pickle.load(handle)
    data = data_inputs()
    protocol = dict(scope='Posthoc matched diagnostic on already-exposed development dates; no independent confirmation.',
        dates=list(development), states='First retained actual-current incoming state in each observable delay group.',
        goal='First original OD destination reachable in the original posterior nominal graph; no future outcome used.',
        projection='Union of retained outgoing candidates; earliest timetable anchor per trip; one common law for all actions.',
        policy='Same max8 exact itinerary/first-failure-rescue controller, original versus common projected Student law.',
        execution='Fixed issued policy, root-frozen thresholds, original seconds/MCT/patience contract; logical clock.',
        parameters=fit, sources={name: (ROOT / name).read_text() for name in (
            'src/trip_marginal_tilt.py', 'src/risk_bounds.py', 'src/recourse_itinerary_router.py',
            'experiments/reconstruction_v6/calibrated_continuation.py')})
    with gzip.open(HERE / 'calibrated_continuation_protocol.json.gz', 'wt') as handle:
        json.dump(protocol, handle)
    records = []
    for day, output in development.items():
        graph, scene, stops, mapping, source, _ = dated_graph(data, day, 'gtfs_station_max')
        by_id = {row['id']: row for row in data['dates'][day]['connections']}
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                      for c in graph.connections}
        menus = defaultdict(dict)
        for menu in ('fixed_schedule', 'actual_current'):
            for row in output[menu]['rows']:
                menus[row['incoming']][row['outgoing']] = row
        selected = {}
        for row in output['actual_current']['rows']:
            selected.setdefault('late' if row['incoming_delay'] > 180 else 'ordinary', row)
        for group, row in selected.items():
            incoming = graph.connections[mapping[row['incoming']]]
            state = CommittedState(incoming.arr_stop, row['clock'], incoming.trip_id, row['clock'],
                unavailable_ids=frozenset({incoming.id}), initialized=True)
            deadline = state.time + 3600
            base = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
            base.observe_connection_event(incoming.id, 'arrival', state.time - incoming.arr_time)
            anchors = {}
            for outgoing in sorted(menus[row['incoming']], key=lambda cid:
                                   (by_id[cid]['scheduled_departure_seconds'], cid)):
                cid = mapping[outgoing]
                anchors.setdefault(graph.connections[cid].trip_id, outgoing)
            targets = {mapping[cid]: calibration.predict(by_id[row['incoming']], by_id[cid], state.time)
                       for cid in anchors.values()}
            common = TripMarginalTilt(base, parameters, targets)
            nominal = base.mean_events(parameters)
            destination = None
            for query in data['od_queries']:
                goal = stops[query['destination']]
                if goal != state.stop and nominal_itinerary(graph, state, goal, nominal,
                        set(mapping.values()), deadline, 1500):
                    destination = goal
                    break
            if destination is None:
                records.append(dict(date=day, group=group, status='no_reachable_original_goal'))
                continue
            for name, belief in (('original', base), ('common_student', common)):
                router = FrozenParameterRouter(graph, parameters, event_belief=belief,
                    max_time=3600, max_itineraries=8)
                router.online_mean = router.online_cancel = False
                router.route(state.stop, destination, state.time)
                router.information_state = state
                before = perf_counter()
                router.select_connection(state.stop, state.time, np.random.default_rng(0))
                elapsed = perf_counter() - before
                prediction = router.predictions[-1]
                policy = FixedWitness(tuple(prediction['primary']), tuple(prediction['rescue']),
                    first_action_now=bool(router._path and router._first_attempt))
                if policy.primary:
                    actual = LazyTripScenario(scene.events, minute=scene.minute, publications={})
                    model = build_committed_model(graph, [(1., actual)], state.stop, destination,
                        state.time, deadline, initial_state=state)
                    step, attempted_state = model.transition, [state]
                    def actual_step(world, context, action):
                        attempted_state[0] = context
                        return step(world, context, action)
                    model.transition = actual_step
                    try:
                        realized = execute_fixed_witness(model, model.scenarios[0], graph, policy,
                            router._fixed_departures, deadline)
                        realized.update(loss_lower=realized['cost'], loss_upper=realized['cost'])
                    except KeyError as missing:
                        if missing.args[0] not in mapping.values():
                            raise
                        realized = dict(cost=None, arrived=None, transitions=None, trace=[],
                            status='unresolved_visited_event', missing_connection=missing.args[0],
                            loss_lower=attempted_state[0].time - state.time, loss_upper=3600.)
                else:
                    realized = dict(cost=3600., arrived=False, transitions=0, trace=[])
                    realized.update(loss_lower=3600., loss_upper=3600.)
                item = dict(date=day, group=group, incoming=row['incoming'], time=state.time,
                    destination=destination, method=name, applied_targets=len(common.applied_cids),
                    primary=list(policy.primary), rescue=list(policy.rescue),
                    model_upper=prediction['value'], model_lower=prediction['full_model_lower'],
                    model_gap=prediction['full_model_gap'], actual=realized,
                    planning_seconds=elapsed, analytic_work=prediction['analytic_work'])
                records.append(item)
                print(day, group, name, item['model_upper'], realized['cost'], flush=True)
    result = dict(protocol_scope=protocol['scope'], rows=records,
        summary={method: dict(n=sum(r.get('method') == method for r in records),
            mean_loss_lower_seconds=float(np.mean([r['actual']['loss_lower'] for r in records if r.get('method') == method])),
            mean_loss_upper_seconds=float(np.mean([r['actual']['loss_upper'] for r in records if r.get('method') == method])))
                 for method in ('original', 'common_student')})
    (HERE / 'calibrated_continuation_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
