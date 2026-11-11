"""Replay the common timeout preceding the nominal coupling counterexample."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import write
from joint_day_run import inputs
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from src.joint_day_belief import JointDayBelief
from src.independent_day_mixture_belief import IndependentDayMixtureBelief


def main():
    paths = {'shared': HERE / 'joint_day_v2_results_cases/2023-10-27_q00_joint_nominal.json',
             'per_trip': HERE / 'joint_day_trip_control_cases/2023-10-27_q00_trip_mixture_nominal.json'}
    traces = {name: json.loads(path.read_text()) for name, path in paths.items()}
    first = traces['shared']['decisions'][0]
    assert first['outcome'] == 'wait_late'
    for key in ('clock_seconds', 'action', 'outcome', 'end_seconds'):
        assert traces['per_trip']['decisions'][0][key] == first[key]
    data, _, fit = inputs()
    graph, _, _, _, source, _ = dated_graph(data, '2023-10-27', 'gtfs_station_max')
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    shared = JointDayBelief(graph, public_occurrences=False, **fit['increments'])
    independent = IndependentDayMixtureBelief(graph, public_occurrences=False, **fit['increments'])
    for model in (shared, independent):
        model.observe_wait(first['action'], first['clock_seconds'], first['end_seconds'])
    delta, weights = shared.day_posterior(parameters)
    targets = {traces[name]['decisions'][1]['action'] for name in traces}
    targets.add(first['action'])
    candidates = []
    for cid in sorted(targets):
        row = source[cid]
        candidates.append({'local_connection': cid, 'source_connection': row['id'], 'route': row['route_name'],
            'scheduled_departure': row['scheduled_departure_seconds'],
            'shared_mean_departure': shared.mean_events(parameters)[cid].departure,
            'per_trip_mean_departure': independent.mean_events(parameters)[cid].departure,
            'actual_departure_for_posthoc_timeline_only': row['departure']['time_seconds']})
    output = {'scope': 'Same archived public timeout and frozen parameters. Actual departure is used only '
              'to explain the realized trajectory, not to estimate a probability truth or fit the model.',
              'history': {key: first[key] for key in ('clock_seconds', 'action', 'outcome', 'end_seconds')},
              'shared_posterior_mean_seconds': float(delta @ weights), 'candidates': candidates,
              'paths': {name: {'source': str(paths[name].relative_to(HERE)), 'loss_seconds': trace['loss_seconds'],
                               'decisions': [{key: row.get(key) for key in ('clock_seconds', 'action', 'source_connection', 'outcome', 'end_seconds')}
                                             for row in trace['decisions']]} for name, trace in traces.items()}}
    write(HERE / 'joint_day_counterexample.json', output)
    print(output['shared_posterior_mean_seconds'], candidates)


if __name__ == '__main__':
    main()
