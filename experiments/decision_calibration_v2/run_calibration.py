"""Public-only selection, then frozen V41 joint-event calibration on14 dates.

Run prepare before acquiring target actual/RT values; evaluate keeps every
registered opportunity, including roots without an observed initial history.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
from pathlib import Path
from time import perf_counter, process_time

from experiments.decision_gap_v1.candidate_coverage import expand_menu
from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.transfer_events import (
    event_probabilities, label_event, public_event_features, register_opportunities,
)
from experiments.decision_calibration_v1.platform_pairs import static_sources, transfer_pair
from experiments.decision_calibration_v2.calibration_summary import summarize
from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes
from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, static_catalog
from experiments.reconstruction_v34.execute import ArchivedWorld, Executor
from experiments.reconstruction_v34.run import relevant_frames, setup
from experiments.reconstruction_v37.controller import node_structure
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v50.menu import wide_root

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def select_public(catalog, dates, maximum):
    selected, counts = [], {}
    for day in dates:
        rows = [r for r in catalog['candidates'] if r['day'] == day]
        groups = [('downstream_only_direct', [r for r in rows if r['has_downstream_only_direct_tail']]),
                  ('shared_across_hubs', [r for r in rows if r['shared_tail_trips']]),
                  ('two_transfers', [r for r in rows if r['has_two_transfer_option']])]
        trips, route_od, chosen = set(), set(), []
        while len(chosen) < maximum:
            before = len(chosen)
            for layer, group in groups:
                candidate = next((r for r in group if r['first_trip'] not in trips
                    and (r['first_route'], r['stop'], r['goal']) not in route_od), None)
                if candidate is not None:
                    chosen.append(dict(candidate, selected_layer=layer,
                        selection_uses_actual_coverage_and_timing=False))
                    trips.add(candidate['first_trip'])
                    route_od.add((candidate['first_route'], candidate['stop'], candidate['goal']))
                if len(chosen) == maximum:
                    break
            if len(chosen) == before:
                break
        counts[day] = dict(candidates=len(rows), selected=len(chosen))
        selected.extend(chosen)
    return dict(selected=selected, by_day=counts, selection_uses_target_values=False)


def prepare(directory):
    manifest = read(directory/'date_manifest.json')
    if not manifest['dates_frozen']:
        raise ValueError('Freeze the metadata date panel first')
    protocol = read(EXPERIMENTS/'reconstruction_v34/protocol.json')
    protocol['main_dates'] = manifest['selected_dates']
    data = dict(stops={}, dates={})
    for row in manifest['days']:
        scheduled = read(directory/'static'/row['static_version']/'scheduled.json.gz')
        day = row['day']
        data['stops'].update(scheduled['stops'])
        connections = scheduled['dates'][day]['connections']
        attach_outcomes(day, connections, [])  # placeholders, no target labels
        data['dates'][day] = dict(connections=connections)
    write(directory/'scheduled.json.gz', data)
    catalog = static_catalog(data, protocol)
    write(directory/'static_catalog.json.gz', catalog)
    selection = select_public(catalog, manifest['selected_dates'], protocol['structure']['max_selected_per_day'])
    ledgers = []
    for index, selected in enumerate(selection['selected']):
        graph = context(data, selected['day'])[0]
        wide, caps = wide_root(graph, selected, protocol['structure'])
        original = root_ride(graph, wide)
        expanded, ledger = expand_menu(graph, original, wide['goal'])
        # This first source exit is fixed from the public physical path.
        cid = next(cid for cid in original.path if original.exits.get(cid) is not None)
        registration = register_opportunities(graph, original, cid, graph.connections[cid].arr_time, expanded)
        registration.update(clock_source='schedule_placeholder_until_initial_history_recovered',
                            root_index=index, day=selected['day'])
        observations = ObservationContract.from_menu(expanded)
        write(directory/'menus'/f'{index:03}.json.gz', dict(root_index=index, root=wide,
            original_node=node_structure(original), expanded_node=node_structure(expanded),
            observation_contract=observations.describe(), registration=registration,
            candidate_ledger=ledger, cap_ledger=caps))
        ledgers.append(dict(root_index=index, day=selected['day'],
            opportunities=len(registration['rows']), candidate_pairs=len(registration['candidate_pairs']),
            no_opportunity=registration['no_opportunity']))
    selection.update(registered=ledgers, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='All public roots/menus/opportunities fixed before reading actual endpoints or RT values')
    write(directory/'selection.json', selection)
    print('public selection frozen', len(ledgers), 'roots', sum(r['opportunities'] for r in ledgers), 'events', flush=True)


class FirstDecision(Exception):
    def __init__(self, cid, clock):
        self.cid, self.clock = cid, clock


class PrefixExecutor(Executor):
    def choose_onboard(self, ride, cid, clock, history):
        raise FirstDecision(cid, clock)


def initial_state(graph, root, node, history, metadata):
    """The unchanged physical executor supplies only the observed root prefix."""
    try:
        outcome = PrefixExecutor(graph, None, root['deadline']).run(
            node, root['start'], history, ArchivedWorld(metadata))
    except FirstDecision as state:
        return dict(status='observed', cid=state.cid, clock=float(state.clock))
    return dict(status=outcome['status'], cid=None, clock=None)


def evaluate_day(directory, day):
    tick, cpu = perf_counter(), process_time()
    manifest = read(directory/'date_manifest.json')
    selection = read(directory/'selection.json')
    source = next(r for r in manifest['days'] if r['day'] == day)
    data = read(directory/'scheduled.json.gz')
    connections = data['dates'][day]['connections']
    actual_path, message_path = directory/'actual'/f'{day}.csv.gz', directory/'messages'/f'{day}.json.gz'
    actual_available = actual_path.exists()
    message_source = read(message_path) if message_path.exists() else dict(frames=[], summary={})
    messages_available = message_source['summary'].get('sampled_snapshots', 0) > 0
    with gzip.open(actual_path, 'rt') if actual_available else open('/dev/null') as handle:
        rows = list(csv.DictReader(handle, delimiter=';')) if actual_available else []
    coverage = attach_outcomes(day, connections, rows)
    graph, _, _, mapping, metadata, _ = context(data, day)
    messages = message_source['frames']
    fit = read(EXPERIMENTS/'reconstruction_v34/model_fit.json')
    emission = read(EXPERIMENTS/'reconstruction_v35/fitted_emission.json')
    static_root = directory/'static'/source['static_version']/'data'
    platform_index, stops, transfers = static_sources(connections, static_root/'zurich_gtfs', static_root/'swiss_gtfs')
    events, roots, pairs = [], [], []
    for index, selected in enumerate(selection['selected']):
        if selected['day'] != day:
            continue
        root_tick, root_cpu = perf_counter(), process_time()
        menu = read(directory/'menus'/f'{index:03}.json.gz')
        root = menu['root']
        node, original = restore_node(menu['expanded_node']), restore_node(menu['original_node'])
        model, parameters = setup(graph, fit)
        frames = relevant_frames(messages, node.trips(model), mapping)
        history = PhysicalHistory(model, parameters, fit=emission, frames=frames, source_to_local=mapping)
        state = initial_state(graph, root, node, history, metadata)
        registered = menu['registration']
        if state['status'] == 'observed':
            if state['cid'] != registered['current_cid']:
                raise AssertionError('Public first exit changed after label acquisition')
            registered = register_opportunities(graph, original, state['cid'], state['clock'], node)
        if len(registered['rows']) != len(menu['registration']['rows']):
            raise AssertionError('A target value changed the registered denominator')
        public = history.public_state()
        forecast_labels = []
        for observation in public:
            if observation.get('evidence') != 'forecast' or not observation.get('emission_selected'):
                continue
            endpoint = metadata[observation['cid']][observation['kind']]
            forecast_labels.append(dict(forecast=observation, endpoint=endpoint,
                actual_minus_forecast_seconds=(endpoint['time_seconds']-observation['predicted_time'])
                    if endpoint['status'] == 'REAL' else None))
        cache = {}
        for event in registered['rows']:
            known_state = state['status'] == 'observed'
            if known_state:
                features = public_event_features(graph, event, public)
                options = len(node.exits[event['source_arrival_cid']].rides)
                features['downstream_action_candidate_count'] = options
                features['groups']['downstream_options'] = str(options)
                try:
                    prediction = event_probabilities(model, history, event, root['deadline'], cache)
                except UnsupportedHistory as error:
                    prediction = dict(status='probability_unresolved', reason=str(error))
                actual = label_event(metadata, event, root['deadline'])
                capture_range = message_source['summary'].get('capture_range')
                if not capture_range or capture_range[0] > state['clock']:
                    prediction = dict(status='probability_unresolved', reason='missing_predecision_message_captures')
            else:
                difference = event['downstream_mct']-event['upstream_mct']
                features = dict(groups=dict(forecast_age='missing_initial_history',
                    mct_difference='nonpositive' if difference <= 0 else 'at_most_180s' if difference <= 180 else 'over_180s',
                    downstream_options=str(len(node.exits[event['source_arrival_cid']].rides))))
                prediction = dict(status='probability_unresolved', reason=state['status'])
                actual = dict(status='label_unresolved', possible_partitions=['00','01','10','11'],
                              reason='initial_history_unavailable')
            events.append(dict(day=day, root_index=index, event_index=event['event_index'],
                incoming_trip=event['incoming_trip'], outgoing_trip=event['outgoing_trip'],
                event=event, initial_state=state, model=prediction, actual=actual,
                groups=features['groups'], public_features=features))
            for role, incoming, outgoing, mct in (
                ('upstream', event['current_cid'], event['upstream_cid'], event['upstream_mct']),
                ('downstream', event['source_arrival_cid'], event['hub_cid'], event['downstream_mct'])):
                pairs.append(dict(day=day, root_index=index, event_index=event['event_index'], role=role,
                    **transfer_pair(metadata[incoming], metadata[outgoing], mct, platform_index, stops, transfers)))
        roots.append(dict(root_index=index, root_id=root['root_id'], day=day,
            origin=root['origin_name'], goal=root['goal_name'], incoming_trip=root['first_trip'],
            initial_state=state, registered=registered, public_history=public,
            selected_forecast_labels=forecast_labels,
            relevant_message_frames=len(frames), observation_contract=menu['observation_contract'],
            cost=dict(process_cpu=process_time()-root_cpu, wall=perf_counter()-root_tick)))
    result = dict(day=day, events=events, roots=roots, platform_pairs=pairs,
        actual_available=actual_available, messages_available=messages_available, coverage=coverage,
        message_coverage=message_source['summary'],
        model_routes=dict(total=len({c.route for c in graph.connections}),
            using_pooled_prior=len({c.route for c in graph.connections if c.route not in fit['route_priors']})),
        cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick))
    write(directory/'results'/f'{day}.json.gz', result)
    print(day, 'roots', len(roots), 'events', len(events), 'states', dict(Counter(r['initial_state']['status'] for r in roots)), flush=True)


def aggregate(directory):
    manifest = read(directory/'date_manifest.json')
    selection = read(directory/'selection.json')
    events, platform_pairs, days, roots = [], [], [], []
    for day in manifest['selected_dates']:
        path = directory/'results'/f'{day}.json.gz'
        result = read(path) if path.exists() else dict(events=[], platform_pairs=[], roots=[])
        expected = sum(r['opportunities'] for r in selection['registered'] if r['day'] == day)
        days.append(dict(day=day, available=path.exists() and result.get('actual_available', False) and result.get('messages_available', False),
                         denominator_known=True, expected_opportunities=expected,
                         actual_available=result.get('actual_available', False), messages_available=result.get('messages_available', False)))
        events.extend(result['events'])
        platform_pairs.extend(result['platform_pairs'])
        roots.extend(result['roots'])
    evaluation_manifest = dict(manifest, days=days)
    write(directory/'evaluation_manifest.json', evaluation_manifest)
    write(directory/'events.json.gz', events)
    summary = summarize(events, evaluation_manifest)
    statuses = Counter(pair['status'] for pair in platform_pairs)
    contradictions = statuses['planned_pair_prohibited']+statuses['fixed_mct_differs_from_planned_pair']
    summary['platform_verification'] = dict(status='fail' if contradictions else 'unresolved', pairs=len(platform_pairs),
        status_counts=dict(statuses), actual_platform_pairs_verified=0,
        reason='The2023 official actual archive has no observed platform identity; published plans are retained separately')
    summary['reconstruction_admitted'] = False
    forecasts = [value for root in roots for value in root['selected_forecast_labels']]
    residuals = [value['actual_minus_forecast_seconds'] for value in forecasts
                 if value['actual_minus_forecast_seconds'] is not None]
    summary['source_diagnostics'] = dict(roots=len(roots),
        initial_state_statuses=dict(Counter(root['initial_state']['status'] for root in roots)),
        roots_without_opportunities=[root['root_index'] for root in roots if not root['registered']['rows']],
        candidate_pair_statuses=dict(Counter(pair['status'] for root in roots for pair in root['registered']['candidate_pairs'])),
        event_model_statuses=dict(Counter(event['model']['status'] for event in events)),
        event_actual_statuses=dict(Counter(event['actual']['status'] for event in events)),
        actual_partition_counts=dict(Counter(event['actual'].get('actual_partition') or 'unknown' for event in events)),
        selected_forecasts=len(forecasts), forecast_endpoint_statuses=dict(Counter(value['endpoint']['status'] for value in forecasts)),
        selected_forecast_observed_residual_count=len(residuals),
        selected_forecast_mean_error_seconds=sum(residuals)/len(residuals) if residuals else None,
        selected_forecast_mean_absolute_error_seconds=sum(abs(value) for value in residuals)/len(residuals) if residuals else None,
        forecast_weighting='Descriptive selected root-trip forecasts, including repeats across roots; no forecast refit',
        source_reach_failure_probability_sums={cause:sum(event['model'].get('source_first_failure_probabilities', {}).get(cause, 0.) for event in events)
            for cause in ('canceled', 'deadline', 'time_order')})
    write(directory/'summary.json', summary)
    write(directory/'platform_pairs.json.gz', platform_pairs)
    write(directory/'exposure_record.json', dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        dates=[row['day'] for row in days if (directory/'results'/f"{row['day']}.json.gz").exists()],
        use='Frozen V41 calibration; these real target dates are now exposed and cannot be reused as undeveloped confirmation dates'))
    print('calibration summary', summary['verdict'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'evaluate', 'summarize'])
    parser.add_argument('--directory', type=Path, default=HERE)
    parser.add_argument('--date')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.directory)
    elif args.command == 'evaluate':
        evaluate_day(args.directory, args.date)
    else:
        aggregate(args.directory)
