"""Freeze the 72-root event denominator, then evaluate labels/probabilities.

    python -m experiments.decision_gap_v1.event_run register
    python -m experiments.decision_gap_v1.event_run evaluate
    python -m experiments.decision_gap_v1.event_run banks

The last command only reconstructs endpoints of saved full128 random banks.
It never executes a policy, draws a new seed or changes a fitted model.
"""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter, process_time

from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.transfer_events import (
    archived_policy_usage, event_probabilities, label_event, partition_saved_bank,
    public_event_features, register_opportunities,
)
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node, save
from experiments.reconstruction_v41.physical_posterior import restore_physical_snapshot
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v51.rollout import StableTripWorlds


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def _find_ride(node, cid, trip, graph):
    if hasattr(node, 'path'):
        if cid in node.path and graph.connections[node.first_cid].trip_id == trip:
            return node
        children = [child for child in node.exits.values() if child is not None]
    else:
        children = node.rides.values()
    for child in children:
        found = _find_ride(child, cid, trip, graph)
        if found is not None:
            return found
    return None


def register(directory=HERE):
    """Freeze all original roots' first legal onboard decisions before scoring."""
    directory = Path(directory)
    data = read(EXPERIMENTS/'reconstruction_v34/data.json.gz')
    rows = []
    for index in range(72):
        old = read(EXPERIMENTS/f'reconstruction_v51/roots/{index:03}_cheap.json.gz')
        menu = read(directory/f'menus/{index:03}.json.gz')
        graph = context(data, old['day'])[0]
        states = [(si, state) for si, state in enumerate(old['states'])
                  if state['kind'] == 'onboard' and 'alight' in state['legal_actions']]
        if not states:
            rows.append(dict(root_index=index, day=old['day'], source_state_index=None,
                rows=[], candidate_pairs=[], no_opportunity=True, reason='no_legal_onboard_state'))
            continue
        si, state = states[0]
        source = restore_node(state['node'])
        expanded = _find_ride(restore_node(menu['expanded_node']), state['cid'],
                              graph.connections[state['cid']].trip_id, graph)
        if expanded is None:
            raise ValueError(f'Frozen source ride absent from expanded root {index}')
        result = register_opportunities(graph, source, state['cid'], state['clock'], expanded)
        result.update(root_index=index, day=old['day'], source_state_index=si,
            deadline=state['deadline'], public_history=state['public_history'],
            source=f'../reconstruction_v51/roots/{index:03}_cheap.json.gz#/states/{si}')
        rows.append(result)
    summary = dict(roots=len(rows), roots_with_opportunities=sum(bool(row['rows']) for row in rows),
        no_opportunity_roots=[row['root_index'] for row in rows if row['no_opportunity']],
        candidate_pairs=sum(len(row['candidate_pairs']) for row in rows),
        opportunities=sum(len(row['rows']) for row in rows),
        same_chain_opportunities=sum(event['same_incoming_outgoing_trip'] for row in rows for event in row['rows']),
        pair_status_counts=dict(Counter(pair['status'] for row in rows for pair in row['candidate_pairs'])))
    output = dict(selection='All 72 original V51 roots, first public legal onboard alight state; no outcome selection',
        domain='Expanded declared candidates and future legal source exits; physical endpoint event separate from action availability',
        model_scope='Root-conditioned V41 P0; future remote messages frozen',
        actual_scope='Developed archived endpoint replay labels; no independent calibration or passenger trajectory claim',
        summary=summary, roots=rows)
    save(directory/'events/registration.json.gz', output)
    print(summary, flush=True)
    return output


def _inputs():
    return (read(EXPERIMENTS/'reconstruction_v34/data.json.gz'),
            read(EXPERIMENTS/'reconstruction_v34/model_fit.json'),
            read(EXPERIMENTS/'reconstruction_v35/fitted_emission.json'))


def groups(directory=HERE):
    """Freeze public covariates separately, without changing event registration."""
    directory = Path(directory)
    registered = read(directory/'events/registration.json.gz')
    data = read(EXPERIMENTS/'reconstruction_v34/data.json.gz')
    rows = []
    for root in registered['roots']:
        index = root['root_index']
        graph = context(data, root['day'])[0]
        menu = read(directory/f'menus/{index:03}.json.gz')
        features = []
        if root['rows']:
            source = _find_ride(restore_node(menu['expanded_node']), root['current_cid'], root['incoming_trip'], graph)
            for frozen in root['rows']:
                event = dict(frozen)
                candidate = source.exits[root['current_cid']].rides[event['upstream_cid']]
                event['outgoing_to_hub_path'] = list(candidate.path[:candidate.path.index(event['hub_cid'])])
                value = public_event_features(graph, event, root['public_history'])
                value.update(event_index=event['event_index'],
                    downstream_action_candidate_count=len(source.exits[event['source_arrival_cid']].rides),
                    downstream_through_candidate_count=len({other['upstream_cid'] for other in root['rows']
                        if other['source_arrival_cid'] == event['source_arrival_cid']}))
                features.append(value)
        rows.append(dict(root_index=index, rows=features))
    output = dict(scope='Frozen from registered public history/graph before P0 integration or actual labels',
        definitions=dict(forecast_age='Model-selected outgoing-trip numeric forecast: none, age <= 300 s, age > 300 s',
            mct_difference='downstream minus upstream MCT: <= 0, (0,180], >180 seconds',
            source_delay='observed source arrival minus schedule: <=0, (0,60], >60 seconds',
            shared_corridor='incoming and outgoing upstream-to-hub paths share at least one directed station edge',
            downstream_counts='All declared hub action candidates; distinct upstream candidates with a through hub departure'),
        roots=rows)
    save(directory/'events/group_registration.json.gz', output)
    print('group registration', len(rows), 'roots', sum(len(row['rows']) for row in rows), 'events', flush=True)
    return output


def evaluate(directory=HERE):
    directory = Path(directory)
    registered = read(directory/'events/registration.json.gz')
    data, fit, emission = _inputs()
    for root in registered['roots']:
        index = root['root_index']
        output_path = directory/f'events/roots/{index:03}.json.gz'
        if output_path.exists():
            continue
        tick, cpu = perf_counter(), process_time()
        graph, _, _, _, metadata, _ = context(data, root['day'])
        output = dict(root_index=index, day=root['day'], no_opportunity=root['no_opportunity'], rows=[])
        if root['rows']:
            model, parameters = setup(graph, fit)
            history = restore_physical_snapshot(model, parameters, emission,
                                                root['public_history'], root['clock'])
            cache = {}
            for event in root['rows']:
                model_result = event_probabilities(model, history, event, root['deadline'], cache)
                actual_result = label_event(metadata, event, root['deadline'])
                output['rows'].append(dict(event_index=event['event_index'],
                    model=model_result, actual=actual_result))
        output['cost'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        save(output_path, output)
        print(index, 'events', len(output['rows']), 'cpu', round(output['cost']['process_cpu'], 3), flush=True)
    return summarize(directory)


def banks(directory=HERE):
    directory = Path(directory)
    registered = read(directory/'events/registration.json.gz')
    data, fit, emission = _inputs()
    for root in registered['roots']:
        index = root['root_index']
        result_path = directory/f'events/banks/{index:03}.json.gz'
        if result_path.exists():
            continue
        output = dict(root_index=index, rows=[], status='no_opportunity' if not root['rows'] else 'pending')
        if not root['rows']:
            save(result_path, output)
            continue
        receipt_path = directory/f'roots/{index:03}_new_full128.json.gz'
        if not receipt_path.exists():
            print(index, 'saved full128 bank not available', flush=True)
            continue
        receipt = read(receipt_path)
        matches = [gate for gate in receipt['gates'] if gate['cid'] == root['current_cid']
                   and gate['clock'] == root['clock']
                   and receipt['states'][gate['state_index']]['public_history'] == root['public_history']]
        if len(matches) != 1:
            output.update(status='bank_unresolved', reason='no_unique_gate_with_frozen_public_history')
            save(result_path, output)
            continue
        gate = matches[0]
        tick, cpu = perf_counter(), process_time()
        menu = read(directory/f'menus/{index:03}.json.gz')
        graph = context(data, root['day'])[0]
        model, parameters = setup(graph, fit)
        history = restore_physical_snapshot(model, parameters, emission, root['public_history'], root['clock'])
        node = restore_node(menu['expanded_node'])
        described = menu['new_observation_contract']
        contract = ObservationContract({row['stop']:row['cids'] for row in described['station_to_cids']})
        bank = StableTripWorlds(history.as_belief(), parameters,
            [*node.paths(), *contract.paths()], gate['count'], gate['seed'], index, model._trips)
        if bank.used_trips != gate['common_tape']['trips'] or len(bank.trip_ranks) != gate['common_tape']['rank_universe_size']:
            raise ValueError(f'Original bank trip-rank identity differs at root {index}')
        output.update(status='resolved', seed=gate['seed'], count=gate['count'],
            source=f'roots/{index:03}_new_full128.json.gz',
            scope='Saved development bank loss decomposition; no new seed or policy execution')
        for event in root['rows']:
            value = (partition_saved_bank(bank.events, event, gate, root['deadline'])
                     if event['source_path'] is not None else
                     dict(status='partition_unresolved', reason='source_path_mismatch'))
            output['rows'].append(dict(event_index=event['event_index'], **value))
        output['cost'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        save(result_path, output)
        print(index, 'bank partitions', len(output['rows']), flush=True)
    return summarize(directory)


def summarize(directory=HERE):
    directory = Path(directory)
    registered = read(directory/'events/registration.json.gz')
    evaluated = [read(path) for path in sorted((directory/'events/roots').glob('*.json.gz'))]
    rows = [row for root in evaluated for row in root['rows']]
    banks_rows = [read(path) for path in sorted((directory/'events/banks').glob('*.json.gz'))]
    result = dict(registration=registered['summary'], evaluated_roots=len(evaluated),
        event_model_status=dict(Counter(row['model']['status'] for row in rows)),
        event_actual_status=dict(Counter(row['actual']['status'] for row in rows)),
        actual_partition_counts=dict(Counter(row['actual']['actual_partition'] or 'unknown' for row in rows)),
        model_cpu=sum(root['cost']['process_cpu'] for root in evaluated),
        bank_roots=len(banks_rows), bank_status=dict(Counter(root['status'] for root in banks_rows)),
        development_scope='Event rows overlap and roots share dated operational events; counts are descriptive, not independent calibration tests')
    _descriptive_groups(directory, registered, evaluated, result)
    write(directory/'events/summary.json', result)
    print(result, flush=True)
    return result


def _descriptive_groups(directory, registered, evaluated, result):
    group_path = directory/'events/group_registration.json.gz'
    if not group_path.exists():
        result['group_status'] = 'public_group_registration_pending'
        return
    features = {(root['root_index'], event['event_index']): event
                for root in read(group_path)['roots'] for event in root['rows']}
    estimates = {(root['root_index'], event['event_index']): event
                 for root in evaluated for event in root['rows']}
    all_events = []
    for root in registered['roots']:
        for event in root['rows']:
            key = root['root_index'], event['event_index']
            if key not in estimates:
                continue
            all_events.append(dict(root_index=root['root_index'], day=root['day'],
                # Distinct named operational events, with one first-registered
                # representative when several roots reuse them. Date dependence remains.
                operational_key=(root['day'], event['incoming_trip'], event['outgoing_trip'],
                    event['current_cid'], event['upstream_cid'], event['hub_cid'], event['source_arrival_cid']),
                features=features[key], **estimates[key]))
            residuals = []
            actual = estimates[key]['actual']['endpoints']
            endpoints = {(actual[name]['cid'], actual[name]['kind']):actual[name] for name in ('upstream', 'hub')}
            for forecast in features[key]['endpoint_forecasts']:
                endpoint = endpoints.get((forecast['cid'], forecast['kind']))
                residuals.append(dict(cid=forecast['cid'], kind=forecast['kind'],
                    forecast_age_seconds=root['clock']-forecast['receipt_seconds'],
                    selected=forecast['emission_selected'],
                    actual_minus_prediction_seconds=endpoint['time']-forecast['predicted_time']
                        if endpoint and endpoint['status'] == 'REAL' and endpoint['time'] is not None else None))
            all_events[-1]['continuous_forecast_residuals'] = residuals
    representatives = {}
    for event in all_events:
        representatives.setdefault(event['operational_key'], event)
    unique = list(representatives.values())
    def aggregate(entries):
        known = [e for e in entries if e['model']['status'] == 'resolved']
        if not known:
            return dict(events=len(entries), model_resolved=0)
        probabilities = [e['model']['partitions']['10'] for e in known]
        indicator_bounds = [e['actual']['partition_bounds']['10'] for e in known]
        brier = []
        for probability, bounds in zip(probabilities, indicator_bounds):
            scores = [(probability-y)**2 for y in range(bounds[0], bounds[1]+1)]
            brier.append([min(scores), max(scores)])
        n = len(known)
        frequency = [sum(row[i] for row in indicator_bounds)/n for i in (0, 1)]
        predicted = sum(probabilities)/n
        return dict(events=len(entries), model_resolved=n, known_labels=sum(a == b for a,b in indicator_bounds),
            roots=len({e['root_index'] for e in entries}), dates=len({e['day'] for e in entries}),
            mean_p_up_only=predicted, observed_frequency_bounds=frequency,
            signed_frequency_minus_probability_bounds=[value-predicted for value in frequency],
            brier_bounds=[sum(row[i] for row in brier)/n for i in (0, 1)])
    grouped = {}
    for unit, entries in (('all_registered_opportunities', all_events), ('distinct_operational_events', unique)):
        grouped[unit] = dict(overall=aggregate(entries),
            by_date={day: aggregate([e for e in entries if e['day'] == day]) for day in sorted({e['day'] for e in entries})},
            by_public_group={name: {value: aggregate([e for e in entries if e['features']['groups'][name] == value])
                for value in sorted({e['features']['groups'][name] for e in entries})}
                for name in ('forecast_age', 'mct_difference', 'source_delay', 'shared_corridor')})
    result['descriptive_groups'] = grouped
    result['event_deduplication'] = dict(representative='First root/event in frozen registration order',
        tuple_fields=['day', 'incoming_trip', 'outgoing_trip', 'current_cid', 'upstream_cid', 'hub_cid', 'source_arrival_cid'],
        scope='Distinct endpoints are not claimed statistically independent; date clusters retained, no iid uncertainty test')
    write(directory/'events/grouped_events.json.gz', dict(rows=all_events,
        representative_rows=[dict(root_index=e['root_index'], event_index=e['event_index']) for e in unique]))


def usage(directory=HERE):
    """Attach recorded actual replay boarding, keeping missing policies explicit."""
    directory = Path(directory)
    registered = read(directory/'events/registration.json.gz')
    data = read(EXPERIMENTS/'reconstruction_v34/data.json.gz')
    rows = []
    for root in registered['roots']:
        index = root['root_index']
        graph = context(data, root['day'])[0]
        paths = {'original_cheap': EXPERIMENTS/f'reconstruction_v51/roots/{index:03}_cheap.json.gz',
                 'original_full128': EXPERIMENTS/f'reconstruction_v51/roots/{index:03}_full128.json.gz',
                 'new_cheap': directory/f'roots/{index:03}_new_cheap.json.gz',
                 'new_full128': directory/f'roots/{index:03}_new_full128.json.gz'}
        receipts = {label: read(path) for label,path in paths.items() if path.exists()}
        rows.append(dict(root_index=index, day=root['day'], rows=[dict(event_index=event['event_index'],
            policies={label: archived_policy_usage(receipts[label], graph, event) if label in receipts
                else dict(status='receipt_pending') for label in paths}) for event in root['rows']]))
    write(directory/'events/policy_usage.json.gz', dict(scope='Actual archived replay actions; four policy domains retained separately', roots=rows))
    print('actual policy usage', len(rows), 'roots', flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('register', 'groups', 'evaluate', 'banks', 'usage', 'summarize'))
    parser.add_argument('--directory', type=Path, default=HERE)
    args = parser.parse_args()
    globals()[args.stage](args.directory)
