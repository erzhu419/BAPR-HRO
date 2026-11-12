"""Attribute the observed information-arm gaps using saved receipts only."""
from collections import Counter
import json

from experiments.reconstruction_v34.discover import HERE, context, read


def action(row):
    return row['kind'], row.get('cid', row.get('stop')), row['clock'], row.get('choice')


def endpoint_identity(cid, kind, graph, metadata):
    connection, source = graph.connections[cid], metadata[cid]
    stop = connection.dep_stop if kind == 'departure' else connection.arr_stop
    return dict(local_cid=cid, source_cid=source['id'], kind=kind, trip_id=connection.trip_id,
        route=source['route_name'], stop=graph.stops[stop].name,
        scheduled_seconds=source['scheduled_'+kind+'_seconds'])


def trace_actions(outcome, graph, metadata):
    result = []
    for row in outcome['trace']:
        item = {k: row[k] for k in ('kind', 'clock', 'choice', 'entry', 'ready', 'visible', 'observed') if k in row}
        cid = row.get('cid')
        if cid is not None:
            item['event'] = endpoint_identity(cid, 'arrival', graph, metadata)
        elif row.get('choice') is not None:
            item['event'] = endpoint_identity(row['choice'], 'departure', graph, metadata)
        else:
            item['stop'] = graph.stops[row['stop']].name
        result.append(item)
    return result


def explain(case):
    index = case['root_index']
    if index == 13:
        return dict(mechanism='At Hardbruecke the messages increased the alighting score and lowered the '
            'continuation score, changing alight to continue. The public arm then transferred at HB to S5; '
            'the own/local arm waited at Hardbruecke for S11 and arrived 343 seconds earlier.',
            highlighted_cids=[139, 182, 193],
            interpretation='The relevant S11/S5 predictions were received 476 seconds before the first '
            'fork. This is a stale-snapshot exposure, not proof that forecast age caused the loss. '
            'The scores also integrate service uncertainty, the frozen trip law, and surrogate tail values.')
    if index == 33:
        return dict(mechanism='At HB both arms saw S7 departure 231 at 30011 with the same visit history. '
            'Messages raised the wait score for S14 227 from 30646.177 to 31207.792, so the public arm '
            'boarded S7. The own/local arm waited six more seconds for S14 and arrived 885 seconds earlier.',
            highlighted_cids=[227, 231, 270],
            interpretation='The S14 departure prediction was 30000, received at 30003 and used at 30011; '
            'the realized departure was 30017. A 17-second endpoint forecast error alone does not explain '
            'the 561.615-second change in expected wait score. The fixed likelihood, discretized joint '
            'trip posterior, no-offer risk and downstream scoring all enter that score.')
    if index == 69:
        return dict(mechanism='At Oerlikon both absolute scores improved after messages, but continuation '
            'improved more: the public arm continued by a 5.371-second modeled margin, while own/local '
            'alighted by 53.359 seconds. Public continued to HB and boarded S12; own/local used S16 '
            'from Oerlikon and arrived 259 seconds earlier.',
            highlighted_cids=[169, 189, 209, 215],
            interpretation='The highlighted forecasts were 157 seconds old. This is a near-boundary '
            'decision in the declared model, with a negative realized outcome; a single realized world '
            'does not establish that the expected-value ranking was mathematically wrong.')
    raise ValueError('An unreviewed difference case must not inherit another case explanation')


def main():
    protocol, data, selection, messages = (read(HERE/name) for name in (
        'protocol.json', 'data.json.gz', 'selection.json', 'messages.json.gz'))
    comparisons, cases = {}, []
    for path in sorted((HERE/'roots').glob('*.json.gz')):
        saved = read(path)
        for method in ('combined_rolling', 'rolling_nominal'):
            public = saved['rows']['public_messages'][method]['outcome']
            own = saved['rows']['own_and_local_only'][method]['outcome']
            if public['loss'] is None or own['loss'] is None:
                comparisons.setdefault(method, []).append(None)
            else:
                comparisons.setdefault(method, []).append(own['loss']-public['loss'])
        public = saved['rows']['public_messages']['combined_rolling']['outcome']
        own = saved['rows']['own_and_local_only']['combined_rolling']['outcome']
        if public['loss'] == own['loss']:
            continue
        index = saved['root_index']
        root = selection['selected'][index]
        graph, _, _, mapping, metadata, _ = context(data, saved['day'])
        fork = next(i for i, (a, b) in enumerate(zip(public['trace'], own['trace'])) if action(a) != action(b))
        a, b = public['trace'][fork], own['trace'][fork]
        hard = lambda row: [r for r in row['public_history'] if r['evidence'] in ('observed', 'unseen')]
        assert hard(a) == hard(b), 'A later-history comparison cannot isolate the first information fork'
        assert [action(r) for r in public['trace'][:fork]] == [action(r) for r in own['trace'][:fork]]
        frame_index = {}
        for frame in messages['dates'][saved['day']]['frames']:
            if frame['receipt_seconds'] <= a['clock']:
                frame_index[frame['trip_id'], frame['receipt_seconds']] = frame
        soft = []
        for record in a['public_history']:
            if record['evidence'] in ('observed', 'unseen'):
                continue
            receipt = record['receipt_seconds']
            frame = frame_index[record['trip'], receipt]
            item = dict(record, age_seconds=a['clock']-receipt,
                publication={k:frame[k] for k in ('archive_member', 'entity_id', 'capture_seconds',
                    'header_seconds', 'update_seconds', 'receipt_seconds')})
            assert item['age_seconds'] >= 0
            if 'cid' in record:
                item['target'] = endpoint_identity(record['cid'], record['kind'], graph, metadata)
            soft.append(item)
        costs = {}
        for arm, row in (('public_messages', a), ('own_and_local_only', b)):
            receipt = row['receipt']
            margin = (receipt['alight']-receipt['continuation'] if row['kind']=='onboard'
                      else receipt['board']-receipt['wait'])
            costs[arm] = dict(choice=row.get('choice'), receipt=receipt,
                immediate_action_minus_wait_or_continue=margin)
        case = dict(root_index=index, root_id=root['root_id'], day=saved['day'],
            source=str(path.relative_to(HERE)), origin=saved['origin'], destination=saved['goal'],
            first_divergence=dict(trace_index=fork, clock=a['clock'], kind=a['kind'],
                local_cid=a.get('cid'), stop_id=a.get('stop'), hard_public_history_equal=True,
                common_hard_public_history=hard(a), costs=costs,
                relative_cost_shift=costs['public_messages']['immediate_action_minus_wait_or_continue']-
                    costs['own_and_local_only']['immediate_action_minus_wait_or_continue'],
                received_relevant_frame_count=a['message_state']['frame_index'],
                active_forecasts=sum(r['evidence']=='forecast' for r in soft),
                nonoperating_notices=[r for r in soft if r.get('status') in ('CANCELED','DELETED','SKIPPED','NO_DATA')],
                current_soft_history_with_publication=soft),
            realized=dict(public_loss=public['loss'], own_local_loss=own['loss'],
                information_gain_seconds=own['loss']-public['loss'],
                public_status=public['status'], own_local_status=own['status'],
                public_trajectory=trace_actions(public, graph, metadata),
                own_local_trajectory=trace_actions(own, graph, metadata)),
            selection_metadata={k:root.get(k) for k in (
                'static_menu_uses_actual', 'selection_uses_actual_coverage_and_timing')})
        case['diagnosis'] = explain(case)
        highlighted = set(case['diagnosis']['highlighted_cids'])
        case['highlighted_forecasts'] = []
        for record in soft:
            if record['evidence'] != 'forecast' or record['cid'] not in highlighted:
                continue
            endpoint = metadata[record['cid']][record['kind']]
            case['highlighted_forecasts'].append(dict(record,
                posthoc_actual_endpoint=dict(status=endpoint['status'], time_seconds=endpoint['time_seconds'],
                    role='Diagnostic comparison only; this endpoint was not supplied to the decision')))
        cases.append(case)
    summary = {}
    for method, values in comparisons.items():
        known = [x for x in values if x is not None]
        summary[method] = dict(total_roots=len(values), paired_known=len(known),
            public_wins=sum(x>0 for x in known), public_harms=sum(x<0 for x in known),
            ties=sum(x==0 for x in known), gain_sum_seconds=sum(known),
            mean_information_gain_seconds=sum(known)/len(known))
    result = dict(scope='Posthoc receipt-only diagnosis. No replay, new sample, model fit, or menu change.',
        comparison='Same combined_rolling control; public_messages versus own_and_local_only.',
        summary=summary, cases=cases,
        findings='All three combined-control first forks have identical hard own/local history and '
            'available message receipts only. No nonoperating notice is active at any first fork. '
            'Received numerical forecasts change the declared conditional scores and action boundary. '
            'No implementation error is established by these three negative realized outcomes.',
        limitations='The arm comparison identifies the effect of the complete message-processing/control '
            'bundle in these fixed realized worlds. Later histories and message use also diverge, so total '
            'loss is not an isolated causal effect of the first changed action or a single forecast. '
            'Forecast-error calibration, posterior discretization, frozen-snapshot tail surrogates and '
            'future revisions are not separated by this audit. Dates are development data, not confirmation.',
        selection='Inherited selection_uses_actual:false metadata has been corrected. Static menus do '
            'not use actuals; final eligibility uses actual coverage/consistency and message timing, '
            'never forecast values or policy loss.')
    (HERE/'information_diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
