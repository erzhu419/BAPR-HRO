"""Frozen joint-event and planned-platform checks from saved control histories."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

from experiments.decision_calibration_v1.platform_pairs import static_sources, transfer_pair
from experiments.decision_calibration_v2.calibration_summary import summarize as calibration
from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.transfer_events import (
    PARTITIONS, event_probabilities, label_event, public_event_features, register_opportunities,
)
from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v41.physical_posterior import restore_physical_snapshot

HERE = Path(__file__).resolve().parent


def source_available(summary):
    return (summary.get('source_parse_complete',False)
            and summary.get('identity_mapping_complete',False)
            and summary.get('sampled_snapshots',0)>0)


def root_events(graph, metadata, menu, receipt, index, physical_fit, emission, message_summary):
    root, frozen = menu['root'], menu['registration']
    node,original = restore_node(menu['expanded_node']),restore_node(menu['original_node'])
    first = next((row for row in receipt['states'] if row['kind']=='onboard'),None)
    public = [] if first is None else first['public_history']
    if first is None:
        state = dict(status=receipt['outcome']['status'],cid=None,clock=None)
        registered = dict(frozen,clock=None,rows=[dict(row,clock=None) for row in frozen['rows']])
    else:
        state = dict(status='observed',cid=first['cid'],clock=first['clock'])
        if first['cid']!=frozen['current_cid']:
            raise ValueError('Saved first decision differs from the registered public exit')
        registered = register_opportunities(graph,original,first['cid'],first['clock'],node)
        strip_clock = lambda rows:[{k:v for k,v in row.items() if k!='clock'} for row in rows]
        if (strip_clock(registered['rows'])!=strip_clock(frozen['rows'])
                or registered['candidate_pairs']!=frozen['candidate_pairs']):
            raise ValueError('Actual history changed the frozen event population')
    if len(registered['rows'])!=len(frozen['rows']):
        raise ValueError('Frozen event denominator changed')
    history = model = None
    source_ok = source_available(message_summary) and receipt['coverage']['source_available']
    capture_range = message_summary.get('capture_range')
    predecision_source = bool(first is not None and source_ok and capture_range
                              and capture_range[0]<=first['clock'])
    if predecision_source:
        model,parameters = setup(graph,physical_fit)
        history = restore_physical_snapshot(model,parameters,emission,public,first['clock'])
    events,cache = [],{}
    for event in registered['rows']:
        options = len(node.exits[event['source_arrival_cid']].rides)
        if first is None:
            difference = event['downstream_mct']-event['upstream_mct']
            features = dict(groups=dict(forecast_age='missing_initial_history',
                mct_difference='nonpositive' if difference<=0 else 'at_most_180s' if difference<=180 else 'over_180s'))
            prediction = dict(status='probability_unresolved',reason='initial_history_unavailable')
            actual = dict(status='label_unresolved',possible_partitions=list(PARTITIONS),
                          reason='initial_history_unavailable')
        else:
            features = public_event_features(graph,event,public)
            # Endpoint labels remain usable even when messages or the model
            # posterior are unknown; no numeric-message requirement filters them.
            actual = label_event(metadata,event,root['deadline'])
            if history is None:
                prediction = dict(status='probability_unresolved',reason='predecision_message_source_unavailable')
            else:
                try:
                    prediction = event_probabilities(model,history,event,root['deadline'],cache)
                except UnsupportedHistory as error:
                    prediction = dict(status='probability_unresolved',reason=str(error))
        features['downstream_action_candidate_count'] = options
        features['groups']['downstream_options'] = str(options)
        events.append(dict(day=root['day'],root_index=index,event_index=event['event_index'],
            incoming_trip=event['incoming_trip'],outgoing_trip=event['outgoing_trip'],event=event,
            initial_state=state,model=prediction,actual=actual,groups=features['groups'],public_features=features))
    return events,dict(root_index=index,root_id=root['root_id'],day=root['day'],
        incoming_trip=root['first_trip'],initial_state=state,registered=registered,public_history=public,
        saved_control=f'roots/{index:03}_new_cheap.json.gz',predecision_source_available=predecision_source)


def evaluate_day(directory, day):
    directory = Path(directory)
    output = directory/'events/results'/f'{day}.json.gz'
    if output.exists():
        return read(output)
    tick,cpu = perf_counter(),process_time()
    manifest,selection = read(directory/'date_manifest.json'),read(directory/'selection.json')
    source = next(row for row in manifest['days'] if row['day']==day)
    data = read(directory/'data.json.gz')
    graph,_,_,_,metadata,_ = context(data,day)
    physical_fit,emission = read(directory/'physical_fit.json'),read(directory/'repaired_fit.json')
    message_path = directory/'messages'/f'{day}.json.gz'
    message_summary = read(message_path)['summary'] if message_path.exists() else {}
    endpoint_coverage = read(directory/'endpoint_coverage.json')['days'][day]
    static = directory/'inputs'/('static_'+source['static_version'])/'data'
    platform_index,stops,transfers = static_sources(data['dates'][day]['connections'],
                                                  static/'zurich_gtfs',static/'swiss_gtfs')
    events,roots,pairs = [],[],[]
    for index,selected in enumerate(selection['selected']):
        if selected['day']!=day:
            continue
        menu = read(directory/'menus'/f'{index:03}.json.gz')
        receipt = read(directory/'roots'/f'{index:03}_new_cheap.json.gz')
        values,root = root_events(graph,metadata,menu,receipt,index,physical_fit,emission,message_summary)
        events.extend(values);roots.append(root)
        for row in values:
            event = row['event']
            for role,incoming,outgoing,mct in (
                ('upstream',event['current_cid'],event['upstream_cid'],event['upstream_mct']),
                ('downstream',event['source_arrival_cid'],event['hub_cid'],event['downstream_mct'])):
                pairs.append(dict(day=day,root_index=index,event_index=event['event_index'],role=role,
                    **transfer_pair(metadata[incoming],metadata[outgoing],mct,platform_index,stops,transfers)))
    result = dict(day=day,events=events,roots=roots,platform_pairs=pairs,
        actual_available=endpoint_coverage['actual_source_available'],
        messages_available=source_available(message_summary),coverage=endpoint_coverage,
        message_coverage=message_summary,cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick),
        scope='Offline exact event integration from saved first-decision histories; no control replay or new sampled worlds.')
    write(output,result)
    return result


def finalize(directory=HERE):
    directory = Path(directory)
    if (directory/'calibration_summary.json').exists():
        raise ValueError('Final calibration already recorded; do not rerun the bootstrap')
    manifest,selection = read(directory/'date_manifest.json'),read(directory/'selection.json')
    events,roots,pairs,days = [],[],[],[]
    # Read every required day before invoking the single final bootstrap.
    for day in manifest['selected_dates']:
        result = read(directory/'events/results'/f'{day}.json.gz')
        expected = sum(r['opportunities'] for r in selection['registered'] if r['day']==day)
        wanted = [i for i,r in enumerate(selection['selected']) if r['day']==day]
        if (result['day']!=day or [r['root_index'] for r in result['roots']]!=wanted
                or len(result['events'])!=expected):
            raise ValueError('Daily events do not cover every frozen root and opportunity')
        days.append(dict(day=day,available=result['actual_available'] and result['messages_available'],
            denominator_known=True,expected_opportunities=expected,
            actual_available=result['actual_available'],messages_available=result['messages_available']))
        events.extend(result['events']);roots.extend(result['roots']);pairs.extend(result['platform_pairs'])
    evaluation = dict(manifest,days=days)
    write(directory/'events.json.gz',events)
    write(directory/'evaluation_manifest.json',evaluation)
    summary = calibration(events,evaluation)
    statuses = Counter(pair['status'] for pair in pairs)
    contradictions = statuses['planned_pair_prohibited']+statuses['fixed_mct_differs_from_planned_pair']
    summary['platform_verification'] = dict(status='fail' if contradictions else 'unresolved',
        pairs=len(pairs),status_counts=dict(statuses),actual_platform_pairs_verified=0,
        reason='Static planned platform rules do not verify actual platforms or realized walking time.')
    summary['source_diagnostics'] = dict(roots=len(roots),
        initial_state_statuses=dict(Counter(root['initial_state']['status'] for root in roots)),
        roots_without_opportunities=[root['root_index'] for root in roots if not root['registered']['rows']],
        candidate_pair_statuses=dict(Counter(pair['status'] for root in roots for pair in root['registered']['candidate_pairs'])),
        event_model_statuses=dict(Counter(row['model']['status'] for row in events)),
        event_actual_statuses=dict(Counter(row['actual']['status'] for row in events)))
    summary.update(reconstruction_admitted=False,
        scope='Original four-partition, incident/day weighting and 20000 whole-day bootstrap gate. These natural discovery dates are now exposed; successful calibration alone is not G2.')
    write(directory/'calibration_summary.json',summary)
    write(directory/'platform_pairs.json.gz',pairs)
    write(directory/'exposure_record.json',dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        dates=manifest['selected_dates'],use='Natural control and event calibration outcomes inspected; no longer unused dates.'))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('day','final'))
    parser.add_argument('--directory',type=Path,default=HERE)
    parser.add_argument('--date')
    args = parser.parse_args()
    if args.stage=='day':
        evaluate_day(args.directory,args.date)
    else:
        finalize(args.directory)
