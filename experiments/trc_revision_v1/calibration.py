"""Contract-specific first-onboard calibration with explicitly named weights."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.decision_calibration_v2.calibration_summary import (
    PARTITIONS, average, described, event_bounds, incidents, unknown_day,
)
from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.transfer_events import (
    event_probabilities, label_event, public_event_features,
)
from experiments.natural_gap_v2.events import root_events
from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v41.physical_posterior import restore_physical_snapshot
from experiments.bounded_gate_5d_v1.pair_contract import DayPlatformIndex

HERE = Path(__file__).resolve().parent


def conditional_summary(events, roots, dates, *, replicates=20000, seed=20261116):
    """Unreached roots stay in availability; reached unknown events keep bounds."""
    conditional = [r for r in events if r['initial_state']['status']=='observed']
    by_day = {day:[r for r in conditional if r['day']==day] for day in dates}
    result = {}
    for weighting in ('equal_incident_then_equal_date','equal_event_then_equal_date'):
        values, daily = [], []
        for day, rows in by_day.items():
            bounds = [event_bounds(r) for r in rows]
            groups = incidents(rows) if weighting.startswith('equal_incident') else [list(range(len(rows)))]
            components = [average([bounds[i] for i in group]) for group in groups if group]
            value = average(components) if components else unknown_day()
            values.append(value)
            daily.append(dict(day=day,reached_events=len(rows),components=len(components),
                model_resolved=sum(r['model_resolved'] for r in bounds),
                label_resolved=sum(r['label_resolved'] for r in bounds),**described(value)))
        matrix = np.array([v['residual'] for v in values])
        draws = np.random.default_rng(seed).integers(0,len(dates),size=(replicates,len(dates)))
        boot = matrix[draws].mean(axis=1)
        result[weighting] = dict(**described(average(values)),days=daily,
            bootstrap=dict(replicates=replicates,seed=seed,unit='whole date',
                marginal95={k:[float(np.quantile(boot[:,:,0][:,i],.025)),
                    float(np.quantile(boot[:,:,1][:,i],.975))] for i,k in enumerate(PARTITIONS)},
                simultaneous4class95={k:[float(np.quantile(boot[:,:,0][:,i],.00625)),
                    float(np.quantile(boot[:,:,1][:,i],.99375))] for i,k in enumerate(PARTITIONS)}))
    return dict(registered_roots=len(roots),registered_events=len(events),
        initial_state_statuses=dict(Counter(r['initial_state']['status'] for r in roots)),
        roots_without_opportunities=[r['root_index'] for r in roots if not r['registered']['rows']],
        reached_events=len(conditional),unreached_events=len(events)-len(conditional),
        reached_model_statuses=dict(Counter(r['model']['status'] for r in conditional)),
        reached_label_statuses=dict(Counter(r['actual']['status'] for r in conditional)),
        weightings=result,independent_confirmation=False,
        scope='Already exposed development dates. Conditional-onboard calibration; all reached unknown events retained. Availability denominator includes roots with no opportunity. No old acceptance verdict is overwritten.')


def pair_events(graph, metadata, values, root, physical, emission, platform_index, tau):
    """Change both event prediction and label to the same explicit pair contract."""
    state = root['initial_state']
    history = model = None
    if state['status']=='observed' and root['predecision_source_available']:
        model,parameters = setup(graph,physical)
        history = restore_physical_snapshot(model,parameters,emission,root['public_history'],state['clock'])
    updated, cache = [], {}
    for row in values:
        event = dict(row['event'])
        up = platform_index.resolve_pair(event['current_cid'],event['upstream_cid'],tau_same_platform=tau)
        down = platform_index.resolve_pair(event['source_arrival_cid'],event['hub_cid'],tau_same_platform=tau)
        event.update(upstream_mct=up['contract_mct_seconds'],downstream_mct=down['contract_mct_seconds'])
        if state['status']=='observed':
            actual = label_event(metadata,event,root['deadline'])
            features = dict(row['public_features'], **public_event_features(graph,event,root['public_history']))
            features['groups'] = dict(row['public_features'].get('groups',{}), **features['groups'])
            features['mct_source'] = 'static_pair_or_declared_same_platform_assumption'
            if history is None:
                prediction = dict(status='probability_unresolved',reason='predecision_message_source_unavailable')
            else:
                try:
                    prediction = event_probabilities(model,history,event,root['deadline'],cache)
                except UnsupportedHistory as error:
                    prediction = dict(status='probability_unresolved',reason=str(error))
        else:
            actual,prediction = row['actual'],row['model']
            difference = event['downstream_mct']-event['upstream_mct']
            features = dict(row['public_features'],mct_source='static_pair_or_declared_same_platform_assumption')
            features['groups'] = dict(features['groups'],mct_difference='nonpositive' if difference<=0
                else 'at_most_180s' if difference<=180 else 'over_180s')
        updated.append(dict(row,event=event,actual=actual,model=prediction,public_features=features,
            groups=features['groups'],mct_contract='platform_pair_static',
            transfer_rules=dict(upstream=up,downstream=down)))
    return updated


def evaluate(directory, source):
    directory,source = Path(directory),Path(source)
    if (directory/'calibration_summary.json').exists():
        raise ValueError('Development calibration already recorded; preserve the first result')
    contract,selection = read(directory/'contract.json'),read(source/'selection.json')['selected']
    manifest,data = read(source/'date_manifest.json'),read(source/'data.json.gz')
    physical,emission = read(source/'physical_fit.json'),read(source/'repaired_fit.json')
    events = {'station_max':[],'platform_pair':[]}
    roots = {'station_max':[],'platform_pair':[]}
    for day in contract['control_dates']:
        graph,_,stop_ids,_,metadata,_ = context(data,day)
        version = next(r['static_version'] for r in manifest['days'] if r['day']==day)
        static = source/'inputs'/f'static_{version}'/'data'
        index = DayPlatformIndex(metadata,static/'zurich_gtfs',static/'swiss_gtfs',
            {str(bpuic):graph.stops[i].min_transfer_time for bpuic,i in stop_ids.items()})
        messages = read(source/'messages'/f'{day}.json.gz')['summary']
        daily = {key:dict(events=[],roots=[]) for key in events}
        for i,selected in enumerate(selection):
            if selected['day']!=day:
                continue
            menu = read(source/'menus'/f'{i:03}.json.gz')
            for key,label in [('station_max','cheap_statmax_ideal'),('platform_pair','cheap_plat_ideal')]:
                receipt = read(directory/'roots'/f'{i:03}_{label}.json.gz')
                values,root = root_events(graph,metadata,menu,receipt,i,physical,emission,messages)
                root.update(deadline=selected['deadline'],saved_control=f'roots/{i:03}_{label}.json.gz')
                if key=='platform_pair':
                    values = pair_events(graph,metadata,values,root,physical,emission,index,
                        contract['mct_contracts']['platform_pair']['tau_same_platform_default_seconds'])
                else:
                    values = [dict(r,mct_contract='station_max') for r in values]
                events[key].extend(values); roots[key].append(root)
                daily[key]['events'].extend(values);daily[key]['roots'].append(root)
        write(directory/'events'/f'{day}.json.gz',daily)
    result = dict(stage='trc_revision_v1',contracts={key:conditional_summary(
        events[key],roots[key],contract['control_dates']) for key in events},
        scope='Both MCT contracts evaluated on their matching saved ideal first-decision histories. No control replay, extra sampled worlds or new model fit; readiness probability excludes compute latency.')
    write(directory/'calibration_summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=HERE)
    p.add_argument('--source-directory',type=Path,required=True)
    a=p.parse_args();evaluate(a.directory,a.source_directory)
