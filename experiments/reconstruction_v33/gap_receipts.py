"""Recover public traces only for differences in the completed paired panel.

This post-hoc diagnostic does not select a new performance panel, alter the
original results, or add evaluation worlds. The original full bank shape and
seed are retained because changing the bank size changes subsequent trip
draws. Complete receipts are reused on later invocations.
"""

import gzip
import json
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.execute import Executor, DrawnWorld
from experiments.reconstruction_v33.history import ConditionalEngine
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.planner import Planner
from experiments.reconstruction_v33.run import setup_day


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


def action_signature(event):
    return event['kind'], event.get('cid', event.get('stop')), event['clock'], event.get('choice')


def first_action_difference(reference, compared):
    left, right = reference['trace'], compared['trace']
    common = 0
    for a, b in zip(left, right):
        if action_signature(a) != action_signature(b):
            break
        common += 1
    return dict(common_action_prefix=common,
        primary=left[common] if common < len(left) else None,
        compared=right[common] if common < len(right) else None)


def attach_public_histories(result, ride, model, parameters, world):
    """Recover observed intervening stops on the already executed path.

    This does not call a controller again. The stored actions determine the
    traversed prefixes; only their already reached event slots are exposed.
    """
    history = ConditionalEngine(model, parameters)
    current_ride, platform, position = ride, None, 0
    observations = []
    for event in result['trace']:
        if event['kind'] in ('onboard', 'destination'):
            until = current_ride.path.index(event['cid'])
            for cid in current_ride.path[position:until+1]:
                for kind in ('departure', 'arrival'):
                    clock, canceled, _ = world.event(cid, kind)
                    if canceled:
                        raise AssertionError('A saved reached prefix contains a canceled observation')
                    history.observe(cid, kind, clock)
                    observations.append(dict(source='ridden_vehicle', cid=cid, kind=kind, clock=clock))
            position = until+1
            if event['kind'] == 'onboard' and event['choice'] == 'alight':
                platform = current_ride.exits[event['cid']]
                current_ride = None
        elif event['kind'] == 'platform':
            if platform.stop != event['stop']:
                raise AssertionError('The stored platform does not follow the chosen alighting exit')
            observed = {int(cid): clock for cid, clock in event['observed'].items()}
            history.observe_platform(tuple(platform.rides), event['entry'], event['clock'], observed)
            observations.append(dict(source='local_platform', entry=event['entry'], clock=event['clock'],
                cids=list(platform.rides), observed=observed))
            if event['choice'] is not None:
                current_ride, position = platform.rides[event['choice']], 0
                platform = None
        event['public_history'] = history.public_state()
        event['public_observation_count'] = len(observations)
    result['public_observations'] = observations


def main():
    began = perf_counter()
    protocol = json.loads((HERE/'protocol.json').read_text())
    selection = json.loads((HERE.parent/'reconstruction_v32/selection_full_hubs.json').read_text())
    primary = protocol['primary']
    chosen, archives = {}, {}
    for path in sorted((HERE/'roots').glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            row = json.load(handle)
        index = row['root_index']
        archives[index] = row
        for method, outcomes in row['sampled'].items():
            for world, (reference, candidate) in enumerate(zip(row['sampled'][primary], outcomes)):
                if candidate['loss'] != reference['loss']:
                    chosen.setdefault((index, world), set()).add(method)
    output = HERE/'gap_receipts.json.gz'
    if output.exists():
        with gzip.open(output, 'rt') as handle:
            existing = json.load(handle)
    else:
        existing = dict(rows=[])
    receipts = {(row['root_index'], row['world']): row for row in existing['rows']}
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    days, banks = {}, {}
    new_replays = 0
    for (index, world), affected in sorted(chosen.items()):
        required = affected | {primary, 'combined_rolling'}
        row = receipts.get((index, world))
        if row is not None and required.issubset(row['outcomes']):
            print('reuse', index, world, flush=True)
        root, archived = selection['selected'][index], archives[index]
        if root['root_id'] != archived['root_id']:
            raise AssertionError('A saved root no longer addresses the frozen selection')
        if root['day'] not in days:
            days[root['day']] = setup_day(data, root['day'], fit, segment)
        graph, _, model, params, _ = days[root['day']]
        if index not in banks:
            ride = root_ride(graph, root)
            bank = TripWorlds(model, params, ride.paths(), archived['evaluation_worlds'],
                              archived['evaluation_seed'])
            banks[index] = ride, bank
        ride, bank = banks[index]
        if row is None:
            row = dict(root_id=root['root_id'], root_index=index, day=root['day'], world=world,
                original_bank_size=archived['evaluation_worlds'],
                original_seed=archived['evaluation_seed'], affected_methods=sorted(affected), outcomes={})
        for name in sorted(required):
            if name in row['outcomes']:
                continue
            method, waiting = protocol['methods'][name]
            planner = Planner(graph, model, params, root['deadline'])
            result = Executor(graph, planner, root['deadline'], method=method, waiting=waiting).run(
                ride, root['start'], ConditionalEngine(model, params), DrawnWorld(bank, world))
            old = archived['sampled'][name][world]
            if (result['loss'], result['status']) != (old['loss'], old['status']):
                raise AssertionError(('Trace reconstruction changed the saved outcome', index, world, name,
                                      result['loss'], old['loss'], result['status'], old['status']))
            result['original_compute_wall'] = old['compute_wall']
            result['original_total_wall'] = old['total_wall']
            row['outcomes'][name] = result
            new_replays += 1
        for result in row['outcomes'].values():
            attach_public_histories(result, ride, model, params, DrawnWorld(bank, world))
        reference = row['outcomes'][primary]
        combined = row['outcomes']['combined_rolling']
        row['comparisons'] = {method: dict(
            compared_loss=row['outcomes'][method]['loss'], primary_loss=reference['loss'],
            gain_from_primary=row['outcomes'][method]['loss']-reference['loss'],
            first_action_difference=first_action_difference(reference, row['outcomes'][method]))
            for method in sorted(affected)}
        for comparison in row['comparisons'].values():
            difference = comparison['first_action_difference']
            comparison['same_public_information_at_divergence'] = (
                difference['primary']['public_history'] == difference['compared']['public_history'])
            if not comparison['same_public_information_at_divergence']:
                raise AssertionError('Compared policies received different information before their first action divergence')
        row['absorbed_by_combined'] = (combined['loss'] == reference['loss'] and
            [action_signature(event) for event in combined['trace']] ==
            [action_signature(event) for event in reference['trace']])
        receipts[index, world] = row
        print(index, world, {name: result['loss'] for name, result in row['outcomes'].items()},
              'combined reproduces primary', row['absorbed_by_combined'], flush=True)
        # Preserve completed rows if a later diagnostic query fails.
        with gzip.open(output, 'wt') as handle:
            json.dump(dict(scope=__doc__, rows=list(receipts.values())), handle)
    rows = [receipts[key] for key in sorted(chosen)]
    totals = {}
    for row in rows:
        for method, comparison in row['comparisons'].items():
            item = totals.setdefault(method, dict(worlds=0, gain_sum=0., first_divergence_types={}))
            item['worlds'] += 1
            item['gain_sum'] += comparison['gain_from_primary']
            event = comparison['first_action_difference']['primary']
            kind = 'no_primary_event' if event is None else event['kind']
            item['first_divergence_types'][kind] = item['first_divergence_types'].get(kind, 0)+1
    original_worlds = sum(row['evaluation_worlds'] for row in archives.values())
    for item in totals.values():
        item['mean_gain_over_original_panel'] = item['gain_sum']/original_worlds
    report = dict(scope=__doc__, original_panel_worlds=original_worlds,
        summary=dict(difference_worlds=len(rows), roots=len({row['root_index'] for row in rows}),
            reconstructed_outcomes=sum(len(row['outcomes']) for row in rows), new_replays_this_run=new_replays,
            all_absorbed_by_combined=all(row['absorbed_by_combined'] for row in rows),
            by_method=totals, wall=perf_counter()-began,
            initial_trace_build_wall=existing.get('summary', {}).get('initial_trace_build_wall',
                existing.get('summary', {}).get('wall', perf_counter()-began))), rows=rows)
    with gzip.open(output, 'wt') as handle:
        json.dump(report, handle)
    compact = []
    for row in rows:
        for method, comparison in row['comparisons'].items():
            difference = comparison['first_action_difference']
            compact.append(dict(root_id=row['root_id'], root_index=row['root_index'], world=row['world'],
                method=method, gain_from_primary=comparison['gain_from_primary'],
                same_public_information=comparison['same_public_information_at_divergence'],
                absorbed_by_combined=row['absorbed_by_combined'],
                first_difference={side: {key: value for key, value in difference[side].items()
                    if key not in ('public_history', 'public_observation_count')}
                    for side in ('primary', 'compared')},
                original_compute_wall={m: result['original_compute_wall'] for m,result in row['outcomes'].items()}))
    (HERE/'gap_receipts.json').write_text(json.dumps(dict(scope=__doc__, summary=report['summary'],
        full_receipts=output.name, cases=compact), indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
