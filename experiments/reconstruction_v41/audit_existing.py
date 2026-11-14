"""Recount existing V40 receipts; no model evaluation or new random worlds."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
from statistics import mean


HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'reconstruction_v40'


def read(path):
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
        return json.load(handle)


def failure_census():
    states, failures, advances, examples = [], Counter(), [], []
    prefixes = {128:dict(unresolved=0, widths=[], crossing=0),
                512:dict(unresolved=0, widths=[], crossing=0)}
    for path in sorted((OLD/'banks').glob('*_frontier.json.gz')):
        bank = read(path)
        counts = Counter()
        for arm, rows in bank['rows'].items():
            for row in rows:
                if row['loss'] is not None:
                    continue
                status = row['status']
                parsed = re.fullmatch(r'projection_incompatible:(\d+):departure=([\d.]+)<known_previous=([\d.]+)', status)
                counts['unresolved_arms'] += 1
                failures['unresolved_arms'] += 1
                failures['continue_action'] += bank['actions'][arm] == 'continue'
                failures['empty_trace'] += not row['trace']
                failures['departure_before_known_previous'] += parsed is not None
                if parsed:
                    advances.append(float(parsed[3])-float(parsed[2]))
                if len(examples) < 5:
                    examples.append(dict(source=str(path.relative_to(HERE.parent.parent)),
                        arm=arm, action=bank['actions'][arm], world_index=row['world_index'], status=status))
        for n, stats in prefixes.items():
            incomplete = sum(a['loss'] is None or b['loss'] is None for a, b in zip(
                bank['rows']['alternative'][:n], bank['rows']['base'][:n]))
            interval = next(x['full_bank_mean_gain_interval'] for x in bank['prefixes'] if x['count'] == n)
            stats['unresolved'] += incomplete
            stats['widths'].append(interval[1]-interval[0])
            stats['crossing'] += interval[0] <= 0 <= interval[1]
        states.append(dict(source=bank['source'], root_index=bank['state']['root_index'],
            clock=bank['state']['clock'], unresolved_pairs=bank['paired']['unresolved_pairs'], **counts))
    return dict(states=len(states), pairs_512=len(states)*512,
        affected_states=sum(x['unresolved_pairs'] > 0 for x in states), failures=dict(failures),
        advance_seconds=dict(minimum=min(advances), mean=mean(advances), maximum=max(advances)),
        prefixes={str(n):dict(pairs=len(states)*n, unresolved=x['unresolved'],
            fraction=x['unresolved']/(len(states)*n), mean_missing_interval_width=mean(x['widths']),
            intervals_crossing_zero=x['crossing']) for n,x in prefixes.items()},
        examples=examples, by_state=states)


def concentration(summary, selection):
    base = {r['root_index']:r['outcome']['loss'] for r in
            summary['root_receipts']['cheap_frontier__combined_rolling']}
    result = {}
    for count in (128, 512):
        grouped = {key:defaultdict(list) for key in ('day', 'od', 'day_od')}
        changed = []
        for row in summary['root_receipts'][f'frontier__pairs{count}']:
            i, day = row['root_index'], row['day']
            root = selection[i]
            od = root['origin_name']+' → '+root['goal_name']
            gain = base[i]-row['outcome']['loss']
            for field, key in [('day',day), ('od',od), ('day_od',day+' | '+od)]:
                grouped[field][key].append(gain)
            if gain:
                changed.append(dict(root_index=i, day=day, od=od, gain_seconds=gain))
        result[str(count)] = dict(changed_roots=changed, **{field:{key:dict(roots=len(gs),
            wins=sum(g>0 for g in gs), harms=sum(g<0 for g in gs), ties=sum(g==0 for g in gs),
            total_gain=sum(gs), mean_gain=mean(gs), positive_gain=sum(g for g in gs if g>0))
            for key,gs in sorted(groups.items())} for field,groups in grouped.items()})
    return result


def root25():
    cheap = read(OLD/'roots/025_cheap_frontier__combined_rolling.json.gz')
    rollout = read(OLD/'roots/025_frontier__pairs128.json.gz')
    first = cheap['outcome']['trace'][0]
    def trace(row):
        return [{key: value for key,value in event.items()
                 if key in ('kind','cid','stop','entry','ready','clock','choice','observed','visible')}
                for event in row['outcome']['trace']]
    return dict(day=cheap['day'], clock=first['clock'], arrival_cid=first['cid'],
        cheap_choice=first['choice'], alight_remaining=first['receipt']['alight']-first['clock'],
        continue_remaining=first['receipt']['continuation']-first['clock'],
        cheap_loss=cheap['outcome']['loss'], rollout_loss=rollout['outcome']['loss'],
        gain=cheap['outcome']['loss']-rollout['outcome']['loss'],
        cheap_trace=trace(cheap), rollout_trace=trace(rollout),
        inference='Saved code and traces establish a first-offer valuation / selective-wait execution mismatch; 468 seconds is an observed path difference, not an isolated causal effect.')


if __name__ == '__main__':
    summary = read(OLD/'summary.json')
    selection = read(HERE.parent/'reconstruction_v34/selection.json')['selected']
    result = dict(source='Frozen V40 receipts, current local files; no new MC or fitting.',
        unavailable_attachments=['V7 linked 15-page Word', 'V7 linked independent-review package'],
        failure_census=failure_census(), concentration=concentration(summary,selection), root25=root25())
    path = HERE/'existing_mechanisms.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(path)
