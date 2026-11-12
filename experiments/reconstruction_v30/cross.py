"""Read-only recomputation of the frozen model-by-policy evidence."""

from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
LAWS = ('original20', 'route_empirical20')


def gain_summary(rows):
    gains = [r['outcomes']['one_step_nominal']['loss']-
             r['outcomes']['single_target_nominal']['loss'] for r in rows]
    return dict(n=len(gains), total_gain=sum(gains), mean_gain=mean(gains),
        wins=sum(g > 0 for g in gains), harms=sum(g < 0 for g in gains),
        ties=sum(g == 0 for g in gains))


def main():
    with gzip.open(HERE.parent/'reconstruction_v25/results.json.gz', 'rt') as handle:
        old = {r['root_id']: r for r in json.load(handle)['results']}
    with gzip.open(HERE.parent/'reconstruction_v28/results.json.gz', 'rt') as handle:
        rows = json.load(handle)['results']
    origins = {law: [r for r in rows if r['law'] == law and r['root_kind'] == 'origin'] for law in LAWS}
    root_ids = [{r['root_id'] for r in group} for group in origins.values()]
    if root_ids[0] != root_ids[1] or len(root_ids[0]) != 48:
        raise AssertionError('Model cross comparison must use the same 48 frozen origins')
    reproduced = 0
    for law, group in origins.items():
        old_policy = 'original_full_nominal' if law == 'original20' else 'empirical_full_nominal'
        for row in group:
            before = old[row['root_id']]['outcomes'][old_policy]
            after = row['outcomes']['one_step_nominal']
            if any(before[key] != after[key] for key in ('action', 'tail', 'loss', 'arrived', 'status')):
                raise AssertionError('V25 one-step behavior changed in V28')
            reproduced += 1
    matrix, blocks, concentration = {}, {}, {}
    for law, group in origins.items():
        matrix[law] = {policy: dict(n=len(group), mean_loss=mean(r['outcomes'][policy]['loss'] for r in group),
                            arrived=sum(r['outcomes'][policy]['arrived'] for r in group))
            for policy in ('one_step_nominal', 'single_target_nominal')}
        by_block = defaultdict(list)
        for row in group:
            by_block[row['day'], row['root_id'].split('_')[1]].append(row)
        blocks[law] = {f'{day}_od{od}': gain_summary(values)
                      for (day, od), values in sorted(by_block.items())}
        all_summary = gain_summary(group)
        construction = gain_summary([r for r in group if r['construction']])
        remaining = gain_summary([r for r in group if not r['construction']])
        concentration[law] = dict(all=all_summary, construction=construction, remaining=remaining,
            construction_fraction_of_net_gain=construction['total_gain']/all_summary['total_gain'])
    model_gain = {policy: matrix['original20'][policy]['mean_loss']-
        matrix['route_empirical20'][policy]['mean_loss'] for policy in matrix['original20']}
    v29 = json.loads((HERE.parent/'reconstruction_v29/results.json').read_text())['results']
    support = {}
    for law in LAWS:
        values = [r for r in v29 if r['law'] == law]
        supports = [r['scores']['support'] for r in values]
        support[law] = dict(n=len(values),
            maximum_cancel_mass=max(s['cancellation_mass'] for s in supports),
            maximum_before_entry_mass=max(s['before_entry_mass'] for s in supports),
            minimum_legal_future_probability=min(s['legal_future_probability'] for s in supports),
            maximum_legal_future_probability=max(s['legal_future_probability'] for s in supports),
            mean_integrated_wait_gain=mean(-r['scores']['integrated']['wait_minus_board'] for r in values))
    summary = dict(sources=['reconstruction_v25/results.json.gz', 'reconstruction_v28/results.json.gz',
                           'reconstruction_v29/results.json'],
        scope='Frozen archive arithmetic, no new policy replay, model fitting, or independent confirmation.',
        origin_model_policy_matrix=matrix, model_gain_at_fixed_policy=model_gain,
        interaction_gain=model_gain['single_target_nominal']-model_gain['one_step_nominal'],
        date_od_gain=blocks, concentration=concentration,
        v25_one_step_outcomes_reproduced=reproduced, v29_target_support=support,
        v29_model_difference_in_mean_integrated_wait_gain=
            support['route_empirical20']['mean_integrated_wait_gain']-
            support['original20']['mean_integrated_wait_gain'])
    (HERE/'cross_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(matrix=matrix, interaction=summary['interaction_gain'],
        primary_concentration=concentration['route_empirical20'], v29_support=support), indent=2))


if __name__ == '__main__':
    main()
