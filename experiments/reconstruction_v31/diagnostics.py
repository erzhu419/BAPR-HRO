"""Read saved V30/V31 evidence without fitting or predicting again."""

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent


def read(version):
    with gzip.open(HERE.parent/f'reconstruction_{version}'/'results.json.gz', 'rt') as handle:
        return json.load(handle)


def action(gain):
    return None if gain is None else 'board_current' if gain <= 1e-9 else 'wait_target'


def pair_summary(rows):
    compared = [row for row in rows if row['baseline_action'] is not None and row['candidate_action'] is not None]
    return dict(rows=len(rows), no_target=sum(row['target_cid'] is None for row in rows),
        supported_comparisons=len(compared), ranking_flips=sum(row['ranking_flip'] for row in compared),
        baseline_actions=dict(Counter(row['baseline_action'] for row in compared)),
        candidate_actions=dict(Counter(row['candidate_action'] for row in compared)),
        mean_baseline_gain=mean(row['baseline_gain'] for row in compared) if compared else None,
        mean_candidate_gain=mean(row['candidate_gain'] for row in compared) if compared else None)


def crps_summary(rows):
    scored = [row for row in rows if row['delta_crps'] is not None]
    return dict(rows=len(rows), scored=len(scored), missing=len(rows)-len(scored),
        improved=sum(row['delta_crps'] < -1e-9 for row in scored),
        worsened=sum(row['delta_crps'] > 1e-9 for row in scored),
        tied=sum(abs(row['delta_crps']) <= 1e-9 for row in scored),
        baseline_crps=mean(row['baseline_crps'] for row in scored) if scored else None,
        candidate_crps=mean(row['candidate_crps'] for row in scored) if scored else None,
        mean_delta_crps=mean(row['delta_crps'] for row in scored) if scored else None)


def policy_comparison(old, new):
    old_rows = {row['root_id']: row for row in old['results'] if row['law'] == 'route_empirical20'}
    fields = ('action', 'tail', 'loss', 'arrived', 'status')

    def trace(outcome):
        return [dict(clock=row['clock'], visible=sorted(row['visible']), seen=sorted(row['seen']),
                     choice=row['choice']) for row in outcome['trace']]

    records = []
    for row in new['policies']:
        for policy in ('one_step_nominal', 'single_target_nominal'):
            baseline, candidate = old_rows[row['root_id']]['outcomes'][policy], row['outcomes'][policy]
            a, b = ({key: outcome[key] for key in fields} for outcome in (baseline, candidate))
            ta, tb = trace(baseline), trace(candidate)
            records.append(dict(root_id=row['root_id'], day=row['day'], policy=policy,
                baseline=a, candidate=b, outcome_equal=a == b,
                baseline_trace=ta, candidate_trace=tb, trace_equal=ta == tb,
                paired_events=min(len(ta), len(tb)),
                equal_events=sum(x == y for x, y in zip(ta, tb)),
                baseline_events=len(ta), candidate_events=len(tb)))
    by_policy = {}
    for policy in ('one_step_nominal', 'single_target_nominal'):
        rows = [r for r in records if r['policy'] == policy]
        by_policy[policy] = dict(roots=len(rows), equal_outcomes=sum(r['outcome_equal'] for r in rows),
            equal_complete_traces=sum(r['trace_equal'] for r in rows),
            equal_events=sum(r['equal_events'] for r in rows),
            baseline_events=sum(r['baseline_events'] for r in rows),
            candidate_events=sum(r['candidate_events'] for r in rows))
    return dict(fields=list(fields), trace_fields=['clock', 'visible', 'seen', 'choice'],
        scope='Public sets canonicalized by sorting; score receipts and wall times are not compared.',
        by_policy=by_policy, rows=records)


def main():
    old, new, replay = read('v30'), read('v31'), read('v28')
    original = {(r['history_id'], r['role'], r['cid']): r for r in old['forecasts']
                if r['law'] == 'route_empirical20'}
    forecasts = {(r['history_id'], r['role'], r['cid']): r for r in new['forecasts']}
    pairs = []
    for pair in old['pairs']:
        board = forecasts[pair['history_id'], 'board', pair['current_cid']]
        target = None if pair['target_cid'] is None else forecasts[pair['history_id'], 'target', pair['target_cid']]
        supported = target is not None and all(r['forecast']['status'] == 'supported' for r in (board, target))
        gain = board['forecast']['cost']-target['forecast']['cost'] if supported else None
        baseline = pair['by_law']['route_empirical20']['integrated_gain']
        before, after = action(baseline), action(gain)
        pairs.append(dict(history_id=pair['history_id'], root_id=pair['root_id'], day=pair['day'],
            tags=pair['tags'], selectors=pair['selectors'], current_cid=pair['current_cid'], target_cid=pair['target_cid'],
            status='no_target' if target is None else 'supported' if supported else 'unsupported',
            baseline_gain=baseline, candidate_gain=gain,
            gain_change=None if gain is None or baseline is None else gain-baseline,
            baseline_action=before, candidate_action=after,
            ranking_flip=None if before is None or after is None else before != after,
            actual_gain=pair['actual_gain'], actual_board=pair['actual_board'], actual_target=pair['actual_target']))
    v29 = [row for row in pairs if 'v29_frozen' in row['selectors']]
    actual_pattern = dict(cases=len(v29), wins=sum(r['actual_gain'] > 0 for r in v29),
                          losses=sum(r['actual_gain'] < 0 for r in v29))
    if actual_pattern != dict(cases=6, wins=2, losses=4):
        raise AssertionError('V29 two-win/four-loss fixed-action evidence changed')
    scored_rows, projection_rows = [], []
    for key, row in forecasts.items():
        source = original[key]
        if 'first_offer' in row['tags']:
            before, after = row['baseline_scores']['final_cost'], row['scores'].get('final_cost')
            scored_rows.append(dict(history_id=row['history_id'], root_id=row['root_id'], day=row['day'],
                role=row['role'], cid=row['cid'], route=source['route'], actual_status=row['actual']['status'],
                baseline_crps=None if before is None else before['crps'],
                candidate_crps=None if after is None else after['crps'],
                delta_crps=None if before is None or after is None else after['crps']-before['crps']))
        if row['forecast']['status'] != 'supported':
            continue
        for name in ('first_arrival', 'final_arrival'):
            distribution = row['forecast']['distributions'][name]
            early = [(t, p) for t, p in zip(distribution['values'], distribution['probabilities'])
                     if t < source['clock'] and p > 0]
            if early:
                projection_rows.append(dict(history_id=row['history_id'], role=row['role'], cid=row['cid'],
                    route=source['route'], distribution=name, method=row['forecast']['method'],
                    public_clock=source['clock'], earliest_arrival=min(t for t, _ in early),
                    mass_before_public_clock=sum(p for _, p in early)))
    days = sorted({r['day'] for r in scored_rows})
    route7 = [r for r in scored_rows if r['route'] == '91-7-G-j23-1']
    output = dict(scope='Development attribution of saved forecasts and policy replay; fixed-action ranking is not a new policy value.',
        sources=['experiments/reconstruction_v30/results.json.gz', 'experiments/reconstruction_v31/results.json.gz',
                 'experiments/reconstruction_v28/results.json.gz'],
        gain_definition='Current BOARD minus fixed-target WAIT cost; positive recommends target waiting.',
        crps_definition='Candidate minus baseline_empirical complete-cost CRPS; negative is improvement.',
        fixed_pairs=dict(first_offer=pair_summary([r for r in pairs if 'first_offer' in r['tags']]),
            v29=pair_summary(v29), v29_actual_pattern=actual_pattern, rows=pairs),
        first_offer_cost_crps=dict(total=crps_summary(scored_rows),
            by_day_role={day: {role: crps_summary([r for r in scored_rows if r['day'] == day and r['role'] == role])
                              for role in ('board', 'target')} for day in days},
            route7=crps_summary(route7), route7_worsened=[r for r in route7 if r['delta_crps'] is not None and r['delta_crps'] > 1e-9],
            rows=scored_rows),
        pre_public_arrival_mass=dict(checked_supported_forecasts=sum(r['forecast']['status'] == 'supported' for r in forecasts.values()),
            affected_distributions=len(projection_rows), affected_forecasts=len({(r['history_id'], r['role'], r['cid']) for r in projection_rows}),
            maximum_mass=max((r['mass_before_public_clock'] for r in projection_rows), default=0.), rows=projection_rows),
        policy_parity=policy_comparison(replay, new))
    (HERE/'diagnostics.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(first_offer=output['fixed_pairs']['first_offer'], v29=output['fixed_pairs']['v29'],
        v29_gains=[r['candidate_gain'] for r in v29], cost_crps=output['first_offer_cost_crps']['total'],
        route7=output['first_offer_cost_crps']['route7'],
        pre_public={k:v for k,v in output['pre_public_arrival_mass'].items() if k != 'rows'},
        policy_parity=output['policy_parity']['by_policy']), indent=2))


if __name__ == '__main__':
    main()
