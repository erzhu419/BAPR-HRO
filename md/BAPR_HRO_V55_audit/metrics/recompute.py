#!/usr/bin/env python3
"""Read-only numerical audit of saved V51/V53--V55 receipts; no solver calls.

Usage: BAPR_HRO_REPO=/path/to/BAPR-HRO python recompute.py
Outputs are placed beside this script. Uses only the Python standard library.
"""
import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('BAPR_HRO_REPO', HERE.parent.parent / 'repos' / 'BAPR-HRO-a2b82be9')).resolve()
COMMIT = 'a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491'
TOL = 1e-8
MODES = ('optimal', 'cheap_p0', 'cheap_working', 'one_step_p0', 'two_step_p0')
KEY_ROOTS = (4, 6, 18, 32, 57, 62, 63)
SOURCES = {
    'v53_compressed': 'experiments/reconstruction_v53/natural',
    'v53_impact': 'experiments/reconstruction_v53/natural_priority',
    'v54_reflow': 'experiments/reconstruction_v54/natural',
    'v55_ordinary': 'experiments/reconstruction_v55/natural',
    'v55_strong': 'experiments/reconstruction_v55/priority/natural',
}


def read(path):
    blob = path.read_bytes()
    return json.loads(gzip.decompress(blob) if path.suffix == '.gz' else blob)


def w(r):
    return r['value']['upper'] - r['value']['lower']


def cpu(r):
    return r['complete_cost']['process_cpu']


def action(a):
    return 'wait' if a is None else str(a)


def digest_paths(paths):
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(REPO)).encode())
        h.update(b'\0')
        h.update(path.read_bytes())
    return {'files': len(paths), 'sha256_path_and_bytes': h.hexdigest()}


def aggregate(rows):
    counter_names = sorted(set().union(*(r['counters'] for r in rows)))
    return {
        'n': len(rows), 'mean_width': mean(map(w, rows)),
        'mean_cpu': mean(map(cpu, rows)),
        'mean_wall': mean(r['complete_cost']['wall'] for r in rows),
        'complete': sum(r['value']['complete'] for r in rows),
        'action_certificates': sum(r['certified_action'] is not None for r in rows),
        'mean_counters': {k: mean(r['counters'].get(k, 0) for r in rows) for k in counter_names},
        'terminal_dp_admitted_queries': sum(r['counters'].get('terminal_dp_admitted', 0) > 0 for r in rows),
        'terminal_dp_nonzero_state_queries': sum(r['counters'].get('terminal_dp_states', 0) > 0 for r in rows),
        'budget_flag_queries': sum(any(k.startswith('budget') and v for k, v in r['reasons'].items()) for r in rows),
        'impact_cutoff_queries': sum(r['reasons'].get('impact_cutoff', 0) > 0 for r in rows),
    }


def initial_bounds(r):
    b = r['root_bounds']
    lo = min(x['lower'] for x in b.values()) - r['clock']
    hi = min(x['upper'] for x in b.values()) - r['clock']
    cert = [a for a, x in b.items() if len(b) > 1 and x['upper'] < min(y['lower'] for aa, y in b.items() if aa != a) - 1e-9]
    assert len(cert) <= 1
    dl, du = r['value']['lower'] - lo, hi - r['value']['upper']
    assert dl >= -1e-7 and du >= -1e-7
    return {
        'query': r['query_index'], 'root': r['root_index'],
        'initial_lower': lo, 'initial_upper': hi,
        'final_lower': r['value']['lower'], 'final_upper': r['value']['upper'],
        'initial_width': hi - lo, 'final_width': w(r),
        'lower_lift': dl, 'upper_drop': du,
        'root_value_entirely_initial': abs(dl) <= TOL and abs(du) <= TOL,
        'initial_action_certificate': cert[0] if cert else None,
        'final_action_certificate': r['certified_action'],
        'cpu': cpu(r),
    }


def compare(left, right):
    rows = []
    for key, a in sorted(left.items()):
        b = right[key]
        for name in ('query_index', 'root_index', 'clock', 'deadline', 'mode', 'budget', 'source'):
            assert a[name] == b[name], (key, name, a[name], b[name])
        assert set(a['actions']) == set(b['actions'])
        assert max(a['value']['lower'], b['value']['lower']) <= min(a['value']['upper'], b['value']['upper']) + TOL
        if a['certified_action'] is not None and b['certified_action'] is not None:
            assert action(a['certified_action']) == action(b['certified_action'])
        dw, dc = w(b) - w(a), cpu(b) - cpu(a)
        rows.append({
            'query': key[0], 'root': b['root_index'], 'budget': key[1], 'mode': key[2],
            'old_width': w(a), 'new_width': w(b), 'width_change': dw,
            'old_cpu': cpu(a), 'new_cpu': cpu(b), 'cpu_change': dc,
            'width_relation': 'narrower' if dw < -TOL else 'wider' if dw > TOL else 'equal',
            'new_no_worse_both': dw <= TOL and dc <= TOL,
            'old_no_worse_both': dw >= -TOL and dc >= -TOL,
            'terminal_dp_admitted': b['counters'].get('terminal_dp_admitted', 0),
            'terminal_dp_states': b['counters'].get('terminal_dp_states', 0),
        })
    return rows


def joint_values(receipts):
    # Independent interval propagation through the declared policy-class order:
    # V* <= V2 <= V1 <= P0, and V* <= W. A root certificate may identify V1=P0.
    # The two paid P0 / one-step queries share forced-root continuations at
    # onboard/platform decision roots. Intersect those action targets first;
    # do NOT intersect their different overall policy values directly.
    receipts = {m: dict(r) for m, r in receipts.items()}
    p0, one = receipts['cheap_p0'], receipts['one_step_p0']
    assert set(p0['actions']) == set(one['actions'])
    common = {a: {'lower': max(p0['actions'][a]['lower'], one['actions'][a]['lower']),
                  'upper': min(p0['actions'][a]['upper'], one['actions'][a]['upper'])}
              for a in p0['actions']}
    assert all(v['lower'] <= v['upper'] + TOL for v in common.values())
    best = min(common, key=lambda a: common[a]['upper'])
    cert = best if len(common) > 1 and common[best]['upper'] < min(v['lower'] for a, v in common.items() if a != best) - 1e-9 else None
    for m in ('cheap_p0', 'one_step_p0'):
        receipts[m]['actions'] = common
        receipts[m]['certified_action'] = cert
    receipts['cheap_p0']['value'] = common[action(p0['cheap_action'])]
    receipts['one_step_p0']['value'] = {'lower': min(v['lower'] for v in common.values()),
                                      'upper': min(v['upper'] for v in common.values())}
    vals = {m: [r['value']['lower'], r['value']['upper']] for m, r in receipts.items()}
    eq = any(receipts[m]['certified_action'] is not None and
             action(receipts[m]['certified_action']) == action(receipts['cheap_p0']['cheap_action'])
             for m in ('cheap_p0', 'one_step_p0'))
    edges = [('optimal', 'two_step_p0'), ('two_step_p0', 'one_step_p0'),
             ('one_step_p0', 'cheap_p0'), ('optimal', 'cheap_working')]
    if eq:
        edges.append(('cheap_p0', 'one_step_p0'))
    for _ in range(6):
        for low, high in edges:
            vals[high][0] = max(vals[high][0], vals[low][0])
            vals[low][1] = min(vals[low][1], vals[high][1])
    assert all(lo <= hi + TOL for lo, hi in vals.values())
    gaps = {}
    for m in ('cheap_p0', 'cheap_working', 'one_step_p0', 'two_step_p0'):
        lo, hi = max(0, vals[m][0] - vals['optimal'][1]), max(0, vals[m][1] - vals['optimal'][0])
        gaps[m] = {'lower': lo, 'upper': hi,
                   'status': 'positive' if lo > TOL else 'zero' if hi <= TOL else 'unresolved'}
    return {'values': vals, 'regrets': gaps, 'P0_equals_one_step': eq,
            'total_cpu': sum(cpu(r) for r in receipts.values())}


def main():
    assert (REPO / 'experiments/reconstruction_v55/RESULTS.md').exists(), REPO
    result = {'commit': COMMIT, 'scope': 'Read-only audit of committed receipts; no new experiments. Seconds. Conditional model targets are not empirical population guarantees.',
              'source_digests': {}, 'aggregate': {}, 'initial_bounds': {}, 'joint_regret': {}}
    banks = {}
    for tag, source in SOURCES.items():
        paths = sorted((REPO / source).glob('*.json.gz'))
        bank = {}
        for p in paths:
            r = read(p)
            if 'budget' not in r or 'query_index' not in r:
                continue
            key = (r['query_index'], r['budget']['name'] if isinstance(r['budget'], dict) else r['budget'], r['mode'])
            assert key not in bank, (tag, key)
            bank[key] = r
            assert r['value']['lower'] <= r['value']['upper'] + TOL
        assert len(bank) == 880, (tag, len(bank))
        banks[tag] = bank
        result['source_digests'][tag] = digest_paths(paths)
        result['aggregate'][tag] = {budget: {mode: aggregate([bank[(q, budget, mode)] for q in range(88)])
                                   for mode in MODES} for budget in ('b2', 'b10')}
        result['initial_bounds'][tag] = {}
        result['joint_regret'][tag] = {}
        for budget in ('b2', 'b10'):
            initial = [initial_bounds(bank[(q, budget, 'optimal')]) for q in range(88)]
            result['initial_bounds'][tag][budget] = {
                'entirely_initial_root_value': sum(x['root_value_entirely_initial'] for x in initial),
                'initial_certificates': sum(x['initial_action_certificate'] is not None for x in initial),
                'final_certificates': sum(x['final_action_certificate'] is not None for x in initial),
                'new_certificates': [{k: x[k] for k in ('query', 'root', 'final_action_certificate')} for x in initial
                                     if x['final_action_certificate'] is not None and x['initial_action_certificate'] is None],
                'tightened_rows': [x for x in initial if not x['root_value_entirely_initial']],
            }
            joint = [joint_values({m: bank[(q, budget, m)] for m in MODES}) for q in range(88)]
            result['joint_regret'][tag][budget] = {
                'mean_total_cpu': mean(x['total_cpu'] for x in joint),
                'P0_equals_one_step': sum(x['P0_equals_one_step'] for x in joint),
                'policy': {m: {
                    'status_counts': dict(Counter(x['regrets'][m]['status'] for x in joint)),
                    'mean_regret_lower': mean(x['regrets'][m]['lower'] for x in joint),
                    'mean_regret_upper': mean(x['regrets'][m]['upper'] for x in joint),
                    'upper_at_most': {str(t): sum(x['regrets'][m]['upper'] <= t + TOL for x in joint) for t in (1, 10, 60, 300)},
                } for m in ('cheap_p0', 'cheap_working', 'one_step_p0', 'two_step_p0')},
            }
    paired = compare(banks['v53_impact'], banks['v55_strong'])
    # Which CPU savings coincide with exhaustion of the new, heterogeneous
    # state counter? Root-value equality is weaker than all-action equality.
    attribution_rows = []
    for q in range(88):
        a, b = banks['v53_impact'][(q, 'b10', 'optimal')], banks['v55_strong'][(q, 'b10', 'optimal')]
        attribution_rows.append({
            'query': q, 'root': b['root_index'],
            'old_states': a['counters']['states'], 'new_states': b['counters']['states'],
            'terminal_dp_states': b['counters'].get('terminal_dp_states', 0),
            'same_value': all(abs(a['value'][k] - b['value'][k]) <= TOL for k in ('lower', 'upper')),
            'same_all_action_intervals': all(abs(a['actions'][act][k] - b['actions'][act][k]) <= TOL
                                             for act in a['actions'] for k in ('lower', 'upper')),
            'same_certificate': a['certified_action'] == b['certified_action'],
            'same_selected_action': a['selected_action'] == b['selected_action'],
            'certificate': b['certified_action'], 'selected_action': b['selected_action'],
            'old_cpu': cpu(a), 'new_cpu': cpu(b), 'saved_cpu': cpu(a) - cpu(b),
            'old_reasons': a['reasons'], 'new_reasons': b['reasons'],
            'old_actions': a['actions'], 'new_actions': b['actions'],
            'source_old': f"{SOURCES['v53_impact']}/{q:03}_b10_optimal.json.gz",
            'source_new': f"{SOURCES['v55_strong']}/{q:03}_b10_optimal.json.gz",
        })
    saved = sum(x['saved_cpu'] for x in attribution_rows)
    attribution = {'total_net_cpu_saved': saved, 'groups': {}, 'rows': attribution_rows}
    predicates = {
        'new_statecap_same_value': lambda x: x['new_states'] >= 20000 and x['same_value'],
        'old_cpu_stop_new_statecap_same_value': lambda x: x['new_states'] >= 20000 > x['old_states'] and x['same_value'],
        'both_statecaps_same_value': lambda x: x['new_states'] >= 20000 and x['old_states'] >= 20000 and x['same_value'],
        'roots32_57': lambda x: x['root'] in (32, 57),
    }
    for name, predicate in predicates.items():
        sub = [x for x in attribution_rows if predicate(x)]
        subtotal = sum(x['saved_cpu'] for x in sub)
        attribution['groups'][name] = {
            'n': len(sub), 'query_indices': [x['query'] for x in sub],
            'net_cpu_saved': subtotal, 'fraction_of_net_cpu_saved': subtotal / saved,
            'same_certificate': sum(x['same_certificate'] for x in sub),
            'same_selected_action': sum(x['same_selected_action'] for x in sub),
            'same_all_action_intervals': sum(x['same_all_action_intervals'] for x in sub),
        }
    (HERE / 'optimal_cpu_attribution.json').write_text(json.dumps(attribution, ensure_ascii=False, indent=2) + '\n')
    result['optimal_cpu_attribution'] = {k: v for k, v in attribution.items() if k != 'rows'}
    result['paired_strong_vs_impact'] = {}
    for budget in ('b2', 'b10'):
        result['paired_strong_vs_impact'][budget] = {}
        for mode in MODES:
            rr = [r for r in paired if r['budget'] == budget and r['mode'] == mode]
            groups = {}
            for label, pred in [('no_terminal_admission', lambda x: not x['terminal_dp_admitted']),
                                ('terminal_admitted', lambda x: bool(x['terminal_dp_admitted'])),
                                ('positive_terminal_states', lambda x: bool(x['terminal_dp_states']))]:
                sub = [r for r in rr if pred(r)]
                groups[label] = {'n': len(sub), 'width_relations': dict(Counter(r['width_relation'] for r in sub)),
                                 'mean_width_change': mean(r['width_change'] for r in sub) if sub else None,
                                 'mean_cpu_change': mean(r['cpu_change'] for r in sub) if sub else None}
            result['paired_strong_vs_impact'][budget][mode] = {
                'width_relations': dict(Counter(r['width_relation'] for r in rr)),
                'new_no_worse_both': sum(r['new_no_worse_both'] for r in rr),
                'old_no_worse_both': sum(r['old_no_worse_both'] for r in rr),
                'groups': groups,
                'largest_widening': sorted(rr, key=lambda x: x['width_change'], reverse=True)[:5],
                'largest_tightening': sorted(rr, key=lambda x: x['width_change'])[:5],
            }
    with (HERE / 'paired_raw_recomputed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    # Same-history panel membership and the original seven root states.
    structure = read(REPO / 'experiments/reconstruction_v55/structure.json')
    result['structure'] = structure['aggregate']
    records = read(REPO / 'experiments/reconstruction_v55/priority/summary.json')['records']
    result['key_roots'] = []
    for root in KEY_ROOTS:
        qs = [r['query_index'] for r in records if r['root_index'] == root]
        q = min(qs)
        st = structure['rows'][q]
        modes = {}
        for m in MODES:
            a, b = banks['v53_impact'][(q, 'b10', m)], banks['v55_strong'][(q, 'b10', m)]
            modes[m] = {'old_value': a['value'], 'new_value': b['value'], 'old_cpu': cpu(a), 'new_cpu': cpu(b),
                        'old_counters': a['counters'], 'new_counters': b['counters'],
                        'new_certificate': b['certified_action'], 'new_reasons': b['reasons']}
        joint = joint_values({m: banks['v55_strong'][(q, 'b10', m)] for m in MODES})
        result['key_roots'].append({'root': root, 'first_query': q, 'all_queries': qs,
            'clock': records[q]['clock'], 'kind': records[q]['kind'],
            'root_shared_trips': st['root_action_overlap'].get('shared_count'),
            'root_semantic_shared_trips': st['root_action_overlap'].get('semantic_shared_count'),
            'first_platform_candidates': st['platforms'][0]['candidate_count'],
            'first_platform_terminal_eligible': st['platforms'][0]['terminal_dp_eligible'],
            'first_platform_noncandidate_observations': st['platforms'][0]['semantic_nonaction_observation_cids'],
            'optimal_initial_bounds': initial_bounds(banks['v55_strong'][(q, 'b10', 'optimal')]),
            'modes': modes, 'joint_regret': joint['regrets']})
    # Recompute all eight V51 journey controls directly from individual archives.
    p51 = REPO / 'experiments/reconstruction_v51/roots'
    policies = ('cheap', 'mean', 'single', 'pair', 'one_step', 'snapshot_csa', 'matched128', 'full128')
    archive51 = {(q, p): read(p51 / f'{q:03}_{p}.json.gz') for q in range(72) for p in policies}
    result['source_digests']['v51_journeys'] = digest_paths(sorted(p51.glob('*.json.gz')))
    result['v51'] = {}
    for p in policies:
        rr = [archive51[(q, p)] for q in range(72)]
        losses = [r['outcome']['loss'] for r in rr]
        changes = [{'root': q, 'cheap_loss': archive51[(q, 'cheap')]['outcome']['loss'], 'policy_loss': loss,
                    'cheap_minus_policy': archive51[(q, 'cheap')]['outcome']['loss'] - loss}
                   for q, loss in enumerate(losses) if loss is not None and loss != archive51[(q, 'cheap')]['outcome']['loss']]
        result['v51'][p] = {
            'n': 72, 'complete': sum(v is not None for v in losses),
            'loss_lower': sum(v if v is not None else 0 for v in losses) / 72,
            'loss_upper': sum(v if v is not None else rr[q]['loss_cap'] for q, v in enumerate(losses)) / 72,
            'mean_cpu': mean(r['cost']['process_cpu_including_menu'] for r in rr),
            'wins': sum(r['cheap_minus_policy'] > 0 for r in changes),
            'losses': sum(r['cheap_minus_policy'] < 0 for r in changes),
            'ties': sum(v == archive51[(q, 'cheap')]['outcome']['loss'] for q, v in enumerate(losses) if v is not None),
            'nonzero_roots': changes, 'unresolved_roots': [q for q, v in enumerate(losses) if v is None],
        }
    result['limitations'] = [
        'CPU measurements are historical single receipts, not fresh replicated timings.',
        'Counters do not identify wall/CPU share of a subroutine. Terminal structural eligibility is not visitation probability or runtime share.',
        'Strict action certificates and whole-policy regret are different targets.',
        'V51 full128 gate banks force a first action then run the original shallow pi with frozen remote messages; the online full128 controller replans and is not that continuation.',
        'Width comparison at a fixed finite CPU/state cap is not an equal-output-quality timing experiment.',
        'Initial root bounds are computed work: _prepare, destination lower bounds and up to 16 feasible routes per action. The commitment search is capped by 15% of query CPU availability; receipts do not separately time those stages.',
    ]
    # Numerical cross-check only: the computations above did not call any
    # project summarizer or use these aggregate values as inputs.
    published = read(REPO / 'experiments/reconstruction_v55/priority/summary.json')
    for ours, theirs in [('v55_strong', 'v55_priority'), ('v53_impact', 'v53_impact')]:
        for budget in ('b2', 'b10'):
            expected = published['matched_action_attribution']['aggregate'][theirs][budget]
            observed = result['joint_regret'][ours][budget]
            assert abs(expected['mean_cpu_per_joint_state'] - observed['mean_total_cpu']) < 1e-7
            for m in ('cheap_p0', 'cheap_working'):
                assert abs(expected['gaps'][m + '_minus_optimal']['mean_upper'] - observed['policy'][m]['mean_regret_upper']) < 1e-7
    out = HERE / 'metrics.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'receipt_counts': {k: len(v) for k, v in banks.items()},
                      'v51_journeys': len(archive51),
                      'v53_initial': result['initial_bounds']['v53_impact'],
                      'v55_initial': result['initial_bounds']['v55_strong']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
