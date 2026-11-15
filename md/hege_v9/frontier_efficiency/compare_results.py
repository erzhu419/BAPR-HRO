"""Read retained receipts only; no natural-state evaluation or repository edits."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from efficient_priority import OUT, REPO


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


paths = dict(
    original=REPO / 'experiments/reconstruction_v47/natural/055_bulk_recycle.json.gz',
    naive=OUT.parent / 'theory_code/priority_diagnostic/055_priority.json',
    efficient=OUT / '055_efficient.json')
receipts = {name: read(path) for name, path in paths.items()}
original, naive, efficient = (receipts[name] for name in ('original', 'naive', 'efficient'))
costs = dict(original=original['complete_cost']['process_cpu'],
             naive=naive['complete_diagnostic_cost']['process_cpu'],
             efficient=efficient['contract_query_cost']['process_cpu'])

rows = []
for name, receipt in receipts.items():
    first = next((snap for snap in receipt['checkpoints'] if snap['certified_action']), None)
    # Charge all outside-oracle query overhead to the first saved certificate,
    # including the tiny final-receipt assembly tail, so this does not hide any
    # setup cost or call oracle-only time a complete query time.
    outside_oracle = costs[name] - receipt['cost']['process_cpu']
    rows.append(dict(
        variant=name, manifest_index=receipt['index'], root_index=receipt['root_index'],
        query_cpu=costs[name], charged=receipt['counters']['charged_compiles'],
        gap=receipt['difference_interval'], certified_action=receipt['certified_action'],
        original_fields=dict(complete=receipt['complete'],
                             alight_mean=receipt['actions']['alight']['mean'],
                             alight_unresolved_probability=receipt['actions']['alight']['unresolved_probability']),
        first_saved_certificate=None if first is None else dict(
            phase=first['phase'], charged=first['counters']['charged_compiles'],
            gap=first['difference_interval'], oracle_cpu=first['cost']['process_cpu'],
            outside_oracle_query_cpu_charged=outside_oracle,
            conservative_complete_query_cpu=first['cost']['process_cpu'] + outside_oracle),
        budget_allocation=receipt['budget_allocation']))

shared_counters = [key for key in naive['counters'] if not key.startswith('priority_')]
counter_differences = {key: dict(naive=naive['counters'][key], efficient=efficient['counters'][key])
                       for key in shared_counters
                       if naive['counters'][key] != efficient['counters'][key]}
checkpoint500 = {name: next(snap for snap in receipt['checkpoints']
                           if snap['counters']['charged_compiles'] == 500)
                 for name, receipt in receipts.items()}
validation = read(OUT / 'small_checks.json')
independent = read(OUT.parent / 'theory_code/heap_adversarial_checks.json')
summary = dict(
    repo_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    repo_status_after=subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
    natural_runs_in_this_prototype=1,
    comparisons_are_descriptive_not_same_machine_formal_benchmarks=True,
    rows=rows,
    same_final_action_receipts_naive_and_efficient=naive['actions'] == efficient['actions'],
    same_final_difference_interval_naive_and_efficient=naive['difference_interval'] == efficient['difference_interval'],
    same_actions_at_500_naive_and_efficient=checkpoint500['naive']['actions'] == checkpoint500['efficient']['actions'],
    core_counter_differences_naive_to_efficient=counter_differences,
    naive_to_efficient_query_cpu_ratio=costs['naive'] / costs['efficient'],
    efficient_vs_original_query_cpu_ratio=costs['efficient'] / costs['original'],
    naive_cpu_reduction_fraction=1 - costs['efficient'] / costs['naive'],
    naive_fullgraph_node_visits=naive['counters']['priority_node_visits'],
    efficient_root_mass_node_visits=efficient['counters']['priority_weight_node_visits'],
    efficient_root_mass_edge_visits=efficient['counters']['priority_weight_edge_visits'],
    efficient_discovery_node_visits=efficient['counters']['priority_discovery_visits'],
    root_mass_only_cpu=efficient['counters']['priority_weight_cpu'],
    entire_priority_next_cpu_including_arithmetic=efficient['counters']['priority_next_cpu'],
    snapshot_cpu=efficient['counters']['priority_snapshot_cpu'],
    cold_process_cost=read(OUT / '055_process_cost.json'),
    small_chain_cases=len(validation['cases']),
    small_chain_checkpoint_count=sum(case['checkpoints_checked'] for case in validation['cases']),
    small_chain_max_relative_mass_error=max(case['max_relative_mass_error'] for case in validation['cases']),
    independent_adversarial_checks=independent,
    evidence={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
              for name, path in paths.items()},
    caveats=[
        'One retrospective root, selected for mechanism diagnosis; no general 7-root success-rate or speed claim.',
        'The archived original DFS query and the two audit runs were not a controlled same-machine repeated timing experiment.',
        'The fast first A phase reaches quota 1000; the naive run hits its 60-second phase limit at 921. Total final A paid work is 1323 in both.',
        'bulk_interval_reductions differs because saved checkpoints intersect C at different work counts; all other common non-priority counters match.',
        'A interval prints equal endpoints but unresolved_probability is positive and mean remains null; no floating-point completion override is applied.',
        'No certificate stopping policy was introduced: all 2000 paid units were still consumed.',
        'No probability is pruned; np.longdouble changes only scheduling weight arithmetic, never value or policy calculations.',
        'Incremental floating point summation order can differ at machine precision; six small paid traces match exactly and root27 final receipts/core counters match, but this is not a proof of universal bitwise scheduling equivalence.'
    ])
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
