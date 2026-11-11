"""Separate witness-first/cutoff experiment; archived full-EA baseline is kept."""
from fractions import Fraction
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ROOT, dated_graph, fit_training, freeze_replay_inputs
from experiments.reconstruction_v5.run_sparse_certificates import setup, witness, bank_work
from src.sparse_certificate_cutoff import SparseCertificateIndex


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    output = Path(__file__).with_name('sparse_certificate_cutoff_results.json')
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    day = data['holdout_dates'][0]
    graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    archive = json.loads(archive_path.read_text())
    archived = next(r for r in archive['rows'] if r['date'] == day and r['query'] == 0
                    and r['method'] == 'evolving_rollout')
    protocol = freeze_replay_inputs(
        output, data=data, forecasts={'dates': {day: {'frames': []}}}, training=fit,
        settings={'contexts': [0, 5], 'archived_public_history': archived,
                  'samples': 4, 'witness_transition_budget': 512,
                  'order': 'causal nominal witness then cutoff lower', 'transfer_rule': 'gtfs_station_max'},
        sources=[str(source), str(archive_path), 'src/sparse_certificate.py',
                 'src/sparse_certificate_cutoff.py', 'src/evolving_trip_belief.py',
                 'src/committed_transit.py', 'src/continuation_router.py',
                 'experiments/reconstruction_v5/run_sparse_certificates.py',
                 'experiments/reconstruction_v5/run_sparse_cutoff.py'])
    baseline = json.loads(Path(__file__).with_name('sparse_certificate_results.json').read_text())
    rows = []
    for number in (0, 5):
        started = perf_counter()
        current = setup(graph, metadata, archived, number, fit, stops[archived['od']['destination']])
        upper = witness(current)
        before_bank = bank_work(current)
        before = perf_counter()
        index = SparseCertificateIndex(graph, current['bank'], current['destination'],
                                       current['state'].time, current['deadline'])
        lower = index.action_lower(current['state'], current['model'].actions(current['state']),
                                   cutoff=Fraction(upper['upper']))
        lower_ms = (perf_counter() - before) * 1000
        reference = next(r for r in baseline['rows'] if r['archived_decision'] == number)
        reference_lower = {tuple(action): value for action, value in reference['action_lower_remaining_seconds']}
        assert all(float(value) <= reference_lower[action] + 1e-9 for action, value in lower.items())
        assert upper['world_loss_or_cap'] == reference['fresh_cold_witness']['world_loss_or_cap']
        bound = float(min(lower.values()))
        assert bound <= upper['upper']
        row = {'archived_decision': number, 'date': day, 'query': 0, 'all_root_actions': len(lower),
               'lower': bound, 'upper': upper['upper'], 'gap': upper['upper'] - bound,
               'removed_actions': [list(a) for a, value in lower.items() if value > Fraction(upper['upper'])],
               'retained_actions': [list(a) for a, value in lower.items() if value <= Fraction(upper['upper'])],
               'action_lower_remaining_seconds': [[list(a), float(v)] for a, v in lower.items()],
               'lower_ms_after_witness': lower_ms, 'work': dict(index.stats),
               'causal_witness': upper, 'context_setup_ms': current['setup_ms'],
               'bank_after_witness': before_bank, 'bank_after_bound': bank_work(current),
               'total_context_witness_certificate_ms': (perf_counter() - started) * 1000}
        rows.append(row)
        output.write_text(json.dumps({'scope': 'Full-city four-world finite-bank, all root actions, separate cutoff variant.',
                                     'frozen_protocol': str(protocol.relative_to(ROOT)),
                                     'baseline': 'experiments/reconstruction_v5/sparse_certificate_results.json',
                                     'rows': rows}, indent=2) + '\n')
        print(json.dumps({key: row[key] for key in ['archived_decision', 'all_root_actions', 'lower', 'upper',
                         'gap', 'lower_ms_after_witness', 'total_context_witness_certificate_ms']} |
                         {'removed': len(row['removed_actions']), 'ea_skipped': row['work'].get('ea_skipped_by_cutoff', 0)}), flush=True)


if __name__ == '__main__':
    main()
