"""Combine existing complete city day records and audit the prior source."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training

JULY = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
OCTOBER = ROOT / 'experiments/reconstruction_data_audit/city_dated_replay.json.gz'
OUTPUT = ROOT / 'experiments/reconstruction_data_audit/city_past_actual.json.gz'


def coverage(graph, fit):
    counts = Counter(c.route for c in graph.connections)
    covered = sorted(route for route in counts if route in fit['route_priors'])
    missing = sorted(set(counts) - set(covered))
    return {'routes': len(counts), 'matched_routes': len(covered), 'matched_route_ids': covered,
            'connections': len(graph.connections), 'matched_connections': sum(counts[r] for r in covered),
            'matched_connection_fraction': sum(counts[r] for r in covered) / len(graph.connections),
            'pooled_fallback_connections': sum(counts[r] for r in missing),
            'pooled_fallback_route_connection_counts': {r: counts[r] for r in missing},
            'increments': fit['increments'], 'pooled': fit['route_priors']['pooled']}


def main():
    with gzip.open(JULY, 'rt') as handle:
        july = json.load(handle)
    with gzip.open(OCTOBER, 'rt') as handle:
        october = json.load(handle)
    combined = {'scope': 'Existing complete city records: July25 actual training, October25 held-out replay.',
                'source_records': [{'path': str(JULY.relative_to(ROOT)), 'date': '2023-07-25'},
                                   {'path': str(OCTOBER.relative_to(ROOT)), 'date': '2023-10-25'}],
                'source_selection': 'Reuse already available past full-city actuals, without download or '
                                    'selection of additional dates after observing performance.',
                'cancellation_identifiability': 'The municipal performed-service training source does not '
                    'identify all cancellations. Route cancellation probabilities remain the declared '
                    'zero assumption, not an estimate identifying true city cancellation risk.',
                'fit_recipe': 'Unchanged fit_training(retain_no_pickup=True): raw route departure '
                              'moments and raw travel/dwell/gap residual moments; no normalized fit.',
                'train_dates': ['2023-07-25'], 'holdout_dates': ['2023-10-25'],
                'dates': {'2023-07-25': july['dates']['2023-07-25'], '2023-10-25': october}}
    if OUTPUT.exists():
        with gzip.open(OUTPUT, 'rt') as handle:
            assert json.load(handle) == combined, 'Use a new output for changed source records'
    else:
        with gzip.open(OUTPUT, 'wt') as handle:
            json.dump(combined, handle, ensure_ascii=False)
    data, original = load_city(OUTPUT)
    old_data, _ = load_city(OCTOBER)
    graph, world, *rest = dated_graph(data, '2023-10-25', 'gtfs_station_max', True)
    old_graph, old_world, *old_rest = dated_graph(old_data, '2023-10-25', 'gtfs_station_max', True)
    assert original['dates']['2023-07-25'] == july['dates']['2023-07-25']
    assert original['dates']['2023-10-25'] == october
    assert graph.connections == old_graph.connections and graph.stops == old_graph.stops
    assert world == old_world and data['od_queries'] == old_data['od_queries']
    assert rest == old_rest
    new_fit = fit_training(data, retain_no_pickup=True)
    archived = json.loads((HERE / 'city_national_logical_results.json').read_text())
    old_fit = archived['training']['2023-10-25']
    assert all(p['cancel_rate'] == 0 for p in new_fit['route_priors'].values())
    nominal_rows = [r for r in archived['rows'] if r['method'] == 'trained_nominal']
    assert all(r['planning_transitions'] == 0 for r in nominal_rows)
    result = {'input': str(OUTPUT.relative_to(ROOT)), 'train_dates': data['train_dates'],
              'holdout_dates': data['holdout_dates'], 'preserved_date_records': True,
              'same_october_graph_actual_world_query_and_mapping': True,
              'cancellation_assumption': combined['cancellation_identifiability'],
              'old_source': 'Archived prior from earlier GTFS-RT predictions/statuses on the old '
                            '2000-edge city template, with declared fallback increment scales.',
              'new_source': combined['fit_recipe'], 'old_proxy': coverage(graph, old_fit),
              'past_actual': coverage(graph, new_fit), 'new_training': new_fit,
              'attribution': 'Nominal comparisons change the complete prior source package, including '
                  'predictions versus realizations, cohorts, seasonal lag and increment fitting. '
                  'Their archived budget512 is inactive (zero planning transitions). '
                  'New UCT64 is a strong baseline reading, not an isolated contrast to old UCT512.'}
    audit = HERE / 'city_past_actual_input_audit.json'
    audit.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('input', 'old_proxy', 'past_actual')}, indent=2))


if __name__ == '__main__':
    main()
