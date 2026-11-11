"""Time-label cuts on the frozen V8 city continuation states."""

from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter, process_time
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.risk_bounds import CommitmentBounds
from src.temporal_frontier import TemporalFrontier, integrate_survival, time_labels


SLACKS = (0, 60, 120, 180, 300)


def evaluate(graph, belief, parameters, state, destination, deadline, case):
    nominal_shift = {r: round(p['mean'] * belief.minute) for r, p in parameters.items()}
    started, cpu = perf_counter(), process_time()
    bounds = CommitmentBounds(belief, parameters, graph, destination, deadline)
    frontier = TemporalFrontier(bounds, state)
    support_seconds = perf_counter() - started
    labels, keys = [], []
    for slack in SLACKS:
        windows = {c.id: (c.dep_time + nominal_shift[c.route] + slack,
                          c.arr_time + nominal_shift[c.route] - slack) for c in graph.connections}
        pair = time_labels(graph, state, destination, deadline, windows, unavailable=state.unavailable_ids)
        for direction, label in zip(('forward', 'backward'), pair):
            if label[destination] > state.time and label not in labels:
                labels.append(label)
                keys.append((slack, direction))
    proposal_seconds = perf_counter() - started - support_seconds
    cuts, work = frontier.evaluate(labels)
    cut_seconds = perf_counter() - started - support_seconds - proposal_seconds
    new = [(c.threshold, c.survival) for c in cuts]
    old = [(t + 1, p) for t, p in zip(case['thresholds'], case['survival_lower'])]
    if abs(max(case['static_lower'], integrate_survival(state.time, old)) - case['lower']) > 1e-8:
        raise AssertionError('Frozen V8 survival grid did not reproduce its lower bound')
    raw_lower = integrate_survival(state.time, new)
    combined = max(case['lower'], integrate_survival(state.time, old + new))
    lower_seconds = perf_counter() - started
    lower_cpu_seconds = process_time() - cpu
    analytic_work = dict(bounds.stats)

    started = perf_counter()
    witness_bounds = CommitmentBounds(belief, parameters, graph, destination, deadline)
    path = tuple(case['witness'])
    upper = witness_bounds.itinerary(state, path).upper if path else deadline - state.time
    witness_evaluate_seconds = perf_counter() - started
    if abs(upper - case['upper']) > 1e-8 or combined > upper + 1e-8:
        raise AssertionError('Incompatible frozen witness or lower bound')
    return dict(frontier_lower_seconds=raw_lower, combined_lower_seconds=combined,
                gain_seconds=combined - case['lower'], upper_seconds=upper, gap_seconds=upper - combined,
                support_seconds=support_seconds, proposal_seconds=proposal_seconds, cut_seconds=cut_seconds,
                lower_wall_seconds=lower_seconds, lower_cpu_seconds=lower_cpu_seconds,
                witness_evaluate_seconds=witness_evaluate_seconds, work=work, analytic_work=analytic_work,
                cuts=[dict(slack_seconds=slack, direction=direction, **asdict(cut))
                      for (slack, direction), cut in zip(keys, cuts)])


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    frozen_path = ROOT / 'experiments/reconstruction_v8/opportunity_city_results.json'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads(archive_path.read_text())
    journeys = {(r['date'], r['query']): r for r in archive['rows'] if r['method'] == 'evolving_rollout'}
    frozen = json.loads(frozen_path.read_text())['rows']
    rows = []
    for day in dict.fromkeys(case['date'] for case in frozen):
        graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                      for c in graph.connections}
        for case in (c for c in frozen if c['date'] == day):
            journey = journeys[day, case['query']]
            destination = stops[journey['od']['destination']]
            deadline = journey['od']['deadline_seconds']
            started = perf_counter()
            belief, state = context(graph, metadata, journey, case['decision'], fit)
            context_wall_seconds = perf_counter() - started
            if (state.stop, state.time, destination) != (case['stop'], case['clock'], case['destination']):
                raise AssertionError('Unmatched city history')
            result = evaluate(graph, belief, parameters, state, destination, deadline, case)
            rows.append(dict(date=day, query=case['query'], decision=case['decision'], stop=state.stop,
                clock=state.time, destination=destination, deadline=deadline,
                context_wall_seconds=context_wall_seconds, v8_lower_seconds=case['lower'],
                v8_lower_wall_seconds=case['opportunity_seconds'], v8_witness_wall_seconds=case['witness_seconds'],
                **result))
            print(f'{day}/{case["query"]}/{case["decision"]}: V8 {case["lower"]:.2f}s, '
                  f'cut {result["frontier_lower_seconds"]:.2f}s, combined {result["combined_lower_seconds"]:.2f}s, '
                  f'gap {result["gap_seconds"]:.2f}s, cut wall {result["lower_wall_seconds"]:.3f}s', flush=True)
    summary = dict(mean_frontier_lower_seconds=mean(r['frontier_lower_seconds'] for r in rows),
                   mean_combined_lower_seconds=mean(r['combined_lower_seconds'] for r in rows),
                   mean_gain_seconds=mean(r['gain_seconds'] for r in rows),
                   improved=sum(r['gain_seconds'] > 1e-8 for r in rows),
                   min_gap_seconds=min(r['gap_seconds'] for r in rows),
                   median_lower_wall_seconds=median(r['lower_wall_seconds'] for r in rows),
                   max_lower_wall_seconds=max(r['lower_wall_seconds'] for r in rows),
                   median_witness_evaluate_seconds=median(r['witness_evaluate_seconds'] for r in rows),
                   gaps_within_seconds={t: sum(r['gap_seconds'] <= t for r in rows) for t in (1, 30, 60)})
    protocol = dict(source=str(source.relative_to(ROOT)), archive=str(archive_path.relative_to(ROOT)),
                    frozen_cases_and_upper=str(frozen_path.relative_to(ROOT)), training_dates=fit['train_dates'],
                    slack_seconds=SLACKS, proposal='Forward and reverse labels on shifted nominal interval timetable.',
                    target='Model-conditional all-action lower bound under the frozen independent trip posterior.',
                    assumptions='Complete graph, zero transfer time, no footpaths. The cut also relaxes pickup restrictions.',
                    timing='Lower includes cold support, label proposal, joint crossing integration. Frozen first-exit '
                           'bound and fixed witness path reused. Exact witness value independently recomputed and timed.')
    (HERE / 'frontier_results.json').write_text(json.dumps(dict(protocol=protocol, summary=summary, rows=rows), indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
