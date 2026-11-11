"""Cold full-city posterior/native query; a 60s terminal feasibility diagnostic."""
import json
from pathlib import Path
import signal
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training, freeze_replay_inputs
from src.durner.native import native_topocsa
from src.evolving_trip_belief import EvolvingTripBelief
from src.native_event_router import minute_snapshot


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    out = HERE / 'native_cold_results.json'
    protocol = freeze_replay_inputs(out, data=data, forecasts={}, training=fit,
        settings={'day': '2023-07-26', 'query': 0, 'deadline_seconds': 60,
                  'candidate_rule': 'all original connections, no source pruning',
                  'quantization': 'nearest minute PMF clocks; fractional timetable order; ceil MCT',
                  'terminal_rule': 'Any cold full posterior+query >60s ends feasibility phase without full journeys'},
        sources=[str(source), str(Path(__file__)), 'src/native_event_router.py', 'src/durner/native.py'],
        implementation='v6_native_event_cold')
    t0 = perf_counter()
    graph, _, stops, _, _, _ = dated_graph(data, '2023-07-26', 'gtfs_station_max', True)
    graph_ms = (perf_counter() - t0) * 1000
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    pars = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    q = data['od_queries'][0]
    record = dict(implementation='v6_native_event_cold', frozen_protocol=str(protocol.relative_to(ROOT)),
                  graph_ms=graph_ms, connections=len(graph.connections), stage='full_event_marginals')
    t0, cpu0 = perf_counter(), process_time()
    def timeout(*_):
        raise TimeoutError('full posterior + native query exceeded predeclared 60 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60.)
    try:
        temp, work = minute_snapshot(belief, pars)
        record.update(snapshot=work, stage='strict_native_topocsa')
        before = perf_counter()
        result = native_topocsa(temp, stops[q['origin']], stops[q['destination']],
                               q['ready_seconds'] // 60, (q['deadline_seconds'] + 59) // 60)
        record.update(status='completed', query_ms=(perf_counter() - before) * 1000,
            native_scans=result.n_connections_processed, native_cuts=len(result.cuts),
            native_labels=sum(map(len, result.stop_labels.values())),
            origin_labels=len(result.stop_labels[stops[q['origin']]]))
    except TimeoutError as error:
        record.update(status='terminal_60s', reason=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    record.update(wall_seconds=perf_counter() - t0, cpu_seconds=process_time() - cpu0)
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
