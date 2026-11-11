"""Replay the exact failed stay-to-stop certificate with the received skip."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v5.calibration_run import data_inputs
from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.risk_bounds import CommitmentBounds
from src.stop_service_status import available_alighting_leg, stop_status_at


def main():
    previous = ROOT / 'experiments/reconstruction_v6'
    diagnosis = json.loads((previous / 'decision_risk_failure_diagnosis.json').read_text())
    receipts = json.loads((HERE / 'service_receipts_results.json').read_text())
    fit = json.loads((previous / 'decision_risk_frozen.json').read_text())['protocol']['configuration']['training']
    target = diagnosis['failure']
    graph, _, _, ids, source, _ = dated_graph(data_inputs(), target['date'], 'gtfs_station_max', True)
    incoming, outgoing = ids[target['incoming']], ids[target['outgoing']]
    before, after = graph.connections[incoming], graph.connections[outgoing]
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    for kind in ('departure', 'arrival'):
        scheduled = getattr(before, 'dep_time' if kind == 'departure' else 'arr_time')
        belief.observe_connection_event(incoming, kind, source[incoming][kind]['time_seconds']-scheduled)
    params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    t = target['time']
    state = CommittedState(before.arr_stop, t, before.trip_id, t, frozenset({incoming}), initialized=True)
    observations = [row for frame in receipts['revisions'] for row in frame['target']]
    cases = []
    for label, stream in [('no_realtime', []), ('archived_realtime', observations)]:
        status = stop_status_at(stream, t, source[outgoing]['arr_sequence'])
        eligible = available_alighting_leg(state, outgoing, source[outgoing]['arr_sequence'], stream)
        gate_state = state if eligible else replace(state, unavailable_ids=state.unavailable_ids | {outgoing})
        cert = CommitmentBounds(belief, params, graph, after.arr_stop, t+3600).transfer_gate(gate_state, (outgoing,))
        cases.append(dict(method=label, stop_status=status, eligible=eligible,
            accepted=cert is not None and cert.regret_bound <= 1., value=asdict(cert) if cert else None))
    if not cases[0]['accepted'] or cases[1]['accepted']:
        raise AssertionError('The observed failure must reproduce and the received skip must veto the target')
    result = dict(date=target['date'], trip=before.trip_id, time=t, incoming=incoming, outgoing=outgoing,
        earliest_known_skip=min(row[0] for row in receipts['explicit_failure_before_cutoff']),
        cases=cases, scope='One fixed observed arrival history. Veto applies to the alighting endpoint; no inference on physical reroute or alternate realized loss.')
    (HERE/'service_gate_replay_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
