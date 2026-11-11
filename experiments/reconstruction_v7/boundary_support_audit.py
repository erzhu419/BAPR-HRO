"""Inspect actual current-grid support before assuming positive physical floors."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.evolving_trip_belief import EvolvingTripBelief, logsumexp


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    graph, _, _, _, _, _ = dated_graph(data, '2023-07-26', 'gtfs_station_max', True)
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    selected = sorted(trip for trip, slots in belief._trips.items() if len(slots) >= 4)[:20]
    rows = []
    before = perf_counter()
    for trip in selected:
        law = belief._law(trip, parameters)
        for slot in range(0, len(law.times), 2):
            cid = belief._trips[trip][slot].cid
            marginal = law.marginal(slot)
            future = law._backward_from(max(slot + 1, law._last_effect), slot + 1)
            log_edge = law._transition(slot) + law.emissions[slot + 1] + future
            norm = logsumexp(log_edge, axis=1, keepdims=True)
            conditional = np.exp(log_edge - np.where(np.isfinite(norm), norm, 0.))
            zero = ((law.times[slot][:, None] == law.times[slot + 1][None, :])
                    & ~law.canceled[slot][:, None] & ~law.canceled[slot + 1][None, :])
            mass = float((marginal[:, None] * conditional)[zero].sum())
            rows.append(dict(trip=trip, connection=cid,
                scheduled_runtime_seconds=graph.connections[cid].scheduled_travel_time,
                current_model_zero_operating_runtime_probability=mass,
                zero_runtime_in_support=mass > 0))
    result = dict(scope='Existing full-grid city model support diagnostic, not a physical runtime estimate.',
        source=str(source.relative_to(ROOT)), date='2023-07-26', training_dates=fit['train_dates'],
        selection='Lexicographic first20 trips with >=2 connections, chosen before support inspection.',
        visibility='No occurrence feed; no incoming ride evidence at this prior root.',
        trip_count=len(selected), connection_count=len(rows),
        zero_runtime_supported_connections=sum(row['zero_runtime_in_support'] for row in rows),
        elapsed_seconds=perf_counter() - before, rows=rows,
        interpretation='Any positive mass at runtime0 forbids replacing its support lower bound by an empirical positive minimum. The first-exit relaxation may therefore be valid but loose; this does not prove every possible structured bound is impossible.')
    (HERE / 'boundary_support_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
