"""Matching-marginal transfer control on the already frozen pair subset."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import DiscreteLaw, law_summary, load, summarize_endpoints, write
from joint_day_run import inputs
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from src.independent_day_mixture_belief import IndependentDayMixtureBelief


def main():
    data, _, fit = inputs()
    original = load(HERE / 'joint_day_local_v2_results.json.gz')
    output = {'scope': 'Same fixed every-twentieth timetable-pair subset. Only incoming arrival is observed. '
              'For the independent per-trip offset model, a different unknown target trip has its unchanged prior law; '
              'using that identity avoids recomputing irrelevant incoming filters. Development only.', 'dates': {}}
    for day, values in original['transfers'].items():
        graph, _, _, mapping, source, _ = dated_graph(data, day, 'gtfs_station_max')
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        model = IndependentDayMixtureBelief(graph, public_occurrences=False, **fit['increments'])
        rows = []
        for row in values['rows']:
            cid = mapping[row['outgoing']]
            law = DiscreteLaw(*model.event_distribution(cid, 'departure', parameters))
            rows.append({**row, 'methods': {'trip_mixture': law_summary(law, row['actual'])},
                         'probability': law.board_probability(row['ready'])})
        known = [row for row in rows if row['caught'] is not None]
        p = np.asarray([row['probability'] for row in known])
        y = np.asarray([row['caught'] for row in known], dtype=float)
        output['dates'][day] = {'raw_interval': summarize_endpoints(rows, 'trip_mixture'),
                                'n_probability': len(known), 'brier': float(np.mean((p - y) ** 2)),
                                'signed_probability_error': float(np.mean(p - y)),
                                'predictions': [{'incoming': row['incoming'], 'outgoing': row['outgoing'],
                                                 'probability': row['probability']} for row in rows]}
        print(day, output['dates'][day]['brier'], flush=True)
    write(HERE / 'joint_day_local_coupling_control.json', output)


if __name__ == '__main__':
    main()
