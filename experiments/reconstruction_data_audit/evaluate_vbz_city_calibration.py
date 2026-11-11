"""Evaluate July25 route fits on later complete-source city peak measurements."""
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_data_audit.evaluate_vbz_calibration import fit, score


def main():
    with gzip.open(HERE / 'city_vbz_dated_replay_v2.json.gz', 'rt', encoding='utf-8') as handle:
        source = json.load(handle)
    data = {'train_dates': source['train_dates'], 'dates': {}}
    for day, rows in source['dates'].items():
        legs = []
        for c in rows['connections']:
            if c['outcome'] != 'MEASURED':
                continue
            leg = {'route_id': c['route_id'], 'mode': c['mode']}
            for kind in ('departure', 'arrival'):
                leg[f'actual_{kind}_seconds'] = c[kind]['time_seconds']
                leg[f'scheduled_{kind}_seconds'] = c[f'scheduled_{kind}_seconds']
                leg[f'provider_scheduled_{kind}_seconds'] = c[kind]['provider_scheduled_seconds']
            legs.append(leg)
        data['dates'][day] = {'legs': legs}
    fits = {basis: fit(data, basis) for basis in ('', 'provider_')}
    results = {}
    for day in source['holdout_dates']:
        legs = data['dates'][day]['legs']
        results[day] = {'coverage': source['dates'][day]['coverage'],
                        'point_and_interval': {basis: {mode: score(legs, fits[basis], basis, mode)
                                                      for mode in ('tram', 'bus')}
                                               for basis in fits}}
        print(day, results[day]['point_and_interval'], flush=True)
    output = {'scope': 'Complete-source peak measurements with exact GTFS identity and explicit unmatched candidates. '
              'Train July25 and test July26/27, retaining both GTFS and provider scheduled-time baselines. '
              'No cancellation probability is inferred from an archive of performed runs.',
              'train_dates': source['train_dates'], 'dates': results}
    (HERE / 'vbz_city_calibration_v2.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
