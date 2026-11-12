"""Describe retained policy disagreements and static-priority failures."""

import gzip
import json
from pathlib import Path

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v25.run import departure


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


def main():
    with gzip.open(HERE/'results.json.gz', 'rt') as handle:
        rows = json.load(handle)['results']
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    metadata = {day: dated_graph(data, day, 'gtfs_station_max', True)[4] for day in {r['day'] for r in rows}}
    changes, failures = [], []
    for row in rows:
        one, single = (row['outcomes'][m] for m in ('one_step_nominal', 'single_target_nominal'))
        if one['action'] != single['action']:
            a, b = ({e['clock']: e for e in out['trace']} for out in (one, single))
            clock = next(t for t in sorted(set(a)&set(b)) if a[t]['choice'] != b[t]['choice'])
            record = dict(law=row['law'], root_id=row['root_id'], menu_size=row['menu_size'],
                construction=row['construction'], first_split_clock=clock,
                one_step_loss=one['loss'], single_loss=single['loss'],
                realized_gain=None if one['loss'] is None or single['loss'] is None else one['loss']-single['loss'],
                one_step_event=a[clock], single_event=b[clock])
            if a[clock]['choice'] is not None and b[clock]['choice'] is None and b[clock]['receipt']['target'] == single['action']:
                board, wait = a[clock]['receipt']['board'], b[clock]['receipt']['wait']
                record['proxy_predicted_gain'] = board-wait
                if record['realized_gain'] is not None:
                    record['board_realization_error'] = board-(row['start']+one['loss'])
                    record['wait_realization_error'] = wait-(row['start']+single['loss'])
            changes.append(record)
        priority = row['outcomes']['schedule_priority']
        if priority['status'] == 'declined_all_offers':
            target = priority['trace'][-1]['receipt']['target']
            clock, canceled = departure(metadata[row['day']][target])
            relation = ('canceled' if canceled else 'before_entry' if clock < row['entry']
                        else 'at_or_after_deadline' if clock >= row['deadline'] else 'during_visit')
            failures.append(dict(law=row['law'], root_id=row['root_id'], last_target=target,
                actual_departure=clock, entry=row['entry'], relation=relation))
    (HERE/'diagnostics.json').write_text(json.dumps(dict(
        scope='Post-replay descriptions only. Prediction versus one realized journey does not establish miscalibration.',
        single_vs_one_step_changes=changes, schedule_priority_failures=failures), indent=2)+'\n')


if __name__ == '__main__':
    main()
