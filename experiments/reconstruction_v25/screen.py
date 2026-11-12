"""Freeze V25 full menus using only the two static timetables."""

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v24.screen import screen_day
from experiments.reconstruction_v23.run_transfer_screen import TRAINING
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATES = ('2023-07-16', '2023-07-17')


def main():
    started = perf_counter()
    schedules = {}
    for day in DATES:
        with gzip.open(HERE/f'city_schedule_{day}.json.gz', 'rt') as handle:
            schedules[day] = json.load(handle)
    stops = {key: value for schedule in schedules.values() for key, value in schedule['stops'].items()}
    with (ROOT/'experiments/reconstruction_data_audit/station_transfer_bounds.json').open() as handle:
        transfer = json.load(handle)['stations']
    data = dict(stops=stops, station_transfer_bounds=transfer, dates={
        day: dict(connections=[dict(row, outcome='UNRESOLVED') for row in schedule['connections']])
        for day, schedule in schedules.items()})
    selected = {}
    for day in DATES:
        graph, *_ = dated_graph(data, day, 'gtfs_station_max', True)
        selected[day] = screen_day(day, graph)
        print(day, selected[day]['counts'])
    selection = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Schedule-only full menus. Origin roots are primary; first-transfer slices are secondary.',
        by_day=selected, schedule_preparation_wall=perf_counter()-started)
    (HERE/'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    training, _ = load_city(TRAINING)
    fit = fit_training(training, retain_no_pickup=True)
    fit['duration_fit'] = json.loads((ROOT/'experiments/reconstruction_v24/route_empirical20/duration_fit.json').read_text())
    fit['scope'] = 'Frozen V24 route-stratified elapsed law and original July 25 route priors; no V25 outcomes fitted.'
    (HERE/'model_fit.json').write_text(json.dumps(fit, indent=2)+'\n')


if __name__ == '__main__':
    main()
