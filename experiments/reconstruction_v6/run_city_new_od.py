"""Frozen-controller new-OD sensitivity on the already-exposed city dates."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import fit_training, freeze_replay_inputs, summarize
from experiments.reconstruction_v6.run_city_confirmation import execute

METHODS = ('trained_nominal', 'evolving_uct', 'compiled_nominal', 'analytic_recourse')
PAIRS = (('Zürich, Paradeplatz', 'Zürich, Sternen Oerlikon'),
         ('Zürich, Heuried', 'Zürich, Goldbrunnenplatz'))


def main():
    source = HERE / 'city_confirmation_actuals.json.gz'
    data, _ = load_city(source)
    names = {value['name']: key for key, value in sorted(data['stops'].items())}
    data['od_queries'] = [dict(origin=names[a], destination=names[b], origin_name=a, destination_name=b,
        ready_seconds=29100, deadline_seconds=32700) for a, b in PAIRS]
    fit = fit_training(data, retain_no_pickup=True)
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    output = HERE / 'city_new_od_charged.json'
    protocol = freeze_replay_inputs(output, data=data, forecasts=forecasts, training=fit,
        settings=dict(scope='New static OD sensitivity after original-date results; not another independent date confirmation.',
            pairs=PAIRS, ready_seconds=29100, deadline_seconds=32700, methods=METHODS,
            clock='charged', max_itineraries=8, uct_budget=64, private_worlds=4),
        sources=[str(source), str(Path(__file__)), str(HERE / 'confirmation_locked_recourse.py'),
            str(HERE / 'run_city_confirmation.py'), 'src/risk_bounds.py'],
        implementation='v6_locked_controller_new_od_sensitivity')
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for day in data['holdout_dates']:
        for qi in range(len(PAIRS)):
            for method in METHODS:
                spec = day, qi, method
                path = folder / f'{day}_{qi}_{method}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                else:
                    pending.append((spec, path))
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(execute, spec, data, forecasts, fit, 'charged'): (spec, path)
                for spec, path in pending}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            row = job.result()
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(*spec, row['status'], row['loss_seconds'], flush=True)
    output.write_text(json.dumps(dict(frozen_protocol=str(protocol.relative_to(ROOT)),
        scope='New OD sensitivity on previously exposed confirmation dates.',
        queries=data['od_queries'], summary=summarize(rows, METHODS), rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
