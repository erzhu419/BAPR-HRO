"""Complete observed labels after one supported stop-identity repair."""
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
from experiments.reconstruction_v6.run_city_new_od import PAIRS

METHODS = ('trained_nominal', 'evolving_uct', 'compiled_nominal', 'analytic_recourse')


def execute_paired(spec, data, forecasts, fit):
    day, qi, method = spec
    context = dict(data, od_queries=data['od_queries'][2:]) if qi >= 2 else data
    stream_index = qi - 2 if qi >= 2 else qi
    row = execute((day, stream_index, method), context, forecasts, fit, 'charged')
    row.update(query=qi, random_stream_query_index=stream_index)
    return row


def main():
    source = HERE / 'city_confirmation_source_completed.json.gz'
    data, _ = load_city(source)
    original, _ = load_city(HERE / 'city_confirmation_actuals.json.gz')
    if data['dates']['2023-07-25'] != original['dates']['2023-07-25']:
        raise ValueError('Source completion must retain the original training record')
    names = {value['name']: key for key, value in sorted(data['stops'].items())}
    data['od_queries'] += [dict(origin=names[a], destination=names[b], origin_name=a, destination_name=b,
        ready_seconds=29100, deadline_seconds=32700) for a, b in PAIRS]
    fit = fit_training(original, retain_no_pickup=True)
    forecasts = {'dates': {day: {'frames': []} for day in data['holdout_dates']}}
    output = HERE / 'city_source_completed_paired_charged.json'
    protocol = freeze_replay_inputs(output, data=data, forecasts=forecasts, training=fit,
        settings=dict(scope='Source completion sensitivity after original confirmation/new OD readout; original fit/controller unchanged.',
            queries=data['od_queries'], clock='charged', methods=METHODS,
            max_itineraries=8, private_worlds=4, uct_budget=64,
            random_streams='Each two-OD panel retains local query indices0/1, matching its original source run.',
            reused_original_od_cases='Identical source, fit, local stream indices and controller from city_source_completed_charged_cases.'),
        sources=[str(source), str(Path(__file__)), str(HERE / 'confirmation_locked_recourse.py'),
            str(HERE / 'run_city_confirmation.py'), 'src/risk_bounds.py'],
        implementation='v6_stop_identity_source_completion')
    folder = output.with_name(output.stem + '_cases')
    folder.mkdir(exist_ok=True)
    rows, pending = [], []
    for day in data['holdout_dates']:
        for qi in range(4):
            for method in METHODS:
                spec = day, qi, method
                path = folder / f'{day}_{qi}_{method}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                elif qi < 2:
                    prior = HERE / 'city_source_completed_charged_cases' / path.name
                    row = json.loads(prior.read_text())
                    if row['od'] != data['od_queries'][qi]:
                        raise ValueError('Reused source-completed case is a different OD')
                    row.update(random_stream_query_index=qi, reused_case=str(prior.relative_to(ROOT)))
                    rows.append(row)
                else:
                    pending.append((spec, path))
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(execute_paired, spec, data, forecasts, fit): (spec, path)
                for spec, path in pending}
        for job in as_completed(jobs):
            spec, path = jobs[job]
            row = job.result()
            path.write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(*spec, row['status'], row['loss_seconds'], flush=True)
    panels = {name: summarize([r for r in rows if (r['query'] < 2) == old], METHODS)
              for name, old in (('original_od', True), ('new_od', False))}
    output.write_text(json.dumps(dict(frozen_protocol=str(protocol.relative_to(ROOT)),
        scope='Source correction sensitivity; no new model or independent date.',
        queries=data['od_queries'], training_record_unchanged=True, summary_by_panel=panels,
        summary=summarize(rows, METHODS), rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
