"""Read new outcomes only after loading the unchanged frozen candidates."""
import gzip
import json
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import data_inputs, load, write
from calibration_relative_run import METHODS, evaluate, summarize
from calibration_relative_variants import NEW_METHODS, add_predictions


def main():
    protocol = json.loads((HERE / 'calibration_relative_confirmation_protocol.json').read_text())
    frozen = json.loads((HERE / 'calibration_frozen_fit.json').read_text())
    relative_protocol = json.loads((HERE / 'calibration_relative_protocol.json').read_text())
    offsets = json.loads((HERE / 'calibration_relative_variants_frozen_offsets.json').read_text())['offset_seconds']
    with (HERE / 'calibration_frozen_model.pkl').open('rb') as handle:
        marginal = pickle.load(handle)
    with (HERE / 'calibration_relative_frozen_model.pkl').open('rb') as handle:
        coefficient_one = pickle.load(handle)
    with (HERE / 'calibration_relative_variants_frozen.pkl').open('rb') as handle:
        models = pickle.load(handle)
    # No training or calibration function is called by this evaluator.
    actuals = load(HERE / 'calibration_relative_confirmation_actuals.json.gz')
    data = data_inputs()
    data['dates'].update(actuals['dates'])
    frames = load(HERE / 'calibration_relative_confirmation_public_frames.json.gz')['dates']
    outcomes = {}
    for day in protocol['confirmation_dates']:
        outcomes[day] = evaluate(data, day, frozen['evolving_fit'], frames[day]['frames'],
                                 marginal, coefficient_one, relative_protocol)
    add_predictions(data, outcomes, models)
    methods = (*METHODS, *NEW_METHODS)
    summaries = {day: summarize(value, offsets, methods) for day, value in outcomes.items()}
    for day in summaries:
        summaries[day]['source_coverage'] = actuals['dates'][day]['coverage']
    result = {'protocol': protocol,
              'scope': 'Independent predeclared dates; all fits, parameters and menu-specific interval offsets '
              'were frozen before inspecting these actual outcomes. No further date or parameter selection.',
              'offset_seconds': offsets, 'confirmation': summaries, 'rows_by_date': outcomes,
              'integration_diagnostics': list(models['heavy_tail_day'].diagnostics.values())}
    with gzip.open(HERE / 'calibration_relative_confirmation_results.json.gz', 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)
    write(HERE / 'calibration_relative_confirmation_summary.json', {key: result[key] for key in ('protocol', 'scope', 'confirmation')})
    for day, value in summaries.items():
        for menu in ('fixed_schedule', 'actual_current'):
            print('CONFIRMATION', day, menu, {name: {'coverage': round(row['date_calibrated']['interval']['coverage'], 4),
                  'width': round(row['date_calibrated']['interval']['mean_width_seconds'], 1),
                  'score': round(row['date_calibrated']['interval']['mean_interval_score'], 1),
                  'mae': round(row['date_calibrated']['interval']['mae_seconds'], 1),
                  'brier': round(row['date_calibrated']['probability']['brier'], 5)}
                  for name, row in value[menu]['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
