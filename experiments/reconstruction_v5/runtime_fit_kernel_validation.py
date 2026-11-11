"""Compare training elapsed moments to the fixed, deployed cold CDF rows."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ordered_duration_belief import elapsed_clock_log_mass


def main():
    folder = Path(__file__).resolve().parent
    fitted = json.loads((folder / 'ordered_duration_training_fit.json').read_text())
    rows = []
    for row in fitted['duration_bins']:
        fit = fitted['fits'][row['kind']]['continuous']
        duration = row['scheduled_seconds']
        shifts = np.unique(np.r_[np.arange(-300, 3841, 60), np.arange(-300, 301, 20)])
        clocks = duration + shifts
        probabilities = np.exp(elapsed_clock_log_mass(np.array([0]), clocks,
                              duration + fit['location_seconds'], fit['scale_seconds'])[0])
        mean = float(clocks @ probabilities)
        sd = float(np.sqrt(probabilities @ (clocks - mean) ** 2))
        rows.append(dict(row, cdf20_kernel_mean=mean, cdf20_kernel_sd=sd,
                         continuous_fit_sd=fit['scale_seconds']))
    result = {'scope': 'Training-only conditional transition row, preceding delay zero; finite cold supports '
                       'and CDF cells are deployed, not infinite continuous moments. '
                       'Full trip marginals additionally condition/order all events.',
              'fine_grid': {'seconds': 20, 'central_delay_seconds': [-300, 300], 'original_tail_seconds': 60},
              'rows': rows}
    output = folder / 'ordered_duration_kernel_validation.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('Saved', output)


if __name__ == '__main__':
    main()
