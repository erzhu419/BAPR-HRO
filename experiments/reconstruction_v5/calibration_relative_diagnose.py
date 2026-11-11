"""Describe residual stability and observable-menu selection after evaluation."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import load, write
from calibration_relative_run import METHODS, metrics


def incoming_bin(delay):
    return 'early' if delay < 0 else '0_to_60' if delay <= 60 else '60_to_180' if delay <= 180 else 'over_180'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'calibration_relative_results.json.gz')
    args = parser.parse_args()
    results = load(args.input)
    offsets = results['offset_seconds']
    methods = list(offsets['fixed_schedule'])
    output = {}
    for day, value in results['rows_by_date'].items():
        detail = {}
        for menu in ('fixed_schedule', 'actual_current'):
            rows = value[menu]['rows']
            real = [row for row in rows if row['actual'] is not None]
            incoming = np.asarray([row['incoming_delay'] for row in real])
            outgoing = np.asarray([row['actual'] - row['scheduled'] for row in real])
            known = [row for row in rows if row['caught'] is not None]
            detail[menu] = {'n_operating_pairs': len(real), 'n_known_outcomes': len(known),
                            'status_counts': dict(Counter(row['status'] for row in rows)),
                            'mean_incoming_delay': float(incoming.mean()), 'mean_outgoing_delay': float(outgoing.mean()),
                            'mean_relative_delay': float((outgoing - incoming).mean()),
                            'std_relative_delay': float((outgoing - incoming).std()),
                            'within_date_correlation': float(np.corrcoef(incoming, outgoing)[0, 1]),
                            'observed_catch_fraction': float(np.mean([row['caught'] for row in known])),
                            'by_incoming_delay': {}}
            for name in ('early', '0_to_60', '60_to_180', 'over_180'):
                group = [row for row in rows if incoming_bin(row['incoming_delay']) == name]
                if any(row['actual'] is not None for row in group):
                    detail[menu]['by_incoming_delay'][name] = {
                        method: metrics(group, method, offsets[menu][method]) for method in methods}
        output[day] = {'menu_comparison': value['menu_comparison'], **detail}
    write(args.input.with_name(args.input.name.replace('_results.json.gz', '_diagnosis.json')), output)
    dates = (results['protocol']['confirmation_dates'] if 'confirmation_dates' in results['protocol']
             else results['protocol']['development_dates'])
    for day in dates:
        value = output[day]
        print(day, value['menu_comparison'])
        for menu in ('fixed_schedule', 'actual_current'):
            item = value[menu]
            print(menu, {key: round(item[key], 3) for key in ('mean_incoming_delay', 'mean_outgoing_delay',
                          'mean_relative_delay', 'std_relative_delay', 'within_date_correlation', 'observed_catch_fraction')})
            print('relative_coverage_by_incoming_delay', {key: (group['relative_incoming']['interval']['n'],
                  round(group['relative_incoming']['interval']['coverage'], 4))
                  for key, group in item['by_incoming_delay'].items()})


if __name__ == '__main__':
    main()
