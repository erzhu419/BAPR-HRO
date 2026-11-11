"""Report stated GTFS station-interior transfer times without changing batches."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def station_id(value):
    return value.split(':')[0].removeprefix('Parent')


def main():
    with gzip.open(HERE / 'dated_replay.json.gz', 'rt', encoding='utf-8') as handle:
        rail = json.load(handle)['stops']
    with gzip.open(HERE / 'city_schedule.json.gz', 'rt', encoding='utf-8') as handle:
        city = json.load(handle)['stops']
    stations = {**city, **rail}
    times, prohibited, examples = defaultdict(Counter), Counter(), defaultdict(list)
    with (ROOT / 'data/swiss_gtfs/transfers.txt').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            a, b = station_id(row['from_stop_id']), station_id(row['to_stop_id'])
            if a != b or a not in stations:
                continue
            if row['transfer_type'] == '3':
                prohibited[a] += 1
            elif row['min_transfer_time']:
                value = int(row['min_transfer_time'])
                times[a][value] += 1
                if len(examples[a]) < 3:
                    examples[a].append(row)
    results = {}
    for station, info in stations.items():
        values = times[station]
        results[station] = {'name': info['name'], 'rail_cohort': station in rail,
                            'n_stated_interior_pairs': sum(values.values()), 'prohibited_interior_pairs': prohibited[station],
                            'minimum_stated_seconds': min(values) if values else None,
                            'conservative_max_stated_seconds': max(values) if values else None,
                            'assumed120_below_minimum': 120 < min(values) if values else None,
                            'assumed120_below_maximum': 120 < max(values) if values else None,
                            'stated_time_counts': dict(sorted(values.items())), 'examples': examples[station]}
    output = {'source': 'data/swiss_gtfs/transfers.txt',
              'rule': 'At station-level resolution the maximum stated applicable interior pair time is a conservative '
              'single-station sensitivity value. Platform-specific pairs and prohibited transfers retain their own meanings. '
              'A station without a stated interior time has no inferred official minimum.',
              'unchanged_original_analysis_assumption_seconds': 120, 'stations': results}
    (HERE / 'station_transfer_bounds.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    for station in rail:
        info = results[station]
        print(station, info['name'], info['minimum_stated_seconds'], info['conservative_max_stated_seconds'],
              info['n_stated_interior_pairs'])


if __name__ == '__main__':
    main()
