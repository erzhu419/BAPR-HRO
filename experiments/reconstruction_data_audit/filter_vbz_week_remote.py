"""Filter the complete municipal weekly file remotely to selected operating peaks."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

SOURCE = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd_2023/download/Fahrzeiten_SOLL_IST_20230723_20230729.csv'
DATES = ('2023-07-25', '2023-07-26', '2023-07-27')
START, END = 7 * 3600 + 30 * 60, 9 * 3600 + 35 * 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.source_cache.exists():
        temporary = args.source_cache.with_suffix('.csv.partial')
        with urlopen(SOURCE, timeout=90) as response, temporary.open('wb') as handle:
            copyfileobj(response, handle, 1024 * 1024)
        temporary.replace(args.source_cache)
    counts = Counter()
    scanned = 0
    with args.source_cache.open(encoding='utf-8-sig', newline='') as handle, gzip.open(
        args.output, 'wt', encoding='utf-8', newline='') as output:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(output, reader.fieldnames)
        writer.writeheader()
        for row in reader:
            scanned += 1
            day = datetime.strptime(row['betriebsdatum'], '%d.%m.%y').date().isoformat()
            if day not in DATES or not row['soll_ab_von']:
                continue
            dep = int(row['soll_ab_von'])
            offset = (datetime.strptime(row['datum_von'], '%d.%m.%y').date()
                      - datetime.strptime(row['betriebsdatum'], '%d.%m.%y').date()).days * 86400
            if not START <= dep + offset < END:
                continue
            writer.writerow(row)
            counts[day, row['linie']] += 1
    audit = {'source': SOURCE, 'remote_source_cache': str(args.source_cache),
             'remote_source_bytes': args.source_cache.stat().st_size, 'scanned_weekly_rows': scanned,
             'local_subset_bytes': args.output.stat().st_size, 'window_seconds': [START, END],
             'rows_by_date': {day: sum(n for (date_, line), n in counts.items() if date_ == day) for day in DATES},
             'rows_by_date_line': [{'date': day, 'line': line, 'rows': n} for (day, line), n in sorted(counts.items())]}
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
