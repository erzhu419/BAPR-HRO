"""Verify the existing endpoint parser against all retained March rows."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from experiments.reconstruction_data_audit.actual_event_parser import parse_actual_endpoint

HERE = Path(__file__).resolve().parent


def main():
    dates = json.loads((HERE/'date_manifest.json').read_text())['selected_dates']
    output, errors = [], []
    for day in dates:
        path = HERE/'actual'/f'{day}.csv.gz'
        audit = json.loads(path.with_suffix('.audit.json').read_text())
        counts, statuses = Counter(), Counter()
        with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle, delimiter=';')
            fields = reader.fieldnames
            rows = 0
            for number, row in enumerate(reader, 2):
                rows += 1
                for kind in ('arrival', 'departure'):
                    try:
                        outcome, value, _, reported, _, raw = parse_actual_endpoint(row, kind, day)
                    except (KeyError, ValueError) as error:
                        errors.append(dict(day=day, row=number, kind=kind, error=str(error)))
                        continue
                    counts[kind, outcome] += 1
                    statuses[kind, raw] += 1
                    if value is not None and (outcome != 'REAL' or reported != value):
                        errors.append(dict(day=day, row=number, kind=kind, error='Nonactual time exposed'))
        if rows != audit['selected_rows'] or not audit['complete_daily_member']:
            errors.append(dict(day=day, error='Daily selection is incomplete'))
        output.append(dict(day=day, rows=rows, selected_bytes=path.stat().st_size,
                           scanned_daily_rows=audit['scanned_daily_rows'],
                           remote_bytes_requested=audit['remote_bytes_requested'], fields=fields,
                           outcomes=[dict(kind=k[0], outcome=k[1], count=n) for k,n in sorted(counts.items())],
                           raw_statuses=[dict(kind=k[0], status=k[1], count=n) for k,n in sorted(statuses.items())]))
    result = dict(days=len(output), selected_rows=sum(row['rows'] for row in output),
                  selected_bytes=sum(row['selected_bytes'] for row in output),
                  remote_bytes_requested=sum(row['remote_bytes_requested'] for row in output),
                  existing_parser_compatible=not errors, model_changed=False, menu_changed=False,
                  errors=errors, rows=output)
    (HERE/'actual_parser_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
