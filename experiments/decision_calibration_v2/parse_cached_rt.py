"""Filter completed server archives after the public menu selection is frozen."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import sleep

from experiments.decision_calibration_v2.fetch_messages import extract_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory/'date_manifest.json').read_text())
    selection_path = args.directory/'selection.json'
    selection = json.loads(selection_path.read_text())
    with gzip.open(args.directory/'scheduled.json.gz', 'rt') as handle:
        scheduled = json.load(handle)
    directory = args.directory/'messages'
    directory.mkdir(exist_ok=True)
    remaining = {row['day']:row['rt'] for row in manifest['days']}
    audit = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        date_freeze_utc=manifest['frozen_at_utc'],
        selection_file=str(selection_path), days={}, errors={})
    audit_path = args.directory/'messages_summary.json'
    jobs = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        while remaining or jobs:
            for day, source in tuple(remaining.items()):
                target = directory/f'{day}.json.gz'
                if target.exists():
                    with gzip.open(target, 'rt') as handle:
                        saved = json.load(handle)
                    target.with_suffix('.audit.json').write_text(json.dumps(dict(day=day,
                        output_complete=True, filtered_bytes=target.stat().st_size,
                        summary=saved['summary']), indent=2)+'\n')
                    audit['days'][day] = saved['summary']
                    del remaining[day]
                    continue
                archive = args.cache/source['filename']
                if not archive.exists() or archive.stat().st_size != source['size_bytes']:
                    continue
                jobs[pool.submit(extract_messages, day,
                    scheduled['dates'][day]['connections'], archive, target)] = day
                del remaining[day]
                print(json.dumps(dict(date=day, status='parsing_complete_cached_archive')), flush=True)
            for future, day in tuple(jobs.items()):
                if not future.done():
                    continue
                try:
                    result = future.result()
                    audit['days'][day] = result['summary']
                    summary = result['summary']
                    print(json.dumps(dict(date=day, status='extracted',
                        captures=summary['sampled_snapshots'], messages=summary['matched_trip_messages'],
                        distinct_trips=summary['distinct_trips'],
                        extraction_seconds=summary['extraction_seconds'])), flush=True)
                except Exception as error:
                    audit['errors'][day] = repr(error)
                    print(json.dumps(dict(date=day, status='failed', error=repr(error))), flush=True)
                del jobs[future]
            audit['waiting_for_archive'] = sorted(remaining)
            audit['parsing_dates'] = sorted(jobs.values())
            audit_path.write_text(json.dumps(audit, indent=2)+'\n')
            if remaining or jobs:
                sleep(5)
    if audit['errors']:
        raise RuntimeError(f'Message extraction failed for {sorted(audit["errors"])}')


if __name__ == '__main__':
    main()
