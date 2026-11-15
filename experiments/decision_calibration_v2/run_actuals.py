"""Run the frozen complete-day actual extraction on the server with four workers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run_day(day):
    output = HERE/'actual'/f'{day}.csv.gz'
    command = [sys.executable, str(HERE/'filter_actuals_remote.py'), '--date', day,
               '--metadata', str(HERE/'actual_metadata_2023-03.json'),
               '--filter', str(HERE/'filter.json'), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    (HERE/'actual'/f'{day}.log').write_text(result.stdout+result.stderr)
    print(json.dumps(dict(day=day, exit_code=result.returncode,
                          completed_at_utc=datetime.now(timezone.utc).isoformat())), flush=True)
    return dict(day=day, exit_code=result.returncode)


def main():
    manifest = json.loads((HERE/'date_manifest.json').read_text())
    if not manifest['dates_frozen']:
        raise ValueError('The declared date freeze must precede value extraction')
    (HERE/'actual').mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run_day, day) for day in manifest['selected_dates']]
        results = [job.result() for job in as_completed(jobs)]
    value = dict(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 dates_frozen_at_utc=manifest['frozen_at_utc'],
                 rows=sorted(results, key=lambda r:r['day']), max_workers=4)
    (HERE/'actual_acquisition.json').write_text(json.dumps(value, indent=2)+'\n')
    if any(row['exit_code'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
