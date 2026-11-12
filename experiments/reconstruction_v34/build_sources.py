"""Assemble the fixed October rail discovery cohort from existing sources.

The optional SSH operation reads metadata and small audit JSON files only.
All actual CSV inputs are already local; absent days retain their schedule.
"""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess

from experiments.reconstruction_data_audit.build_dated_replay import (
    ARCHIVE, DEADLINE, DEPARTURE, END, OD_PAIRS, START, STATIONS,
    attach_outcomes, dated_connections,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / 'experiments/reconstruction_data_audit'
REMOTE_ROOT = '/home/erzhu419/BAPR-HRO'
REMOTE_CACHE = REMOTE_ROOT + '/experiments/reconstruction_data_audit/remote_actual_20231025'
DATES = tuple(f'2023-10-{d:02}' for d in (*range(12, 29), 30, 31))
MAIN_DATES = tuple(f'2023-10-{d:02}' for d in (25, 26, 27, 28, 30, 31))
SOURCES = {
    12: [('reconstruction_data_audit/rail_replay_sources/2023-10-12.csv.gz', False)],
    13: [('reconstruction_v5/calibration_confirmation_sources/rail_confirmation_2023-10-13.csv.gz', False)],
    14: [('reconstruction_v6/calibration_confirmation_sources/rail_v6_confirmation_2023-10-14.csv.gz', False)],
    15: [('reconstruction_v6/decision_risk_sources/rail_decision_risk_2023-10-15.csv.gz', True)],
    16: [('reconstruction_v5/calibration_relative_confirmation_sources/rail_relative_confirmation_2023-10-16.csv.gz', False)],
    17: [('reconstruction_v6/calibration_confirmation_sources/rail_v6_confirmation_2023-10-17.csv.gz', False)],
    18: [('reconstruction_v6/calibration_confirmation_sources/rail_v6_confirmation_2023-10-18.csv.gz', False)],
    19: [('reconstruction_v6/decision_risk_sources/rail_decision_risk_2023-10-19.csv.gz', True)],
    20: [('reconstruction_v5/calibration_relative_confirmation_sources/rail_relative_confirmation_2023-10-20.csv.gz', False)],
    21: [],
    22: [('reconstruction_v6/decision_risk_sources/rail_decision_risk_2023-10-22.csv.gz', True)],
    23: [('reconstruction_v6/calibration_confirmation_sources/rail_v6_confirmation_2023-10-23.csv.gz', False)],
    24: [('reconstruction_v6/calibration_confirmation_sources/rail_v6_confirmation_2023-10-24.csv.gz', False)],
    25: [('reconstruction_data_audit/rail_replay_sources/2023-10-25.csv.gz', False),
         ('reconstruction_data_audit/city_actual_rows.csv.gz', True)],
    26: [('reconstruction_v5/calibration_confirmation_sources/rail_confirmation_2023-10-26.csv.gz', False)],
    27: [('reconstruction_data_audit/rail_replay_sources/2023-10-27.csv.gz', False)],
    28: [('reconstruction_v6/calibration_confirmation_sources/rail_v6_confirmation_chunked_2023-10-28.csv.gz', True)],
    30: [('reconstruction_v5/calibration_confirmation_sources/rail_confirmation_2023-10-30.csv.gz', False)],
    31: [('reconstruction_v5/calibration_relative_confirmation_sources/rail_relative_confirmation_2023-10-31.csv.gz', False)],
}


def file_info(path):
    stat = path.stat()
    return dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size,
                modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())


def ssh_json(code, filename):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
        'jtl110gpu2', 'python3 -'], input=code, text=True, capture_output=True, check=True)
    (HERE / filename).write_text(result.stdout)
    return len(result.stdout.encode())


def refresh_remote():
    paths = [REMOTE_CACHE, REMOTE_ROOT + '/experiments/reconstruction_data_audit/rail_replay_sources',
             REMOTE_ROOT + '/experiments/reconstruction_data_audit/rail_calibration_sources',
             REMOTE_ROOT + '/data']
    directory_bytes = ssh_json('''from pathlib import Path
import json
out={}
for name in PATHS:
 p=Path(name)
 out[name]=[{"name":f.name,"bytes":f.stat().st_size if f.is_file() else None,
 "type":"file" if f.is_file() else "directory","modified_ns":f.stat().st_mtime_ns}
 for f in sorted(p.iterdir())] if p.exists() else None
print(json.dumps(out))
'''.replace('PATHS', repr(paths)), 'remote_inventory_raw.json')
    audit_bytes = ssh_json('''from pathlib import Path
import json
root=Path(CACHE)
parts=root/'actual_member_parts'
print(json.dumps({'audits':{f.name:json.loads(f.read_text()) for f in sorted(root.glob('rail*.audit.json'))},
'actual_member_parts':[{'name':f.name,'bytes':f.stat().st_size,'modified_ns':f.stat().st_mtime_ns}
for f in sorted(parts.iterdir()) if f.is_file()] if parts.exists() else []}))
'''.replace('CACHE', repr(REMOTE_CACHE)), 'remote_rail_audits.json')
    return directory_bytes + audit_bytes


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle, delimiter=';'))


def main(refresh=False):
    HERE.mkdir(exist_ok=True)
    if refresh:
        refresh_remote()
    remote = json.loads((HERE / 'remote_inventory_raw.json').read_text())
    audits = json.loads((HERE / 'remote_rail_audits.json').read_text())
    stops, scheduled = dated_connections(DATES)
    data = dict(scope='Fixed SBB physical adjacency among the original 16 Zurich stations, '
        '07:30–09:35 service clocks. All scheduled actions are retained, including missing outcomes.',
        timezone='Europe/Zurich', discovery_dates=DATES, main_discovery_dates=MAIN_DATES,
        retrospective_coverage_dates=[d for d in DATES if d not in MAIN_DATES],
        train_dates=[], holdout_dates=[],
        archive=ARCHIVE, static_feed=(ROOT / 'data/swiss_gtfs/feed_info.txt').read_text().strip(),
        stops=stops, od_queries=[dict(origin=a, destination=b, ready_seconds=DEPARTURE,
                                    deadline_seconds=DEADLINE) for a, b in OD_PAIRS], dates={})
    daily = {}
    for day in DATES:
        rows, sources = [], []
        for relative, complete in SOURCES[int(day[-2:])]:
            path = ROOT / 'experiments' / relative
            rows.extend(read_rows(path))
            audit_path = path.with_suffix('.audit.json')
            if not audit_path.exists():
                audit_path = path.with_suffix('').with_suffix('.audit.json')
            source = dict(**file_info(path), complete_daily_member=complete,
                          new_transfer_bytes=0, audit=file_info(audit_path))
            if path.name == 'city_actual_rows.csv.gz':
                source['completeness_evidence'] = ('Existing full-member range and 2,226,994 scanned '
                    'rows in audit; filter_actuals_remote.py default has no prefix limit. '
                    'Combined with the rail subset; exact endpoint conflicts stay ambiguous.')
            sources.append(source)
        connections = scheduled[day]
        coverage = attach_outcomes(day, connections, rows)
        partition = 'main_discovery' if day in MAIN_DATES else 'retrospective_coverage'
        data['dates'][day] = dict(connections=connections, coverage=coverage, sources=sources,
                                 partition=partition,
                                 source_available=bool(sources))
        daily[day] = dict(sources=sources, coverage=coverage, partition=partition, source_available=bool(sources),
            gtfs_rt_archive=file_info(ROOT / 'data/swiss_rt/gtfs-rt' / f'{day}.tar.bz2'))
        print(day, coverage, flush=True)
    with gzip.open(HERE / 'data.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(',', ':'))
    static_files = ['feed_info.txt', 'calendar.txt', 'calendar_dates.txt', 'routes.txt', 'stops.txt']
    inventory = dict(
        scope='Existing-source inventory and fixed development discovery cohort; no selection by realized gain.',
        discovery_dates=DATES, main_discovery_dates=MAIN_DATES,
        retrospective_coverage_dates=[d for d in DATES if d not in MAIN_DATES],
        partition_rule='Primary discovery dates are strictly after static feed version date 2023-10-24. '
            'Earlier dates remain retrospective coverage only; all partitions are development data.',
        available_actual_dates=[d for d in DATES if daily[d]['source_available']],
        absent_actual_dates=[d for d in DATES if not daily[d]['source_available']],
        source_rules=dict(operator_id='85:11', stations=sorted(STATIONS),
            planned_departure_interval=[START, END], endpoint_join='Service date, operator, train '
            'number, BPUIC, event type and scheduled minute; static trip/stop-sequence ambiguity retained.',
            actual_status='Only REAL is a realized time. CANCELED, PROGNOSE, ambiguity and missing stay distinct.',
            additional_runs='ZUSATZFAHRT_TF=true remains excluded under the original identity contract.',
            source_priority='Complete daily-member filtered sources replace prefixes where present; '
            'Oct25 combines existing city and rail subsets without choosing among conflicting outcomes.'),
        static_files=[file_info(ROOT / 'data/swiss_gtfs' / name) for name in static_files]
            + [file_info(ROOT / 'data/zurich_gtfs' / name) for name in ['feed_info.txt','trips.txt','stop_times.txt']],
        static_feed_version='20231024', static_validity=['2022-12-11','2023-12-09'],
        static_time_contract='Fixed historical-validity snapshot; not evidence of the snapshot published before Oct24.',
        message_time_contract='Raw GTFS-RT archives are local; actual labels are never publication receipts. '
            'Message availability must be established from archive capture and feed/header/update times separately.',
        local_gtfs_rt_archives=[file_info(p) for p in sorted((ROOT / 'data/swiss_rt/gtfs-rt').glob('*.tar.bz2'))],
        remote=dict(host='jtl110gpu2', cache=REMOTE_CACHE,
            metadata_file='remote_inventory_raw.json', audit_file='remote_rail_audits.json',
            available_rail_actual_dates=sorted({v['date'] for v in audits['audits'].values()}),
            directory_entries=remote, retained_json_payload_bytes=sum((HERE / name).stat().st_size for name
                in ['remote_inventory_raw.json','remote_rail_audits.json']),
            payload_accounting='Retained JSON responses only; SSH overhead and an earlier directory preview excluded.',
            new_csv_transfer_bytes=0, full_csv_or_checkpoint_downloaded=False,
            operation='Read existing directories and audit JSON; no new provider download or remote scientific run.'),
        dates=daily, data_file=file_info(HERE / 'data.json.gz'),
        limitations=['Coverage describes the fixed query region and time window, not the full network.',
            'Oct21 has no actual source in the known local/server cache and remains an all-missing scheduled date.',
            'Most available days are selected rows from a 1 MiB compressed archive prefix; missing is not cancellation.',
            'Every date is development discovery data; historical confirmation filenames confer no new holdout status.'])
    (HERE / 'source_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-remote', action='store_true')
    main(parser.parse_args().refresh_remote)
