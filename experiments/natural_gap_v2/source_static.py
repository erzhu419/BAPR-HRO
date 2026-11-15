"""Prepare the two metadata-selected public feeds in an isolated server runtime."""
import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from experiments.decision_calibration_v2.prepare_static import (
    dated_connections_from_static, extract_static)
from experiments.training_identity_v1.public_catalog import build_public_catalog
from experiments.validation_identity_v1.static_acquire import download_archive, save_gzip


def freeze_public_sources(directory):
    directory = Path(directory)
    versions = {}
    for version in ('2023-03-22', '2023-03-29'):
        value = json.loads((directory / f'static_metadata_{version}.json').read_text())
        versions[version] = {key: value[key] for key in ('url', 'archive_bytes', 'feed_info')}
    dates = [(date(2023, 3, 23) + timedelta(days=i)).isoformat() for i in range(14)]
    value = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        public_schedule_dates=dates, target_dates_frozen=False,
        source_version_by_date={day: '2023-03-22' if day <= '2023-03-29' else '2023-03-29'
                                for day in dates}, versions=versions,
        metadata_evidence='static_source_metadata.json',
        rule='Collector next-ZIP mtime day14:00; retain prior release on each switch-day morning.',
        preparation_scope='Public schedules only; service dates and menus require separate main-thread freezes before actual values.',
        original_stations=16, agency='11/SBB', time_window_seconds=[27000, 34500],
        actual_or_rt_values_read=False)
    target = directory / 'static_source_manifest.json'
    if target.exists():
        raise ValueError('Public-source assignment is already frozen')
    target.write_text(json.dumps(value, indent=2) + '\n')
    return value


def acquire(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'static_source_manifest.json').read_text())
    versions = manifest['source_version_by_date']
    source_catalogs = directory / 'source_catalogs'
    source_catalogs.mkdir(parents=True, exist_ok=True)
    scheduled = dict(dates={}, source_version_by_date=versions,
                     public_source_frozen_at_utc=manifest['frozen_at_utc'],
                     actual_or_rt_values_read=False)
    catalog = dict(dates={}, source_version_by_date=versions,
                   public_source_frozen_at_utc=manifest['frozen_at_utc'],
                   actual_or_rt_values_read=False)
    receipts = []
    for version, meta in manifest['versions'].items():
        dates = [day for day, source in versions.items() if source == version]
        archive = directory / 'cache' / meta['url'].rsplit('/', 1)[1]
        transport = download_archive(meta['url'], meta['archive_bytes'], archive)
        runtime = directory / 'inputs' / f'static_{version}'
        preparation = extract_static(archive, runtime)
        if preparation['feed_info'] != meta['feed_info']:
            raise ValueError('Extracted static feed differs from frozen metadata')
        stops, connections = dated_connections_from_static(dates, runtime)
        if 'stops' in scheduled and scheduled['stops'] != stops:
            raise ValueError('Original16 station directory changed across public feeds')
        scheduled['stops'] = stops
        scheduled['dates'].update({day: dict(connections=rows, static_release=version)
                                   for day, rows in connections.items()})
        public = build_public_catalog(runtime, dates)
        public.update(static_release=version, feed_info=meta['feed_info'])
        save_gzip(source_catalogs / f'{version}.json.gz', public)
        catalog['dates'].update(public['dates'])
        receipt = dict(static_release=version, dates=dates, runtime=str(runtime),
            source_url=meta['url'], transport=transport, preparation=preparation,
            scheduled_counts={day: len(rows) for day, rows in connections.items()},
            catalog_counts={day: dict(trips=len(rows['trips']), endpoints=len(rows['endpoints']))
                            for day, rows in public['dates'].items()})
        receipts.append(receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != 'preparation'}), flush=True)
    save_gzip(directory / 'scheduled.json.gz', scheduled)
    save_gzip(directory / 'active_public_endpoints.json.gz', catalog)
    value = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        public_source_frozen_at_utc=manifest['frozen_at_utc'], versions=receipts,
        dates=sorted(scheduled['dates']),
        scheduled_bytes=(directory / 'scheduled.json.gz').stat().st_size,
        public_catalog_bytes=(directory / 'active_public_endpoints.json.gz').stat().st_size,
        original_files_changed=False, actual_or_rt_values_read=False)
    (directory / 'static_acquisition.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(stage='complete', dates=value['dates'],
                         scheduled_bytes=value['scheduled_bytes'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze-public', 'acquire'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    (freeze_public_sources if args.stage == 'freeze-public' else acquire)(args.directory)
