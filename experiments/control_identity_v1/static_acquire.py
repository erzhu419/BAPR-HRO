"""Acquire the missing public release, then merge metadata-selected catalogs."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from experiments.decision_calibration_v2.prepare_static import extract_static
from experiments.training_identity_v1.public_catalog import build_public_catalog
from experiments.validation_identity_v1.static_acquire import download_archive, save_gzip


def acquire(directory):
    """Run on the server: only Oct18 is missing from the existing static data."""
    directory = Path(directory)
    manifest = json.loads((directory / 'static_source_manifest.json').read_text())
    version = manifest['missing_release']
    meta = manifest['versions'][version]
    dates = [day for day, source in manifest['source_version_by_date'].items() if source == version]
    archive = directory / 'cache' / meta['url'].rsplit('/', 1)[1]
    transport = download_archive(meta['url'], meta['archive_bytes'], archive)
    runtime = directory / 'inputs' / f'static_{version}'
    preparation = extract_static(archive, runtime)
    if preparation['feed_info'] != meta['feed_info']:
        raise ValueError('Prepared feed metadata differs from frozen source')
    catalog = build_public_catalog(runtime, dates)
    catalog.update(static_release=version, feed_info=meta['feed_info'])
    target = directory / 'source_catalogs' / f'{version}.json.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    save_gzip(target, catalog)
    receipt = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        static_release=version, dates=dates, runtime=str(runtime), source_url=meta['url'],
        source_metadata_frozen_at_utc=manifest['frozen_at_utc'], transport=transport,
        preparation=preparation, catalog_file=str(target), catalog_bytes=target.stat().st_size,
        counts={day: dict(trips=len(row['trips']), endpoints=len(row['endpoints']))
                for day, row in catalog['dates'].items()},
        actual_rt_control_values_or_models_read=False)
    (directory / 'static_acquisition_server.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'preparation'}), flush=True)


def merge(directory):
    """Reuse the verified Oct25 release locally; no archive/CSV is transferred."""
    directory = Path(directory)
    manifest = json.loads((directory / 'static_source_manifest.json').read_text())
    version_by_date = manifest['source_version_by_date']
    reused = manifest['reused_local_release']['static_release']
    original_path = directory / 'original_active_public_endpoints.json.gz'
    with gzip.open(original_path, 'rt') as handle:
        original = json.load(handle)
    dates = [day for day, source in version_by_date.items() if source == reused]
    reused_catalog = dict(dates={day: original['dates'][day] for day in dates},
        static_release=reused, feed_info=manifest['versions'][reused]['feed_info'],
        reused_catalog=str(original_path),
        provenance=manifest['reused_local_release'], actual_rt_control_values_or_models_read=False)
    catalog_dir = directory / 'source_catalogs'
    catalog_dir.mkdir(parents=True, exist_ok=True)
    save_gzip(catalog_dir / f'{reused}.json.gz', reused_catalog)
    merged = dict(dates={}, source_version_by_date=version_by_date,
        source_metadata_frozen_at_utc=manifest['frozen_at_utc'],
        scope='Full-day active SBB named trips and planned endpoints at the original16 stations',
        actual_rt_control_values_or_models_read=False)
    for version in sorted(set(version_by_date.values())):
        with gzip.open(catalog_dir / f'{version}.json.gz', 'rt') as handle:
            catalog = json.load(handle)
        expected = {day for day, source in version_by_date.items() if source == version}
        if set(catalog['dates']) != expected:
            raise ValueError('Catalog dates differ from metadata-frozen source assignment')
        merged['dates'].update(catalog['dates'])
    target = directory / 'active_public_endpoints.json.gz'
    save_gzip(target, merged)
    receipt = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_metadata_frozen_at_utc=manifest['frozen_at_utc'],
        source_version_by_date=version_by_date, merged_catalog=str(target),
        merged_catalog_bytes=target.stat().st_size,
        server_acquisition_receipt='static_acquisition_server.json',
        local_reuse_provenance=manifest['reused_local_release'],
        counts={day: dict(trips=len(row['trips']), endpoints=len(row['endpoints']))
                for day, row in merged['dates'].items()},
        large_csv_or_zip_pulled_from_server=False,
        old_data_models_menus_or_previous_stage_changed=False,
        actual_rt_control_values_or_models_read=False)
    (directory / 'static_acquisition.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('acquire', 'merge'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    (acquire if args.stage == 'acquire' else merge)(args.directory)
