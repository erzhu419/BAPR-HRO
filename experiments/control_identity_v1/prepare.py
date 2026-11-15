"""Reconstruct the six frozen control-date publication streams by public identity.

The complete source connection panel is retained. All public mappings are
written before any RT archive is read; this module does not fit or run controls.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from experiments.training_identity_v1.audit import audit_day
from experiments.training_identity_v1.public_catalog import write_public_catalog
from experiments.training_identity_v1.run import extract_day

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V34 = HERE.parent/'reconstruction_v34'


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as handle:
        return json.load(handle)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(path, 'wt') if str(path).endswith('.gz') else path.open('w')) as handle:
        json.dump(value, handle, allow_nan=False,
                  separators=(',', ':') if str(path).endswith('.gz') else None,
                  indent=None if str(path).endswith('.gz') else 2)
        handle.write('\n')


def check_frame_identity(day, connections, frames):
    """Detect a raw/source ID leaking into the frozen controller's graph."""
    canonical = {row['trip_id'] for row in connections}
    endpoints = {(row['id'], kind) for row in connections for kind in ('arrival', 'departure')}
    for frame in frames:
        if frame['trip_id'] not in canonical or any(t not in canonical for t, _ in frame['trip_statuses']):
            raise ValueError(f'{day}: a publication uses a noncanonical trip')
        if any((cid, kind) not in endpoints for cid, kind, *_ in frame['endpoint_updates']):
            raise ValueError(f'{day}: a publication addresses a noncanonical endpoint')


def main():
    contract = read(HERE/'contract.json')
    dates = contract['control_dates']
    data = read(V34/'data.json.gz')
    original_messages = read(V34/'messages.json.gz')
    connections = {day: data['dates'][day]['connections'] for day in dates}
    original_catalog_path = HERE/'original_active_public_endpoints.json.gz'
    if not original_catalog_path.exists():
        write_public_catalog(ROOT, dates, original_catalog_path)
    original_catalog = read(original_catalog_path)['dates']
    source = read(HERE/'active_public_endpoints.json.gz')
    evidence = {day: audit_day(day, connections[day], original_catalog[day], source['dates'][day])
                for day in dates}
    write(HERE/'public_identity_mapping.json.gz', dict(days=evidence))
    write(HERE/'public_identity_summary.json', {day: evidence[day]['summary'] for day in dates})
    write(HERE/'mapping_receipt.json', dict(
        written_at_utc=datetime.now(timezone.utc).isoformat(),
        contract_file=str(HERE/'contract.json'),
        mapping_file=str(HERE/'public_identity_mapping.json.gz'),
        control_dates=dates, source_version_by_date=source['source_version_by_date'],
        original_source_connections=sum(len(rows) for rows in connections.values()),
        original_endpoint_queries=sum(len(day['rows']) for day in evidence.values()),
        all_six_mappings_written_before_rt_decode=True,
        source_connections_filtered_to_menu_subset=False,
        mapping_uses_actual_or_forecast_values=False,
        unknown_identity_literal_id_fallback=False))
    print(json.dumps(dict(status='all_public_mappings_written', dates=dates)), flush=True)

    messages = {}
    unchanged_frame_dates = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(extract_day, day, evidence[day],
            ROOT/f'data/swiss_rt/gtfs-rt/{day}.tar.bz2'): day for day in dates}
        for future in as_completed(jobs):
            day = jobs[future]
            messages[day] = future.result()
            check_frame_identity(day, connections[day], messages[day]['frames'])
            if set(evidence[day]['summary']['statuses']) == {'same_literal_id'}:
                comparable = [{k: v for k, v in frame.items() if k != 'raw_trip_id'}
                              for frame in messages[day]['frames']]
                if comparable != original_messages['dates'][day]['frames']:
                    raise ValueError(f'{day}: unchanged public identities did not reproduce original frames')
                unchanged_frame_dates.append(day)
            write(HERE/'message_days'/f'{day}.json.gz', messages[day])
            print(json.dumps(dict(date=day, **messages[day]['summary'])), flush=True)

    metadata = {k: v for k, v in original_messages.items() if k != 'dates'}
    metadata['scope'] = ('Six fixed control dates for the 72 frozen expanded-menu roots; '
        'complete original source connection panel, decoded through source-date public identities.')
    metadata['identity_source'] = 'control_identity_v1/public_identity_mapping.json.gz'
    write(HERE/'messages.json.gz', dict(**metadata, dates={day: messages[day] for day in dates}))
    receipt = read(HERE/'source_receipts.json')
    receipt.update(
        scope='Six control-date publication streams reconstructed from existing local archives; '
            'all public mappings written before RT decoding. Complete source panels retained.',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), raw_archives_decoded=True,
        contract_file=str(HERE/'contract.json'),
        public_source_catalog=str(HERE/'active_public_endpoints.json.gz'),
        original_catalog=str(original_catalog_path),
        mapping_receipt=str(HERE/'mapping_receipt.json'),
        public_identity_summaries={day: evidence[day]['summary'] for day in dates},
        message_summaries={day: messages[day]['summary'] for day in dates},
        unchanged_frame_dates=sorted(unchanged_frame_dates),
        unchanged_frame_comparison='All original frame fields compared exactly; only added raw_trip_id excluded.',
        raw_forecast_values_modified=False, raw_availability_clocks_modified=False,
        canonical_consumed_histories='Source identities corrected; resulting posterior histories may change.',
        source_connection_ids_and_canonical_trips_preserved=True,
        original_v34_inputs_modified=False, fitted_parameters_changed=False,
        menus_modified=False, controls_executed=False, new_control_scores_read=False,
        missing_sources_dropped=False)
    write(HERE/'source_receipts.json', receipt)
    print(json.dumps(dict(status='control_messages_complete', dates=dates,
        matched_trip_messages=sum(len(value['frames']) for value in messages.values()))), flush=True)


if __name__ == '__main__':
    main()
