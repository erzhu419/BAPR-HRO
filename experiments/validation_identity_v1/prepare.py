"""Prepare the fixed nine-day V35 validation inputs with public source identities.

All public mappings are written before opening an RT archive. The corrected
training rows are copied unchanged; this stage does not train a model.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from experiments.reconstruction_v35.prepare_data import endpoint_rows
from experiments.training_identity_v1.audit import audit_day, compare_queries
from experiments.training_identity_v1.public_catalog import write_public_catalog
from experiments.training_identity_v1.run import extract_day

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = HERE.parent/'training_identity_v1'
V35 = HERE.parent/'reconstruction_v35'


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


def main():
    # Reading the frozen protocol is required before source reconstruction.
    protocol = read(HERE/'protocol.json')
    original_events = read(V35/'events.json.gz')
    dates = original_events['validation_dates']
    before = [row for row in original_events['rows'] if row['split']=='validation']
    corrected_training = read(TRAINING/'corrected_training_events.json.gz')
    training_messages = read(TRAINING/'corrected_messages.json.gz')['dates']
    original_catalog_path = HERE/'original_active_public_endpoints.json.gz'
    if not original_catalog_path.exists():
        write_public_catalog(ROOT, dates, original_catalog_path)
    original_catalog = read(original_catalog_path)['dates']
    source = read(HERE/'active_public_endpoints.json.gz')
    actuals = read(HERE.parent/'reconstruction_data_audit/rail_calibration_dates_v2.json.gz')
    connections = {day:actuals['dates'][day]['connections'] for day in dates}
    evidence = {day:audit_day(day, connections[day], original_catalog[day], source['dates'][day])
                for day in dates}
    write(HERE/'public_identity_mapping.json.gz', dict(days=evidence))
    write(HERE/'public_identity_summary.json', {day:evidence[day]['summary'] for day in dates})
    mapping_receipt = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_file=str(HERE/'protocol.json'), mapping_file=str(HERE/'public_identity_mapping.json.gz'),
        all_validation_dates=list(dates), original_query_endpoints=len(before),
        source_version_by_date=source.get('source_version_by_date'),
        all_nine_mappings_written_before_rt_decode=True,
        mapping_uses_actual_or_forecast_values=False,
        model_fitted=False, candidate_modified=False)
    write(HERE/'mapping_receipt.json', mapping_receipt)
    print(json.dumps(dict(status='all_public_mappings_written', dates=dates,
                          query_endpoints=len(before))), flush=True)

    messages = dict(training_messages)
    for day, value in training_messages.items():
        write(HERE/'message_days'/f'{day}.json.gz', value)
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(extract_day, day, evidence[day],
            ROOT/f'data/swiss_rt/gtfs-rt/{day}.tar.bz2'):day for day in dates}
        for future in as_completed(jobs):
            day = jobs[future]
            messages[day] = future.result()
            write(HERE/'message_days'/f'{day}.json.gz', messages[day])
            print(json.dumps(dict(date=day, **messages[day]['summary'])), flush=True)
    legacy = read(HERE.parent/'reconstruction_v5/calibration_public_frames.json.gz')['dates']
    after = [row for day in dates for row in endpoint_rows(
        day, connections[day], messages[day]['frames'], legacy[day]['frames'])]
    for row in after:
        publication = row['forecast_publication']
        if publication is not None:
            frame = messages[row['date']]['frames'][publication['frame_index']]
            publication['raw_trip_id'] = frame['raw_trip_id']
    comparison = compare_queries(before, after, evidence)
    write(HERE/'query_comparison.json.gz', comparison)
    write(HERE/'events.json.gz', dict(train_dates=corrected_training['train_dates'],
        validation_dates=dates, rows=corrected_training['rows']+after,
        scope='Unmodified corrected training rows and all original validation queries/labels; '
            'validation publications decoded through source-date public endpoint identities.'))
    write(HERE/'messages.json.gz', dict(dates=messages,
        scope='Original V34 sampling and V35 latest-valid-publication logic. '
            'Only source identities repaired; no fitted model or query/label changes.'))
    receipt = read(HERE/'source_receipts.json')
    receipt.update(scope='Fixed validation-source reconstruction; every original query and unresolved '
        'source retained. All nine public mappings were written before RT decoding.',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), raw_archives_decoded=True,
        protocol_file=str(HERE/'protocol.json'),
        public_source_catalog=str(HERE/'active_public_endpoints.json.gz'),
        original_catalog=str(original_catalog_path),
        mapping_receipt=str(HERE/'mapping_receipt.json'),
        corrected_training_rows=len(corrected_training['rows']),
        corrected_training_rows_copied_without_changes=True,
        validation_queries_preserved_by_compare_queries=True,
        validation_query_comparison_summary=comparison['summary'],
        message_summaries={day:messages[day]['summary'] for day in dates},
        fitted_or_candidate_parameters_changed=False, validation_scores_read=False,
        original_v35_inputs_modified=False, missing_queries_dropped=False)
    write(HERE/'source_receipts.json', receipt)
    print(json.dumps(dict(status='validation_inputs_complete',
        training_rows=len(corrected_training['rows']), validation_rows=len(after),
        original_queries_and_labels_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
