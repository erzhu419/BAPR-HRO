"""Preserve the literal-ID receipts, repair four decoded inputs, then summarize."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from experiments.decision_gap_v1.panel import read, write
from experiments.decision_calibration_v2.run_calibration import evaluate_day, aggregate
from experiments.decision_calibration_v2.platform_announcements import run as announcements

HERE = Path(__file__).resolve().parent


def physical_receipt(result):
    return dict(roots=[(root['root_index'], root['initial_state'], root['registered'])
                       for root in result['roots']],
        events=[(row['root_index'], row['event_index'], row['event'], row['actual'])
                for row in result['events']], platform_pairs=result['platform_pairs'])


def run(directory=HERE):
    manifest = read(directory/'date_manifest.json')
    completed = read(directory/'evaluation_run.json')
    if set(completed['days']) != set(manifest['selected_dates']) or completed['errors']:
        raise ValueError('Finish the entire registered literal-ID panel first')
    correction = read(directory/'messages_corrected/summary.json')
    backup = directory/'literal_id_receipts'
    if backup.exists():
        raise ValueError('Finalization already started; retain its original receipts')
    rows = []
    for row in correction['days']:
        day = row['day']
        old = read(directory/'results'/f'{day}.json.gz')
        for folder, names in (
            ('messages', [f'{day}.json.gz', f'{day}.json.audit.json']),
            ('results', [f'{day}.json.gz'])):
            (backup/folder).mkdir(parents=True, exist_ok=True)
            for name in names:
                shutil.copy2(directory/folder/name, backup/folder/name)
        for name in (f'{day}.json.gz', f'{day}.json.audit.json'):
            shutil.copy2(directory/'messages_corrected'/name, directory/'messages'/name)
        evaluate_day(directory, day)
        new = read(directory/'results'/f'{day}.json.gz')
        if physical_receipt(old) != physical_receipt(new):
            raise AssertionError(f'An input identity repair changed physical receipts on {day}')
        rows.append(dict(day=day, physical_receipts_identical=True,
            original_matched_trips=old['message_coverage']['distinct_trips'],
            corrected_matched_trips=new['message_coverage']['distinct_trips'],
            registered_events=len(new['events']), old_receipt=f'literal_id_receipts/results/{day}.json.gz'))
    write(directory/'source_corrections.json', dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(), days=rows,
        cause='Mirror directory dates were an incorrect proxy for source GTFS activation. '
            'Literal IDs were both renamed and reused for other trains.',
        publication_evidence='https://github.com/traines-source/public-transport-statistics/blob/master/ingest/extract-gtfsrt.js:getValidUntil',
        correction='Use the independently verified public service bijection for all raw IDs on March16–19. '
            'Operator, train number, route, every regional station/sequence/clock/pickup/dropoff and platform agree.',
        mapping='static_identity_mapping.json.gz', platform_rule_check='static_transfer_version_audit.json',
        changed=['RT source-ID decoding'], unchanged=['dates','model parameters','graph','menu',
            'source connection IDs','numerical source forecasts','receipt clocks','MCT','horizon','actual labels',
            'acceptance thresholds','bootstrap budget'],
        probability_or_loss_used_to_choose_mapping=False,
        original_manifest_preserved='date_manifest.json', original_receipts='literal_id_receipts/'))
    aggregate(directory)
    announcements(directory)


if __name__ == '__main__':
    run()
