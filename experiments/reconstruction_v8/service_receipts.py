"""Check the archived feed for contemporaneous evidence of the Oct27 failure."""
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import tarfile
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v5.calibration_run import data_inputs
from src.stop_service_status import stop_status_at


def main():
    diagnosis = json.loads((ROOT/'experiments/reconstruction_v6/decision_risk_failure_diagnosis.json').read_text())
    failure, future = diagnosis['failure'], diagnosis['future_source']
    day, cutoff = failure['date'], failure['time']
    tid = future['trip_id']
    epoch = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    source = ROOT/'data'/'swiss_rt'/'gtfs-rt'/f'{day}.tar.bz2'
    with gzip.open(ROOT/'experiments/reconstruction_v6/decision_risk_development_results.json.gz','rt') as handle:
        selected = [row for row in json.load(handle)['dates'][day]['rows']
                    if row.get('selected') and row['group'] == 'late']
    connections = {row['id']: row for row in data_inputs()['dates'][day]['connections']}
    stops_by_trip, updates_by_trip = {}, {}
    for row in selected:
        connection = connections[row['outgoing']]
        stops_by_trip.setdefault(row['trip_id'], {}).update({
            connection['dep_sequence']:connection['dep_stop'], connection['arr_sequence']:connection['arr_stop']})
        updates_by_trip[row['trip_id']] = []
    frames, revisions, prior = [], [], None
    first, last = min(future['scheduled_departure_seconds'], cutoff)-3600, cutoff+3600
    scan_end = max(last, max(row['time'] for row in selected))
    with tarfile.open(source, 'r|bz2') as archive:
        for member in archive:
            if not member.name.endswith('.gtfsrt'):
                continue
            capture = int(datetime.fromisoformat(Path(member.name).stem).timestamp())-epoch
            if not first <= capture <= scan_end:
                continue
            feed = pb.FeedMessage.FromString(archive.extractfile(member).read())
            header = feed.header.timestamp-epoch if feed.header.HasField('timestamp') else capture
            observations = []
            for entity in feed.entity:
                if not entity.HasField('trip_update'):
                    continue
                update = entity.trip_update
                trip = update.trip.trip_id
                if trip not in stops_by_trip or update.trip.start_date != day.replace('-',''):
                    continue
                endpoints = stops_by_trip[trip]
                stamp = update.timestamp-epoch if update.HasField('timestamp') else capture
                receipt = max(capture, header, stamp)
                trip_status = pb.TripDescriptor.ScheduleRelationship.Name(update.trip.schedule_relationship)
                stops = []
                for stop in update.stop_time_update:
                    if stop.stop_sequence not in endpoints or (stop.stop_id and
                            stop.stop_id.split(':')[0] != endpoints[stop.stop_sequence]):
                        continue
                    status = pb.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(stop.schedule_relationship)
                    estimates = {}
                    for kind in ('arrival', 'departure'):
                        if stop.HasField(kind):
                            value = getattr(stop, kind)
                            estimates[kind] = dict(time=value.time-epoch if value.HasField('time') else None,
                                                   delay=value.delay if value.HasField('delay') else None)
                    stops.append(dict(sequence=stop.stop_sequence, stop_id=stop.stop_id,
                                      status=status, forecasts=estimates))
                entry = dict(receipt_seconds=receipt, status=trip_status, stops=stops)
                updates_by_trip[trip].append(entry)
                if trip == tid:
                    observations.append(entry)
            if capture <= last:
                frames.append(dict(capture_seconds=capture, archive_member=member.name,
                                   header_seconds=header, target=observations))
    frames.sort(key=lambda row: row['capture_seconds'])
    precutoff = []
    for frame in frames:
        status = [{k:v for k,v in row.items() if k!='receipt_seconds'} for row in frame['target']]
        if status != prior:
            revisions.append(frame)
            prior = status
        precutoff.extend(x for x in frame['target'] if x['receipt_seconds'] <= cutoff)
    flags = [(x['receipt_seconds'], x['status'], tuple(s['status'] for s in x['stops']))
             for x in precutoff if x['status'] in ('CANCELED','DELETED')
             or any(s['status']=='SKIPPED' for s in x['stops'])]
    after = [x['receipt_seconds'] for frame in frames for x in frame['target']
             if x['receipt_seconds'] > cutoff and (x['status'] in ('CANCELED','DELETED')
                 or any(s['status']=='SKIPPED' for s in x['stops']))]
    cohort = []
    for row in selected:
        connection = connections[row['outgoing']]
        available = [x for x in updates_by_trip[row['trip_id']] if x['receipt_seconds']<=row['time']]
        current = max(available, key=lambda x:x['receipt_seconds'], default=None)
        status = stop_status_at(available, row['time'], connection['arr_sequence'])
        veto = status == 'SKIPPED' or (current is not None and current['status'] in ('CANCELED','DELETED'))
        cohort.append(dict(incoming=row['incoming'],outgoing=row['outgoing'],time=row['time'],
            trip_id=row['trip_id'], stop_status=status, feed_updates=len(available), veto=veto,
            outcome=row['outcome'], risk_lower=row['risk_lower'], risk_upper=row['risk_upper']))
    accepted = [row for row in cohort if not row['veto']]
    summary = dict(original_n=len(cohort), retained_n=len(accepted),
        vetoed_outcomes=dict(Counter(row['outcome'] for row in cohort if row['veto'])),
        retained_mean_risk_bounds=[sum(row[key] for row in accepted)/len(accepted)
                                   for key in ('risk_lower','risk_upper')])
    result = dict(source=str(source.relative_to(ROOT)), scope='All captured snapshots in the declared local window, exact dated trip and platform-aggregated stop sequence.',
        trip_id=tid, scoring_label=future['arrival']['status'], cutoff=cutoff,
        window=[first,last], captures=len(frames), captures_at_or_before_cutoff=sum(f['capture_seconds']<=cutoff for f in frames),
        available_exact_trip_updates=len(precutoff), precutoff_trip_status_counts=dict(Counter(x['status'] for x in precutoff)),
        precutoff_stop_status_counts=dict(Counter(s['status'] for x in precutoff for s in x['stops'])),
        explicit_failure_before_cutoff=flags, first_explicit_failure_after_cutoff=min(after,default=None),
        revisions=revisions, late_selected=cohort, late_selected_summary=summary,
        late_selected_capture_window=[first,scan_end])
    (HERE/'service_receipts_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('revisions','late_selected')},indent=2),flush=True)


if __name__ == '__main__':
    main()
