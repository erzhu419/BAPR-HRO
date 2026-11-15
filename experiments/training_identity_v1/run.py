"""Audit V35 sources and rebuild an isolated same-estimator training candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import json
from pathlib import Path
import tarfile
from time import perf_counter
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2 as pb

from experiments.reconstruction_v34.messages import SAMPLE_SECONDS, WINDOW, snapshot_frames
from experiments.reconstruction_v35.prepare_data import endpoint_rows
from experiments.reconstruction_v35.fit import fit_emissions
from experiments.training_identity_v1.audit import audit_day, decoder_maps, canonical_frames, compare_queries

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATES = ('2023-10-01', '2023-10-02')


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as handle:
        return json.load(handle)


def write(path, value):
    with (gzip.open(path,'wt') if str(path).endswith('.gz') else Path(path).open('w')) as handle:
        json.dump(value, handle, allow_nan=False, indent=None if str(path).endswith('.gz') else 2)
        handle.write('\n')


def extract_day(day, evidence, archive):
    tick = perf_counter()
    trips, canonical = decoder_maps(evidence)
    midnight = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Zurich')).timestamp())
    frames, captures = [], []
    members = 0
    with tarfile.open(archive, 'r|bz2') as source:
        for member in source:
            if not member.name.endswith('.gtfsrt'):
                continue
            members += 1
            captured = int(datetime.fromisoformat(Path(member.name).stem).timestamp())-midnight
            if not WINDOW[0] <= captured < WINDOW[1]+60 or captured % SAMPLE_SECONDS >= 60:
                continue
            captures.append(captured)
            feed = pb.FeedMessage.FromString(source.extractfile(member).read())
            frames.extend(snapshot_frames(feed, day, midnight, captured, member.name, trips))
    frames.sort(key=lambda f:(f['receipt_seconds'],f['capture_seconds'],f['archive_member'],f['entity_id']))
    canonical_frames(frames, canonical)
    return dict(day=day, frames=frames, source=str(archive),
        summary=dict(archive_members=members, sampled_snapshots=len(captures),
                     matched_trip_messages=len(frames), distinct_trips=len({f['trip_id'] for f in frames}),
                     extraction_seconds=perf_counter()-tick),
        source_kind='User-approved third-party traines capture of the official Swiss GTFS-RT feed',
        information='Original V34 sampling and max(capture, header, TripUpdate) receipt; raw numeric forecasts unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-catalog', type=Path, default=HERE/'active_public_endpoints.json.gz')
    parser.add_argument('--original-catalog', type=Path, default=HERE/'original_active_public_endpoints.json.gz')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    original_catalog, source_catalog = read(args.original_catalog), read(args.source_catalog)
    actuals = read(ROOT/'experiments/reconstruction_data_audit/rail_calibration_dates_v2.json.gz')
    connections = {d:actuals['dates'][d]['connections'] for d in DATES}
    evidence = {d:audit_day(d, connections[d], original_catalog['dates'][d], source_catalog['dates'][d]) for d in DATES}
    write(HERE/'public_identity_mapping.json.gz', dict(days=evidence))
    write(HERE/'public_identity_summary.json', {d:evidence[d]['summary'] for d in DATES})
    print(json.dumps({d:evidence[d]['summary'] for d in DATES}), flush=True)
    if args.audit_only:
        return
    messages = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(extract_day, d, evidence[d], ROOT/f'data/swiss_rt/gtfs-rt/{d}.tar.bz2'):d for d in DATES}
        for future in as_completed(jobs):
            d = jobs[future]
            messages[d] = future.result()
            print(d,json.dumps(messages[d]['summary']),flush=True)
    write(HERE/'corrected_messages.json.gz', dict(dates=messages))
    legacy = read(ROOT/'experiments/reconstruction_v5/calibration_public_frames.json.gz')
    corrected = [row for d in DATES for row in endpoint_rows(
        d, connections[d], messages[d]['frames'], legacy['dates'][d]['frames'])]
    for row in corrected:
        publication = row['forecast_publication']
        if publication is not None:
            publication['raw_trip_id'] = messages[row['date']]['frames'][publication['frame_index']]['raw_trip_id']
    write(HERE/'corrected_training_events.json.gz', dict(train_dates=DATES, rows=corrected,
          scope='Original queries and actual labels; source-version-correct training messages only.'))
    old_rows = [r for r in read(ROOT/'experiments/reconstruction_v35/events.json.gz')['rows'] if r['split']=='training']
    comparison = compare_queries(old_rows, corrected, evidence)
    write(HERE/'query_comparison.json.gz', comparison)
    protocol = read(HERE/'protocol.json')['fit']
    kwargs = dict(training_dates=DATES, pseudocount=protocol['route_pseudocount'], minimum_sd=protocol['minimum_sd_seconds'])
    original_fit = read(ROOT/'experiments/reconstruction_v35/fitted_emission.json')
    reproduced = fit_emissions(old_rows, **kwargs)
    if reproduced != original_fit:
        raise ValueError('Original V35 estimator could not reproduce its frozen fit')
    candidate = fit_emissions(corrected, **kwargs)
    candidate['source'] = 'training_identity_v1/corrected_training_events.json.gz: original training dates and eligibility rule'
    candidate['status'] = 'isolated diagnostic candidate; not promoted to V41'
    write(HERE/'candidate_emission.json', candidate)
    parameters = []
    for scope in ('pooled','route_kind'):
        groups = [('',original_fit[scope],candidate[scope])] if scope=='pooled' else [
            (r,original_fit[scope].get(r,{}),candidate[scope].get(r,{})) for r in sorted(original_fit[scope].keys()|candidate[scope].keys())]
        for route, before, after in groups:
            for kind in sorted(before.keys()|after.keys()):
                parameters.append(dict(scope=scope,route=route,kind=kind,original=before.get(kind),corrected=after.get(kind)))
    result = dict(**comparison['summary'], original_estimator_reproduced=True,
        by_date={d:dict(identity=evidence[d]['summary'],messages=messages[d]['summary'],
                       original_eligible=original_fit['eligible_errors_by_date'][d],
                       corrected_eligible=candidate['eligible_errors_by_date'].get(d,0)) for d in DATES},
        fit_parameters=parameters, original_model_overwritten=False, physical_model_changed=False,
        validation_or_march_used_for_tuning=False, model_promoted=False, G2_passed=False)
    write(HERE/'summary.json',result)
    print(json.dumps(comparison['summary']),flush=True)


if __name__ == '__main__':
    main()
