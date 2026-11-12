"""Bounded information coverage and missing-endpoint recovery for V32.

Reads existing local archives. Optional --remote scans the known weekly CSV
on its existing server and transfers only counts and selected endpoint rows.
Neither archive endpoints nor their municipal extraction date are presented
as a message publication time.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import gzip
import json
from pathlib import Path
import subprocess
from time import perf_counter

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_data_audit import build_vbz_replay_sample as join
from experiments.reconstruction_data_audit.build_vbz_city_replay import STOP_NAME_ALIASES


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V32 = HERE.parent/'reconstruction_v32'
SOURCE = HERE.parent/'reconstruction_v25'
AUDIT = HERE.parent/'reconstruction_data_audit'
REMOTE_CACHE = '/home/erzhu419/BAPR-HRO/experiments/reconstruction_data_audit/remote_actual_20231025/vbz_week_20230716_20230722.csv'


def load_json(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as handle:
        return json.load(handle)


def blockers(root, outcome, metadata):
    if outcome['status'] == 'unresolved_first_departure':
        candidates = [root['first_cid']]
    elif outcome['tail_trace']:
        last = outcome['tail_trace'][-1]
        hub = root['hubs'][outcome['trace'][-1]['stage']]
        option = hub['options'][last['variant']]
        candidates = list(option['first_path'])+[p[0] for p in option['tail_paths']]
        if last.get('final_tail') is not None:
            candidates += next(p for p in option['tail_paths'] if p[0] == last['final_tail'])
    else:
        hub = root['hubs'][outcome['trace'][-1]['stage']]
        candidates = list(map(int, hub['variants']))
    return [cid for cid in sorted(set(candidates))
            if any(metadata[cid][kind]['status'] != 'REAL' for kind in ('departure', 'arrival'))]


def provider(row):
    return row['fahrt_id'], row['fahrzeug']


def evidence_match(target, metadata, raw, crosswalk):
    """Recover only an exact planned-minute row with an anchored vehicle run.

    The frozen archive supplies the named GTFS trip -> municipal vehicle run
    identity through its already matched edges. Source sequence offsets must
    agree across those anchors and the proposed edge. This repairs the later
    stop-name table's missing historical alias without changing the clocks.
    """
    known = [r for r in metadata.values() if r['trip_id'] == target['trip_id']
             and r['departure']['status'] == 'REAL' and r['arrival']['status'] == 'REAL']
    identities = {r['departure']['official_journey'] for r in known}
    candidates = [r for r in raw if r['linie'] == target['route_name']
        and int(r['soll_ab_von'])//60 == target['scheduled_departure_seconds']//60
        and int(r['soll_an_nach'])//60 == target['scheduled_arrival_seconds']//60]
    linked = [r for r in candidates if any(identity.endswith(':'+':'.join(provider(r))) for identity in identities)]
    anchors, offsets = [], set()
    for edge in known:
        matches = [r for r in raw if edge['departure']['official_journey'].endswith(':'+':'.join(provider(r)))
            and r['ist_ab_von'] and r['ist_an_nach1']
            and int(r['ist_ab_von']) == edge['departure']['time_seconds']
            and int(r['ist_an_nach1']) == edge['arrival']['time_seconds']]
        if len(matches) == 1:
            raw_edge = matches[0]
            offset = (int(raw_edge['seq_von'])-edge['dep_sequence'],
                      int(raw_edge['seq_nach'])-edge['arr_sequence'])
            offsets.add(offset)
            anchors.append(dict(cid=edge['id'], departure=edge['departure']['time_seconds'],
                arrival=edge['arrival']['time_seconds'], sequence_offset=offset))
    proof = dict(known_vehicle_runs=sorted(identities), exact_minute_candidate_count=len(candidates),
                 anchored_candidate_count=len(linked), anchor_count=len(anchors), sequence_offsets=sorted(offsets))
    if len(linked) != 1 or len(offsets) != 1:
        return None, proof
    row = linked[0]
    offset = int(row['seq_von'])-target['dep_sequence'], int(row['seq_nach'])-target['arr_sequence']
    if offset not in offsets or not row['ist_ab_von'] or not row['ist_an_nach1']:
        return None, proof
    if int(row['ist_ab_von']) > int(row['ist_an_nach1']):
        return None, proof
    for side in ('von', 'nach'):
        mapped = crosswalk.get(row['halt_id_'+side])
        expected = target['dep_stop' if side == 'von' else 'arr_stop']
        if mapped is not None and mapped != expected:
            return None, proof
    proof.update(source_row=row, anchors=anchors,
        missing_historical_stop_mappings={row['halt_id_'+side]: target['dep_stop' if side == 'von' else 'arr_stop']
            for side in ('von', 'nach') if row['halt_id_'+side] not in crosswalk})
    return row, proof


def remote_audit(request):
    code = '''import csv,json
from collections import Counter
from pathlib import Path
request=REQUEST
source=Path(request['source'])
counts=Counter(); runs={key:[] for key in request['runs']}; scanned=0
with source.open(encoding='utf-8-sig',newline='') as handle:
 for row in csv.DictReader(handle):
  scanned+=1
  if row['betriebsdatum'] not in request['dates']:continue
  if row['linie'] in request['routes']:counts[row['betriebsdatum']+'|'+row['linie']]+=1
  key='|'.join((row['betriebsdatum'],row['fahrt_id'],row['fahrzeug']))
  if key in runs:runs[key].append(row)
summary={}
for key,rows in runs.items():
 rows.sort(key=lambda row:int(row['seq_von']))
 summary[key]={'rows':len(rows),'first':rows[0] if rows else None,'last':rows[-1] if rows else None}
rt=Path('/home/erzhu419/BAPR-HRO/data/swiss_rt/gtfs-rt')
print(json.dumps({'source':str(source),'source_bytes':source.stat().st_size,'scanned_rows':scanned,
 'whole_day_line_rows':dict(counts),'vehicle_runs':summary,
 'rt_archive_directory_exists':rt.exists(),'rt_archive_dates':[p.name for p in rt.glob('*.tar.bz2')]}))
'''.replace('REQUEST', repr(request))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
        'jtl110gpu2', 'python3 -'], input=code, text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    output['transferred_json_bytes'] = len(result.stdout.encode())
    return output


def followup_endpoint():
    """The newly reached root9 event: retain conflicting arrival labels."""
    day, cid = '2023-07-17', 10818
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    metadata = dated_graph(data, day, 'gtfs_station_max', True)[4]
    target = metadata[cid]
    with gzip.open(SOURCE/'v25_peak_rows.csv.gz', 'rt') as handle:
        raw = [r for r in csv.DictReader(handle) if r['betriebsdatum'] == '17.07.23']
    path = AUDIT/'vbz_stop_mapping.csv'
    crosswalk = join.stop_crosswalk(ROOT/'data/swiss_gtfs', path.read_bytes())
    with path.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    _, proof = evidence_match(target, metadata, raw, crosswalk)
    identities, offsets = set(proof['known_vehicle_runs']), set(map(tuple, proof['sequence_offsets']))
    candidates = [r for r in raw if r['linie'] == target['route_name']
        and crosswalk.get(r['halt_id_von']) == target['dep_stop']
        and crosswalk.get(r['halt_id_nach']) == target['arr_stop']
        and int(r['soll_ab_von'])//60 == target['scheduled_departure_seconds']//60
        and int(r['soll_an_nach'])//60 == target['scheduled_arrival_seconds']//60
        and f'municipal:{day}:{r["fahrt_id"]}:{r["fahrzeug"]}' in identities
        and (int(r['seq_von'])-target['dep_sequence'],int(r['seq_nach'])-target['arr_sequence']) in offsets]
    supplement = dict(day=day,local_city_cid=cid,trip_id=target['trip_id'],
                      dep_sequence=target['dep_sequence'],arr_sequence=target['arr_sequence'])
    endpoint_values = {}
    for kind, actual, planned in [('departure','ist_ab_von','soll_ab_von'),('arrival','ist_an_nach1','soll_an_nach')]:
        endpoint_values[kind] = sorted({int(r[actual]) for r in candidates if r[actual]})
        values = {(r[actual],r[planned],r['fahrt_id'],r['fahrzeug']) for r in candidates}
        if len(values) == 1 and all(value[0] for value in values) and len(identities) == len(offsets) == 1:
            actual_time, planned_time, provider_trip, vehicle = next(iter(values))
            supplement[kind] = dict(status='REAL',time_seconds=int(actual_time),
                provider_scheduled_seconds=int(planned_time),official_journey=f'municipal:{day}:{provider_trip}:{vehicle}')
    recovered = [kind for kind in ('departure','arrival') if kind in supplement]
    result = dict(root_id='onboard_2023-07-17_8313_84',day=day,local_city_cid=cid,
        identity={k:target[k] for k in ('trip_id','route_name','dep_stop','arr_stop','dep_sequence','arr_sequence')},
        source='Existing v25_peak_rows.csv.gz; zero additional network transfer.',
        proof=proof,matched_source_rows=candidates,endpoint_values=endpoint_values,
        recovered_endpoints=recovered,
        interpretation='The whole-connection join rejected conflicting arrivals. The two source records agree on departure, so only that endpoint is restored. Arrival remains unresolved; no source row is selected by outcome.')
    (HERE/'followup_endpoint_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (HERE/'endpoint_supplement_followup.json').write_text(json.dumps(dict(
        source='Local endpoint-specific consensus within the uniquely anchored same-vehicle source records.',
        records=[supplement] if recovered else []),ensure_ascii=False,indent=2)+'\n')
    if recovered:
        combined = load_json(HERE/'endpoint_supplement.json')
        if not any((row['day'],row['local_city_cid']) == (day,cid) for row in combined['records']):
            combined['records'].append(supplement)
        combined['partial_endpoint_rule'] = ('The root9 follow-up restores departure only when both '
            'conflicting same-vehicle rows agree; the disputed arrival remains absent.')
        (HERE/'endpoint_supplement.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cid=cid,matched_rows=len(candidates),recovered=recovered,endpoint_values=endpoint_values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--followup', action='store_true')
    args = parser.parse_args()
    if args.followup:
        followup_endpoint()
        return
    tick = perf_counter()
    roots = load_json(V32/'selection_full_hubs.json')['selected']
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    dates = {day: dated_graph(data, day, 'gtfs_station_max', True) for day in data['dates']}
    source_by_day = defaultdict(list)
    with gzip.open(SOURCE/'v25_peak_rows.csv.gz', 'rt') as handle:
        for row in csv.DictReader(handle):
            source_by_day[row['betriebsdatum']].append(row)
    mapping_path = AUDIT/'vbz_stop_mapping.csv'
    crosswalk = join.stop_crosswalk(ROOT/'data/swiss_gtfs', mapping_path.read_bytes())
    stop_names = {}
    with mapping_path.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            stop_names[row['halt_id']] = row['halt_lang']
            if row['halt_lang'] in STOP_NAME_ALIASES:
                crosswalk[row['halt_id']] = STOP_NAME_ALIASES[row['halt_lang']]
    records, supplements, missing_roots, runs = [], [], [], set()
    for index, root in enumerate(roots):
        outcome = load_json(V32/'roots_full'/f'{index:02}.json.gz')['actual']['pair_one_step_rollout']
        if outcome['loss'] is not None:
            continue
        day = root['day']
        date_key = date.fromisoformat(day).strftime('%d.%m.%y')
        metadata = dates[day][4]
        cids = blockers(root, outcome, metadata)
        missing_roots.append(dict(root_id=root['root_id'], status=outcome['status'], required_cids=cids))
        for cid in cids:
            target = metadata[cid]
            matched, proof = evidence_match(target, metadata, source_by_day[date_key], crosswalk)
            record = dict(root_id=root['root_id'], day=day, local_city_cid=cid,
                identity={k:target[k] for k in ('trip_id','route_id','route_name','agency_id','dep_stop','arr_stop',
                    'dep_sequence','arr_sequence','scheduled_departure_seconds','scheduled_arrival_seconds')},
                local_source_line_rows=sum(r['linie'] == target['route_name'] for r in source_by_day[date_key]),
                recovered=matched is not None, proof=proof)
            for identity in proof['known_vehicle_runs']:
                runs.add('|'.join((date_key, *identity.split(':')[-2:])))
            if matched is not None:
                official = f'municipal:{day}:{matched["fahrt_id"]}:{matched["fahrzeug"]}'
                endpoints = {kind:dict(status='REAL', time_seconds=int(matched[key]), official_journey=official,
                    provider_scheduled_seconds=int(matched[planned])) for kind,key,planned in
                    [('departure','ist_ab_von','soll_ab_von'),('arrival','ist_an_nach1','soll_an_nach')]}
                supplements.append(dict(day=day, local_city_cid=cid, trip_id=target['trip_id'],
                    dep_sequence=target['dep_sequence'], arr_sequence=target['arr_sequence'], **endpoints))
                record['reason'] = 'Historical stop-name mapping missing; exact minute clocks and existing same-vehicle sequence anchors identify one source edge.'
                record['current_stop_names'] = {s:stop_names[s] for s in proof['missing_historical_stop_mappings']}
            else:
                record['reason'] = ('No rows for this line in the already transferred operating peaks.'
                    if record['local_source_line_rows'] == 0 else 'No unique same-vehicle source edge at the required scheduled minutes.')
            records.append(record)
    request = dict(source=REMOTE_CACHE, dates=sorted(source_by_day),
                   routes=sorted({r['identity']['route_name'] for r in records}), runs=sorted(runs))
    remote_path = HERE/'remote_endpoint_audit.json'
    if args.remote:
        remote = remote_audit(request)
        remote_path.write_text(json.dumps(remote, ensure_ascii=False, indent=2)+'\n')
    else:
        remote = load_json(remote_path) if remote_path.exists() else None
    if remote is not None:
        for record in records:
            key = date.fromisoformat(record['day']).strftime('%d.%m.%y')+'|'+record['identity']['route_name']
            record['remote_whole_day_line_rows'] = remote['whole_day_line_rows'].get(key, 0)
            if not record['recovered'] and not record['remote_whole_day_line_rows']:
                record['reason'] = 'The complete cached municipal week has no records for this line on this date; peak extraction did not cause the missingness.'
    local_dates = sorted(p.name.removesuffix('.tar.bz2') for p in (ROOT/'data/swiss_rt/gtfs-rt').glob('*.tar.bz2'))
    published = load_json(AUDIT/'dated_public_forecasts.json.gz')
    forecast_dates = sorted(published['dates'])
    signal_cases = [dict(root_id=r['root_id'], day=r['day'], first_trip=r['first_trip'],
        irreversible_arrival_cids=[h['arrival_cid'] for h in r['hubs']],
        available_same_date_public_forecasts=0) for r in roots]
    headroom = load_json(ROOT/'md/BAPR_HRO_GPT_diagnosis_v5/board_gap/headroom_45.json')
    signal_cases.append(dict(root_id=headroom['root_id'], day='2023-07-17',
        case='45-second retrospective alighting opportunity',
        irreversible_clock=headroom['rolling']['trace'][-1]['observed_arrival'],
        named_trips=sorted({r['trip'] for r in headroom['connections'].values()}),
        available_same_date_public_forecasts=0))
    if set(data['dates']) & (set(local_dates) | set(forecast_dates)):
        raise AssertionError('Same-date forecast files exist and require trip/stop/time matching')
    summary = dict(signal_cases=len(signal_cases), matching_archive_dates=0,
        unresolved_roots=len(missing_roots), audited_blocking_connections=len(records),
        recovered_connections=len(supplements), recovered_endpoints=2*len(supplements),
        remote_json_bytes=remote.get('transferred_json_bytes') if remote else None,
        seconds=perf_counter()-tick)
    result = dict(summary=summary, public_information=dict(local_raw_archive_dates=local_dates,
        extracted_forecast_dates=forecast_dates, extracted_source='dated_replay.json.gz; connection IDs belong to that graph and service date',
        municipal_endpoint_fields=sorted(data['dates']['2023-07-17']['connections'][0]['departure']),
        remote_project_rt_directory_exists=remote['rt_archive_directory_exists'] if remote else None,
        contract='Own arrival and ideal local departure observation only. No same-date archived external prediction publication is available. REAL municipal endpoints have no message publication timestamp.'),
        signal_cases=signal_cases, missing_roots=missing_roots, connections=records,
        acquisition='Existing local compressed peak rows plus a remote scan of the cached week; only compact JSON crossed the network.')
    (HERE/'endpoint_supplement.json').write_text(json.dumps(dict(
        source='Existing municipal 2023-07-16/22 peak subset; no model or old archive changed.',
        matching='Named trip/provider vehicle identity, exact planned minutes, and consistent source sequence offsets.',
        records=supplements), ensure_ascii=False, indent=2)+'\n')
    (HERE/'information_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
