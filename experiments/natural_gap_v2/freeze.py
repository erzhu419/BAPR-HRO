"""Fix the first fourteen metadata-qualified dates before reading target values."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from experiments.decision_gap_v1.panel import read, write

HERE = Path(__file__).resolve().parent


def freeze(directory):
    directory = Path(directory)
    if (directory/'date_manifest.json').exists():
        raise ValueError('Date manifest already frozen; retain its original population')
    rules = read(directory/'protocol.json')
    exposed = set(read(directory/'exposure_inventory.json')['known_exposed_service_dates'])
    rt = read(directory/'rt_source_metadata.json')
    static = read(directory/'static_source_manifest.json')
    actual = {row['day']:row for row in read(directory/'actual_sources.json')['days']}
    # This public source completion receipt contains no target endpoints/messages.
    public = read(directory/'static_acquisition.json')
    ready_static = set(public['dates'])
    considered = [dict(day=row['date'],reasons=[row['status']],source_metadata=row,
                       target_values_opened=False)
                  for row in rt.get('considered_dates',[]) if row['status'].startswith('excluded')]
    selected = []
    for source in sorted(rt['candidate_days'],key=lambda row:row['date']):
        day = source['date']
        reasons = []
        clock_path = directory/'rt_clocks'/f'{day}.json'
        clocks = read(clock_path) if clock_path.exists() else {}
        if day in exposed:
            reasons.append('already_developed')
        if actual.get(day,{}).get('official_daily_member_count') != 1:
            reasons.append('official_actual_member_not_unique_or_absent')
        if day not in ready_static or day not in static['source_version_by_date']:
            reasons.append('matching_public_static_unavailable')
        if not clocks.get('capture_coverage_complete',False):
            reasons.append('original_capture_window_incomplete_or_unverified')
        considered.append(dict(day=day,reasons=reasons,rt=source,
            captured_bins=clocks.get('matching_bins',[]),missing_bins=clocks.get('missing_bins'),
            target_values_opened=False))
        if not reasons:
            selected.append(dict(day=day,static_version=static['source_version_by_date'][day],
                rt=source,actual_archive=f'https://archive.opentransportdata.swiss/istdaten/{day[:4]}/ist-daten-{day[:7]}.zip',
                actual_member=actual[day]['members'][0]['name'],capture_metadata=f'rt_clocks/{day}.json'))
        if len(selected)==rules['date_selection']['target_days']:
            break
    write(directory/'date_consideration.json',dict(rows=considered,selected_count=len(selected),
        excluded_exposure_dates=sorted(exposed),target_values_opened=False))
    if len(selected)!=14:
        raise ValueError('Not fourteen qualified dates; retain considered failures and finish metadata acquisition')
    manifest = dict(stage='natural_gap_v2',dates_frozen=True,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        selected_dates=[row['day'] for row in selected],days=selected,
        target_values_opened_before_freeze=False,rules='protocol.json',
        rt_capture_rule='rt_metadata_rule.json',sources='Official actual/static plus accepted traines captures',
        prior_exposure_checked=True,replacement_after_freeze=False,original_72_additional_budget=0)
    write(directory/'date_manifest.json',manifest)
    print('dates frozen',manifest['selected_dates'],flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=HERE)
    freeze(parser.parse_args().directory)
