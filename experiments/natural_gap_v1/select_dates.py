"""Apply the frozen availability/exposure rule without opening target values."""
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def select_dates(rows, exposed, target_days):
    records, eligible = [], []
    for row in sorted(rows, key=lambda item: item['day']):
        day = row['day']
        reasons = []
        if day in exposed:
            reasons.append('previously_developed_date')
        else:
            if not row['actual_available']:
                reasons.append('missing_actual_archive')
            elif row.get('actual_domain_verified') is not True:
                reasons.append('v41_rail_actual_source_unverified')
            if not row['rt_available']:
                reasons.append('missing_rt_archive')
            elif row.get('capture_coverage_verified') is not True:
                reasons.append('predecision_capture_coverage_unverified')
            if row['static_available'] is not True:
                reasons.append('matching_historical_static_unverified')
        if not reasons:
            eligible.append(day)
        records.append(dict(day=day, eligible=not reasons, reasons=reasons,
                            source_metadata=row))
    complete = len(eligible) >= target_days
    return dict(target_days=target_days, considered_dates=len(records), rows=records,
        eligible_dates=eligible, selected_dates=eligible[:target_days] if complete else [],
        dates_frozen=complete,
        status='ready_to_freeze_public_roots' if complete else 'insufficient_unused_sources',
        target_values_opened=False, controls_executed=0, original_72_additional_budget=0)


def main():
    source = json.loads((HERE/'source_inventory.json').read_text())
    exposure = json.loads((HERE/'exposure_inventory.json').read_text())
    protocol = json.loads((HERE/'protocol.json').read_text())
    result = select_dates(source['rows'], set(exposure['known_exposed_service_dates']),
                          protocol['date_selection']['target_days'])
    result.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol='protocol.json', sources='source_inventory.json',
                  exposure='exposure_inventory.json')
    (HERE/'date_selection.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(result['status'], 'eligible=',len(result['eligible_dates']),
          'selected=',len(result['selected_dates']))


if __name__ == '__main__':
    main()
