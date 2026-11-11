"""Keep official national event statuses and cancellation conflicts distinct."""
from datetime import datetime

FIELDS = {
    'departure': ('ABFAHRTSZEIT', 'AB_PROGNOSE', 'AB_PROGNOSE_STATUS'),
    'arrival': ('ANKUNFTSZEIT', 'AN_PROGNOSE', 'AN_PROGNOSE_STATUS'),
}


def seconds_of_service(value, service_date):
    parsed = datetime.strptime(value, '%d.%m.%Y %H:%M:%S' if value.count(':') == 2 else '%d.%m.%Y %H:%M')
    return int((parsed - datetime.fromisoformat(service_date)).total_seconds())


def parse_actual_endpoint(row, kind, service_date):
    _, actual, status = FIELDS[kind]
    reported = seconds_of_service(row[actual], service_date) if row[actual] else None
    raw_status = row[status]
    canceled = row['FAELLT_AUS_TF'] == 'true'
    if canceled and raw_status == 'REAL' and reported is not None:
        outcome, time = 'CONFLICT_REAL_CANCELED', None
    elif canceled:
        outcome, time = 'CANCELED', None
    elif row['DURCHFAHRT_TF'] == 'true':
        outcome, time = 'SKIPPED', None
    elif raw_status == 'REAL' and reported is not None:
        outcome, time = 'REAL', reported
    else:
        outcome, time = raw_status or 'NO_STATUS', None
    return outcome, time, row['FAHRT_BEZEICHNER'], reported, canceled, raw_status
