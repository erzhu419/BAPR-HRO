from copy import deepcopy

from experiments.reconstruction_v35.prepare_data import endpoint_rows


def connection():
    return dict(id=0, trip_id='T', route_id='R', route_name='route', dep_stop='A', arr_stop='B',
        dep_sequence=1, arr_sequence=2, scheduled_departure_seconds=27000,
        scheduled_arrival_seconds=27300,
        departure=dict(status='REAL', time_seconds=27050, official_journey='T'),
        arrival=dict(status='REAL', time_seconds=27340, official_journey='T'))


def frame(receipt, prediction, *, captured=None, status='SCHEDULED', event='FORECAST'):
    return dict(receipt_seconds=receipt, capture_seconds=receipt if captured is None else captured,
        header_seconds=receipt, update_seconds=None, archive_member='fixture.gtfsrt', entity_id='T',
        trip_id='T', trip_statuses=[['T', status]],
        endpoint_updates=[] if status == 'CANCELED' else [[0, 'departure', event, prediction]])


def departure_row(c, frames):
    return next(row for row in endpoint_rows('2023-10-01', [c], frames, frames)
                if row['kind'] == 'departure')


def test_event_table_uses_availability_clock_and_preserves_physical_identity():
    rows = [frame(26600, 27100), frame(26800, 30000, captured=26650)]
    result = departure_row(connection(), rows)
    assert result['forecast_seconds'] == 27100  # Later availability overrides its earlier capture.
    assert result['forecast_receipt'] == 26600 and result['forecast_age'] == 100
    assert result['forecast_publication']['frame_index'] == 0
    assert result['source_connection_id'] == 0 and result['stop_sequence'] == 1
    assert result['error_eligible'] and result['error_seconds'] == -50


def test_notice_withdrawal_and_later_forecast_restore_do_not_change_actual_label():
    frames = [frame(26500, 27100), frame(26600, None, status='CANCELED')]
    withdrawn = departure_row(connection(), frames)
    assert withdrawn['forecast_seconds'] is None and not withdrawn['error_eligible']
    assert withdrawn['actual_status'] == 'REAL' and withdrawn['actual_seconds'] == 27050
    frames.append(frame(26650, 27020))
    restored = departure_row(connection(), frames)
    assert restored['forecast_seconds'] == 27020 and restored['error_seconds'] == 30
    frames.append(frame(26670, None, event='NO_DATA'))
    assert departure_row(connection(), frames)['forecast_seconds'] is None


def test_ineligible_early_or_unresolved_endpoints_remain_in_table_without_errors():
    c = connection()
    c['departure']['time_seconds'] = 26700  # Exactly at cutoff is not a future endpoint.
    rows = endpoint_rows('2023-10-01', [c], [frame(26600, 27100)], [])
    assert len(rows) == 2
    departure = next(row for row in rows if row['kind'] == 'departure')
    assert not departure['cutoff_before_actual'] and not departure['error_eligible']
    assert departure['error_seconds'] is None
    missing = deepcopy(c)
    missing['departure'] = dict(status='MISSING', time_seconds=None, official_journey=None)
    retained = departure_row(missing, [frame(26600, 27100)])
    assert retained['actual_status'] == 'MISSING' and retained['forecast_seconds'] == 27100
    assert retained['actual_seconds'] is None and retained['error_seconds'] is None
