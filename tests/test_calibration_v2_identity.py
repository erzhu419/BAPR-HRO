"""Regression cases for the observed GTFS version identity issue."""
from experiments.decision_calibration_v2.static_identity_audit import compare_day


def connection(trip, sequence=1, **changes):
    row = dict(operator_id='85:11', vehicle_number='100', route_id='R', trip_id=trip,
        dep_stop=str(sequence), arr_stop=str(sequence+1), dep_sequence=sequence,
        arr_sequence=sequence+1, scheduled_departure_seconds=27000+60*sequence,
        scheduled_arrival_seconds=27060+60*sequence, pickup_allowed=True,
        dropoff_allowed=True, dep_platform_id=f'{sequence}:1', arr_platform_id=f'{sequence+1}:1')
    return dict(row, **changes)


def test_whole_trip_needed_for_rename():
    old = [connection('old'), connection('old',2)]
    identical = [connection('new'),connection('new',2)]
    result = compare_day('2023-03-16',old,identical)
    assert result['rows'][0]['mapped_trip_id'] == 'new'
    assert result['rows'][0]['status'] == 'renamed_id'
    changed = [connection('new'),connection('new',2,scheduled_arrival_seconds=27200)]
    result = compare_day('2023-03-16',old,changed)
    assert result['rows'][0]['mapped_trip_id'] is None
    assert result['rows'][0]['status'] == 'changed_public_service'


def test_duplicate_public_signatures_are_not_arbitrarily_resolved():
    result = compare_day('2023-03-16',[connection('a'),connection('b')],[connection('c')])
    assert result['summary']['exact_unique_matches'] == 0
    assert {row['status'] for row in result['rows']} == {'ambiguous_signature'}


def test_reused_literal_id_cannot_hide_different_service():
    result = compare_day('2023-03-16',[connection('old')],
                         [connection('new'),connection('old',vehicle_number='999')])
    assert result['rows'][0]['mapped_trip_id'] == 'new'
    assert result['summary']['literal_id_changed_signature'] == 1
    assert result['reused_literal_id_changes'][0]['new_identity']['vehicle_number'] == '999'


def test_planned_platform_change_is_distinguished_from_pure_rename():
    result = compare_day('2023-03-16',[connection('old')],[connection('new',arr_platform_id='2:7')])
    assert result['rows'][0]['status'] == 'renamed_id_with_platform_change'
    assert result['rows'][0]['planned_platform_sequence_equal'] is False
