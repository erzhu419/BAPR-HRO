from experiments.decision_calibration_v2.platform_announcements import index_announcements, received


def test_platform_record_must_be_received_and_is_never_actual_truth():
    def frame(clock, stop):
        return dict(receipt_seconds=clock, capture_seconds=clock-2, archive_member='capture',
            entity_id='entity', trip_id='J', platform_stop_updates=[dict(stop_sequence=2,
                published_stop_id=stop, stop_status='SCHEDULED')])
    index = index_announcements([frame(5, 'station:1'), frame(21, 'station:2')])
    endpoint = dict(trip_id='J', stop_sequence=2, stop_id='station:1')
    now = received(endpoint, 20, index)
    assert now['announcement']['published_stop_id'] == 'station:1'
    assert now['matches_scheduled_platform'] and not now['actual_platform_verified']
    assert received(endpoint, 21, index)['matches_scheduled_platform'] is False
    assert received(endpoint, 4, index)['status'] == 'no_received_platform_record'
    assert received(endpoint, None, index)['status'] == 'initial_clock_unknown'
