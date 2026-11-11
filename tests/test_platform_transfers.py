from src.platform_transfers import PlatformTransfers


def test_direction_specificity_and_unknown_are_preserved():
    stops = [dict(stop_id=p, parent_station='station') for p in ('a', 'b', 'c')]
    rows = [dict(from_stop_id='station', to_stop_id='station', transfer_type='2', min_transfer_time='420'),
            dict(from_stop_id='a', to_stop_id='b', transfer_type='2', min_transfer_time='60'),
            dict(from_stop_id='b', to_stop_id='a', transfer_type='3', min_transfer_time='')]
    index = PlatformTransfers(stops, rows)
    assert index('a', 'b') == 60
    assert index('b', 'a') is None and index.resolve('b', 'a').allowed is False
    assert index('a', 'c') == 420
    assert index.resolve('a', 'c').physical_lower_seconds == 0
    assert index(None, 'a') is None and index('missing', 'elsewhere') is None


def test_conflicting_declarations_are_not_silently_taken_as_a_physical_max():
    rows = [dict(from_stop_id='a', to_stop_id='b', transfer_type='2', min_transfer_time=str(t)) for t in (60, 120)]
    index = PlatformTransfers([], rows)
    assert index.resolve('a', 'b').kind == 'conflicting_declarations'
    assert index('a', 'b') is None
