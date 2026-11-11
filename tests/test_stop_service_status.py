from src.committed_transit import CommittedState
from src.stop_service_status import available_alighting_leg


def test_received_skip_vetoes_only_the_matching_alighting_stop_and_can_be_revised():
    messages = [{'receipt_seconds': time, 'stops': [{'sequence': 4, 'status': status}]}
                for time, status in [(20, 'SCHEDULED'), (30, 'SKIPPED'), (45, 'SCHEDULED')]]
    state = lambda t: CommittedState(0, t, 'same-trip', t)
    assert available_alighting_leg(state(29), 17, 4, messages)
    assert not available_alighting_leg(state(30), 17, 4, messages)
    assert available_alighting_leg(state(30), 16, 3, messages)
    assert available_alighting_leg(state(45), 17, 4, messages)
