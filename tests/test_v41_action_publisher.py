from threading import Event, Thread
from types import SimpleNamespace

import pytest

from experiments.reconstruction_v41.action_publisher import ActionPublisher, ActionToken


def fixture():
    clock = SimpleNamespace(wall=10., cpu=20., thread=1.)
    publisher = ActionPublisher(wall_clock=lambda: clock.wall,
        process_clock=lambda: clock.cpu, thread_clock=lambda: clock.thread)
    token = ActionToken('onboard:7', ('alight', 'continue'), 10., 11., 'declared test window')
    return clock, publisher, token


def test_external_consumer_receives_fallback_while_worker_is_blocked_and_result_is_not_publication():
    clock, publisher, token = fixture()
    request = publisher.request(token, 'continue')
    assert request.updates.get_nowait()['action'] == 'continue'
    started, release = Event(), Event()

    def compute(submit):
        started.set()
        assert release.wait(2)
        return 'alight'

    request.start(compute)
    assert started.wait(2)
    observed = []
    consumer = Thread(target=lambda: observed.append(request.take()))
    consumer.start()
    consumer.join(2)
    assert not consumer.is_alive()
    assert observed[0]['action'] == 'continue'
    release.set()
    receipt = request.finish()
    assert receipt['background_result'] == 'alight'
    assert receipt['consumed']['action'] == 'continue'
    assert len(receipt['attempts']) == 1


def test_timely_worker_submission_replaces_fallback_and_is_consumed_independently():
    clock, publisher, token = fixture()
    request = publisher.request(token, 'continue')
    done = Event()

    def compute(submit):
        clock.wall = 10.4
        clock.thread = 1.2
        assert submit('alight', {'pairs': 128})['accepted']
        done.set()

    assert request.start(compute)
    assert done.wait(2)
    accepted = request.take()
    assert accepted['action'] == 'alight'
    assert accepted['consumed_wall'] < token.closes_at
    receipt = request.finish()
    assert receipt['cost']['worker_thread_cpu'] == pytest.approx(.2)
    assert receipt['cost']['worker_late_cpu_interval'] == [0., 0.]


def test_late_work_is_rejected_and_fully_charged_including_work_after_rejection():
    clock, publisher, token = fixture()
    request = publisher.request(token, 'continue')

    def compute(submit):
        clock.wall, clock.cpu, clock.thread = 11.2, 20.4, 1.4
        assert submit('alight')['reason'] == 'deadline_expired'
        clock.wall, clock.cpu, clock.thread = 11.4, 20.5, 1.5

    request.start(compute)
    receipt = request.finish()
    assert receipt['latest_published']['action'] == 'continue'
    assert receipt['cost']['all_worker_cpu_charged'] == pytest.approx(.5)
    assert receipt['cost']['request_to_finish_process_cpu'] == pytest.approx(.5)
    assert receipt['cost']['worker_late_wall'] == pytest.approx(.4)
    assert receipt['cost']['worker_late_cpu_interval'] == pytest.approx([.1, .5])
    assert request.take() is None  # An unconsumed old instruction is not backdated.


@pytest.mark.parametrize('opens,closes,reason', [(10., 10., 'empty_window'),
    (9., 10., 'deadline_expired'), (11., 12., 'not_open')])
def test_no_legal_window_does_not_publish_or_start_work(opens, closes, reason):
    clock, publisher, token = fixture()
    token = ActionToken(token.token_id, token.legal_actions, opens, closes, token.source)
    request = publisher.request(token, 'continue')
    assert request.current() is None
    assert not request.start(lambda submit: pytest.fail('No legal request exists'))
    receipt = request.finish()
    assert not receipt['fallback_published']
    assert receipt['attempts'][0]['reason'] == reason
    assert receipt['cost']['all_worker_cpu_charged'] == 0.


def test_wrong_token_illegal_action_and_post_consumption_replacement_rejected():
    clock, publisher, token = fixture()
    request = publisher.request(token, 'continue')
    assert request.submit('alight', token_id='other')['reason'] == 'wrong_token'
    assert request.submit('board')['reason'] == 'illegal_action'
    assert request.take()['action'] == 'continue'
    assert request.submit('alight')['reason'] == 'already_consumed'
    assert request.take() is None
    assert request.finish()['consumed']['action'] == 'continue'


def test_worker_failure_preserves_available_fallback_and_exposes_error():
    clock, publisher, token = fixture()
    request = publisher.request(token, 'continue')

    def compute(submit):
        raise ValueError('bad posterior')

    request.start(compute)
    receipt = request.finish()
    assert receipt['background_error'] == {'type': 'ValueError', 'message': 'bad posterior'}
    assert request.take()['action'] == 'continue'


def test_deadline_is_exclusive_for_submission_and_external_consumption():
    clock, publisher, token = fixture()
    request = publisher.request(token, 'continue')
    clock.wall = 11.
    assert request.submit('alight')['reason'] == 'deadline_expired'
    assert request.take() is None
