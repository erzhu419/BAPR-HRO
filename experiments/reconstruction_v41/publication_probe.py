"""Small wall-clock publisher/consumer probe, with declared model windows only."""

import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

from experiments.reconstruction_v41.action_publisher import ActionPublisher, ActionToken


def measure(case):
    publisher, release = ActionPublisher(), Event()
    start = perf_counter()
    width = 0. if case == 'empty' else 1. if case == 'timely' else .03
    token = ActionToken(f'probe:{case}', ('continue', 'alight'), start,
                        start+width, 'fixed synthetic window; no archive endpoints')
    request = publisher.request(token, 'continue')
    fallback_seen = request.current()
    candidate_ready = Event()

    def compute(submit):
        if case == 'late':
            # Simulate work completing after the declared closing instant.
            Event().wait(max(0., token.closes_at-perf_counter())+.01)
        submit('alight')
        candidate_ready.set()
        assert release.wait(2.)
        return {'computation': 'protocol-only; no model scoring'}

    launched = request.start(compute)
    consumed = []

    def consumer():
        if case == 'timely':
            assert candidate_ready.wait(2.)
        consumed.append(request.take())

    receiver = Thread(target=consumer, name=f'consumer-{case}')
    receiver.start()
    receiver.join(2.)
    assert not receiver.is_alive()
    consumer_finished_before_worker = launched and request._thread.is_alive()
    if launched:
        assert candidate_ready.wait(2.)
    release.set()
    result = request.finish()
    result.update(case=case, fallback_seen_before_launch=fallback_seen,
                  consumer_finished_before_worker=consumer_finished_before_worker)
    if case == 'timely':
        assert result['consumed']['action'] == 'alight'
        assert consumer_finished_before_worker
    elif case == 'late':
        assert result['consumed']['action'] == 'continue'
        assert result['attempts'][1]['reason'] == 'deadline_expired'
        assert result['cost']['worker_late_wall'] > 0
        assert consumer_finished_before_worker
    else:
        assert not result['fallback_published'] and not result['worker_started']
        assert result['consumed'] is None
    return result


if __name__ == '__main__':
    output = Path(__file__).with_name('publication_receipts.json')
    receipts = [measure(case) for case in ('timely', 'late', 'empty')]
    output.write_text(json.dumps(dict(
        scope='Real threads and wall clocks; fixed model windows, not transit door data or policy timing.',
        cases=receipts), indent=2)+'\n')
    print(output)
