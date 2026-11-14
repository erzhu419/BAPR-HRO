"""An external action channel with replaceable, explicitly expiring tokens.

The controller supplies a legal window; this module does not infer door times.
Publish a fallback first, then start work. The consumer never waits for finish().
"""

from dataclasses import asdict, dataclass
from queue import SimpleQueue
from threading import Lock, Thread
from time import perf_counter, process_time, thread_time


@dataclass(frozen=True)
class ActionToken:
    token_id: str
    legal_actions: tuple
    opens_at: float
    closes_at: float
    source: str

    def rejection(self, action, now):
        if action not in self.legal_actions:
            return 'illegal_action'
        if self.closes_at <= self.opens_at:
            return 'empty_window'
        if now < self.opens_at:
            return 'not_open'
        if now >= self.closes_at:
            return 'deadline_expired'
        return None


class ActionPublisher:
    """Resident metadata belongs to the caller, outside per-request timing."""

    def __init__(self, *, wall_clock=perf_counter, process_clock=process_time,
                 thread_clock=thread_time):
        self.wall_clock = wall_clock
        self.process_clock = process_clock
        self.thread_clock = thread_clock

    def request(self, token, fallback):
        """Synchronously expose fallback; no planning is started by this call."""
        return ActionRequest(self, token, fallback)


class ActionRequest:
    def __init__(self, publisher, token, fallback):
        self.publisher, self.token = publisher, token
        self.started_wall = publisher.wall_clock()
        self.started_cpu = publisher.process_clock()
        self.updates = SimpleQueue()
        self._lock = Lock()
        self._thread = None
        self._current = None
        self._taken = None
        self._attempts = []
        self._consumption = []
        self._worker_points = []
        self._result = None
        self._error = None
        self._launch_reason = None
        self._offer(fallback, 'fallback', token.token_id, None)

    def _offer(self, action, origin, token_id, payload):
        with self._lock:
            now = self.publisher.wall_clock()
            reason = ('wrong_token' if token_id != self.token.token_id else
                      self.token.rejection(action, now) or
                      ('already_consumed' if self._taken is not None else None))
            row = dict(sequence=len(self._attempts), token_id=token_id,
                       action=action, origin=origin, received_wall=now,
                       accepted=reason is None, reason=reason, payload=payload)
            self._attempts.append(row)
            if reason is None:
                self._current = row
                self.updates.put(dict(row))
            return dict(row)

    def submit(self, action, payload=None, *, token_id=None):
        """Independent worker channel; check actual reception, not compute start."""
        row = self._offer(action, 'background', self.token.token_id if token_id is None
                          else token_id, payload)
        return row

    def current(self):
        """Read the latest published instruction without waiting for the worker.

        Publication is not execution. After expiry, take() rejects consumption.
        """
        with self._lock:
            return None if self._current is None else dict(self._current)

    def take(self):
        """Atomically consume a currently legal instruction, irrevocably."""
        with self._lock:
            now = self.publisher.wall_clock()
            reason = ('already_consumed' if self._taken is not None else
                      'no_published_action' if self._current is None else
                      self.token.rejection(self._current['action'], now))
            row = dict(received_wall=now, accepted=reason is None, reason=reason,
                       action=None if self._current is None else self._current['action'])
            self._consumption.append(row)
            if reason is not None:
                return None
            self._taken = dict(self._current, consumed_wall=now)
            return dict(self._taken)

    def start(self, compute):
        """Run compute(submit) asynchronously, after fallback is externally visible.

        compute may emit several candidates. Its returned value is retained only
        as a diagnostic result; returning a value does not publish an action.
        """
        with self._lock:
            if self._thread is not None:
                raise RuntimeError('This request already started its worker')
            now = self.publisher.wall_clock()
            self._launch_reason = ('already_consumed' if self._taken is not None else
                'no_published_fallback' if self._current is None else
                self.token.rejection(self._current['action'], now))
            if self._launch_reason is not None:
                return False
            self._thread = Thread(target=self._work, args=(compute,),
                                  name=f'action-{self.token.token_id}')
        self._thread.start()
        return True

    def _checkpoint(self):
        self._worker_points.append((self.publisher.wall_clock(),
                                    self.publisher.thread_clock()))

    def _work(self, compute):
        self._checkpoint()

        def submit(action, payload=None, *, token_id=None):
            row = self.submit(action, payload, token_id=token_id)
            self._checkpoint()
            return row

        try:
            self._result = compute(submit)
        except Exception as exc:
            self._error = dict(type=type(exc).__name__, message=str(exc))
        finally:
            self._checkpoint()

    def finish(self):
        """Join only for final accounting, after the consumer chose its action."""
        if self._thread is not None:
            self._thread.join()
        end_wall, end_cpu = self.publisher.wall_clock(), self.publisher.process_clock()
        points, cutoff = self._worker_points, self.token.closes_at
        late_lo, late_hi = 0., 0.
        for (wa, ca), (wb, cb) in zip(points, points[1:]):
            if wa >= cutoff:
                late_lo += cb-ca
                late_hi += cb-ca
            elif wb > cutoff:
                late_hi += cb-ca
        worker_cpu = points[-1][1]-points[0][1] if points else 0.
        worker_wall = points[-1][0]-points[0][0] if points else 0.
        late_wall = max(0., points[-1][0]-max(points[0][0], cutoff)) if points else 0.
        with self._lock:
            return dict(token=asdict(self.token), started_wall=self.started_wall,
                fallback_published=self._attempts[0]['accepted'],
                attempts=[dict(row) for row in self._attempts],
                consumption=[dict(row) for row in self._consumption],
                consumed=None if self._taken is None else dict(self._taken),
                latest_published=None if self._current is None else dict(self._current),
                worker_started=self._thread is not None, launch_rejection=self._launch_reason,
                background_result=self._result, background_error=self._error,
                cost=dict(request_to_finish_process_cpu=end_cpu-self.started_cpu,
                    request_to_finish_wall=end_wall-self.started_wall,
                    worker_thread_cpu=worker_cpu, worker_wall=worker_wall,
                    worker_late_wall=late_wall, worker_late_cpu_interval=[late_lo, late_hi],
                    all_worker_cpu_charged=worker_cpu,
                    interpretation='Process span includes concurrent consumer work; worker CPU is a subset, not an additive fee. Late CPU is bracketed when a work segment crosses the wall deadline.'))
