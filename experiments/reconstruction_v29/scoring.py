"""Evaluate one frozen visible-versus-unseen target comparison."""

import numpy as np
from time import perf_counter


def _comparison(current, waiting):
    return dict(current=float(current), wait=float(waiting), wait_minus_board=float(waiting-current),
                action='board_current' if current <= waiting+1e-9 else 'wait_target')


def evaluate_pair(compiler, current_cid, target_cid, clock, entry, ready):
    """Compare existing nominal and integrated values on identical support.

    The current vehicle is publicly visible and legal; the target has not had
    an observed departure/cancellation since entry. The waiting alternative
    is this fixed target, not the full-menu optimal WAIT action. Target scores
    use only positive-mass atoms of the frozen departure law. Off-grid current
    scores retain the existing bracket projection, not a continuous posterior.
    """
    started = perf_counter()
    deadline = compiler.deadline
    wall = dict(nominal_preparation=0., current_integration=0., target_integration=0., bracket=0.)
    result = dict(status='unsupported', reason=None, current_cid=current_cid, target_cid=target_cid,
                  clock=clock, entry=entry, ready=ready, deadline=deadline,
                  nominal=None, integrated=None, mean_arrival=None,
                  current_clock_method=None, current_bracket=None, support=None, wall=wall)

    def finish(reason=None):
        result['reason'] = reason
        wall['total'] = perf_counter()-started
        return result

    if not entry <= ready <= clock < deadline:
        return finish('current_not_legally_ready')
    phase = perf_counter()
    times, mass, canceled = compiler.departure_distribution(target_cid)
    before_entry = float(mass[times < entry].sum())
    future = (times > clock) & (mass > 0)
    future_mass = float(mass[future].sum())
    eligible = future & (times >= ready) & (times < deadline)
    legal_mass = float(mass[eligible].sum())
    q = float(canceled+before_entry+future_mass)
    result['support'] = dict(denominator=q, cancellation_mass=float(canceled),
        before_entry_mass=before_entry, future_mass=future_mass, legal_future_mass=legal_mass,
        after_deadline_mass=float(mass[future & (times >= deadline)].sum()),
        legal_future_probability=legal_mass/q if q > 0 else None,
        positive_future_atoms=int(eligible.sum()))
    if q <= 0:
        wall['nominal_preparation'] = perf_counter()-phase
        return finish('target_zero_survival')
    # Prepare the original cheap comparison first. Its probability laws and
    # duration/mark work must not be hidden inside a warm exact-score query.
    nominal_current = compiler.clock_nominal_score(current_cid, clock)
    nominal_gain = sum(probability*(deadline-compiler.clock_nominal_score(target_cid, int(time)))
                       for time, probability in zip(times[eligible], mass[eligible], strict=True))
    wall['nominal_preparation'] = perf_counter()-phase
    phase = perf_counter()
    current = compiler.board_score(current_cid, clock)
    wall['current_integration'] = perf_counter()-phase
    if current is None:
        return finish('current_outside_score_support')
    phase = perf_counter()
    gains = dict(nominal=nominal_gain, integrated=0., mean_arrival=0.)
    for time, probability in zip(times[eligible], mass[eligible], strict=True):
        row = compiler.board_score(target_cid, int(time))
        if row is None or row.method != 'on_grid':
            wall['target_integration'] = perf_counter()-phase
            return finish('target_atom_outside_score_support')
        gains['integrated'] += probability*(deadline-row.exact)
        gains['mean_arrival'] += probability*(deadline-row.mean_arrival)
    waiting = {name: float(deadline-gain/q) for name, gain in gains.items()}
    wall['target_integration'] = perf_counter()-phase
    result['nominal'] = _comparison(nominal_current, waiting['nominal'])
    result['integrated'] = _comparison(current.exact, waiting['integrated'])
    result['mean_arrival'] = _comparison(current.mean_arrival, waiting['mean_arrival'])
    result['current_clock_method'] = current.method
    phase = perf_counter()
    clocks, probability, _ = compiler.departure_distribution(current_cid)
    clocks = clocks[probability > 0]
    index = np.searchsorted(clocks, clock)
    endpoints = (clock,) if current.method == 'on_grid' else (int(clocks[index-1]), int(clocks[index]))
    bracket = []
    for endpoint in endpoints:
        row = compiler.board_score(current_cid, endpoint)
        bracket.append(dict(clock=endpoint, exact=float(row.exact), mean_arrival=float(row.mean_arrival),
            integrated_comparison=_comparison(row.exact, waiting['integrated']),
            mean_arrival_comparison=_comparison(row.mean_arrival, waiting['mean_arrival'])))
    result['current_bracket'] = dict(endpoints=bracket,
        integrated_ranking_sensitive=len({r['integrated_comparison']['action'] for r in bracket}) > 1,
        mean_arrival_ranking_sensitive=len({r['mean_arrival_comparison']['action'] for r in bracket}) > 1,
        interpretation='Endpoint board scores are compared against the same actual-clock target wait. '
                       'This is projection sensitivity, not a continuous-clock exact certificate.')
    wall['bracket'] = perf_counter()-phase
    result['status'] = 'supported'
    return finish()
