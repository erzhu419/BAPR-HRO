"""Independent checks of bounded value curves and the frozen paired banks.

Sampling estimates are never used as population truth. Different valid partial
expansions may cross; only checkpoints of one incremental run must nest.
"""
from math import isfinite

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v41.continuation import science
from experiments.reconstruction_v41.run import normalized
from experiments.reconstruction_v45.summarize import same_bank


ARMS = ('alight', 'continue')
TOL = 1e-8


def certified(interval):
    lower, upper = interval
    return 'alight' if upper < -1e-9 else 'continue' if lower > 1e-9 else None


def values(point, *, clock, deadline):
    """Reject an invalid value, lost range or false action certificate."""
    actions = point['actions']
    for arm in ARMS:
        value = actions[arm]
        lo, hi, mass = (value[k] for k in ('lower', 'upper', 'unresolved_probability'))
        if not all(isfinite(v) for v in (lo, hi, mass)):
            raise ValueError(f'Nonfinite bounded value: {arm}')
        if lo < clock-TOL or hi > deadline+TOL or lo > hi+TOL:
            raise ValueError(f'Invalid physical interval: {arm}')
        if mass < -TOL or mass > 1+TOL:
            raise ValueError(f'Invalid unresolved probability: {arm}')
        mean = value['mean']
        if mean is not None and (mass != 0 or abs(mean-lo) > TOL or abs(hi-lo) > TOL):
            raise ValueError(f'Inexact value advertised as exact: {arm}')
        if mass == 0 and (mean is None or abs(hi-lo) > TOL):
            raise ValueError(f'Complete mass without exact value: {arm}')
    a, c = (actions[arm] for arm in ARMS)
    gap = [a['lower']-c['upper'], a['upper']-c['lower']]
    if any(abs(x-y) > TOL for x, y in zip(gap, point['difference_interval'], strict=True)):
        raise ValueError('Action gap does not match the two arm intervals')
    if point['certified_action'] != certified(gap):
        raise ValueError('Incorrect deterministic action certification')
    return gap


def curve(checkpoints, *, clock, deadline, max_compiles=2000):
    """Validate refinement on a single run, including cumulative accounting."""
    if not checkpoints:
        raise ValueError('No bounded checkpoints retained')
    previous = None
    for point in checkpoints:
        values(point, clock=clock, deadline=deadline)
        count = point['counters']['charged_compiles']
        if not isinstance(count, int) or count < 0 or count > max_compiles:
            raise ValueError('Compilation quota exceeded')
        if previous is not None:
            if count < previous['counters']['charged_compiles']:
                raise ValueError('Cumulative work counter decreased')
            for arm in ARMS:
                old, new = previous['actions'][arm], point['actions'][arm]
                if (new['lower'] < old['lower']-TOL or new['upper'] > old['upper']+TOL
                        or new['unresolved_probability'] > old['unresolved_probability']+TOL):
                    raise ValueError(f'Incremental interval widened: {arm}')
            for kind in ('process_cpu', 'wall'):
                if point['cost'][kind] < previous['cost'][kind]-TOL:
                    raise ValueError('Cumulative query cost decreased')
        previous = point
    first, last = checkpoints[0], checkpoints[-1]
    return dict(checkpoints=len(checkpoints), nested=True,
        charged_compiles=last['counters']['charged_compiles'],
        first_gap_width=first['difference_interval'][1]-first['difference_interval'][0],
        final_gap_width=last['difference_interval'][1]-last['difference_interval'][0],
        certified_action=last['certified_action'])


def matching_values(left, right):
    """Value equivalence used for the shared prefix and the legacy rerun."""
    for arm in ARMS:
        a, b = left['actions'][arm], right['actions'][arm]
        for field in ('lower', 'upper', 'unresolved_probability', 'mean'):
            x, y = a[field], b[field]
            if ((x is None) != (y is None)) or (x is not None and abs(x-y) > TOL):
                raise ValueError(f'Paired deterministic value changed: {arm}/{field}')
    if left['certified_action'] != right['certified_action']:
        raise ValueError('Paired deterministic certificate changed')
    return True


def exact_legacy_arms(fresh, legacy):
    """Every new interval must contain any previously established exact arm."""
    checked = []
    for arm in ARMS:
        truth = legacy['actions'][arm]['mean']
        if truth is not None:
            now = fresh['actions'][arm]
            if not now['lower']-TOL <= truth <= now['upper']+TOL:
                raise ValueError(f'New interval excludes established exact value: {arm}')
            checked.append(arm)
    return checked


def overlapping_legacy_bounds(fresh, legacy):
    """Different traversals may cross, but valid bounds cannot be disjoint."""
    for arm in ARMS:
        a, b = fresh['actions'][arm], legacy['actions'][arm]
        if max(a['lower'], b['lower']) > min(a['upper'], b['upper'])+TOL:
            raise ValueError(f'Disjoint full-pi population intervals: {arm}')
    return True


def regret_upper(point, action):
    """Root-local full-pi bound for an already chosen action, not trip gain."""
    other = 'continue' if action == 'alight' else 'alight'
    return max(0., point['actions'][action]['upper']-point['actions'][other]['lower'])


def frozen_samples(index, fresh_markov, fresh_full, experiments):
    old_markov = read(experiments/'reconstruction_v45/banks'/f'{index:03}_markov128.json.gz')
    old_full = read(experiments/'reconstruction_v41/banks'/f'{index:03}_shallow.json.gz')
    same_bank(normalized(fresh_markov), old_markov)
    if fresh_markov['selected_action'] != old_markov['selected_action']:
        raise ValueError('V45 selected action changed')
    if normalized(science(fresh_full)) != normalized(science(old_full)):
        raise ValueError('Full-pi original sampled result changed')
    return dict(markov128_same_bank=True, full128_same_bank=True)
