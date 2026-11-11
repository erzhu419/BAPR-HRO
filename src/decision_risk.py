"""Date-block calibration of an already issued same-vehicle fragment.

The regret comparison uses the project's fixed transfer-buffer execution
contract, not an identified lower bound on every real platform transfer.
"""
from collections import Counter
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RealizedRisk:
    lower: float
    upper: float
    outcome: str


def realized_macro_risk(time, gate, deadline, departure, arrival, *, canceled=False,
                        same_actual_trip=True, patience=720):
    """Partially identified pathwise regret upper surrogate for one next leg.

    Success gives (arrival-gate)+; failure gives deadline-gate. A missing
    future endpoint retains the entire feasible risk interval. No unknown
    target is dropped from the selected population.
    """
    cap = float(deadline - gate)
    if canceled:
        return RealizedRisk(cap, cap, 'failed_canceled')
    if not same_actual_trip:
        return RealizedRisk(0., cap, 'unresolved')
    end = min(deadline, time + patience)
    if departure is not None and not time <= departure <= end:
        return RealizedRisk(cap, cap, 'failed_departure_window')
    if arrival is not None and (arrival < time or arrival > deadline):
        return RealizedRisk(cap, cap, 'failed_arrival_window')
    if (same_actual_trip and departure is not None and arrival is not None
            and departure <= arrival):
        value = float(max(0, arrival - gate))
        return RealizedRisk(value, value, 'succeeded')
    return RealizedRisk(0., cap, 'unresolved')


def summarize_day(rows, groups=('ordinary', 'late')):
    result = {}
    for group in groups:
        chosen = [row for row in rows if row['selected'] and row['group'] == group]
        if not chosen:
            result[group] = {'n_selected': 0, 'mean_lower': None, 'mean_upper': None}
            continue
        n = len(chosen)
        result[group] = {
            'n_selected': n,
            'mean_lower': sum(row['risk_lower'] for row in chosen) / n,
            'mean_upper': sum(row['risk_upper'] for row in chosen) / n,
            'mean_model_regret_bound': sum(row['model_regret_bound'] for row in chosen) / n,
            'outcomes': dict(Counter(row['outcome'] for row in chosen)),
            'transfer_sources': dict(Counter(row['transfer_source'] for row in chosen)),
            'zero_buffer_mean_lower': sum(row['zero_buffer_risk_lower'] for row in chosen) / n,
            'zero_buffer_mean_upper': sum(row['zero_buffer_risk_upper'] for row in chosen) / n,
        }
    present = [value for value in result.values() if value['n_selected']]
    return {'n_contexts': len(rows), 'groups': result,
            'score_lower': max((value['mean_lower'] for value in present), default=0.),
            'score_upper': max((value['mean_upper'] for value in present), default=0.)}


def calibrate_daily_groups(days, alpha=.1):
    """Upper quantile of date maxima; the nine-date90% case uses the maximum.

    With a fixed selector and exchangeable *latent date blocks*, this upper
    score dominates the true group-mean score. Empty groups contribute no
    restriction on that date. This is not coverage conditional on a gate call.
    """
    scores = sorted(day['score_upper'] for day in days.values())
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    bound = scores[rank - 1] if rank <= len(scores) else math.inf
    return {'risk_upper_seconds': bound, 'rank': rank, 'n_calibration_dates': len(scores),
            'nonempty_dates_by_group': {group: sum(day['groups'][group]['n_selected'] > 0
                for day in days.values()) for group in ('ordinary', 'late')}}


def assess_day(day, bound):
    states = {}
    for group, values in day['groups'].items():
        if not values['n_selected']:
            states[group] = 'no_selected_context'
        elif values['mean_upper'] <= bound:
            states[group] = 'verified_below_bound'
        elif values['mean_lower'] > bound:
            states[group] = 'verified_above_bound'
        else:
            states[group] = 'unidentified_from_missing_measurements'
    return states


def binomial_failure_upper(failures, samples, alpha=.1):
    """One-sided Clopper–Pearson upper bound for iid Bernoulli units.

    A unit is explicitly chosen by the caller: the primary service guard uses
    independent date blocks, never the correlated contexts as extra dates.
    """
    from scipy.stats import beta

    if not samples or failures == samples:
        return 1.
    if not failures:
        return 1. - alpha ** (1. / samples)
    return float(beta.ppf(1. - alpha, failures + 1, samples - failures))


def service_guard(clock_tail_conditional_success, service_failure_upper, cap, tolerance=1.):
    """Authorize only if the clock-plus-service *upper* bound is small.

    This is conditional on the supplied clock-tail bound and transfer contract
    being valid for the target cohort. A large q upper bound means insufficient
    evidence to authorize; it is not a lower bound on actual failure risk.
    """
    upper = min(cap, clock_tail_conditional_success + cap * service_failure_upper)
    return {'risk_upper': upper, 'authorized': upper <= tolerance}


def required_zero_failure_blocks(cap, tolerance=1., alpha=.1):
    """Necessary n for the service term alone to fit inside tolerance."""
    return math.ceil(math.log(alpha) / math.log1p(-tolerance / cap))
