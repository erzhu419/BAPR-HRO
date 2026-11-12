"""Compare bounded upper limits on the original V13 paired outer-bank gaps."""

from itertools import product
import json
from math import ceil, isfinite, log, log1p, sqrt
from pathlib import Path
from statistics import mean, variance
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continuation_confidence import continuation_lower

SOURCE = ROOT / 'experiments/reconstruction_v13/factored_results.json'
ALPHA = .05 / 6
GATE_SECONDS = 60


def eb_log_wealth_terms(values, *, alpha, budget):
    """Predictable empirical-Bernstein e-process for E[X_t | past] <= m.

    X and the lagged prediction p are in [0, 1]. For 0 < lambda < 1 and
    z in [-1, 1], log(1 + lambda*z) >= lambda*z - psi(lambda)*z**2, where
    psi(lambda) = -log(1-lambda) - lambda. This follows by bounding the
    terms of order >= 2 in the log series by their absolute values.
    Taking z = X - p gives

        E exp(lambda*(X-m) - psi*(X-p)**2) <=
        exp(lambda*(p-m)) * (1+lambda*(E[X|past]-p)) <= 1.

    Both p and lambda depend only on previous samples. The product of the
    factors is a nonnegative supermartingale under the null, so 1/alpha
    is a valid rejection threshold, including any prefix of this fixed
    budget's tape. Alpha, budget, pseudocounts and the lambda cap are fixed
    independently of the tape. Return terms (weight, intercept) of
    log wealth at m, which is sum(intercept - weight*m).
    """
    if not 0 < alpha < 1 or not 1 <= len(values) <= budget:
        raise ValueError('Use nonempty data within the fixed budget and 0 < alpha < 1')
    observations = []
    total = residual_sq = 0.0
    for index, value in enumerate(values, start=1):
        if not 0 <= value <= 1:
            raise ValueError('Normalize observations to a known [0, 1] range')
        prediction = (.5 + total) / index
        predicted_variance = (.25 + residual_sq) / index
        weight = min(.9, sqrt(2 * log(1 / alpha) / (budget * predicted_variance)))
        residual = (value - prediction) ** 2
        psi = -log1p(-weight) - weight
        observations.append((weight, weight * value - psi * residual))
        total += value
        residual_sq += residual
    return observations


def eb_upper(values, *, low, high, alpha, budget):
    """One-sided upper limit on a bounded loss or signed paired difference."""
    width = high - low
    if not isfinite(width) or width <= 0:
        raise ValueError('Use a finite, nondegenerate range [low, high]')
    complement = [(high - x) / width for x in values]
    terms = eb_log_wealth_terms(complement, alpha=alpha, budget=budget)
    weight = sum(term[0] for term in terms)
    intercept = sum(term[1] for term in terms)
    lower = max(0.0, (intercept - log(1 / alpha)) / weight)
    return dict(upper=high - width * lower, predictable_bet_weights=[term[0] for term in terms])


def bernoulli_exact_checks():
    """Enumerate all eight-draw Bernoulli tapes at boundary null means."""
    budget, alpha = 8, .2
    thresholds = (.05, .25, .5, .75, .95)
    rows = []
    for mu in thresholds:
        rejected = 0.0
        for tape in product((0., 1.), repeat=budget):
            log_wealth = 0.0
            for weight, intercept in eb_log_wealth_terms(tape, alpha=alpha, budget=budget):
                log_wealth += intercept - weight * mu
                if log_wealth >= log(1 / alpha):
                    successes = int(sum(tape))
                    rejected += mu ** successes * (1 - mu) ** (budget - successes)
                    break
        assert rejected <= alpha + 1e-12
        rows.append(dict(null_mean=mu, probability_any_prefix_rejects=rejected))
    # Check the adaptive rule away from the small-n weight cap and verify the
    # same affine mapping on a signed support and its nonnegative shift.
    quiet = eb_log_wealth_terms([.5] * 16, alpha=ALPHA, budget=512)
    varied = eb_log_wealth_terms([0., 1.] * 8, alpha=ALPHA, budget=512)
    assert quiet[-1][0] > varied[-1][0]
    signed = eb_upper([-5.] * budget, low=-10., high=10., alpha=alpha, budget=budget)
    shifted = eb_upper([5.] * budget, low=0., high=20., alpha=alpha, budget=budget)
    assert abs(signed['upper'] + 10 - shifted['upper']) < 1e-12
    return dict(alpha=alpha, budget=budget, rows=rows,
                adaptive_probe=dict(budget=512, step=16, quiet_weight=quiet[-1][0],
                                    varied_weight=varied[-1][0]))


def case_result(receipt, case):
    group = sorted((row for row in receipt['rows'] if row['arm'] == 'product'
                    and [row['date'], row['query'], row['decision']] == case),
                   key=lambda row: row['bank'])
    banks = receipt['protocol']['banks']
    expected_rows = receipt['protocol']['atoms_per_hidden_trip'] ** 2
    if [row['bank'] for row in group] != list(range(banks)) or any(
            row['status'] != 'complete' or row['bank_size'] != expected_rows for row in group):
        raise ValueError(f'Missing or incomplete original banks for {case}')
    cap = group[0]['cap']
    if any(row['cap'] != cap for row in group):
        raise ValueError(f'Mixed loss cap in {case}')
    lower = [row['value'] for row in group]
    policy = [row['policies']['first_visible']['mean_loss'] for row in group]
    deltas = [u - z for u, z in zip(policy, lower, strict=True)]
    if any(not 0 <= delta <= cap for delta in deltas):
        raise ValueError(f'Bank optimality or common loss cap failed for {case}')
    paired_upper = cap - continuation_lower([cap - d for d in deltas], cap, ALPHA)
    adaptive = eb_upper(deltas, low=0., high=cap, alpha=ALPHA, budget=banks)
    original_lower = continuation_lower(lower, cap, ALPHA)
    summary = next(row for row in receipt['summary'] if row['arm'] == 'product'
                   and [row['date'], row['query'], row['decision']] == case)
    assert abs(original_lower - summary['lower_confidence']) < 1e-6
    assert paired_upper >= mean(deltas) and adaptive['upper'] >= mean(deltas)

    # Even observing all zeros cannot yield a distribution-free upper bound
    # at the gate if the all-zero tape has probability > alpha under a
    # two-point null with mean just above the gate and mass at the loss cap.
    best_case_outer_n_necessary = ceil(log(ALPHA) / log1p(-GATE_SECONDS / cap))
    return dict(case=case, independent_banks=banks, dependent_rows_per_bank=expected_rows,
                cap=cap, per_bank_policy_loss=policy, per_bank_relaxation_value=lower,
                per_bank_delta=deltas, sample_mean_delta=mean(deltas),
                sample_variance_delta=variance(deltas), original_mean_relaxation=mean(lower),
                original_relaxation_lower_limit=original_lower,
                original_relaxation_statistical_margin=mean(lower) - original_lower,
                paired_existing_upper=paired_upper, paired_predictable_eb=adaptive,
                extra_upper_margin_existing=paired_upper - mean(deltas),
                extra_upper_margin_eb=adaptive['upper'] - mean(deltas),
                best_case_outer_n_necessary=best_case_outer_n_necessary)


def main():
    receipt = json.loads(SOURCE.read_text())
    result = dict(
        source=str(SOURCE.relative_to(ROOT)), alpha_per_root=ALPHA,
        scope='Fixed comparison of methods on the original V13 product banks only. '
              'Each method uses the prior per-root alpha allocation separately; '
              'the smaller limit is not selected as a combined claim.',
        estimand='Delta=U-Z on one matched independent outer bank. The 4x4 rows share factors. '
                 'Under the frozen model and unbiased fixed-policy evaluation, E[U]=J_F. '
                 'The admissible information relaxation and exact empirical solve give E[Z]<=V*. '
                 'The policy remains feasible in each bank, so 0<=Delta<=cap and E[Delta] '
                 'upper bounds J_F-V*.',
        precision='Extra outer banks narrow uncertainty about E[Delta], without changing its '
                  'inner-bank or information-relaxation bias. The best-case sample counts are '
                  'necessary under the loss range alone, for an all-zero tape, and do not '
                  'project a benefit or specify a data collection target.',
        rows=[case_result(receipt, case) for case in receipt['protocol']['cases']],
        bernoulli_exact_checks=bernoulli_exact_checks())
    output = HERE / 'confidence_results.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
