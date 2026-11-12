"""Descriptive and paired summaries with explicit bounded missing outcomes."""

from collections import Counter
from statistics import mean


METHODS = ('full_first_offer', 'one_step_nominal', 'schedule_priority',
           'single_target_nominal', 'combined_wait_nominal')
PAIRS = (('one_step_nominal', 'single_target_nominal'),
         ('full_first_offer', 'single_target_nominal'),
         ('schedule_priority', 'single_target_nominal'),
         ('one_step_nominal', 'combined_wait_nominal'),
         ('single_target_nominal', 'combined_wait_nominal'))


def _paired(rows, baseline, candidate):
    complete = [row for row in rows if all(row['outcomes'][method]['loss'] is not None
                                           for method in (baseline, candidate))]
    gains = [row['outcomes'][baseline]['loss']-row['outcomes'][candidate]['loss'] for row in complete]
    lower, upper, one_unknown, both_unknown = 0., 0., 0, 0
    for row in rows:
        cap = row['deadline']-row['start']
        left, right = (row['outcomes'][method]['loss'] for method in (baseline, candidate))
        left_low, left_high = (0., cap) if left is None else (left, left)
        right_low, right_high = (0., cap) if right is None else (right, right)
        lower += left_low-right_high
        upper += left_high-right_low
        one_unknown += (left is None) != (right is None)
        both_unknown += left is None and right is None
    return dict(baseline=baseline, candidate=candidate, complete_pairs=len(complete),
        unknown_pairs=len(rows)-len(complete), one_unknown=one_unknown, both_unknown=both_unknown,
        mean_gain=mean(gains) if gains else None,
        wins=sum(gain > 0 for gain in gains), harms=sum(gain < 0 for gain in gains),
        ties=sum(gain == 0 for gain in gains),
        changed=sum(row['outcomes'][baseline]['action'] != row['outcomes'][candidate]['action'] for row in complete),
        mean_loss={method: mean(row['outcomes'][method]['loss'] for row in complete) if complete else None
                   for method in (baseline, candidate)},
        arrivals={method: sum(row['outcomes'][method]['arrived'] for row in complete)
                  for method in (baseline, candidate)},
        win_root_ids=[row['root_id'] for row, gain in zip(complete, gains, strict=True) if gain > 0],
        harm_root_ids=[row['root_id'] for row, gain in zip(complete, gains, strict=True) if gain < 0],
        missingness_bounds=dict(population_roots=len(rows),
            mean_gain_lower=lower/len(rows) if rows else None,
            mean_gain_upper=upper/len(rows) if rows else None,
            interpretation='All supplied roots; each missing loss lies in [0, deadline-start]. '
                           'These are missing-outcome bounds, not confidence intervals.'))


def summarize(rows):
    """Keep each known subset distinct from common and pairwise populations.

    Empty menus and failed prefixes retain their supplied losses and statuses.
    No independence of roots, standard errors, or confidence intervals are
    assumed. Pair gains are baseline loss minus candidate loss.
    """
    rows = list(rows)
    methods = {}
    for method in METHODS:
        outcomes = [row['outcomes'][method] for row in rows]
        known = [outcome for outcome in outcomes if outcome['loss'] is not None]
        methods[method] = dict(known=len(known), unknown=len(rows)-len(known),
            arrivals=sum(outcome['arrived'] for outcome in outcomes),
            statuses=dict(Counter(outcome['status'] for outcome in outcomes)),
            mean_known_loss=mean(outcome['loss'] for outcome in known) if known else None)
    common = [row for row in rows if all(row['outcomes'][method]['loss'] is not None for method in METHODS)]
    return dict(roots=len(rows), methods=methods,
        known_means_scope='Each method uses its own known subset; these means are not a paired comparison.',
        all_method_common=dict(roots=len(common),
            mean_loss={method: mean(row['outcomes'][method]['loss'] for row in common) if common else None
                       for method in METHODS},
            arrivals={method: sum(row['outcomes'][method]['arrived'] for row in common) for method in METHODS}),
        comparisons={f'{candidate}_vs_{baseline}': _paired(rows, baseline, candidate)
                     for baseline, candidate in PAIRS})
