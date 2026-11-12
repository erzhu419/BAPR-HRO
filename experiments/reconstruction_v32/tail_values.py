"""Integrate small public-offer controls against common fixed boarding marks.

The exact reference is the independent-root waiting problem. If a boarding
mark approximates a continuation that shares a root vehicle, neither these
policy integrals nor the waiting optimum are exact physical journey values.
"""

import numpy as np

from src.full_menu_waiting_dp import FullMenuWaitingDP


def integrate(reference, scheduled_arrivals=None, dp=None):
    """Return ``(summary, tables)`` for one entry/readiness pair, K <= 4.

    All policies use the same boarding marks. ``nominal`` is SchedulePriority
    and is included only when its scheduled arrival map is supplied. Other
    keys are first, one_step, single, combined, pair and exact. The pair policy
    is TwoTargetOpportunity, including its combined-wait fallback.

    ``choices[method][j, remaining, present]`` is an int8 position in ``cids``;
    -1 means WAIT and -2 means an unsupported public branch. The clock index
    j addresses ``times``. ``wait`` stores post-event thresholds; ``values``
    stores pre-event integrals, with a final deadline row. Policies observe
    only public departures from entry onwards, including pre-ready receipts.

    The size-two columns of the full waiting DP equal separately compiled
    two-vehicle DPs: independent excluded vehicles generate no information
    about the retained pair. Reusing those columns avoids recompilation.
    """
    cids = tuple(reference.cids)
    count, deadline = len(cids), float(reference.deadline)
    if count > 4:
        raise ValueError('The V32 small-menu integrator is limited to four root vehicles')
    dp = FullMenuWaitingDP(reference) if dp is None else dp
    if dp.reference is not reference:
        raise ValueError('The supplied DP must use this exact reference')
    times, supported = dp.times, dp.supported
    total, masks = len(times), 1 << count
    methods = ['first', 'one_step', 'single', 'combined', 'pair', 'exact']
    if scheduled_arrivals is not None:
        if set(scheduled_arrivals) != set(cids):
            raise ValueError('Schedule priority must cover the common menu')
        methods.append('nominal')
    values = {m: np.full(supported.shape, np.nan) for m in methods}
    choices = {m: np.full((total, masks, masks), -2, dtype=np.int8) for m in methods}
    for table in values.values():
        table[-1, supported[-1]] = deadline
    waiting = {m: np.full((total, masks), deadline)
               for m in ('one_step', 'single', 'combined', 'pair', 'exact')}
    board = np.full((total, count), deadline)
    target_wait = np.full((total, count), deadline)
    for i, cid in enumerate(cids):
        clocks, probability, scores = reference.marks[cid]
        clocks, probability, scores = np.asarray(clocks), np.asarray(probability), np.asarray(scores)
        legal = (times >= reference.ready) & (dp.hazards[:, i] > 0)
        board[legal, i] = scores[np.searchsorted(clocks, times[legal])]
        gains = probability*(deadline-scores)
        gains = np.r_[np.cumsum(gains[::-1])[::-1], 0.]
        survival = dp.survival[1:, i]
        target_wait[:, i] -= np.divide(gains[np.searchsorted(clocks, times, side='right')],
            survival, out=np.zeros(total), where=survival > 0)
    bits = [[i for i in range(count) if mask & (1 << i)] for mask in range(masks)]
    for mask in range(1, masks):
        bit = mask & -mask
        waiting['single'][:, mask] = np.minimum(waiting['single'][:, mask ^ bit],
                                                target_wait[:, bit.bit_length()-1])

    # Every possible public branch is (previous unseen mask, newly present
    # subset). At most 3^K rows are integrated together at each clock.
    parents, present, remaining = [], [], []
    for mask in range(masks):
        subset = mask
        while True:
            parents.append(mask)
            present.append(subset)
            remaining.append(mask ^ subset)
            if subset == 0:
                break
            subset = (subset-1) & mask
    parents, present, remaining = (np.asarray(a, dtype=int) for a in (parents, present, remaining))
    probability = np.ones((total, len(parents)))
    for i in range(count):
        active = (parents & (1 << i)) != 0
        occurs = (present & (1 << i)) != 0
        hazard = dp.hazards[:, i, None]
        probability[:, active] *= np.where(occurs[active][None, :], hazard, 1.-hazard)
    first_index = np.array([min(group, key=lambda i: reference.ranks[cids[i]]) if group else -1
                            for group in bits])
    priority = None if scheduled_arrivals is None else np.array([
        min(group, key=lambda i: (scheduled_arrivals[cids[i]], cids[i])) if group else -1 for group in bits])
    pair_masks = [mask for mask in range(masks) if len(bits[mask]) == 2]
    mask_ids = np.arange(masks)
    for j in range(total-1, -1, -1):
        legal = times[j] >= reference.ready
        waiting['one_step'][j] = values['first'][j+1]
        waiting['combined'][j] = np.minimum(waiting['single'][j], waiting['one_step'][j])
        waiting['pair'][j] = waiting['combined'][j]
        for pair_mask in pair_masks:
            if not supported[j+1, pair_mask]:
                continue
            contains = (mask_ids & pair_mask) == pair_mask
            waiting['pair'][j, contains] = np.minimum(waiting['pair'][j, contains],
                                                     dp.values[j+1, pair_mask])
        waiting['exact'][j] = dp.values[j+1]
        best_index = np.array([min(group, key=lambda i: (board[j, i], reference.ranks[cids[i]]))
                               if group else -1 for group in bits])
        positive = (probability[j] > 0) & supported[j, parents]
        old, new, left = parents[positive], present[positive], remaining[positive]
        p = probability[j, positive]
        visible = (new != 0) & legal
        best = best_index[new]
        best_cost = np.full(len(new), deadline)
        best_cost[visible] = board[j, best[visible]]
        for method in methods:
            selected = np.full(len(new), -1, dtype=np.int8)
            if method == 'first':
                selected[visible] = first_index[new[visible]]
            elif method == 'nominal':
                target = priority[old]
                take = visible & ((new & np.left_shift(1, np.maximum(target, 0))) != 0)
                selected[take] = target[take]
            else:
                take = visible & (best_cost <= waiting[method][j, left]+1e-9)
                selected[take] = best[take]
            choices[method][j, left, new] = selected
            cost = values[method][j+1, left].copy()
            take = selected >= 0
            cost[take] = board[j, selected[take]]
            if np.any(~np.isfinite(cost)):
                raise AssertionError('A positive public branch has an unsupported continuation')
            expected = np.bincount(old, weights=p*cost, minlength=masks)
            values[method][j, supported[j]] = expected[supported[j]]
    error = float(np.max(np.abs(values['exact'][supported]-dp.values[supported])))
    excess = {m: float(np.max((dp.values-values[m])[supported])) for m in methods}
    if error > 1e-6 or any(value > 1e-6 for value in excess.values()):
        raise AssertionError(('Waiting policy integration disagrees with the optimal reference', error, excess))
    summary = dict(scope='fixed-mark independent-root waiting surrogate',
        expected_arrival={m: float(table[0, -1]) for m, table in values.items()},
        supported_states=int(supported.sum()), dp_evaluation_max_error=error,
        dp_max_excess_over_control=excess)
    tables = dict(cids=cids, times=times, supported=supported, values=values,
                  wait=waiting, choices=choices, board=board, dp=dp)
    return summary, tables
