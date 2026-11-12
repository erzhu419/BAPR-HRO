"""Exact policy integration on the frozen independent offer law."""

import numpy as np


METHODS = ('full_first_offer', 'one_step_nominal', 'multistep_nominal')


def integrate(reference, dp):
    """Evaluate all three policies, separately from the DP's min recursion."""
    times, cids = dp.times, dp.cids
    count, masks = len(cids), 1 << len(cids)
    values = {m: np.full((len(times)+1, masks), float(reference.deadline)) for m in METHODS}
    supported = np.ones((len(times)+1, masks), dtype=bool)
    one_wait = np.full((len(times), masks), float(reference.deadline))
    hazards = np.zeros((len(times), count))
    for j, clock in enumerate(np.r_[times, reference.deadline]):
        positive = []
        for i, cid in enumerate(cids):
            clocks, mass, cancel = reference.distributions[cid]
            survival = cancel+mass[(clocks < reference.entry) | (clocks >= clock)].sum()
            if j < len(times):
                hazards[j, i] = mass[clocks == clock].sum()/survival if survival > 0 else 0.
            positive.append(survival > 0)
        for mask in range(masks):
            supported[j, mask] = all(positive[i] for i in range(count) if mask & (1 << i))
            if j < len(times) and clock >= reference.ready:
                seen = {cid for i, cid in enumerate(cids) if not mask & (1 << i)}
                one_wait[j, mask] = reference.wait_score(clock, seen)
    for j in range(len(times)-1, -1, -1):
        clock, legal = times[j], times[j] >= reference.ready
        costs = {cid: reference.score(cid, int(clock)) for cid in cids} if legal else {}
        for mask in range(masks):
            expected = dict.fromkeys(METHODS, 0.)
            present = mask
            while True:
                probability = 1.
                for i in range(count):
                    if mask & (1 << i):
                        probability *= hazards[j, i] if present & (1 << i) else 1.-hazards[j, i]
                if probability > 0:
                    remaining = mask ^ present
                    branches = {m: values[m][j+1, remaining] for m in METHODS}
                    if present and legal:
                        visible = [cid for i, cid in enumerate(cids) if present & (1 << i)]
                        first = min(visible, key=reference.ranks.__getitem__)
                        best = min(visible, key=lambda cid: (costs[cid], reference.ranks[cid]))
                        branches['full_first_offer'] = costs[first]
                        if costs[best] <= one_wait[j, remaining]+1e-9:
                            branches['one_step_nominal'] = costs[best]
                        if costs[best] <= dp.values[j+1, remaining]+1e-9:
                            branches['multistep_nominal'] = costs[best]
                    for method in METHODS:
                        expected[method] += probability*branches[method]
                if present == 0:
                    break
                present = (present-1) & mask
            for method in METHODS:
                values[method][j, mask] = expected[method]
    if not np.array_equal(supported, dp.supported) or not np.all(np.isfinite(dp.values[supported])):
        raise AssertionError('DP support or supported values disagree with independent survival calculation')
    error = float(np.max(np.abs(values['multistep_nominal'][supported]-dp.values[supported])))
    excess = {m: float(np.max((dp.values-values[m])[supported])) for m in METHODS[:-1]}
    # Detect a policy/evaluator mismatch or an invalid optimality claim; stop,
    # retain the run log and repair the implementation before producing results.
    if not np.isfinite(error) or error > 1e-6 or any(not np.isfinite(value) or value > 1e-6 for value in excess.values()):
        raise AssertionError(dict(dp_evaluation_error=error, domination_excess=excess))
    return dict(expected_arrival={m: float(v[0, masks-1]) for m, v in values.items()},
        dp_evaluation_max_error=error, supported_states=int(supported.sum()),
        dp_max_excess_over_control=excess), dict(times=times, supported=supported,
            one_step_wait=one_wait, **values)
