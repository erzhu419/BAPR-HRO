"""Integrate new public-event policies against the archived optimal proxy table."""

import numpy as np


def integrate(reference, controls, archive):
    cids = reference.cids
    times = np.array(sorted({int(t) for cid in cids
        for t, p in zip(*reference.distributions[cid][:2], strict=True)
        if p > 0 and reference.entry <= t < reference.deadline}))
    masks = 1 << len(cids)
    if not np.array_equal(times, archive['times']):
        raise AssertionError('Reconstructed clock support differs from V26')
    supported = np.ones((len(times)+1, masks), dtype=bool)
    hazards = np.zeros((len(times), len(cids)))
    for j, clock in enumerate(np.r_[times, reference.deadline]):
        for i, cid in enumerate(cids):
            clocks, mass, cancel = reference.distributions[cid]
            survival = cancel+mass[(clocks < reference.entry) | (clocks >= clock)].sum()
            supported[j, (np.arange(masks) & (1 << i)) != 0] &= survival > 0
            if j < len(times) and survival > 0:
                hazards[j, i] = mass[clocks == clock].sum()/survival
    if not np.array_equal(supported, archive['supported']):
        raise AssertionError('Reconstructed public-history support differs from V26')
    values = {method: np.full(supported.shape, np.nan) for method in controls}
    for table in values.values():
        table[-1, supported[-1]] = reference.deadline
    for j in range(len(times)-1, -1, -1):
        clock = int(times[j])
        for mask in range(masks):
            if not supported[j, mask]:
                continue
            expected = dict.fromkeys(controls, 0.)
            present = mask
            while True:
                probability = 1.
                for i in range(len(cids)):
                    if mask & (1 << i):
                        probability *= hazards[j, i] if present & (1 << i) else 1.-hazards[j, i]
                if probability > 0:
                    remaining = mask ^ present
                    visible = tuple(cid for i, cid in enumerate(cids) if present & (1 << i))
                    seen = tuple(cid for i, cid in enumerate(cids) if not remaining & (1 << i))
                    for method, controller in controls.items():
                        choice = (controller.choose(visible, seen, clock)[0]
                                  if visible and clock >= reference.ready else None)
                        cost = (values[method][j+1, remaining] if choice is None
                                else reference.score(choice, clock))
                        if not np.isfinite(cost):
                            raise AssertionError('Positive-probability branch has an unsupported continuation')
                        expected[method] += probability*cost
                if present == 0:
                    break
                present = (present-1) & mask
            for method in controls:
                values[method][j, mask] = expected[method]
    dp = archive['dp']
    if not np.all(np.isfinite(dp[supported])):
        raise AssertionError('Archived optimal table contains unsupported selected values')
    excess = {method: float(np.max((dp-table)[supported])) for method, table in values.items()}
    if any(not np.isfinite(v) or v > 1e-6 for v in excess.values()):
        raise AssertionError(('New feasible proxy policy beats archived DP', excess))
    return dict(expected_arrival={m: float(v[0, masks-1]) for m, v in values.items()},
        supported_states=int(supported.sum()), dp_max_excess_over_control=excess), values
