"""Simultaneous partial identification on the three frozen V13 model roots.

For a fixed policy, its empirical cost averages to its population cost under
independent outer draws. An exact empirical optimum is downward in expectation:
for any population-feasible policy, the empirical optimum is no larger than
that policy's empirical cost. Infimize the latter expectation over policies.
The same argument applies to a known rest realization, conditional on it, or
to independent empirical trip products whose individual rows have the correct
joint law. None of these optimizations identifies a population optimum.

Bounds use single-sided e-test inversions with a fixed Bonferroni allocation.
They are combined by closure of difference constraints, including
0 <= hindsight <= E Z4 <= V_R,tau <= V_R <= V* <= J_F <= cap. Missing exact
solves use only deterministic valid fallback directions for the entire cohort.
This avoids conditioning inference on solve completion.
"""

import json
from math import inf, isfinite
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continuation_confidence import continuation_lower

V13 = ROOT / 'experiments/reconstruction_v13/factored_results.json'
V14 = ROOT / 'experiments/reconstruction_v14/menu_info.json'
INVENTORY = ROOT / 'experiments/reconstruction_v12/transfer_inventory.json'
FACTOR_SOLVES = HERE / 'v13_exact_banks.json'
IID_SOLVES = HERE / 'iid_causal_banks.json'
FINITE = HERE / 'finite_attribution.json'
FAMILY_ALPHA = .05
GATE_SECONDS = 60
NAMES = ('zero', 'hindsight', 'ell', 'v_r_tau', 'v_r', 'v_star', 'j')
TERMS = {
    'policy_regret': ('v_star', 'j'),
    'other_vehicle_information': ('v_r', 'v_star'),
    'hindsight_tail_information': ('v_r_tau', 'v_r'),
    'empirical_bank_optimism': ('ell', 'v_r_tau'),
}

# Reserve all slots before reading optional solver output. Dependence between
# tests, and reuse of worlds by tests within a panel, do not require independence.
TEST_SIDES = {
    'v13_ell': 'both',
    'v13_total': 'both',
    'iid_first_visible': 'both',
    'iid_hindsight': 'both',
    'iid_hindsight_gap': 'upper',
    'v13_public_failure_witness': 'lower',
    'r_empirical_minus_z4': 'lower',
    'j_minus_r_empirical': 'upper',
    'j_minus_rest_only': 'lower',
    'rest_only_minus_z4': 'upper',
    'j_minus_prefix_witness': 'lower',
    'prefix_witness_minus_z4': 'upper',
    'r_empirical_minus_prefix_witness': 'lower',
    'prefix_witness': 'upper',
    'z8_minus_z4': 'lower',
    'iid_full_empirical': 'lower',
    'j_minus_iid_full_empirical': 'upper',
}
TOTAL_SIDES_PER_ROOT = sum(2 if side == 'both' else 1 for side in TEST_SIDES.values())


class DifferenceBounds:
    """Encode x[right]-x[left] <= distance[left][right] and close the system."""

    def __init__(self, floor, cap):
        self.idx = {name: idx for idx, name in enumerate(NAMES)}
        self.distance = [[0. if i == j else inf for j in range(len(NAMES))]
                         for i in range(len(NAMES))]
        for lower, higher in zip(NAMES, NAMES[1:]):
            self.add(lower, higher, 'lower', 0.)
        self.add('zero', 'hindsight', 'lower', floor)
        self.add('zero', 'j', 'upper', cap)

    def add(self, left, right, side, value):
        if side == 'lower':
            left, right, value = right, left, -value
        i, j = self.idx[left], self.idx[right]
        self.distance[i][j] = min(value, self.distance[i][j])

    def close(self):
        for k in range(len(NAMES)):
            for i in range(len(NAMES)):
                for j in range(len(NAMES)):
                    candidate = self.distance[i][k] + self.distance[k][j]
                    self.distance[i][j] = min(self.distance[i][j], candidate)
        if any(self.distance[i][i] < -1e-7 for i in range(len(NAMES))):
            raise ValueError('The ordered population confidence constraints are inconsistent')

    def interval(self, left, right):
        i, j = self.idx[left], self.idx[right]
        return [-self.distance[j][i], self.distance[i][j]]


class RootAnalysis:
    def __init__(self, floor, cap, alpha):
        self.floor, self.cap, self.alpha = floor, cap, alpha
        self.width = cap - floor
        self.bounds = DifferenceBounds(floor, cap)
        self.receipts = {}

    def add(self, label, left, right, values, support):
        low, high = support
        if (label in self.receipts or not values or not low < high
                or any(not isfinite(x) or x < low - 1e-7 or x > high + 1e-7 for x in values)):
            raise ValueError(f'Mixed or invalid observations for {label} on {support}: {values}')
        # Round only floating discrepancies at a deterministic endpoint.
        values = [min(high, max(low, x)) for x in values]
        side = TEST_SIDES[label]
        receipt = dict(left=left, right=right, n=len(values), sample_mean=mean(values), support=support)
        if side in ('lower', 'both'):
            lower = low + continuation_lower([x-low for x in values], high-low, self.alpha)
            self.bounds.add(left, right, 'lower', lower)
            receipt['lower'] = lower
        if side in ('upper', 'both'):
            upper = high - continuation_lower([high-x for x in values], high-low, self.alpha)
            self.bounds.add(left, right, 'upper', upper)
            receipt['upper'] = upper
        self.receipts[label] = receipt

    def report(self):
        self.bounds.close()
        interval = self.bounds.interval
        return dict(population_values={name: interval('zero', name) for name in NAMES[1:]},
                    population_four_terms={name: interval(*pair) for name, pair in TERMS.items()},
                    combined_terms=dict(total=interval('ell', 'j'),
                        policy_and_other_vehicle=interval('v_r', 'j'),
                        all_except_policy_regret=interval('ell', 'v_star'),
                        tail_and_empirical_bank=interval('ell', 'v_r'),
                        policy_other_vehicle_and_tail=interval('v_r_tau', 'j')),
                    constraints=self.receipts)


def difference(high, low):
    return [u - v for u, v in zip(high, low, strict=True)]


def banks_for(source, case, n):
    if source is None:
        return None
    match = next((row for row in source['roots'] if row['case'] == case), None)
    if match is None:
        return None
    banks = sorted(match['banks'], key=lambda row: row['bank'])
    return banks if [row['bank'] for row in banks] == list(range(n)) else None


def values_with_fallback(rows, field, fallback):
    raw = [row.get(field) for row in rows]
    return [default if value is None else value for value, default in zip(raw, fallback, strict=True)], sum(
        value is not None for value in raw)


def add_factor_solves(analysis, rows, j, z):
    counts = {}
    if rows is None:
        return counts
    if any(abs(row['j']-u) > 1e-6 or abs(row['z']-l) > 1e-6
           for row, u, l in zip(rows, j, z, strict=True)):
        raise ValueError('Exact-factor receipt differs from the matched V13 bank')
    positive, signed = (0., analysis.width), (-analysis.width, analysis.width)
    loss = (analysis.floor, analysis.cap)
    r_emp, counts['v_r_empirical'] = values_with_fallback(rows, 'v_r_empirical', z)
    prefix, counts['v_r_tau_policy'] = values_with_fallback(rows, 'v_r_tau_policy', j)
    rest, counts['v_r_rest_only'] = values_with_fallback(rows, 'v_r_rest_only', [analysis.cap]*len(j))
    z8_fallback = [row.get('z8_lower_sample', analysis.floor) for row in rows]
    z8, counts['z8_empirical'] = values_with_fallback(rows, 'z8_empirical', z8_fallback)
    if counts['v_r_empirical']:
        analysis.add('r_empirical_minus_z4', 'ell', 'v_r', difference(r_emp, z), positive)
        analysis.add('j_minus_r_empirical', 'v_r', 'j', difference(j, r_emp), positive)
    if counts['v_r_rest_only']:
        analysis.add('j_minus_rest_only', 'v_r', 'j', difference(j, rest), signed)
        analysis.add('rest_only_minus_z4', 'ell', 'v_r', difference(rest, z), positive)
    if counts['v_r_tau_policy']:
        analysis.add('j_minus_prefix_witness', 'v_r_tau', 'j', difference(j, prefix), positive)
        analysis.add('prefix_witness_minus_z4', 'ell', 'v_r_tau', difference(prefix, z), positive)
        analysis.add('prefix_witness', 'zero', 'v_r_tau', prefix, loss)
    if counts['v_r_empirical'] and counts['v_r_tau_policy']:
        analysis.add('r_empirical_minus_prefix_witness', 'v_r_tau', 'v_r', difference(r_emp, prefix), signed)
    if counts['z8_empirical'] or all('z8_lower_sample' in row for row in rows):
        analysis.add('z8_minus_z4', 'ell', 'v_r_tau', difference(z8, z), signed)
    return counts


def add_iid_solves(analysis, rows, j_worlds, h_worlds, n):
    if rows is None:
        return 0
    if len(j_worlds) % n:
        raise ValueError('Use disjoint equal-sized fixed outer groups')
    size = len(j_worlds) // n
    j = [mean(j_worlds[i:i+size]) for i in range(0, len(j_worlds), size)]
    h = [mean(h_worlds[i:i+size]) for i in range(0, len(h_worlds), size)]
    if any(abs(row['j']-u) > 1e-6 or abs(row['clairvoyant_mean']-l) > 1e-6
           for row, u, l in zip(rows, j, h, strict=True)):
        raise ValueError('IID exact-bank receipt differs from the fixed V14 world partition')
    v, completed = values_with_fallback(rows, 'v_star_empirical', h)
    if completed:
        analysis.add('iid_full_empirical', 'zero', 'v_star', v, (analysis.floor, analysis.cap))
        analysis.add('j_minus_iid_full_empirical', 'v_star', 'j', difference(j, v), (0., analysis.width))
    return completed


def case_bounds(v13, menu, root, factor_solves, iid_solves, alpha):
    case = menu['case']
    original_banks = v13['protocol']['banks']
    rows = sorted((row for row in v13['rows'] if row['arm'] == 'product'
                   and [row['date'], row['query'], row['decision']] == case), key=lambda row: row['bank'])
    worlds = sorted(menu['rows'], key=lambda row: row['world'])
    cap, floor = menu['deadline'] - menu['time'], menu['earliest_board'] - menu['time']
    if ([row['bank'] for row in rows] != list(range(original_banks))
            or any(row['status'] != 'complete' or row['cap'] != cap for row in rows)
            or any(row['bank_size'] * row['policies']['first_visible']['mean_pipeline_seconds'] > floor
                   for row in rows)
            or [row['world'] for row in worlds] != list(range(64))
            or any(row['policies']['first_visible']['status'] == 'censored' for row in worlds)
            or max(menu['publication_clocks']) > menu['earliest_board']
            or menu['stop'] == root['root']['destination'] or not 0 < floor < cap):
        raise ValueError(f'Original and IID panel contracts are incomplete or differ for {case}')
    analysis = RootAnalysis(floor, cap, alpha)
    loss, positive, signed = (floor, cap), (0., analysis.width), (-analysis.width, analysis.width)
    z = [row['value'] for row in rows]
    j = [row['policies']['first_visible']['mean_loss'] for row in rows]
    p = [row['policies']['public_failure']['mean_loss'] for row in rows]
    j_worlds = [row['policies']['first_visible']['loss_upper'] for row in worlds]
    h_worlds = [row['full_hindsight'] for row in worlds]
    analysis.add('v13_ell', 'zero', 'ell', z, loss)
    analysis.add('v13_total', 'ell', 'j', difference(j, z), positive)
    analysis.add('iid_first_visible', 'zero', 'j', j_worlds, loss)
    analysis.add('iid_hindsight', 'zero', 'hindsight', h_worlds, loss)
    analysis.add('iid_hindsight_gap', 'hindsight', 'j', difference(j_worlds, h_worlds), positive)
    analysis.add('v13_public_failure_witness', 'v_star', 'j', difference(j, p), signed)

    factor_counts = add_factor_solves(analysis, banks_for(factor_solves, case, original_banks), j, z)
    iid_completed = add_iid_solves(analysis, banks_for(iid_solves, case, original_banks),
                                  j_worlds, h_worlds, original_banks)
    return dict(case=case, cap=cap, boarding_floor=floor, v13_independent_banks=original_banks,
                iid_worlds=len(worlds), additional_factor_bank_completed=factor_counts,
                additional_iid_bank_completed=iid_completed, **analysis.report())


def read(path):
    return json.loads(path.read_text())


def finite_support_result(receipt):
    if receipt is None:
        return None
    rows = [dict(case=root['case'], blocks=[dict(block=b['block'], status=b['status'],
                    terms=b['terms']) for b in root['blocks']]) for root in receipt['roots']]
    return dict(source=FINITE.name, receipt_status=receipt['status'], scope=receipt['scope'],
                attribution=receipt['attribution'], roots=rows)


def main():
    v13, v14, inventory = read(V13), read(V14), read(INVENTORY)
    factor_solves = read(FACTOR_SOLVES) if FACTOR_SOLVES.is_file() else None
    iid_solves = read(IID_SOLVES) if IID_SOLVES.is_file() else None
    finite = read(FINITE) if FINITE.is_file() else None
    assert sorted(menu['case'] for menu in v14['roots']) == sorted(v13['protocol']['cases'])
    roots = {(r['date'], r['query'], r['decision']): r for r in inventory['rows']}
    alpha = FAMILY_ALPHA / (len(roots) * TOTAL_SIDES_PER_ROOT)
    result = dict(
        scope='Simultaneous model-population bounds at the three fixed alighted roots. '
              'All statements condition on the frozen posterior, contracts, policies and probes. '
              'No independent-date inference or numerical point decomposition.',
        evidence=dict(v13=str(V13.relative_to(ROOT)), iid=str(V14.relative_to(ROOT)),
                      factor_exact=FACTOR_SOLVES.name if factor_solves is not None else None,
                      iid_exact=IID_SOLVES.name if iid_solves is not None else None),
        support='No walking and destination differs from origin. An alighted root with a prior '
                'connection must respect its 120-second initial transfer time, also in the '
                'root hindsight scan. Every loss is therefore between that physical-contract '
                'floor and the declared timeout cap. No sample extrema shrink the range.',
        samples='The V13 empirical product contains 16 dependent rows per independent outer bank. '
                'The original 64 IID worlds per root are a distinct tape. Optional full causal '
                'solves on that tape use eight disjoint fixed groups. The union bound does not '
                'assume independence between tests or between tapes.',
        missing='Missing V_R exact solves use the matched old prefix Z4 as a pointwise '
                'lower observation. Missing V* empirical solves use matched hindsight. Missing '
                'Z8 use matched 8x8 hindsight or the deterministic loss floor. Missing prefix witness uses J_b. '
                'Missing R-only witness uses the timeout cap. Complete-only means are excluded.',
        order='J_F >= V* >= V_R >= V_R,tau >= ell = E[Z4] >= H. A difference constraint '
              'x_j - x_i <= d is edge i->j of weight d. Shortest path sums give the strongest '
              'implied upper limits, and reversing the pair gives lower limits.',
        family_alpha=FAMILY_ALPHA, side_alpha=alpha, reserved_sides_per_root=TOTAL_SIDES_PER_ROOT,
        threshold_seconds=GATE_SECONDS,
        roots=[case_bounds(v13, menu, roots[tuple(menu['case'])], factor_solves, iid_solves, alpha)
               for menu in v14['roots']],
        distinct_finite_support_attribution=finite_support_result(finite))
    output = HERE / 'population_bounds.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
