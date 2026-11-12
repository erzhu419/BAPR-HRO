"""Update the ordered V15 population constraints with new, held-out pairs."""

import json
from pathlib import Path
from statistics import mean

from scipy.stats import t

from experiments.reconstruction_v14.run_confidence import eb_upper
from experiments.reconstruction_v15 import run_population_bounds as v15
from src.continuation_confidence import continuation_lower


HERE = Path(__file__).resolve().parent
SPLIT_ALPHA = .025
SIDES_NEW = {
    'new_hindsight_gap': 'upper',
    'new_hindsight': 'lower',
    'new_first_visible': 'both',
    'new_executable_policy': 'upper',
    'new_policy_regret_upper': 'upper',
    'new_paired_gain': 'lower',
    'new_conditional_averaged_increment': 'lower',
}
# Reserve one more side for the empirical-Bernstein inversion and one for
# selecting the held-out or precision stage for the new policy's own upper.
TOTAL_NEW = 2+sum(2 if side == 'both' else 1 for side in SIDES_NEW.values())


def read(path):
    return json.loads(path.read_text())


def update(case, old, chosen, proof, fresh, alpha):
    analysis = v15.RootAnalysis(case['floor'],case['cap'],alpha)
    for label,row in old['constraints'].items():
        analysis.receipts[label] = row
        for side in ('lower','upper'):
            if side in row:
                analysis.bounds.add(row['left'],row['right'],side,row[side])
    held_out = case['rows']
    assert [r['world'] for r in held_out] == list(range(128))
    pairs = held_out
    if fresh is not None:
        assert fresh['seed'] != case['seed'] and fresh['cap']==case['cap'] and fresh['floor']==case['floor']
        assert [r['world'] for r in fresh['rows']] == list(range(512))
        pairs = [*held_out,*fresh['rows']]
    assert len({r['policies']['first_visible']['publication'] for r in pairs})==1
    assert len({r['policies'][chosen]['publication'] for r in held_out})==1
    if fresh is not None:
        assert len({r['policies'][chosen]['publication'] for r in fresh['rows']})==1
    values = [r['policies']['first_visible']['loss'] for r in pairs]
    alt = [r['policies'][chosen]['loss'] for r in pairs]
    h = [r['hindsight'] for r in pairs]
    delta = v15.difference(values,alt)
    support = (case['floor'],case['cap'])
    width = analysis.width
    signed = (-width,width)
    analysis.add('new_hindsight_gap','hindsight','j',v15.difference(values,h),(0.,width))
    analysis.add('new_hindsight','zero','hindsight',h,support)
    analysis.add('new_first_visible','zero','j',values,support)
    analysis.add('new_executable_policy','zero','v_star',alt,support)
    analysis.add('new_paired_gain','v_star','j',delta,signed)
    # Use a single frozen evaluation stage for the new policy's own gap. The
    # precision sample used a possibly different charged release than holdout.
    # The choice to extend is made before the independent precision tape.
    latest = held_out if fresh is None else fresh['rows']
    policy_hindsight_gaps = [r['policies'][chosen]['loss']-r['hindsight'] for r in latest]
    upper = width-continuation_lower([width-x for x in policy_hindsight_gaps],width,alpha)
    initial_gain = delta[:len(held_out)]
    eb = -eb_upper([-d for d in initial_gain],low=-width,high=width,alpha=alpha,
                   budget=len(initial_gain))['upper']
    analysis.bounds.add('v_star','j','lower',eb)
    analysis.receipts['new_paired_gain_eb'] = dict(left='v_star',right='j',n=len(initial_gain),
                                                sample_mean=mean(initial_gain),support=signed,lower=eb)
    if proof is not None:
        assert [row['bank'] for row in proof]==list(range(8))
        if all('averaged_increment' in r for r in proof):
            analysis.add('new_conditional_averaged_increment','ell','v_r_tau',
                         [r['averaged_increment'] for r in proof],(0.,width))
    fresh_bounds = v15.DifferenceBounds(case['floor'],case['cap'])
    for label,row in analysis.receipts.items():
        if label.startswith('new_'):
            for side in ('lower','upper'):
                if side in row:
                    fresh_bounds.add(row['left'],row['right'],side,row[side])
    fresh_bounds.close()
    gain = case['summary'][chosen]
    radius = float(t.ppf(.975,len(held_out)-1))*gain['paired_standard_error']
    return dict(case=case['case'], held_out_worlds=len(held_out), comparison=gain,
                precision_worlds=0 if fresh is None else len(fresh['rows']),
                precision_comparison=None if fresh is None else fresh['summary'][chosen],
                mean_original_loss=mean(values[:len(held_out)]),
                new_policy_gap=dict(n=len(latest),sample_mean_policy_minus_hindsight=mean(policy_hindsight_gaps),
                                    true_regret_upper=upper),
                fresh_only_policy_regret=fresh_bounds.interval('v_star','j'),
                fresh_only_other_three_terms=fresh_bounds.interval('ell','v_star'),
                paired_t_interval_descriptive=[gain['mean_gain']-radius,gain['mean_gain']+radius],
                previous_total_same_alpha_family=old['combined_terms']['total'], **analysis.report())


def main():
    validation = read(HERE/'held_out_policy.json')
    assert validation['status']=='complete'
    chosen = str(validation['margins'][0])
    proof_path = HERE/'conditional_bound_fresh_results.json'
    proof = {}
    if proof_path.is_file():
        conditional = read(proof_path)
        if conditional['status']=='complete':
            assert conditional['panel']=='V16 independent fitted-model tape'
            for root in conditional['roots']:
                proof[tuple(root['case'])] = []
                for row in root['rows']:
                    entry = dict(bank=row['bank'])
                    if 'conditional_difference' in row:
                        entry['averaged_increment'] = row['conditional_difference']['value']
                    proof[tuple(root['case'])].append(entry)
    precision_path = HERE/'precision_policy.json'
    precision = {}
    if precision_path.is_file():
        fresh = read(precision_path)
        if fresh['status']=='complete':
            assert fresh['margins']==validation['margins']
            precision = {tuple(r['case']):r for r in fresh['roots']}
    original, menu = read(v15.V13), read(v15.V14)
    inventory = {tuple(r[k] for k in ('date','query','decision')):r for r in read(v15.INVENTORY)['rows']}
    factor, iid = read(v15.FACTOR_SOLVES), read(v15.IID_SOLVES)
    n = len(validation['roots'])
    v15.TEST_SIDES.update(SIDES_NEW)
    previous = {tuple(r['case']):v15.case_bounds(original,r,inventory[tuple(r['case'])],factor,iid,
                     SPLIT_ALPHA/(n*v15.TOTAL_SIDES_PER_ROOT)) for r in menu['roots']}
    output = dict(status='complete',family_alpha=.05,prior_family_alpha=SPLIT_ALPHA,
        fresh_family_alpha=SPLIT_ALPHA, reserved_new_sides_per_root=TOTAL_NEW,chosen_margin=int(chosen),
        evidence='True model intervals conditional on frozen fitted model, policies, clocks and three roots. '
                 'New pairs use independent held-out posterior worlds. t intervals are descriptive and not part '
                 'of the familywise bounds. Conditional estimator uses only complete fixed cohorts or valid fallbacks.',
        roots=[update(r,previous[tuple(r['case'])],chosen,proof.get(tuple(r['case'])),
                      precision.get(tuple(r['case'])),
                      SPLIT_ALPHA/(n*TOTAL_NEW)) for r in validation['roots']])
    (HERE/'population_bounds.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
