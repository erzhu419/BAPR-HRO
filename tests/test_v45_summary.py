from copy import deepcopy

import pytest

from experiments.reconstruction_v45 import summarize


def bank():
    return dict(seed=7,count=128,generated_count=128,future_messages=0,clock=10.,deadline=100.,
        public_history=[dict(cid=3,clock=10.)],cid=3,source=None,base_action='alight',selected_action='continue',
        rows={arm:[dict(world_index=i,prefix=dict(kind='terminal',status='destination',trace=[],entry=None),
            value=summarize.point(value)) for i in range(128)] for arm,value in (('alight',20.),('continue',15.))})


def test_same_bank_requires_every_original_prefix_and_local_value():
    saved=bank();fresh=deepcopy(saved)
    assert summarize.same_bank(fresh,saved)
    fresh['rows']['continue'][127]['prefix']['status']='deadline'
    with pytest.raises(ValueError,match='prefix or target'):
        summarize.same_bank(fresh,saved)
    fresh=deepcopy(saved);fresh['rows']['continue'][127]['value']['mean']+=.1
    with pytest.raises(ValueError,match='prefix or target'):
        summarize.same_bank(fresh,saved)
    fresh=deepcopy(saved);fresh['rows']['alight'].pop()
    with pytest.raises(ValueError,match='Missing paired worlds'):
        summarize.same_bank(fresh,saved)


def test_population_minus_sample_is_descriptive_and_has_no_monte_carlo_interval():
    population=summarize.scalar_summary(summarize.point(20.))
    sample=dict(mean=30.,mean_interval=[30.,30.],paired_se=5.)
    result=summarize.summary_difference(population,sample)
    assert result==dict(mean=-10.,mean_interval=[-10.,-10.])
    assert population['paired_se'] is None
    # A finite-bank mean outside a population bound is not a scientific failure.
    population_bound=dict(mean=None,mean_interval=[15.,25.])
    assert summarize.summary_difference(population_bound,sample)['mean_interval']==[-15.,-5.]


def test_common_population_states_do_not_require_a_sample_seed_or_unchanged_action():
    sampled=bank()
    population={k:v for k,v in sampled.items() if k!='rows'}
    population.update(seed=None,count=0,generated_count=0,selected_action='alight',
        actions={'alight':summarize.point(12.),'continue':summarize.point(15.)})
    result=summarize.gate_comparison(dict(gates=[population]),dict(gates=[sampled]),0,'absorbing','matched128')
    assert len(result['common'])==1
    assert not result['common'][0]['same_selected_action']
    assert result['common'][0]['subject_minus_baseline_gap']['mean']==-8.
    population['public_history']=[]
    result=summarize.gate_comparison(dict(gates=[population]),dict(gates=[sampled]),0,'absorbing','matched128')
    assert result['unmatched_subject_indices']==[0] and result['unmatched_baseline_indices']==[0]


def test_same_bank_action_change_is_rejected_even_when_values_are_equal():
    sampled=bank();candidate=deepcopy(sampled);candidate['selected_action']='alight'
    with pytest.raises(ValueError,match='actual action'):
        summarize.gate_comparison(dict(gates=[candidate]),dict(gates=[sampled]),0,'markov128','matched128')


def test_exact_population_conserves_unresolved_mass_and_retains_sampling_fallback():
    value=dict(count=0,generated_count=0,selected_action='alight',
        atoms={'alight':[dict(kind='platform',probability=1.)],
               'continue':[dict(kind='terminal',probability=.7),dict(kind='unresolved_prefix',probability=.3)]},
        full_pi_bounds={'alight':dict(lower=0.,upper=100.),'continue':dict(lower=20.,upper=90.)},
        full_pi_difference_interval=[-90.,80.],residual_interval=[-95.,75.])
    summarize.absorbing_validation(value)
    result=summarize.population_diagnostics(value)
    assert result['unresolved_prefix_probability']['continue']==.3
    assert not result['full_pi_rank_certified'] and result['selected_action_regret_upper']==80.
    value['atoms']['continue'].pop()
    with pytest.raises(ValueError,match='probability mass'):
        summarize.population_diagnostics(value)
    fallback=bank()
    summarize.absorbing_validation(fallback)
    assert not summarize.population_mode(fallback)
    assert summarize.values_summary(fallback,'absorbing')[1]['mean']==5.


def test_all_declared_receipts_are_required_before_publishing(tmp_path,monkeypatch):
    monkeypatch.setattr(summarize,'HERE',tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(SystemExit,match='Awaiting 804 required V45 receipts'):
        summarize.main()
    assert not (tmp_path/'summary.json').exists()
