import pytest

from experiments.reconstruction_v43 import summarize


def test_unresolved_bridge_keeps_full_world_denominator_and_error_bounds():
    values=[summarize.point(4.),dict(lower=2.,upper=10.,mean=None,unresolved_probability=.2)]
    result=summarize.distribution_summary(values)
    assert result['worlds']==2 and result['resolved_worlds']==1
    assert result['mean'] is None and result['paired_se'] is None
    assert result['mean_interval']==[3.,7.]
    assert result['mean_unresolved_probability_bound']==.1
    error=summarize.distribution_summary([summarize.difference(v,summarize.point(5.)) for v in values])
    aggregate=summarize.error_aggregate([error])
    assert error['mean_interval']==[-2.,2.]
    assert aggregate['mean_absolute_error_interval']==[0.,2.]
    assert aggregate['denominator']==1 and aggregate['fully_resolved_states']==0


def test_paired_bridge_uses_covariance_and_telescopes_with_signed_gap():
    left=[summarize.point(10.),summarize.point(20.)]
    right=[summarize.point(7.),summarize.point(17.)]
    bridge=summarize.distribution_summary([summarize.difference(a,b) for a,b in zip(left,right)])
    assert bridge['mean']==3. and bridge['paired_se']==0.
    levels=[summarize.point(v) for v in (4.,9.,7.,12.)]
    components=[summarize.difference(levels[i+1],levels[i])['mean'] for i in range(3)]
    assert sum(components)==summarize.difference(levels[-1],levels[0])['mean']==8.


def test_prefix_validation_rejects_changed_policy_before_first_platform():
    events=[dict(kind='onboard',cid=3,clock=10,choice='alight'),
            dict(kind='platform',stop=4,clock=20,choice=8)]
    original=dict(trace=events,status='arrived')
    prefix=dict(kind='platform',trace=events[:1],stop=4)
    assert summarize.prefix_matches(prefix,original)
    prefix['trace']=[dict(events[0],choice='continue')]
    assert not summarize.prefix_matches(prefix,original)
    prefix=dict(kind='terminal',trace=events[:1],status='arrived')
    assert not summarize.prefix_matches(prefix,original)


def test_missing_reference_receipts_cannot_emit_partial_summary(tmp_path,monkeypatch):
    monkeypatch.setattr(summarize,'HERE',tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'existing_attribution.json').write_text('{}')
    with pytest.raises(SystemExit,match='Awaiting 436 required V43 receipts'):
        summarize.main()
    assert not (tmp_path/'summary.json').exists()


def test_zero_gap_retains_base_and_unresolved_straddling_gap_is_not_a_flip():
    assert summarize.rank(dict(mean_interval=[0.,0.]),'continue')=='continue'
    assert summarize.rank(dict(mean_interval=[-2.,4.]),'alight')=='undetermined'
    assert summarize.rank(dict(mean_interval=[-2.,-1.]),'continue')=='alight'


def test_terminal_prefix_loss_and_forced_alight_entry_are_not_inferred_from_trace_only():
    direct=dict(kind='ride',first_cid=8,path=[8],exits=[dict(cid=8,next=None)])
    platform=dict(kind='platform',stop=4,rides=[dict(cid=8,next=direct)])
    state=dict(cid=2,clock=10,node=dict(kind='ride',first_cid=2,path=[2],exits=[dict(cid=2,next=platform)]))
    original=dict(trace=[],status='arrived',loss=20.)
    row=dict(prefix=dict(kind='terminal',trace=[],status='arrived',decisions=dict(platform=0,onboard=0)),
             **{level:summarize.point(21.) for level in ('root','updated','working')})
    assert not summarize.validate_reference_prefix(row,original,state,'continue')
    row={level:summarize.point(20.) for level in ('root','updated','working')}
    row['prefix']=dict(kind='platform',trace=[],status='next_platform',entry=11,stop=4,
        decisions=dict(platform=0,onboard=0),node_structure=summarize.serialized_structure(platform))
    assert not summarize.validate_reference_prefix(row,original,state,'alight')
    row['prefix']['entry']=10
    assert summarize.validate_reference_prefix(row,original,state,'alight')
