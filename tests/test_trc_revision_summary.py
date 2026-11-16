"""Guard the actual attribution errors found in the BG5 manuscript audit."""
from experiments.trc_revision_v1.calibration import conditional_summary,pair_events
from experiments.trc_revision_v1.summarize import execution_diagnostics,fixed_policy_hindsight,paired_comparisons


def event(root,incoming,outgoing,observed='10',prediction='00'):
    return dict(day='2023-04-06',root_index=root,incoming_trip=incoming,outgoing_trip=outgoing,
        initial_state={'status':'observed'},
        model=dict(status='resolved',partitions={k:float(k==prediction) for k in ('00','01','10','11')}),
        actual=dict(status='resolved',actual_partition=observed,possible_partitions=[observed]))


def test_incident_and_event_weights_are_distinct_and_named():
    rows=[event(0,'a','b'),event(1,'a','c'),event(2,'d','e',observed='00')]
    roots=[dict(root_index=i,initial_state={'status':'observed'},registered={'rows':[i]}) for i in range(3)]
    summary=conditional_summary(rows,roots,['2023-04-06'],replicates=100)
    weight=summary['weightings']
    assert weight['equal_incident_then_equal_date']['bias_bounds']['10']==[.5,.5]
    assert weight['equal_event_then_equal_date']['bias_bounds']['10']==[2/3,2/3]


def test_availability_counts_no_opportunity_roots_and_preserves_unknowns():
    row=event(0,'a','b');row['model']={'status':'probability_unresolved'}
    row['actual']={'status':'label_unresolved','possible_partitions':['00','01','10','11']}
    roots=[dict(root_index=0,initial_state={'status':'observed'},registered={'rows':[1]}),
           dict(root_index=1,initial_state={'status':'unresolved:missing'},registered={'rows':[]})]
    summary=conditional_summary([row],roots,['2023-04-06'],replicates=100)
    assert summary['registered_roots']==2 and summary['roots_without_opportunities']==[1]
    assert summary['initial_state_statuses']['unresolved:missing']==1
    assert summary['reached_events']==1
    assert summary['weightings']['equal_incident_then_equal_date']['bias_bounds']['10']==[-1.,1.]


def test_pair_contract_changes_actual_label_and_metadata(monkeypatch):
    from experiments.trc_revision_v1 import calibration
    seen=[]
    monkeypatch.setattr(calibration,'label_event',lambda metadata,event,deadline:
        seen.append((event['upstream_mct'],event['downstream_mct'])) or dict(status='resolved'))
    monkeypatch.setattr(calibration,'public_event_features',lambda *a:dict(groups={}))
    class Pairs:
        def resolve_pair(self,a,b,**kw):return {'contract_mct_seconds':25 if a==1 else 60}
    original=dict(current_cid=1,upstream_cid=2,source_arrival_cid=3,hub_cid=4,
        upstream_mct=420,downstream_mct=420)
    value=dict(event=original,actual={},model={},public_features={})
    root=dict(initial_state={'status':'observed','clock':1},predecision_source_available=False,
        public_history=[],deadline=999)
    result=pair_events(None,None,[value],root,None,None,Pairs(),90)
    assert seen==[(25,60)]
    assert result[0]['public_features']['mct_source']=='static_pair_or_declared_same_platform_assumption'
    assert result[0]['model']['status']=='probability_unresolved'
    assert original['upstream_mct']==420


def test_hindsight_stays_two_complete_policies_and_retains_unknown_scope():
    def row(i,y):return dict(root_index=i,outcome={'loss':y})
    banks=[{'a':row(0,100),'b':row(0,90)}, {'a':row(1,100),'b':row(1,120)},
           {'a':row(2,None),'b':row(2,95)}]
    value=fixed_policy_hindsight(banks,'a','b')
    assert value['complete_pairs']==2 and value['complete_mean_saving_seconds']==5
    assert value['positive_roots']==[{'root_index':0,'gain':10}]
    assert 'state-wise switching' in value['scope']


def test_deadline_default_is_not_counted_as_executed_rollout_proposal():
    def gate(accepted,selected):
        return dict(completed_pairs=1,count=1,cost={'process_cpu':.1},
            paired={'unresolved_pairs':0},rows={arm:[dict(loss=0,status='arrived')] for arm in ('alight','continue')},
            base_action='alight',proposed_action='continue',selected_action=selected,
            publication_accepted=accepted,budget=dict(requested_process_cpu=1.,
                overshoot_seconds=0.,interrupted=False,stopped_phase='complete'))
    row=dict(root_index=0,states=[],publications=[],control_stats={},
        gates=[gate(False,'continue'),gate(True,'continue'),gate(False,'alight')])
    result=execution_diagnostics([row])
    assert result['proposed_action_changes']==3
    assert result['accepted_proposal_changes']==1
    assert result['deadline_default_changes']==1


def test_every_registered_causal_direction_has_a_summary_key():
    def row(loss,cpu):
        return dict(root_index=0,day='2023-04-06',loss_cap=100,outcome={'loss':loss},
            cost={'process_cpu_including_menu':cpu})
    banks=[{'async':row(12,2),'ideal':row(10,1)}]
    causal=[('ideal','async')]
    result=paired_comparisons(banks,['async','ideal'],causal)
    assert all(f'{a}__vs__{b}' in result for a,b in causal)
    assert result['ideal__vs__async']['complete_mean_saving_seconds']==-2
    assert result['async__vs__ideal']['complete_mean_saving_seconds']==2
