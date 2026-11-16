import pytest

from experiments.decision_gap_v1.panel import read,write
from experiments.trc_revision_v1.audit import audit


def receipt():
    return dict(root_index=0,label='cheap',
        states=[dict(kind='onboard',cid=2,clock=100.,selected_action='alight',
            public_history=[dict(evidence='forecast',receipt_seconds=90.)])],
        publications=[dict(kind='onboard_publication',opens_at=100.,closes_at=101.,
            consumed=dict(consumed_wall=100.2))],
        outcome=dict(trace=[dict(kind='onboard',cid=2,clock=100.),
            dict(kind='platform_command',entry=100.2),
            dict(kind='platform',clock=102.,choice=7,entry=100.2,
                command=dict(issued_at=100.3,ready_by_cid={'7':101.}))]))


@pytest.mark.parametrize('failure',[None,'late_command','lost_alight_time','future_forecast'])
def test_saved_clock_audit_detects_the_reachable_contract_violations(tmp_path,failure):
    row=receipt()
    if failure=='late_command':row['outcome']['trace'][2]['command']['issued_at']=102.
    if failure=='lost_alight_time':row['outcome']['trace'][1]['entry']=100.
    if failure=='future_forecast':row['states'][0]['public_history'][0]['receipt_seconds']=101.
    write(tmp_path/'contract.json',dict(roots=1,arms=[dict(label='cheap',clock_contract='async')]))
    write(tmp_path/'roots/000_cheap.json.gz',row)
    if failure:
        with pytest.raises(ValueError,match='causal receipt checks failed'):audit(tmp_path)
        assert len(read(tmp_path/'execution_audit.json')['failures'])==1
    else:
        assert audit(tmp_path)['status']=='pass'


def test_mandatory_exit_has_no_query_publication_and_keeps_arrival_clock(tmp_path):
    row=receipt()
    row['states'].append(dict(kind='onboard',cid=3,clock=110.,selected_action='alight',
        decision_type='mandatory_menu_exit',public_history=[]))
    row['outcome']['trace'].extend([dict(kind='onboard',cid=3,clock=110.,arrival_clock=110.,
        consumed_at=110.,choice='alight',receipt={'method':'mandatory_menu_exit'}),
        dict(kind='platform_command',entry=110.)])
    write(tmp_path/'contract.json',dict(roots=1,arms=[dict(label='cheap',clock_contract='async')]))
    write(tmp_path/'roots/000_cheap.json.gz',row)
    result=audit(tmp_path)
    assert result['status']=='pass'
    assert result['checked']['onboard_clock_matches_publication']==1
    assert result['checked']['mandatory_exit_at_arrival']==1
