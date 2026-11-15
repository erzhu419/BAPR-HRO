"""Natural-date denominators, missing computation and residual-event attribution."""
from copy import deepcopy

import pytest

from experiments.decision_gap_v1.panel import NEW_LABELS, read, write
from experiments.natural_gap_v2 import summarize as reporting


@pytest.fixture
def completed(tmp_path):
    directory = tmp_path/'natural'
    dates = ['day1','day2','day3','day4','no_public_roots']
    root_days = ['day1','day1','day2','day2','day3','day4','day1']
    selected = [dict(root_id=f'root{i}',day=day,start=0,deadline=300)
                for i,day in enumerate(root_days)]
    write(directory/'contract.json',dict(roots=len(selected),
        policies=[label.removeprefix('new_') for label in NEW_LABELS],
        rollout_limits=dict(worlds_per_query=2,process_cpu_seconds=120,seed=510015)))
    write(directory/'selection.json',dict(selected=selected))
    write(directory/'date_manifest.json',dict(selected_dates=dates))
    for index,root in enumerate(selected):
        for label in NEW_LABELS:
            loss = 100.
            if label in reporting.CHALLENGERS:
                loss = {0:90.,1:110.,2:90.,6:80.}.get(index,loss)
            if index==2 and label=='new_snapshot_csa':
                loss = 80.
            if index in (3,5):
                loss = None
            if index==4:
                loss = 300.
            status = {3:'unresolved:initial arrival:MISSING',4:'no_initial_service',
                      5:'missing_source:RT'}.get(index,'arrived')
            state = dict(kind='onboard',root_index=index,cid=2 if index==1 else 1,
                         clock=30. if index==1 else 20.,node={},selected_action='continue',
                         base_action='continue',public_history=[dict(evidence='observed',
                         kind='arrival',cid=2 if index==1 else 1,trip='same incoming train')])
            row = dict(root_index=index,root_id=root['root_id'],day=root['day'],label=label,
                origin='O',goal=f'G{index}',loss_cap=300.,executed=index!=5,
                outcome=dict(loss=loss,status=status),actions=[],
                states=[] if index in (3,4,5) else [state],gates=[],
                observation=dict(station_to_cids=[dict(stop=1,cids=[3])]),
                coverage=dict(source_complete=index!=5,numeric_forecast_available=index!=6,
                              initial_prefix_status=status if index in (3,4,5) else 'observed'),
                cost=dict(process_cpu=None if index==5 else 2.,wall=None if index==5 else 3.,
                          shared_menu=dict(process_cpu=.5,wall=.6),
                          process_cpu_including_menu=None if index==5 else 2.5,
                          wall_including_menu=None if index==5 else 3.6))
            if index in (0,1) and label in reporting.CHALLENGERS:
                complete = index==0
                action = 'alight' if complete else 'continue'
                state.update(gate_index=0,selected_action=action,base_action='continue')
                row['gates'] = [dict(count=2,completed_pairs=2 if complete else 0,future_messages=0,
                    seed=510015,used_kernel='full_common_observation_pi',fallback=None,
                    state_index=0,base_action='continue',selected_action=action,
                    paired={'full_bank_mean_gain_interval':[-3.,-3.] if complete else [-270.,270.]},
                    rows={arm:[dict(world_index=j,loss=value if complete else None,
                                    status='arrived' if complete else 'cpu_unresolved') for j in range(2)]
                          for arm,value in [('alight',5.),('continue',8.)]},
                    common_tape=dict(seed=510015))]
            write(directory/'roots'/f'{index:03}_{label}.json.gz',row)
    return directory


def test_every_registered_root_and_empty_date_retained_with_unknown_compute(completed):
    result = reporting.summarize(completed)
    assert result['roots']==7 and result['archive_receipts']==56 and result['executed_calls']==48
    cheap = result['aggregate']['new_cheap']
    assert cheap['complete']==5 and cheap['unresolved']==2
    assert cheap['all_root_mean_loss_bounds']==pytest.approx([700/7,1300/7])
    assert cheap['mean_cpu'] is None
    assert cheap['cpu_measurement']==dict(mean=None,measured_mean=2.5,measured=6,unknown=1)
    assert cheap['unexecuted_roots']==[5]
    assert result['by_day']['no_public_roots']['registered_roots']==0
    assert result['by_day']['day3']['aggregate']['new_cheap']['all_root_mean_loss_bounds']==[300.,300.]
    assert result['by_day']['day4']['aggregate']['new_cheap']['all_root_mean_loss_bounds']==[0.,300.]
    assert result['by_day']['day4']['aggregate']['new_cheap']['mean_cpu'] is None
    assert result['coverage'][6]['coverage']['numeric_forecast_available'] is False
    assert not result['g2_passed']


def test_paired_quality_keeps_positive_negative_unknown_and_cost_denominators(completed):
    result = reporting.summarize(completed)
    pair = result['comparisons']['new_cheap__new_full128']
    assert pair['positive_roots']==[0,2,6] and pair['negative_roots']==[1]
    assert pair['unresolved_roots']==[3,5]
    assert pair['all_root_mean_gain_bounds']==pytest.approx([-570/7,630/7])
    assert pair['complete_pair_mean_gain']==6.
    assert pair['mean_cpu_delta'] is None
    assert pair['cpu_difference_measurement']==dict(mean=None,measured_mean=0.,measured=6,unknown=1)
    import csv
    with (completed/'paired_journeys.csv').open() as handle:
        row = next(r for r in csv.DictReader(handle) if r['root_index']=='5' and r['policy']=='full128')
    assert row['loss']==row['cpu']==''
    assert float(row['loss_lower'])==0. and float(row['loss_upper'])==300.


def test_residual_candidate_requires_every_fixed_baseline_and_duplicate_events_are_descriptive(completed):
    result = reporting.summarize(completed)
    for challenger in reporting.CHALLENGERS:
        residual = result['residual_diagnostic'][challenger]
        assert residual['strictly_better_than_every_cheap_policy_roots']==[0,6]
        assert residual['versus_each_fixed_cheap_policy']['new_snapshot_csa']['negative_roots']==[1,2]
    groups = result['event_groups']
    assert next(r for r in groups['first_event_groups'] if r['day']=='day1' and r['cid']==1)['roots']==[0,6]
    assert next(r for r in groups['incoming_trip_groups'] if r['day']=='day1')['roots']==[0,1,6]
    assert [r['root_index'] for r in groups['roots_without_first_onboard_event']]==[3,4,5]


def test_original_gate_audit_still_accounts_for_unfinished_worlds(completed):
    result = reporting.summarize(completed)
    gates = result['gate_audit']
    assert gates['total']==4 and gates['same_state_full128_comparisons']==1
    assert gates['cpu_unfinished_by_policy']=={'new_matched128':1,'new_full128':1}
    path = completed/'roots/001_new_full128.json.gz'
    row = read(path);row['gates'][0]['rows']['continue'].pop();write(path,row)
    with pytest.raises(AssertionError,match='Unresolved worlds were removed'):
        reporting.summarize(completed)


def test_missing_result_is_not_silently_removed_from_panel(completed):
    (completed/'roots/003_new_full128.json.gz').unlink()
    with pytest.raises(FileNotFoundError):
        reporting.summarize(completed)


@pytest.mark.parametrize('change',['fake_time','observation','menu_cost'])
def test_reject_changed_information_or_fabricated_unexecuted_timing(completed,change):
    path = completed/'roots/005_new_full128.json.gz';row=read(path)
    if change=='fake_time':
        row['cost'].update(process_cpu=0.,process_cpu_including_menu=.5)
    elif change=='observation':
        row['observation']['station_to_cids'][0]['cids'].append(4)
    else:
        row['cost']['shared_menu']['process_cpu'] += 1.
    write(path,row)
    with pytest.raises(ValueError):
        reporting.summarize(completed)


def test_fully_measured_panel_reports_menu_inclusive_whole_panel_timing(completed):
    selection=read(completed/'selection.json');selection['selected']=selection['selected'][:1]
    contract=read(completed/'contract.json');contract['roots']=1
    write(completed/'selection.json',selection);write(completed/'contract.json',contract)
    row=read(completed/'roots/000_new_full128.json.gz')
    row['cost'].update(process_cpu=13.,process_cpu_including_menu=13.5)
    write(completed/'roots/000_new_full128.json.gz',row)
    result=reporting.summarize(completed)
    assert result['aggregate']['new_full128']['mean_cpu']==13.5
    assert result['comparisons']['new_cheap__new_full128']['mean_cpu_delta']==11.
