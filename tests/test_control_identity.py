"""Fixed-fit cold control wiring, common messages, and separate result receipts."""
from copy import deepcopy
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.control_identity_v1 import run as runner
from experiments.control_identity_v1 import summarize as reporting
from experiments.decision_gap_v1.panel import read, write
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v37.controller import node_structure
from test_v51_baselines import deterministic_case


def test_rotated_panel_has_every_fit_policy_once_with_adjacent_fit_pairs():
    expected = {(model, policy) for model in runner.MODELS for policy in runner.POLICIES}
    first_models = []
    for index in range(72):
        order = runner.arm_order(index)
        assert len(order) == len(set(order)) == 16 and set(order) == expected
        for left, right in zip(order[::2], order[1::2], strict=True):
            assert left[1] == right[1] and left[0] != right[0]
            first_models.append(left[0])
    assert first_models.count('old_fit') == first_models.count('repaired_fit') == 288


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    directory = tmp_path/'experiments/control_identity_v1'
    gap = directory.parent/'decision_gap_v1'
    roots = [dict(root_index=i, root_id=f'root{i}', day='day', start=10, deadline=100)
             for i in range(72)]
    budgets = dict(worlds_per_query=128, process_cpu_seconds=120, seed=510015,
                   matched_maximum_candidates=6)
    contract = dict(policies=list(runner.POLICIES), models=list(runner.MODELS), roots=72,
        rollout_limits=budgets, root_manifest=roots, control_dates=['day'],
        emission_sources={m:f'experiments/source/{m}.json' for m in runner.MODELS},
        emission_snapshots={m:f'{m}.json' for m in runner.MODELS})
    write(directory/'contract.json', contract)
    write(gap/'contract.json', dict(rollout_limits=budgets))
    write(directory.parent/'reconstruction_v34/selection.json', dict(selected=roots))
    write(directory/'messages.json.gz', dict(dates={'day':dict(frames=[])}))
    for model in runner.MODELS:
        fit = dict(fixed_model=model)
        write(directory/contract['emission_snapshots'][model], fit)
        write(tmp_path/contract['emission_sources'][model], fit)
    for row in roots:
        write(gap/'menus'/f"{row['root_index']:03}.json.gz",
              dict(root_index=row['root_index'], root=row))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'HERE', directory)
    monkeypatch.setattr(runner, 'GAP', gap)
    return directory, contract


def test_fixed_inputs_accept_both_fits_and_all_72_original_roots(frozen_inputs):
    directory, contract = frozen_inputs
    assert runner.check_inputs(directory) == contract


@pytest.mark.parametrize('change', ['fit', 'budget', 'stream_date', 'root_clock'])
def test_input_check_rejects_changed_fit_budget_stream_or_root(frozen_inputs, change):
    directory, contract = frozen_inputs
    if change == 'fit':
        path = directory/'repaired_fit.json'; value = {'fixed_model':'changed'}
    elif change == 'stream_date':
        path = directory/'messages.json.gz'; value = {'dates':{}}
    else:
        path = directory/'contract.json'; value = deepcopy(contract)
        if change == 'budget':
            value['rollout_limits']['worlds_per_query'] = 256
        else:
            value['root_manifest'][15]['deadline'] += 1
    write(path, value)
    with pytest.raises(ValueError):
        runner.check_inputs(directory)


def test_all_16_arms_are_cold_and_keep_fit_in_frozen_simulation_history(monkeypatch):
    graph, _, parameters, original, platform = deterministic_case(monkeypatch)
    node = platform.rides[0]
    old_fit = original.fit
    repaired = deepcopy(old_fit)
    repaired['pooled']['arrival']['bias_seconds'] = 12.
    frames = [dict(trip_id=trip, receipt_seconds=clock, trip_statuses=[],
                   endpoint_updates=[[cid,'arrival','FORECAST',value]])
              for trip,clock,cid,value in [('A',1,1,40),('C',2,3,30),
                                          ('B',3,2,35),('A',50,1,100)]]
    budgets = dict(worlds_per_query=128, process_cpu_seconds=120, seed=510015,
                   matched_maximum_candidates=6)
    resident = dict(fit=dict(increments=dict(travel_sd=.05,dwell_sd=.05,cancel_persistence=.8),
                            route_priors=dict(parameters, pooled=parameters['R'])),
        emissions={'old_fit':old_fit,'repaired_fit':repaired},
        messages=dict(dates={'day':dict(frames=frames)}),
        contract=dict(rollout_limits=budgets, emission_snapshots={m:m+'.json' for m in runner.MODELS}))
    menu = dict(root=dict(root_id='root',day='day',start=9,deadline=60,
                         origin_name='origin',goal_name='goal'),
        expanded_node=node_structure(node),
        new_observation_contract=dict(station_to_cids=[dict(stop=1,cids=[3])]),
        cost=dict(process_cpu=7.,wall=11.))
    calls = []

    class ProbeExecutor:
        def __init__(self, graph, planner, deadline, **kwargs):
            self.planner, self.kwargs = planner, kwargs
            self.states, self.gates, self.publications, self.decision_costs = [], [], [], []
            self.control = SimpleNamespace(stats={})

        def run(self, restored, start, history, world):
            history.advance(start)
            snapshot = freeze_snapshot(history, start)
            assert snapshot.fit is history.fit and snapshot.clone().fit is history.fit
            assert snapshot.frames == () and snapshot.frame_index == 0
            assert history.frame_index == 2
            assert [frame['trip_id'] for frame in history.frames] == ['A','C','A']
            assert history.frames[-1]['receipt_seconds'] == 50
            assert self.kwargs['rollout_limits'] is budgets
            assert self.kwargs['observation_contract'].cids_at(1) == (3,)
            # The retained candidate affects the forecast likelihood in clones,
            # while future remote publications remain unavailable to simulation.
            likelihood = snapshot.forecast_log_emission(1,'arrival',40,5,np.array([40.]))[0]
            calls.append((self.kwargs['label'], history, self.planner, restored, likelihood))
            return dict(loss=None,status='unresolved:synthetic missing endpoint',trace=[])

    monkeypatch.setattr(runner, 'PolicyExecutor', ProbeExecutor)
    monkeypatch.setattr(runner, 'process_time', lambda:next(cpu))
    monkeypatch.setattr(runner, 'perf_counter', lambda:next(wall))
    cpu, wall = count(100.,2.), count(1000.,5.)
    graph_context = (graph,None,None,{cid:cid for cid in range(4)},{},None)
    results = [runner.one_arm(0,model,policy,menu,graph_context,resident)
               for model,policy in runner.arm_order(0)]
    assert len({id(row[1]) for row in calls}) == 16
    assert len({id(row[1].belief) for row in calls}) == 16
    assert len({id(row[2]) for row in calls}) == len({id(row[3]) for row in calls}) == 16
    for result, (_,history,_,_,likelihood) in zip(results,calls,strict=True):
        name = result['emission_model']
        assert history.fit is resident['emissions'][name]
        assert likelihood == (0. if name == 'old_fit' else -8.)
        assert result['outcome']['loss'] is None and result['loss_cap'] == 51
        assert result['cost']['process_cpu_including_menu'] == 9.
        assert result['cost']['wall_including_menu'] == 16.
        assert result['cost']['shared_menu'] == menu['cost']
        assert result['relevant_remote_frames'] == 3


def test_root_results_are_isolated_by_fit_and_existing_receipt_is_not_replayed(frozen_inputs, monkeypatch):
    directory, contract = frozen_inputs
    resident = dict(directory=directory,contract=contract,graphs={'day':object()},load_cost={})
    monkeypatch.setattr(runner,'RESIDENT',resident)
    existing = directory/'models/old_fit/roots/000_new_cheap.json.gz'
    write(existing,dict(original_receipt='preserved'))
    called = []

    def one(index, model, policy, *args):
        called.append((model,policy))
        return dict(emission_model=model,label='new_'+policy,
                    outcome=dict(loss=None,status='unresolved:missing'),
                    cost=dict(process_cpu_including_menu=1.))

    monkeypatch.setattr(runner,'one_arm',one)
    receipt = runner.run_root(0)
    assert len(called) == len(receipt['completed']) == 15
    assert ('old_fit','cheap') not in called
    assert read(existing) == {'original_receipt':'preserved'}
    for model,policy in called:
        row = read(directory/'models'/model/'roots'/f'000_new_{policy}.json.gz')
        assert row['emission_model'] == model and row['outcome']['loss'] is None


@pytest.fixture
def completed_panel(tmp_path, monkeypatch):
    directory, gap = tmp_path/'controls', tmp_path/'menus'
    monkeypatch.setattr(reporting,'GAP',gap)
    write(directory/'contract.json',dict(roots=3,rollout_limits={'worlds_per_query':2},
                                       measurement='cold query plus fixed menu charge'))
    write(directory/'source_receipts.json',dict(unchanged_frame_dates=[]))
    observation = {'station_to_cids':[{'stop':1,'cids':[3]}]}
    menu_cost = {'process_cpu':.5,'wall':.6}
    for index in range(3):
        write(gap/'menus'/f'{index:03}.json.gz',dict(root={'root_id':f'root{index}'},
              new_observation_contract=observation,cost=menu_cost))
        for model_name in runner.MODELS:
            for policy in runner.POLICIES:
                loss = ([90.,110.,None][index]
                        if model_name=='repaired_fit' and policy=='cheap' else 100.)
                row = dict(root_index=index,root_id=f'root{index}',day='day',origin=0,goal=2,
                    loss_cap=300.,label='new_'+policy,emission_model=model_name,
                    relevant_remote_frames=3,observation=observation,
                    outcome=dict(loss=loss,status='complete' if loss is not None else 'unresolved:missing',trace=[]),
                    actions=[],states=[],gates=[],cost=dict(process_cpu=2.,wall=3.,shared_menu=menu_cost,
                        process_cpu_including_menu=2.5,wall_including_menu=3.6))
                if index==0 and policy in ('matched128','full128'):
                    # Same fit/state must agree; different fits intentionally
                    # yield different sampled clocks/values on the common tape.
                    value = 10. if model_name=='old_fit' else 20.
                    row['states'] = [dict(kind='onboard',root_index=index,cid=1,clock=10.,node={},
                        public_history={},selected_action='alight',base_action='alight',gate_index=0)]
                    row['gates'] = [dict(future_messages=0,count=2,
                        rows={'alight':[{'loss':value}]*2,'continue':[{'loss':value+2}]*2},
                        paired={'full_bank_mean_gain_interval':[-2.,-2.]},selected_action='alight',
                        base_action='alight',state_index=0,used_kernel='full_common_observation_pi',
                        fallback=None,completed_pairs=2,common_tape={'seed':510015})]
                write(directory/'models'/model_name/'roots'/f'{index:03}_new_{policy}.json.gz',row)
    return directory


def test_summary_separates_fit_banks_and_retains_wins_harms_and_unknown_roots(completed_panel):
    result = reporting.summarize(completed_panel,make_plot=False)
    assert result['archive_calls'] == 48
    for model in runner.MODELS:
        assert result['models'][model]['gate_audit']['same_state_full128_comparisons'] == 1
    aggregate = result['models']['repaired_fit']['aggregate']['new_cheap']
    assert aggregate['roots'] == 3 and aggregate['complete'] == 2 and aggregate['unresolved'] == 1
    assert aggregate['all_root_mean_loss_bounds'] == pytest.approx([200/3,500/3])
    assert aggregate['complete_mean_loss'] == 100.
    compared = result['model_comparisons']['cheap']
    assert compared['positive_roots'] == [0] and compared['negative_roots'] == [1]
    assert compared['unresolved_roots'] == [2] and compared['complete_pair_mean_gain'] == 0.
    assert compared['all_root_mean_gain_bounds'] == pytest.approx([-200/3,100/3])
    assert not result['g2_passed']
    import csv
    with (completed_panel/'paired_journeys.csv').open() as handle:
        unknown = next(row for row in csv.DictReader(handle)
                       if row['model']=='repaired_fit' and row['policy']=='cheap' and row['root_index']=='2')
    assert unknown['loss']=='' and float(unknown['loss_lower'])==0. and float(unknown['loss_upper'])==300.


def test_summary_still_rejects_inconsistent_banks_within_one_fit(completed_panel):
    path = completed_panel/'models/repaired_fit/roots/000_new_full128.json.gz'
    row = read(path)
    row['gates'][0]['rows']['alight'][0]['loss'] += 1.
    write(path,row)
    with pytest.raises(AssertionError,match='Same public-state full rollout bank differs'):
        reporting.summarize(completed_panel,make_plot=False)


def test_unchanged_source_regression_keeps_missing_loss_and_checks_partial_trace(completed_panel):
    row = read(completed_panel/'models/old_fit/roots/002_new_cheap.json.gz')
    row['outcome'] = dict(loss=None,status='unresolved:missing',
        trace=[dict(kind='onboard',cid=1,clock=20,choice='continue')])
    old_path = reporting.GAP/'roots/002_new_cheap.json.gz'
    write(old_path,row)
    result = reporting.unchanged_source_regression([{'new_cheap':row}],['day'])
    assert result['checked'] == result['exact_trajectory_matches'] == 1
    assert result['unexpected_complete_bank_differences'] == []
    old = deepcopy(row)
    old['outcome']['trace'][0]['clock'] = 21
    write(old_path,old)
    changed = reporting.unchanged_source_regression([{'new_cheap':row}],['day'])
    assert changed['rows'][0]['equal']['loss'] is True
    assert changed['rows'][0]['equal']['public_physical_trace'] is False
    assert len(changed['unexpected_complete_bank_differences']) == 1
    assert row['outcome']['loss'] is None
