import gzip
import json
from types import SimpleNamespace

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v37 import controller
from src.transit_graph import Connection, Stop, TransitGraph


def fixture():
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'R', 'a', 1, 2, 30, 40))
    positions = {(cid, kind): ('a', slot) for slot, (cid, kind) in
                 enumerate(((0, 'departure'), (0, 'arrival'), (1, 'departure'), (1, 'arrival')))}
    model = SimpleNamespace(graph=graph, _positions=positions)
    parameters = {'R': {'std': 1.}}
    fit = dict(pooled={kind: dict(bias_seconds=5., sd_seconds=30.) for kind in ('departure', 'arrival')}, route_kind={})
    history = SingleMessageHistory(model, parameters, fit=fit)
    history.apply_frame(dict(receipt_seconds=1, trip_statuses=[['a', 'SCHEDULED']], endpoint_updates=[
        [0, 'arrival', 'FORECAST', 10], [1, 'arrival', 'FORECAST', 40]]), 1)
    history.observe_platform([1], 2, 9, {})
    ride = Ride(0, (0, 1), {0: Platform(1), 1: None})
    planner = SimpleNamespace(model=model, parameters=parameters, deadline=100,
        compile_ride=lambda ride, history, method: SimpleNamespace(
            decision=lambda cid, clock: (True, dict(method=method))))
    return graph, planner, history, ride


def fake_check(calls):
    def evaluate(graph, model, parameters, deadline, node, row, history, alternative, base, *, count, seed):
        calls.append(dict(alternative=alternative, base=base, count=count, seed=seed))
        return dict(seed=seed, count=count, actions=dict(nominal=alternative, combined=base),
            rows=dict(nominal=[dict(loss=1.)], combined=[dict(loss=2.)]),
            paired=dict(full_bank_mean_gain_interval=[-1., -1.], unresolved_pairs=0),
            selected_action=alternative,
            cost=dict(total_process_cpu=4., total_wall=5., action_continuations=dict(nominal={}, combined={})))
    return evaluate


def test_continue_eligibility_is_only_the_remaining_retained_exit_structure():
    _, _, _, ride = fixture()
    assert controller.has_legal_continue(ride, 0)
    assert not controller.has_legal_continue(ride, 1)
    ride.exits = {0: Platform(1)}
    assert not controller.has_legal_continue(ride, 0)


def test_restore_preserves_single_type_fit_raw_messages_and_nonoccurrence_windows():
    _, planner, history, _ = fixture()
    stored = json.loads(json.dumps(history.public_state()))
    restored = controller.restore_single_snapshot(planner.model, planner.parameters, history.fit, stored, 9)
    assert isinstance(restored, SingleMessageHistory) and restored.fit is history.fit
    assert json.loads(json.dumps(restored.public_state())) == stored
    assert restored.frames == () and restored.frame_index == 0
    assert restored.selected_forecast('a') == history.selected_forecast('a')


def test_semantic_state_includes_unselected_raw_values_not_only_the_current_likelihood():
    _, planner, history, ride = fixture()
    diag = controller.current_diagnostics(planner, ride, 0, 9, history)
    arguments = dict(root_index=0, day='2023-10-25', count=128, diagnostics=diag)
    state = controller.onboard_state(ride, 0, 9, history, 100, **arguments)
    signature = history.signature()
    history.apply_frame(dict(receipt_seconds=1, trip_statuses=[],
        endpoint_updates=[[1, 'arrival', 'FORECAST', 70]]), 9)
    assert history.signature() == signature  # Current selected likelihood did not change.
    assert controller.onboard_state(ride, 0, 9, history, 100, **arguments) != state
    assert 'seed' not in state and 'call_index' not in state


def test_paid_receipt_roundtrip_matches_tuple_windows_and_keeps_original_seed(tmp_path, monkeypatch):
    graph, planner, history, ride = fixture()
    calls = []
    monkeypatch.setattr(controller, 'conditional_action_check', fake_check(calls))
    store = controller.ReceiptStore(0, tmp_path/'reference', tmp_path/'policy')
    first = controller.OnboardRolloutExecutor(graph, planner, 100, root_index=0,
        day='2023-10-25', store=store)
    action, _ = first.choose_onboard(ride, 0, 9, history)
    assert action is False and len(calls) == 1
    assert first.gates[0]['cheap_agreed'] and first.gates[0]['actions']['nominal_diagnostic'] == 'alight'
    assert first.gates[0]['rows'].keys() == {'base', 'alternative'}
    assert first.reused_evaluation_cpu == 0.  # The miss was already measured in actual execution.

    stored = json.loads(json.dumps(history.public_state()))
    restored = controller.restore_single_snapshot(planner.model, planner.parameters, history.fit, stored, 9)
    repeat = controller.OnboardRolloutExecutor(graph, planner, 100, root_index=0,
        day='2023-10-25', store=store)
    repeat.eligible_calls = 5
    action, receipt = repeat.choose_onboard(ride, 0, 9, restored)
    assert action is False and len(calls) == 1
    assert receipt['reused'] and receipt['seed'] == 37050 and receipt['proposed_seed'] == 37055
    assert repeat.gates[0]['source_evaluation_call_index'] == 0
    assert repeat.reused_evaluation_cpu == 4. and repeat.reused_evaluation_wall == 5.


def test_agreement_never_screens_an_eligible_arrival_but_final_exit_does_not_sample(tmp_path, monkeypatch):
    graph, planner, history, ride = fixture()
    calls = []
    monkeypatch.setattr(controller, 'conditional_action_check', fake_check(calls))
    executor = controller.OnboardRolloutExecutor(graph, planner, 100, root_index=2,
        day='2023-10-25', store=controller.ReceiptStore(2, tmp_path/'reference', tmp_path/'policy'))
    executor.choose_onboard(ride, 0, 9, history)
    assert calls == [dict(alternative='continue', base='alight', count=128, seed=39050)]
    _, receipt = executor.choose_onboard(ride, 1, 40, history)
    assert not receipt['triggered'] and len(calls) == 1 and executor.eligible_calls == 1


def test_reference_match_uses_fixed_source_order_not_the_estimated_gain(tmp_path):
    reference, policy = tmp_path/'reference', tmp_path/'policy'
    reference.mkdir()
    policy.mkdir()
    state = {'root_index': 0, 'public_history': [{'windows': [[2., 9.]]}]}
    for path, seed, value in ((reference/'000_003.json.gz', 37053, 10.), (policy/'000_000.json.gz', 37050, -10.)):
        with gzip.open(path, 'wt') as handle:
            json.dump(dict(state=state, seed=seed, paired={'full_bank_mean_gain_interval': [value, value]}), handle)
    store = controller.ReceiptStore(0, reference, policy)
    receipt, source = store.obtain(state, 8, lambda: (_ for _ in ()).throw(AssertionError('must reuse')))
    assert receipt['seed'] == 37053 and source['reused']
    assert source['receipt_source'] == str(reference/'000_003.json.gz')
