from src import issued_continuation as publisher
from src.committed_transit import CommittedState, build_committed_model
from src.issued_continuation import ContinuationUpdate, replay_continuation
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


def case(departure):
    graph = TransitGraph()
    for stop in range(2):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for trip, time in [('A', departure), ('B', 9)]:
        graph.add_connection(Connection(-1, trip, trip, 0, 1, time, time + 10))
    events = {0: VehicleEvent(0, False, departure, departure + 10), 1: VehicleEvent(0, False, 9, 19)}
    initial = CommittedState(0, 0, None, 0, initialized=True)
    world = JourneyScenario(events, minute=60, publications={})
    model = build_committed_model(graph, [(1., world)], 0, 1, 0, 1000, initial_state=initial)
    return graph, model


def test_worker_result_cannot_interrupt_an_incumbent_commitment(monkeypatch):
    graph, model = case(6)
    times = iter((0., 4.2))
    monkeypatch.setattr(publisher, 'perf_counter', lambda: next(times))
    incumbent = TailRescuePlan((0,))
    replacement = TailRescuePlan((1,))
    result = replay_continuation(model, model.scenarios[0], incumbent,
        lambda state, active, ordinal: ContinuationUpdate(plan=replacement),
        lambda *args: None, lambda *args: True, lambda *args: True, graph=graph, deadline=1000, minute=60)
    state, missing, decisions, actions = result
    assert missing is None and state.time == 16
    assert decisions[0]['ready_at'] == 5 and decisions[0]['publication'] == 'closed_during_compute'
    assert actions[0]['connection'] == 0 and actions[0]['while_computing']


def test_publisher_revalidates_replacement_before_old_wait_ends(monkeypatch):
    graph, model = case(780)
    times = iter((0., 4.2))
    monkeypatch.setattr(publisher, 'perf_counter', lambda: next(times))
    incumbent = TailRescuePlan((0,), first_action_now=False, departures=((0, 780),))
    replacement = TailRescuePlan((1,))
    checked = []

    def applicable(state, plan):
        checked.append((state.time, plan.primary))
        return True

    state, missing, decisions, actions = replay_continuation(model, model.scenarios[0], incumbent,
        lambda state, active, ordinal: ContinuationUpdate(plan=replacement),
        lambda *args: None, applicable, lambda *args: True, graph=graph, deadline=1000, minute=60)
    assert missing is None and state.time == 19
    assert checked == [(5, (1,))] and decisions[0]['publication'] == 'installed'
    assert actions[0]['connection'] == 1 and actions[0]['clock'] == 5
