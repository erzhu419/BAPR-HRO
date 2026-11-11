import numpy as np
import pytest

from src.committed_transit import CommittedState, build_committed_model
from src.continuation_gate import check_continuation
from src.evolving_trip_belief import EvolvingTripBelief
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescueBounds, freeze_tail_plan
from src.temporal_corridor import TemporalCorridor
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks
from src.transit_graph import Connection, Stop, TransitGraph


@pytest.mark.parametrize('same_car', [False, True])
@pytest.mark.parametrize('force_sampling', [False, True])
def test_middle_gate_uses_transfer_and_bounds_frozen_policy(tmp_path, monkeypatch, same_car, force_sampling):
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=15))
    rides = [('A', 0, 1, 0, 10), ('A', 1, 3, 15, 22), ('A', 3, 2, 30, 40)] if same_car else [
        ('A', 0, 1, 0, 10), ('B', 1, 2, 30, 40), ('C', 1, 2, 20, 25)]
    for trip, origin, target, dep, arr in rides:
        graph.add_connection(Connection(-1, trip, trip, origin, target, dep, arr))
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    for c in graph.connections:
        belief.observe_connection_event(c.id, 'departure', 0)
        belief.observe_connection_event(c.id, 'arrival', 0)
    parameters = {c.route: dict(mean=0., std=1., cancel_rate=0.) for c in graph.connections}
    state = CommittedState(1, 10, 'A', 10, frozenset({0}), initialized=True)
    bounds = TailRescueBounds(belief, parameters, graph, 2, 42)
    plan = freeze_tail_plan(bounds, (1, 2) if same_car else (1,))
    kernel = build_temporal_kernel(tmp_path / 'temporal.so')
    corridor = TemporalCorridor(TemporalScan(graph, kernel))
    scenes = [(1., JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                                  for c in graph.connections}, minute=1, publications={}))]
    exact = ExactPlanner(build_committed_model(graph, scenes, 1, 2, 10, 42, initial_state=state)).solve_compressed().value
    options = dict(tolerance=5, samples=128, alpha=.05)
    rng = np.random.default_rng(42)
    old = check_continuation(bounds, state, plan, corridor, rng, allow_middle=False, **options)
    if same_car:
        assert old['kind'] == 'terminal' and not old['accepted']
    else:
        assert old is None

    draws = []
    if force_sampling:
        # Zero remains a valid, weaker support floor. It forces the sequential
        # path on this exact finite model, whose optimal loss is known above.
        original_corridor = corridor.connections_and_floor
        monkeypatch.setattr(corridor, 'connections_and_floor',
                            lambda *args, **kwargs: (original_corridor(*args, **kwargs)[0], 0.))

        def record_draws(belief, parameters, rng, n, **kwargs):
            draws.append(n)
            return sample_clocks(belief, parameters, rng, n, **kwargs)

        monkeypatch.setattr('src.continuation_gate.sample_clocks', record_draws)
        options.update(alpha=.00625, sample_sizes=(8, 32, 128))
    else:
        def no_draws(*args, **kwargs):
            raise AssertionError('A deterministic support certificate must not draw worlds')

        monkeypatch.setattr('src.continuation_gate.sample_clocks', no_draws)
    result = check_continuation(bounds, state, plan, corridor, rng, allow_middle=True, **options)
    assert result['accepted'] and result['kind'] == 'middle'
    assert exact == result['upper'] == 30
    if force_sampling:
        assert draws == [8, 24]
        assert result['worlds'] == 32 and result['alpha'] == .00625
        assert [row['worlds'] for row in result['sample_checkpoints']] == [8, 32]
        assert not result['sample_checkpoints'][0]['accepted']
        assert result['oracle_mean'] == exact
        assert result['lower'] <= exact
    else:
        assert result['support_floor'] == result['lower'] == exact
        assert result['worlds'] == 0 and result['alpha'] == 0.
