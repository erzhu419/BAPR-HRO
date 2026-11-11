from functools import lru_cache
from itertools import product

import pytest

from src.causal_prefix import causal_prefix_lower
from src.factored_lower import factored_prefix_lower, product_trip_bank
from src.information_planner import ExactPlanner, FiniteScenarioModel, Scenario
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from tests.test_risk_bounds import graph


def product_model(primary=(6, 10), rescue=(7, 11), *, alighted=False):
    timetable = graph([('R', 'I', 3, 0, -2, 0), ('R', 'I', 0, 1, 1, 2),
                       ('R', 'A', 1, 2, 4, 6), ('R', 'B', 1, 2, 5, 7)], transfer=2)
    common = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                              for c in timetable.connections}, minute=1, publications={})
    draws = {'A': tuple({2: VehicleEvent(0, False, 4, arrival)} for arrival in primary),
             'B': tuple({3: VehicleEvent(0, False, 5, arrival)} for arrival in rescue)}
    bank, labels = product_trip_bank(timetable, common, draws)
    initial = (PlatformState(1, 2, None, 2, last_connection=1, used_connections=frozenset({0, 1}))
               if alighted else PlatformState(0, 0, 'I', 0, last_connection=0,
                                               used_connections=frozenset({0})))
    model = build_platform_model(timetable, bank, initial.stop, 2, initial.time, 14,
                                 initial_state=initial, poll_interval=1)

    @lru_cache(None)
    def oracle(index, state):
        world = model.scenarios[index]
        single = FiniteScenarioModel((Scenario(world.name, 1., world.factors),), state,
            model.actions, model.transition, model.horizon, model.terminal_cost)
        return ExactPlanner(single).solve_compressed().value

    return timetable, model, oracle, labels, common, draws


@pytest.mark.parametrize('alighted', [False, True])
def test_product_support_keeps_unobserved_rescue_hidden_through_primary_ride(alighted):
    timetable, model, oracle, labels, _, _ = product_model(alighted=alighted)
    result = factored_prefix_lower(model, oracle, graph=timetable, factor_labels=labels,
                                   elapsed_horizon=12)
    elapsed = model.initial_state.time
    assert result.clairvoyant_root_mean == 7.25-elapsed
    assert result.value == ExactPlanner(model).solve_compressed().value == 8-elapsed
    assert result.boundary_kind_mass == {'changed_vehicle': 1.}
    assert result.boundary_factor_support_mass == {'1x2': 1.}
    assert result.factor_singleton_boundary_probability == {'A': 1., 'B': 0.}
    assert result.factor_singleton_before_boundary_probability == {'A': 0., 'B': 0.}
    assert result.singleton_boundary_probability == 0.
    if not alighted:
        first_ride = causal_prefix_lower(model, oracle, elapsed_horizon=12)
        assert first_ride.value == 7.25 < result.value
        assert first_ride.expected_boundary_elapsed == 2


def test_iid_empirical_factor_product_is_downward_biased_without_cross_factor_revelation():
    values = []
    for primary, rescue in product(product((6, 10), repeat=2), product((7, 11), repeat=2)):
        timetable, model, oracle, labels, _, _ = product_model(primary, rescue)
        result = factored_prefix_lower(model, oracle, graph=timetable, factor_labels=labels,
                                       elapsed_horizon=12)
        values.append(result.value)
        # Some empirical supports choose B; observing that ride still leaves A.
        assert result.singleton_boundary_probability == 0.
    assert sum(values) / len(values) == 7.5625 < 8


def test_zipped_factor_support_and_incomplete_trip_overrides_are_rejected():
    timetable, model, oracle, labels, common, draws = product_model()
    selected = (model.scenarios[0], model.scenarios[-1])
    zipped = FiniteScenarioModel(tuple(Scenario(w.name, .5, w.factors) for w in selected),
        model.initial_state, model.actions, model.transition, model.horizon, model.terminal_cost)
    with pytest.raises(ValueError, match='Cartesian'):
        factored_prefix_lower(zipped, oracle, graph=timetable,
            factor_labels={'A': (0, 1), 'B': (0, 1)}, elapsed_horizon=12)
    with pytest.raises(ValueError, match='whole named trip'):
        product_trip_bank(timetable, common, {'A': ({},), 'B': draws['B']})


def test_full_twelve_minute_wait_uses_iterative_exact_search():
    timetable = graph([('R', 'I', 3, 0, -2, 0), ('R', 'A', 0, 2, 900, 1000),
                       ('R', 'B', 0, 2, 910, 1010)], transfer=2)
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in timetable.connections}
    common = JourneyScenario(events, minute=60, publications={})
    bank, labels = product_trip_bank(timetable, common,
        {'A': ({1: events[1]}, {1: events[1]}), 'B': ({2: events[2]}, {2: events[2]})})
    initial = PlatformState(0, 0, None, 0, last_connection=0, used_connections=frozenset({0}))
    model = build_platform_model(timetable, bank, 0, 2, 0, 800, initial_state=initial, poll_interval=1)
    result = factored_prefix_lower(model, lambda index, state: 800-state.time,
        graph=timetable, factor_labels=labels, elapsed_horizon=720, max_nodes=50000)
    assert result.value == 800 and result.expected_boundary_elapsed == 720
    assert result.boundary_kind_mass == {'clock_limit': 1.}
    assert result.boundary_factor_support_mass == {'2x2': 1.}
