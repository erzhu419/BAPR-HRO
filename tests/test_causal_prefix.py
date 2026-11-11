from functools import lru_cache
from itertools import product

import pytest

from src.causal_prefix import PrefixBudgetExceeded, causal_prefix_lower
from src.information_planner import ExactPlanner, FiniteScenarioModel, Scenario
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from tests.test_risk_bounds import graph


def connection_choice(arrivals, *, beacon=None):
    timetable = graph([('R', 'A', 3, 0, -2, 0), ('R', 'A', 0, 2, 1, 3),
                       ('R', 'B', 0, 2, 2, 5)], transfer=2)
    if beacon is not None:
        from src.transit_graph import Connection
        timetable.add_connection(Connection(-1, 'R', 'beacon', 3, 0, -1, 0))
    bank = []
    for i, arrival in enumerate(arrivals):
        events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                  for c in timetable.connections}
        events[1] = VehicleEvent(0, False, 1, arrival)
        if beacon is not None:
            events[3] = VehicleEvent(0, False, -1, beacon[i])
        bank.append((1 / len(arrivals), JourneyScenario(events, minute=1, publications={})))
    initial = PlatformState(0, 0, 'A', 0, last_connection=0, used_connections=frozenset({0}))
    model = build_platform_model(timetable, bank, 0, 2, 0, 10, initial_state=initial, poll_interval=1)

    @lru_cache(None)
    def oracle(index, state):
        world = model.scenarios[index]
        single = FiniteScenarioModel((Scenario(world.name, 1., world.factors),), state,
            model.actions, model.transition, model.horizon, model.terminal_cost)
        return ExactPlanner(single).solve_compressed().value

    return model, oracle


@pytest.mark.parametrize('horizon', [1, 2, 4])
def test_causal_prefix_retains_root_nonanticipativity_and_transfer_readiness(horizon):
    # Alight at0 to catch B at2, or stay for A at1 before its arrival is known.
    # Waiting onboard until1 then alighting cannot catch B: MCT starts at1.
    model, oracle = connection_choice((3, 9))
    result = causal_prefix_lower(model, oracle, elapsed_horizon=horizon)
    assert result.clairvoyant_root_mean == 4
    assert result.value == ExactPlanner(model).solve_compressed().value == 5
    assert result.boundary_support_mass == {2: 1.}
    assert result.singleton_boundary_probability == result.singleton_before_boundary_probability == 0
    assert sum(result.boundary_kind_mass.values()) == pytest.approx(1)
    assert result.information_nodes > 0 and result.transition_evaluations > 0
    assert result.expected_boundary_elapsed <= 5


def test_small_iid_banks_are_downward_biased_not_the_population_causal_optimum():
    # Enumerate every IID bank of size2 from the two-world population.
    values, root_oracles = [], []
    for arrivals in product((3, 9), repeat=2):
        model, oracle = connection_choice(arrivals)
        result = causal_prefix_lower(model, oracle, elapsed_horizon=2)
        values.append(result.value)
        root_oracles.append(result.clairvoyant_root_mean)
    assert values == [3, 5, 5, 5]
    assert sum(root_oracles) / 4 == 4
    assert sum(values) / 4 == 4.5 < 5


def test_exact_local_observations_can_accidentally_identify_a_bank_world():
    # In this two-world bank a local arrival reveals A's remote future.
    # Expanding to the independent four-world population removes that link.
    model, oracle = connection_choice((3, 9), beacon=(0, 1))
    small = causal_prefix_lower(model, oracle, elapsed_horizon=2)
    assert small.value == small.clairvoyant_root_mean == 4
    assert small.singleton_before_boundary_probability == 1
    assert small.boundary_support_mass == {1: 1.}
    population, population_oracle = connection_choice((3, 3, 9, 9), beacon=(0, 1, 0, 1))
    full = causal_prefix_lower(population, population_oracle, elapsed_horizon=2)
    assert full.value == 5 and full.singleton_before_boundary_probability == 0
    assert full.boundary_support_mass == {2: 1.}


def test_exhausted_search_returns_no_partial_lower_bound():
    model, oracle = connection_choice((3, 9))
    with pytest.raises(PrefixBudgetExceeded) as caught:
        causal_prefix_lower(model, oracle, elapsed_horizon=2, max_nodes=1)
    assert caught.value.nodes == 2 and caught.value.transitions > 0
