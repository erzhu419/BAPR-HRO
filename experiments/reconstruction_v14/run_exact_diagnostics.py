"""Solve four known finite transit populations under matched platform rules."""

from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_prefix import causal_prefix_lower
from src.information_planner import ExactPlanner
from src.platform_transit import ALIGHT, OBSERVE, WAIT, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def graph(timetable):
    result = TransitGraph()
    for stop in sorted({s for row in timetable for s in row[1:3]}):
        result.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for trip, origin, destination, departure, arrival in timetable:
        result.add_connection(Connection(-1, trip, trip, origin, destination, departure, arrival))
    return result


def scenario(timetable, overrides):
    events = {}
    for cid, (_, _, _, scheduled_departure, scheduled_arrival) in enumerate(timetable):
        departure, arrival = overrides.get(cid, (scheduled_departure, scheduled_arrival))
        events[cid] = VehicleEvent(departure - scheduled_departure, False, departure, arrival)
    return JourneyScenario(events, publications={})


def platform(timetable, population, goal=1, cap=30):
    """The entire declared uniform population, or a specified empirical bank."""
    network = graph(timetable)
    bank = [(1 / len(population), scene) for scene in population]
    # All local departures and arrivals interrupt WAIT. No remote events or
    # forecasts enter these fixtures, so extra clock polls carry no information.
    return build_platform_model(network, bank, 0, goal, 0, cap, poll_interval=cap)


def solve(model):
    started = perf_counter()
    planner = ExactPlanner(model)
    result = planner.solve_compressed()
    return dict(value=result.value, seconds=perf_counter() - started,
                nodes=planner.stats['information_nodes_solved'],
                transitions=planner.stats['transition_evaluations'])


def reveal(model, labels):
    """Reveal exactly the selected scenario label at the root."""
    groups = defaultdict(list)
    for world, label in zip(model.scenarios, labels, strict=True):
        groups[label].append(world)
    started = perf_counter()
    value, nodes, transitions = 0.0, 0, 0
    for group in groups.values():
        mass = sum(w.probability for w in group)
        posterior = tuple(replace(w, probability=w.probability / mass) for w in group)
        solved = solve(replace(model, scenarios=posterior))
        value += mass * solved['value']
        nodes += solved['nodes']
        transitions += solved['transitions']
    return dict(value=value, seconds=perf_counter() - started, nodes=nodes,
                transitions=transitions, revealed_groups=len(groups))


def restrict_boarding(model, allowed):
    def actions(state):
        return tuple(a for a in model.actions(state) if a[0] != 'board' or a[1] in allowed)

    return replace(model, actions=actions)


def take_visible(model, allowed):
    """A fixed policy that reads the public action menu only."""
    started = perf_counter()
    mean, losses, transitions = 0.0, [], 0
    for world in model.scenarios:
        state, loss = model.initial_state, 0
        while (actions := model.actions(state)):
            board = next((a for a in actions if a[0] == 'board' and a[1] in allowed), None)
            if board is not None:
                action = board
            elif ALIGHT in actions:
                action = ALIGHT
            elif WAIT in actions:
                action = WAIT
            else:
                action = actions[0]
            outcome = model.transition(world, state, action)
            state, loss = outcome.state, loss + outcome.cost
            transitions += 1
        loss += model.terminal_cost(world, state)
        losses.append(loss)
        mean += world.probability * loss
    return dict(value=mean, world_losses=losses, seconds=perf_counter() - started,
                transitions=transitions)


def after_first_departure(model):
    """Group histories by the actually observed departure, then value both choices."""
    groups = defaultdict(list)
    for world in model.scenarios:
        initial = model.transition(world, model.initial_state, OBSERVE).state
        departure = model.transition(world, initial, WAIT).state
        assert departure.visible_connections == (0,)
        groups[departure].append(world)
    results = {}
    for state, group in groups.items():
        mass = sum(world.probability for world in group)
        posterior = tuple(replace(world, probability=world.probability / mass) for world in group)
        solution = ExactPlanner(replace(model, scenarios=posterior, initial_state=state)).solve()
        results[state.time] = dict(board_A=state.time + solution.action_values['board', 0],
                                   wait_for_B=state.time + solution.action_values[WAIT],
                                   choose=solution.action[0])
    return results


def late_suffix_and_empirical_bias():
    # Reconstructs the Word diagnosis §3 without estimating a population from a sample.
    timetable = [('A', 0, 1, 2, 3), ('B', 0, 1, 4, 10)]
    atoms = tuple(product((2, 3), (1, 20)))
    population = tuple(scenario(timetable, {0: (dep, dep + duration)}) for dep, duration in atoms)
    model = platform(timetable, population)
    causal = solve(model)
    wait_B = take_visible(model, {1})
    clairvoyant = reveal(model, range(len(population)))
    assert causal['value'] == wait_B['value'] == 10
    assert clairvoyant['value'] == 6.75

    estimates = []
    for count, expected in [(1, 6.75), (2, 7.5625), (4, 8.52734375)]:
        started = perf_counter()
        histogram = Counter()
        nodes = transitions = 0
        for bank in product(population, repeat=count):
            # Use the same observed departure time and physical action menu as
            # in the population solve. Only the empirical law changes.
            solved = solve(platform(timetable, bank))
            histogram[solved['value']] += 1
            nodes += solved['nodes']
            transitions += solved['transitions']
        banks = len(population) ** count
        mean = sum(value * occurrences for value, occurrences in histogram.items()) / banks
        assert mean == expected
        estimates.append(dict(atoms_per_bank=count, banks=banks, mean_value=mean,
                              optimism=causal['value'] - mean, nodes=nodes,
                              transitions=transitions, seconds=perf_counter() - started,
                              value_counts=dict(sorted(histogram.items()))))

    aliased = platform(timetable, (population[0], population[3]))
    aliased_bank = solve(aliased)
    assert aliased_bank['value'] == 6.5  # departure 2 means fast, 3 means slow in this sample
    true_choices = after_first_departure(model)
    aliased_choices = after_first_departure(aliased)
    assert true_choices == {2: dict(board_A=12.5, wait_for_B=10, choose='wait'),
                            3: dict(board_A=13.5, wait_for_B=10, choose='wait')}
    assert aliased_choices == {2: dict(board_A=3, wait_for_B=10, choose='board'),
                               3: dict(board_A=23, wait_for_B=10, choose='wait')}
    scaled_gap = 100 * estimates[-1]['optimism']
    assert scaled_gap == 147.265625 > 60
    return dict(timetable=timetable, independent_atoms=atoms, true_causal=causal,
                fixed_wait_for_B=wait_B, true_clairvoyant=clairvoyant,
                conditional_duration_support={dep: [1, 20] for dep in (2, 3)},
                example_aliased_atoms=(atoms[0], atoms[3]), example_aliased_bank=aliased_bank,
                true_choices_after_departure=true_choices, aliased_choices_after_departure=aliased_choices,
                iid_bank_enumeration=estimates, uniform_time_scale=100,
                scaled_gap_seconds=scaled_gap,
                scaled_decomposition_seconds=dict(policy_regret=0, rest_information=0,
                    hindsight_tail=0, empirical_bank_optimism=scaled_gap))


def slow_first_visible():
    timetable = [('B', 0, 1, 2, 20), ('A', 0, 1, 3, 9)]
    model = platform(timetable, (scenario(timetable, {}),))
    first_visible, wait_A, causal = take_visible(model, {0, 1}), take_visible(model, {1}), solve(model)
    assert first_visible['value'] == 20
    assert wait_A['value'] == causal['value'] == 9
    return dict(timetable=timetable, first_visible=first_visible, wait_for_A=wait_A,
                true_causal=causal, policy_regret=first_visible['value'] - causal['value'])


def third_vehicle():
    timetable = [('A', 0, 1, 5, 12), ('B', 0, 1, 4, 10), ('R', 0, 1, 2, 3)]
    arrivals = (3, 15)
    model = platform(timetable, tuple(scenario(timetable, {2: (2, arrival)}) for arrival in arrivals))
    restricted = restrict_boarding(model, {0, 1})
    restricted_causal, restricted_revealed = solve(restricted), reveal(restricted, arrivals)
    full_causal, full_revealed = solve(model), reveal(model, arrivals)
    old, expanded = take_visible(model, {0, 1}), take_visible(model, {0, 1, 2})
    assert old['value'] == restricted_causal['value'] == restricted_revealed['value'] == 10
    assert expanded['value'] == full_causal['value'] == 9
    assert full_revealed['value'] == 6.5
    return dict(timetable=timetable, equiprobable_R_arrivals=arrivals,
                restricted_causal=restricted_causal, restricted_revealed=restricted_revealed,
                full_causal=full_causal, full_revealed=full_revealed,
                first_visible_AB=old, first_visible_ABR=expanded,
                causal_candidate_gain=restricted_causal['value'] - full_causal['value'],
                full_information_gain=full_causal['value'] - full_revealed['value'],
                additional_candidate_gain_with_information=(restricted_revealed['value']
                    - full_revealed['value']) - (restricted_causal['value'] - full_causal['value']))


def prefix_to_rides(model, rides):
    @lru_cache(None)
    def tail(index, state):
        world = replace(model.scenarios[index], probability=1)
        return ExactPlanner(replace(model, scenarios=(world,), initial_state=state)).solve_compressed().value

    started = perf_counter()
    result = causal_prefix_lower(model, tail, elapsed_horizon=30,
        stop_when=lambda state: len(state.used_connections) >= rides)
    return dict(value=result.value, seconds=perf_counter() - started,
                nodes=result.information_nodes, transitions=result.transition_evaluations,
                tail_solves=result.oracle_evaluations,
                expected_boundary_time=result.expected_boundary_elapsed,
                boundary_kind_mass=result.boundary_kind_mass)


def hindsight_after_boarding():
    timetable = [('I', 0, 1, 1, 2), ('E', 1, 2, 3, 5), ('B', 1, 2, 4, 10)]
    model = platform(timetable, tuple(scenario(timetable, {1: (3, arrival)}) for arrival in (5, 20)), goal=2)
    fixed = take_visible(model, {0, 2})
    causal = solve(model)
    after_first = prefix_to_rides(model, 1)
    after_transfer = prefix_to_rides(model, 2)
    assert fixed['value'] == causal['value'] == after_transfer['value'] == 10
    assert after_first['value'] == 7.5
    return dict(timetable=timetable, equiprobable_E_arrivals=(5, 20), fixed_wait_for_B=fixed,
                true_causal=causal, clairvoyance_after_first_ride=after_first,
                clairvoyance_after_second_ride=after_transfer,
                restored_value=after_transfer['value'] - after_first['value'])


def run():
    return dict(scope='Analytically specified finite populations. Values are exact for these models, '
                      'not city bounds. Timings are single local diagnostic runs.',
                information='Local occurrence and boarding facts only. Candidate restrictions are '
                            'diagnostic and can increase the complete problem value.',
                late_suffix_and_empirical_bias=late_suffix_and_empirical_bias(),
                slow_first_visible=slow_first_visible(), third_vehicle=third_vehicle(),
                hindsight_after_boarding=hindsight_after_boarding())


if __name__ == '__main__':
    results = run()
    path = Path(__file__).with_name('exact_results.json')
    path.write_text(json.dumps(results, indent=2) + '\n')
    print(path)
