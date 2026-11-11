from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from src.committed_transit import CommittedState, WAIT, build_committed_model
from src.evolving_trip_belief import EvolvingTripBelief, logsumexp
from src.information_planner import ExactPlanner
from src.risk_bounds import CommitmentBounds, _prefix_window_mass, transfer_gate_certificate
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


class SmallGrid(EvolvingTripBelief):
    def _grid(self, slot, shifts, anchors, cancellation_possible):
        return super()._grid(slot, np.array([-1, 0, 1]) * self.minute, anchors,
                             cancellation_possible)


def graph(rows, transfer=60):
    result = TransitGraph()
    for sid in sorted({s for row in rows for s in row[2:4]}):
        result.add_stop(Stop(sid, str(sid), min_transfer_time=transfer))
    for row in rows:
        result.add_connection(Connection(-1, *row))
    return result


def bank(belief, parameters):
    trips = []
    for trip, slots in belief._trips.items():
        law = belief._law(trip, parameters)
        normalizer = float(logsumexp(law.forward[-1]))
        choices = []
        for indices in product(*(range(len(times)) for times in law.times)):
            value = law.initial[indices[0]] + law.emissions[0][indices[0]]
            for i, (before, after) in enumerate(zip(indices, indices[1:])):
                value += law._transition(i)[before, after] + law.emissions[i + 1][after]
            if not np.isfinite(value):
                continue
            events = {}
            for i in range(0, len(slots), 2):
                cid = slots[i].cid
                canceled = bool(law.canceled[i][indices[i]])
                departure, arrival = int(law.times[i][indices[i]]), int(law.times[i + 1][indices[i + 1]])
                events[cid] = VehicleEvent(999 if canceled else departure - slots[i].scheduled,
                                           canceled, departure, arrival)
            choices.append((float(np.exp(value - normalizer)), events))
        trips.append(choices)
    worlds = []
    for combination in product(*trips):
        probability = float(np.prod([item[0] for item in combination]))
        events = {cid: event for _, mapping in combination for cid, event in mapping.items()}
        worlds.append((probability, JourneyScenario(events, minute=belief.minute, publications={})))
    return worlds


def fixed_policy_cost(model, world, path, nominal, graph_, minute, first_now=True):
    state, cost = model.initial_state, 0
    for number, cid in enumerate(path):
        c = graph_.connections[cid]
        transfer = 0 if state.current_trip is None or state.current_trip == c.trip_id else graph_.stops[state.stop].min_transfer_time
        if not (number == 0 and first_now):
            # Determine the fixed-root nominal attempt threshold, then execute
            # actual WAIT transitions. No future world clock chooses an action.
            while c.dep_time > state.time + 25 * minute:
                outcome = model.transition(world, state, WAIT)
                state, cost = outcome.state, cost + outcome.cost
            while nominal[cid].departure is not None and nominal[cid].departure > max(state.time, state.last_arrival + transfer) + 12 * minute:
                outcome = model.transition(world, state, WAIT)
                state, cost = outcome.state, cost + outcome.cost
        outcome = model.transition(world, state, ('select', cid))
        state, cost = outcome.state, cost + outcome.cost
        if not state.feedback or state.feedback[-1][0] != 'board' or outcome.terminal:
            break
    return cost + model.terminal_cost(world, state), state


def test_commitment_integral_matches_all_exact_direct_action_values():
    timetable = graph([('R', 'A', 0, 1, 1, 2)], transfer=1)
    belief = SmallGrid(timetable, minute=1, public_occurrences=False)
    parameters = {'R': {'mean': 0., 'std': .7, 'cancel_rate': .2}}
    state = CommittedState(0, 0, None, 0, initialized=True)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 0, 1, 0, 4, initial_state=state)
    result = CommitmentBounds(belief, parameters, timetable, 1, 4).commit(state, 0)
    exact = ExactPlanner(model).solve()
    assert result.expected_first_cost == pytest.approx(exact.action_values['select', 0])
    assert result.direct_stop_upper == pytest.approx(result.expected_first_cost)
    expected = sum(probability for probability, scene in worlds
                   if not scene.events[0].canceled and 0 <= scene.events[0].departure <= 4)
    assert result.catch_probability == pytest.approx(expected)


def test_calibrated_common_posterior_values_match_its_joint_enumeration():
    from src.conditional_risk import StudentClockLaw
    from src.trip_marginal_tilt import TripMarginalTilt
    timetable = graph([('R', 'A', 0, 1, 1, 2), ('R', 'A', 1, 2, 3, 4)], transfer=1)
    base = SmallGrid(timetable, minute=1, public_occurrences=False)
    parameters = {'R': {'mean': 0., 'std': .7, 'cancel_rate': .1}}
    common = TripMarginalTilt(base, parameters, {0: StudentClockLaw(1.4, .5, .1)})
    state = CommittedState(0, 0, None, 0, initialized=True)
    worlds = bank(common, parameters)
    model = build_committed_model(timetable, worlds, 0, 2, 0, 6, initial_state=state)
    nominal = common.mean_events(parameters)
    expected = sum(world.probability * fixed_policy_cost(model, world, (0, 1), nominal, timetable, 1)[0]
                   for world in model.scenarios)
    value = CommitmentBounds(common, parameters, timetable, 2, 6).itinerary(state, (0, 1))
    assert value.upper == pytest.approx(expected)
    from src.first_branch_belief import FirstBranchSampler
    branch = FirstBranchSampler(common, parameters, 0, 0, 6)
    risk = CommitmentBounds(common, parameters, timetable, 2, 6).commit(state, 0)
    assert branch.probability(True) == pytest.approx(risk.catch_probability)
    times, weights, _ = branch.conditional_distribution(1, 'arrival')
    assert times @ weights == pytest.approx(common._mean_event(1, branch.conditional_law(True)).arrival)


@pytest.mark.parametrize('q', [0., .2])
def test_contiguous_same_trip_itinerary_is_exact_under_all_evidence(q):
    timetable = graph([('R', 'A', 0, 1, 60, 120), ('R', 'A', 1, 2, 120, 180)])
    belief = SmallGrid(timetable, minute=60, public_occurrences=False,
                       travel_sd=.5, dwell_sd=.5)
    belief.observe_forecast(1, 'arrival', 210, 45)
    parameters = {'R': {'mean': 0., 'std': .6, 'cancel_rate': q}}
    state = CommittedState(0, 0, None, 0, initialized=True)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 0, 2, 0, 600, initial_state=state)
    nominal = belief.mean_events(parameters)
    expected = sum(p * fixed_policy_cost(model, scenario, (0, 1), nominal, timetable, 60)[0]
                   for p, scenario in zip((p for p, _ in worlds), model.scenarios))
    value = CommitmentBounds(belief, parameters, timetable, 2, 600).itinerary(state, (0, 1))
    assert value.upper == pytest.approx(expected, abs=1e-9)
    assert value.success_probability == pytest.approx(value.arrival_mass.sum())
    assert 0 <= value.success_probability <= 1 + 1e-12


def test_first_failure_rescue_exact_joint_and_family_lower():
    timetable = graph([('R', 'A', 0, 1, 0, 60), ('R', 'RESCUE', 0, 2, 900, 960),
                       ('R', 'B', 1, 2, 120, 240)])
    belief = SmallGrid(timetable, minute=60, public_occurrences=False,
                       travel_sd=.5, dwell_sd=.5)
    parameters = {'R': {'mean': 0., 'std': .6, 'cancel_rate': .2}}
    state = CommittedState(0, 0, None, 0, initialized=True)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 0, 2, 0, 1200, initial_state=state)
    nominal = belief.mean_events(parameters)
    expected = 0.
    for (p, _), scenario in zip(worlds, model.scenarios):
        first = model.transition(scenario, state, ('select', 0))
        caught = first.state.feedback[-1][0] == 'board'
        if caught:
            value, _ = fixed_policy_cost(model, scenario, (0, 2), nominal, timetable, 60)
        else:
            child = SimpleNamespace(initial_state=first.state, transition=model.transition,
                                    terminal_cost=model.terminal_cost)
            remaining, _ = fixed_policy_cost(child, scenario, (1,), nominal, timetable, 60,
                                             first_now=False)
            value = first.cost + remaining
        expected += p * value
    index = CommitmentBounds(belief, parameters, timetable, 2, 1200)
    value = index.with_rescue(state, (0, 2), (1,))
    assert value.upper == pytest.approx(expected, abs=1e-8)
    assert value.commitment_lower <= value.fixed_rescue_family_lower <= value.upper
    with pytest.raises(ValueError, match='exclude'):
        index.with_rescue(state, (0, 2), (0, 2))


def test_window_prefix_sorts_ready_not_nonmonotone_arrival_phase():
    # Arrival0 must wait to60; arrival1 can act at1. Prefixing arrivals is wrong.
    result = _prefix_window_mass(np.array([60, 1]), np.array([.3, .7]),
                                  np.array([5, 60, 61]), 10, 100)
    np.testing.assert_allclose(result, [.7, .3, .3])


def test_public_occurrence_model_is_not_silently_treated_as_hidden_failure():
    timetable = graph([('R', 'A', 0, 1, 60, 120)])
    with pytest.raises(TypeError, match='no occurrence'):
        CommitmentBounds(SmallGrid(timetable), {}, timetable, 1, 500)


@pytest.mark.parametrize('q', [0., .2])
def test_transfer_gate_encloses_full_causal_optimum_not_only_a_scenario_bank(q):
    timetable = graph([('R', 'ONBOARD', 0, 1, 0, 60), ('R', 'OTHER', 0, 1, 60, 60)], transfer=120)
    belief = SmallGrid(timetable, minute=60, public_occurrences=False,
                       travel_sd=.5, dwell_sd=.5)
    parameters = {'R': {'mean': 0., 'std': .6, 'cancel_rate': q}}
    state = CommittedState(0, 0, 'ONBOARD', 0, initialized=True)
    worlds = bank(belief, parameters)
    model = build_committed_model(timetable, worlds, 0, 1, 0, 300, initial_state=state)
    certificate = transfer_gate_certificate(belief, parameters, state, (0,), 1, 300)
    optimum = ExactPlanner(model).solve()
    assert certificate is not None
    assert certificate.transfer_gate == 120
    assert certificate.lower <= optimum.value + 1e-9
    assert certificate.upper + 1e-9 >= optimum.value
    assert certificate.upper == pytest.approx(optimum.action_values['select', 0])
    assert certificate.regret_bound + 1e-9 >= optimum.action_values['select', 0] - optimum.value


def test_transfer_gate_rejects_reachable_same_vehicle_shortcuts_and_old_arrival():
    timetable = graph([('R', 'A', 0, 1, 0, 60), ('R', 'A', 0, 1, 30, 40)], transfer=120)
    parameters = {'R': {'mean': 0., 'std': .6, 'cancel_rate': .2}}
    belief = SmallGrid(timetable, minute=60, public_occurrences=False)
    index = CommitmentBounds(belief, parameters, timetable, 1, 300)
    state = CommittedState(0, 0, 'A', 0, initialized=True)
    assert index.transfer_gate(state, (0,)) is None
    assert index.transfer_gate(CommittedState(0, 60, 'A', 0, initialized=True), (0,)) is None


def test_transfer_gate_mixes_shared_or_independent_day_trip_posteriors():
    from src.joint_day_belief import JointDayBelief
    from src.independent_day_mixture_belief import IndependentDayMixtureBelief
    timetable = graph([('R', 'A', 0, 1, 0, 60), ('R', 'B', 2, 3, 600, 660)], transfer=120)
    parameters = {'R': {'mean': 0., 'std': .6, 'cancel_rate': .01}}
    state = CommittedState(0, 0, 'A', 0, initialized=True)
    for cls in (JointDayBelief, IndependentDayMixtureBelief):
        belief = cls(timetable, minute=60, public_occurrences=False,
                     deltas_seconds=[-60, 60], prior_weights=[.5, .5], travel_sd=.5)
        belief.observe_connection_event(1, 'arrival', 60)
        result = transfer_gate_certificate(belief, parameters, state, (0,), 1, 300)
        if cls is JointDayBelief:
            _, weights = belief.day_posterior(parameters)
        else:
            _, weights = belief.trip_posterior(0, parameters)
        exact = [CommitmentBounds(component, pars, timetable, 1, 300).transfer_gate(state, (0,))
                 for component, pars in zip(belief.components, belief._shifted_parameters)]
        assert result.lower == pytest.approx(sum(w * v.lower for w, v in zip(weights, exact)))
        assert result.upper == pytest.approx(sum(w * v.upper for w, v in zip(weights, exact)))


def test_two_leg_gate_lower_even_encloses_worldwise_clairvoyant_routes():
    timetable = graph([('R', 'A', 0, 1, 0, 60), ('R', 'A', 1, 2, 60, 120),
                       ('R', 'B', 1, 2, 120, 120)], transfer=120)
    belief = SmallGrid(timetable, minute=60, public_occurrences=False,
                       travel_sd=.5, dwell_sd=.5)
    parameters = {'R': {'mean': 0., 'std': .6, 'cancel_rate': 0.}}
    state = CommittedState(0, 0, 'A', 0, initialized=True)
    certificate = CommitmentBounds(belief, parameters, timetable, 2, 360).transfer_gate(state, (0, 1))
    oracle, witness = 0., 0.
    for probability, scene in bank(belief, parameters):
        model = build_committed_model(timetable, [(1., scene)], 0, 2, 0, 360, initial_state=state)
        oracle += probability * ExactPlanner(model).solve().value
        current, cost = state, 0
        for cid in (0, 1):
            outcome = model.transition(model.scenarios[0], current, ('select', cid))
            current, cost = outcome.state, cost + outcome.cost
            if current.feedback[-1][0] != 'board' or outcome.terminal:
                break
        witness += probability * (cost + model.terminal_cost(model.scenarios[0], current))
    assert certificate.lower <= oracle + 1e-9
    assert certificate.upper == pytest.approx(witness)
