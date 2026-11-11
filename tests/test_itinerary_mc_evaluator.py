from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v6.itinerary_mc_evaluator import FixedWitness, execute_fixed_witness
from src.committed_transit import CommittedState, build_committed_model
from src.evolving_trip_belief import EvolvingTripBelief, logsumexp
from src.risk_bounds import CommitmentBounds
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


class SmallGrid(EvolvingTripBelief):
    def _grid(self, slot, shifts, anchors, cancellation_possible):
        return super()._grid(slot, np.array([-1, 0, 1]) * self.minute, anchors, cancellation_possible)


def full_bank(belief, parameters):
    possibilities = []
    for trip, slots in belief._trips.items():
        law = belief._law(trip, parameters)
        normalizer = logsumexp(law.forward[-1])
        choices = []
        for states in product(*(range(len(times)) for times in law.times)):
            log_mass = law.initial[states[0]] + law.emissions[0][states[0]]
            for i in range(1, len(states)):
                log_mass += law._transition(i - 1)[states[i - 1], states[i]] + law.emissions[i][states[i]]
            if not np.isfinite(log_mass):
                continue
            events = {}
            for i in range(0, len(slots), 2):
                cid = slots[i].cid
                canceled = bool(law.canceled[i][states[i]])
                departure, arrival = int(law.times[i][states[i]]), int(law.times[i + 1][states[i + 1]])
                events[cid] = VehicleEvent(999 if canceled else departure - slots[i].scheduled,
                                           canceled, departure, arrival)
            choices.append((float(np.exp(log_mass - normalizer)), events))
        possibilities.append(choices)
    return [(float(np.prod([item[0] for item in values])),
             JourneyScenario({cid: event for _, events in values for cid, event in events.items()},
                             minute=belief.minute, publications={}))
            for values in product(*possibilities)]


@pytest.mark.parametrize('q', [0., .2])
@pytest.mark.parametrize('rescue', [False, True])
def test_actual_committed_policy_matches_analytic_itinerary_and_root_failure_rescue(q, rescue):
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=61))
    for row in [('R', 'A', 0, 1, 0, 60), ('R', 'B', 1, 2, 120, 240),
                ('R', 'RESCUE', 0, 2, 900, 960)]:
        graph.add_connection(Connection(-1, *row))
    belief = SmallGrid(graph, public_occurrences=False, travel_sd=.5, dwell_sd=.5)
    belief.observe_forecast(1, 'arrival', 275, 45)
    parameters = {'R': {'mean': 0., 'std': .6, 'cancel_rate': q}}
    state = CommittedState(0, 1, None, 1, initialized=True)
    bank = full_bank(belief, parameters)
    model = build_committed_model(graph, bank, 0, 2, 1, 1201, initial_state=state)
    nominal = belief.mean_events(parameters)
    departures = {cid: round(nominal[cid].departure) for cid in range(3)}
    policy = FixedWitness((0, 1), (2,) if rescue else ())
    actual = sum(probability * execute_fixed_witness(model, world, graph, policy, departures, 1201)['cost']
                 for (probability, _), world in zip(bank, model.scenarios))
    index = CommitmentBounds(belief, parameters, graph, 2, 1201)
    analytic = (index.with_rescue(state, policy.primary, policy.rescue).upper if rescue
                else index.itinerary(state, policy.primary).upper)
    assert actual == pytest.approx(analytic, abs=1e-8)


def test_future_wait_phase_uses_original_root_mean_and_same_trip_block():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=60))
    for row in [('R', 'A', 0, 1, 0, 60), ('R', 'A', 1, 2, 1020, 1080)]:
        graph.add_connection(Connection(-1, *row))
    belief = SmallGrid(graph, public_occurrences=False, travel_sd=.5, dwell_sd=.5)
    parameters = {'R': {'mean': 0., 'std': .5, 'cancel_rate': 0.}}
    state = CommittedState(0, 1, None, 1, initialized=True)
    bank = full_bank(belief, parameters)
    model = build_committed_model(graph, bank, 0, 2, 1, 1801, initial_state=state)
    nominal = belief.mean_events(parameters)
    departures = {cid: round(nominal[cid].departure) for cid in range(2)}
    policy = FixedWitness((0, 1))
    values = [execute_fixed_witness(model, world, graph, policy, departures, 1801)
              for world in model.scenarios]
    expected = sum(probability * value['cost'] for (probability, _), value in zip(bank, values))
    assert expected == pytest.approx(CommitmentBounds(belief, parameters, graph, 2, 1801).itinerary(state, (0, 1)).upper)
    assert any(step[2] == ('wait',) for value in values for step in value['trace'])
