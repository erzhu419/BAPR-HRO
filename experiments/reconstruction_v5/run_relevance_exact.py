"""Small exact full/H receipts for the clock-support relevance implementation."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.committed_transit import OBSERVE, WAIT, build_committed_model
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph
from src.transit_relevance import compute_relevance


def graph_of(rows):
    graph = TransitGraph()
    for stop in {s for row in rows for s in row[:2]}:
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid, (source, target, departure, arrival, trip) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, source, target, departure, arrival))
    return graph


def evaluate(graph, bank, start, deadline, public):
    def support(cid, kind):
        values = [getattr(scene.events[cid], kind) for _, scene in bank if not scene.events[cid].canceled]
        return (min(values), max(values)) if values else None
    result = compute_relevance(graph, 0, 2, start, deadline, support, public_occurrences=public)
    model = build_committed_model(graph, bank, 0, 2, start, deadline)
    menu = lambda info: tuple(action for action in model.actions(info.state)
                              if action in (OBSERVE, WAIT) or action[1] in result.retained_connection_ids)
    planners = {'full': ExactPlanner(model), 'all_future_H': ExactPlanner(model, menu)}
    evidence = {}
    for name, planner in planners.items():
        before = perf_counter()
        solution = planner.solve()
        evidence[name] = {'value_remaining': solution.value,
                          'value_total': start + solution.value,
                          'root_action': solution.action,
                          'wall_ms': (perf_counter() - before) * 1000,
                          'work': dict(planner.stats)}
    return {'transport_edges': sorted(result.transport_edge_ids),
            'retained_edges': sorted(result.retained_connection_ids),
            'required_trips': sorted(result.required_trip_ids), 'public_occurrences': public,
            'worlds': len(bank), 'solutions': evidence}


def main():
    graph = graph_of([(0, 1, 0, 1, 'incoming'), (1, 2, 2, 4, 'target'),
                      (0, 2, 0, 7, 'direct'), (0, 3, 0, 1, 'irrelevant')])
    bank = []
    for target_cancel in (False, True):
        for nuisance_cancel in (False, True):
            events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
            events[1] = replace(events[1], canceled=target_cancel)
            events[3] = replace(events[3], canceled=nuisance_cancel)
            bank.append((.25, JourneyScenario(events, publications={})))
    independent = evaluate(graph, bank, 0, 8, False)
    graph = graph_of([(3, 0, -1, 0, 'useful'), (0, 2, 2, 4, 'useful'),
                      (0, 2, 1, 6, 'direct')])
    bank = [(0.5, JourneyScenario({0: VehicleEvent(shift, False, -1 + shift, shift),
                                  1: VehicleEvent(0, False, 2, 4 if shift == 0 else 8),
                                  2: VehicleEvent(0, False, 1, 6)})) for shift in (0, 1)]
    same_trip = evaluate(graph, bank, 1, 9, True)
    no_feed = evaluate(graph, [(p, JourneyScenario(s.events, publications={})) for p, s in bank], 1, 9, False)
    output = {'scope': 'Finite committed models with exact common information and all-future H. True example retains all trips, demonstrating necessary whole-trip evidence rather than a True pruning theorem.',
              'independent_false': independent, 'remote_same_trip_true': same_trip,
              'same_worlds_without_external_receipts': no_feed}
    path = Path(__file__).with_name('relevance_exact_results.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: {method: record['value_total'] for method, record in case['solutions'].items()}
                      for name, case in output.items() if isinstance(case, dict)}))


if __name__ == '__main__':
    main()
