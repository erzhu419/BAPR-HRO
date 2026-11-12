"""Retrospectively check overtaking in the selected pair and its wider timetable menu."""

from collections import Counter

from experiments.reconstruction_v23.run_transfer_screen import (
    HERE, STEPS, TRAINING, build_trip_blocks, choose_conditional, choose_nominal,
    dated_graph, departure_marks, downstream_catalog, fit_training, load_city,
    model, read_gz, replay_outcome, transfer_case, trip_key, write_json,
)


def measured_offers(graph, actual, metadata, row, case, options):
    """Collect known costs and identify coverage holes in the counterfactual menu."""
    known, incomplete = [], False
    for option in options:
        _, duration, cid, path = option
        event = actual.events.get(cid)
        if event is None:
            incomplete = True
            continue
        if event.canceled or not case['ready'] <= event.departure < row['deadline']:
            continue
        arrival, _ = replay_outcome(graph, actual, metadata, path, event.departure, row['deadline'])
        if arrival is None:
            incomplete = True
            continue
        known.append((event.departure, duration, cid, arrival))
    return sorted(known), incomplete


def score_witness(graph, actual, metadata, base, params, row, options, first, faster):
    witness = dict(row, options=[o for o in options if o[2] in (first[2], faster[2])])
    case = transfer_case(graph, actual, metadata, witness)
    assert case['clock'] == first[0] and case['other'] == faster[2]
    clocks = case['entry'], case['clock'], row['deadline']
    trips = {trip_key(graph.connections[o[2]]) for o in witness['options']}
    decisions = {}
    for step in STEPS:
        belief = base.with_grid_step(step, trips=trips)
        marks_for = departure_marks(belief, params)
        marks = {cid: marks_for(option.path) for cid, option in case['options'].items()}
        nominal, nominal_gap = choose_nominal(case['visible'], marks, *clocks, case['other'])
        conditional, conditional_gap, fallback = choose_conditional(
            graph, belief, params, witness, case['root'], case['visible'], marks, case['options'],
            case['clock'], case['other'],
        )
        decisions[step] = dict(nominal=nominal, conditional=conditional, conditional_fallback=fallback,
                               nominal_wait_minus_board=nominal_gap,
                               conditional_wait_minus_board=conditional_gap)
    return decisions


def audit_day(data, day, rows, fit):
    graph, actual, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    blocks = build_trip_blocks(graph)
    base, params = model(graph, fit)
    counts, witnesses = Counter(), []
    for row in rows:
        case = transfer_case(graph, actual, metadata, row)
        if case['status'] != 'offers':
            continue
        counts['witnessed_first_offer'] += 1
        if case['actual_legal_options'] == 2:
            counts['selected_pair_both_legal'] += 1
            selected, _ = measured_offers(graph, actual, metadata, row, case, row['options'])
            if len(selected) == 2:
                counts['selected_pair_both_tails_known'] += 1
                counts['selected_pair_second_arrives_earlier'] += selected[1][-1] < selected[0][-1]
        incoming = graph.connections[row['prefix'][-1]]
        catalog = downstream_catalog(graph, blocks, row['stop'], incoming.arr_time+row['delay'], row['deadline'])
        options = [o for o in catalog[row['goal']]
                   if trip_key(graph.connections[o[2]]) != trip_key(incoming)]
        counts['catalog_with_more_than_two_options'] += len(options) > 2
        measured, incomplete = measured_offers(graph, actual, metadata, row, case, options)
        if len(measured) < 2:
            continue
        counts['catalog_with_two_known_legal_paths'] += 1
        counts['catalog_with_two_known_legal_paths_and_no_missing_endpoint'] += not incomplete
        first = measured[0]
        faster = min(measured, key=lambda o: o[-1])
        if faster[-1] >= first[-1]:
            continue
        counts['catalog_later_vehicle_arrives_earlier'] += 1
        counts['catalog_later_vehicle_arrives_earlier_no_missing_endpoint'] += not incomplete
        counts['catalog_later_vehicle_arrives_earlier_named_od'] += row['primary']
        counts['catalog_faster_vehicle_outside_selected_pair'] += faster[2] not in case['options']
        decisions = None if faster[0] == first[0] else score_witness(
            graph, actual, metadata, base, params, row, options, first, faster,
        )
        witnesses.append(dict(incoming=incoming.id, stop=row['stop'], destination=row['destination'],
                              first=first[2], first_line=metadata[first[2]]['route_name'],
                              faster=faster[2], faster_line=metadata[faster[2]]['route_name'],
                              first_clock=first[0], faster_clock=faster[0], hindsight_gain=first[-1]-faster[-1],
                              catalog_endpoints_complete=not incomplete, narrowed_pair_decisions=decisions))
    return dict(counts=dict(counts), witnesses=witnesses)


def main():
    results = read_gz(HERE/'transfer_results.json.gz')
    data, _ = load_city(HERE/'city_transfer_endpoints.json.gz')
    training, _ = load_city(TRAINING)
    fit = fit_training(training, retain_no_pickup=True)
    output = {day: audit_day(data, day, by_step['10']['rows'], fit)
              for day, by_step in results['by_day'].items()}
    write_json('posthoc_menu_audit.json', dict(
        scope='Outcome-aware sensitivity. Faster vehicles are picked retrospectively and then scored '
              'as a narrowed two-route causal decision. Results do not modify the frozen cohort.',
        days=output,
    ))


if __name__ == '__main__':
    main()
