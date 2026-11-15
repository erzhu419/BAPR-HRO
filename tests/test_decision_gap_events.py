"""Catch-event registration, joint-law integration and retained unknown labels."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.decision_gap_v1.transfer_events import (
    archived_policy_usage, event_probabilities, label_event, partition_saved_bank,
    public_event_features, register_opportunities,
)
from experiments.reconstruction_v33.nodes import Platform, Ride
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def chain(times, *, canceled=None, initial=None, transitions=None):
    times = [np.array(t, dtype=float) for t in times]
    count = len(times[0])
    if initial is None:
        initial = np.ones(count)/count
    if transitions is None:
        transitions = [np.eye(count) for _ in times[:-1]]
    with np.errstate(divide='ignore'):
        logs = [np.log(t) for t in transitions]
        log_initial = np.log(initial)
    return _ChainLaw(times, log_initial, lambda slot: logs[slot],
        [np.zeros(len(t)) for t in times],
        [np.zeros(len(t), dtype=bool) for t in times] if canceled is None else canceled,
        (True,)*(len(times)-1))


def case():
    graph = TransitGraph()
    for i, mct in enumerate((0, 20, 0, 5, 0)):
        graph.add_stop(Stop(i, str(i), min_transfer_time=mct))
    specs = [('B', 0, 1, 0, 10), ('B', 1, 2, 11, 30), ('B', 2, 3, 31, 50),
             ('J', 1, 2, 20, 35), ('J', 2, 3, 36, 45), ('J', 3, 4, 60, 80)]
    for cid, (trip, start, end, departure, arrival) in enumerate(specs):
        graph.add_connection(Connection(cid, 'R', trip, start, end, departure, arrival))
    candidate = Ride(3, (3, 4, 5), {5: None})
    hub = Platform(3, {5: Ride(5, (5,), {5: None})})
    source = Ride(0, (0, 1, 2), {0: Platform(1, {3: candidate}), 2: hub})
    incoming = chain([[0], [10], [11], [30], [31], [50]])
    outgoing = chain([[20, 40], [25, 41], [30, 42], [40, 45], [60, 50], [80, 80]])
    laws = {'B': incoming, 'J': outgoing}
    model = SimpleNamespace(_positions={(cid, kind): (trip, 2*offset+j)
        for trip, cids in (('B', (0, 1, 2)), ('J', (3, 4, 5)))
        for offset, cid in enumerate(cids) for j, kind in enumerate(('departure', 'arrival'))})
    history = SimpleNamespace(law=lambda trip: laws[trip])
    event = register_opportunities(graph, source, 0, 10.)['rows'][0]
    metadata = {cid: {kind: dict(status='REAL', time_seconds=when, official_journey=trip)
        for kind, when in (('departure', dep), ('arrival', arr))}
        for cid, (trip, _, _, dep, arr) in enumerate(specs)}
    return graph, source, model, history, event, metadata, laws


def test_registration_is_public_complete_and_keeps_no_opportunity_roots():
    graph, source, _, _, event, metadata, _ = case()
    first = register_opportunities(graph, source, 0, 10.)
    metadata[5]['departure']['status'] = 'CANCELED'
    assert register_opportunities(graph, source, 0, 10.) == first
    assert first['candidates'] == len(first['candidate_pairs']) == len(first['rows']) == 1
    assert event['hub_in_action_menu']
    source.exits[2].rides.clear()
    assert not register_opportunities(graph, source, 0, 10.)['rows'][0]['hub_in_action_menu']
    source.exits[0].rides[3].path = (3, 4)
    empty = register_opportunities(graph, source, 0, 10.)
    assert empty['no_opportunity'] and empty['rows'] == []
    assert empty['pair_counts'] == {'no_hub_departure': 1}
    source.exits[1] = None
    assert register_opportunities(graph, source, 0, 10.)['candidate_pairs'] == []


def test_exact_joint_partition_matches_independent_world_enumeration():
    _, _, model, history, event, _, laws = case()
    result = event_probabilities(model, history, event, 100.)
    expected = dict.fromkeys(('00', '01', '10', '11'), 0.)
    # Both outgoing endpoints share one regime; the source is deterministic.
    for outgoing_regime in (0, 1):
        law = laws['J']
        up = law.times[0][outgoing_regime] >= 30.
        hub = law.times[4][outgoing_regime] >= 55.
        expected[f'{int(up)}{int(hub)}'] += .5
    assert result['partitions'] == pytest.approx(expected)
    assert result['partitions']['11'] == 0.
    assert result['upstream_catch']*result['hub_catch'] == .25


def test_source_arrival_randomness_cancellation_mct_and_hidden_slots():
    _, _, model, history, event, _, laws = case()
    laws['B'] = chain([[0, 0], [10, 10], [11, 11], [30, 30], [31, 31], [40, 70]])
    result = event_probabilities(model, history, event, 100.)
    assert result['partitions'] == pytest.approx({'00': .25, '01': .25, '10': .25, '11': .25})
    laws['B'].canceled[2][1] = True
    canceled = event_probabilities(model, history, event, 100.)
    assert canceled['source_reach_probability'] == .5
    assert canceled['source_first_failure_probabilities'] == {'canceled': .5, 'deadline': 0., 'time_order': 0.}
    # cid 1 is a latent chain segment when not executed, not a boarded failure.
    event['source_path'] = [2]
    laws['B'].canceled[2][:] = True
    assert event_probabilities(model, history, event, 100.)['source_reach_probability'] == 1.
    event['downstream_mct'] = 100.
    assert event_probabilities(model, history, event, 100.)['hub_catch'] == 0.
    event['same_incoming_outgoing_trip'] = True
    assert event_probabilities(model, history, event, 100.)['status'] == 'probability_unresolved'


def test_actual_missing_canceled_mct_and_inconsistent_identity_are_distinct():
    _, _, _, _, event, metadata, _ = case()
    assert label_event(metadata, event, 100.)['actual_partition'] == '01'
    metadata[5]['departure'] = dict(status='MISSING')
    unknown = label_event(metadata, event, 100.)
    assert unknown['possible_partitions'] == ['00', '01']
    assert unknown['partition_bounds']['01'] == [0, 1]
    metadata[1]['departure']['status'] = 'CANCELED'
    assert label_event(metadata, event, 100.)['actual_partition'] == '00'
    metadata[1]['departure']['status'] = 'REAL'
    metadata[5]['departure'] = dict(status='REAL', time_seconds=60, official_journey='other')
    assert label_event(metadata, event, 100.)['possible_partitions'] == ['00', '01', '10', '11']


def test_saved_bank_partition_preserves_all_pairs_and_unknown_loss_bounds():
    _, _, _, _, event, _, _ = case()
    events = {(0, 'arrival'): (np.array([10., 10.]), np.zeros(2, dtype=bool))}
    for cid, dep, arr in ((1, 11, 30), (2, 31, 50)):
        for kind, value in (('departure', dep), ('arrival', arr)):
            events[cid, kind] = np.full(2, value), np.zeros(2, dtype=bool)
    events[3, 'departure'] = np.array([20., 40.]), np.zeros(2, dtype=bool)
    events[5, 'departure'] = np.array([60., 50.]), np.zeros(2, dtype=bool)
    gate = dict(rows={'alight': [dict(world_index=0, loss=40.), dict(world_index=1, loss=None)],
                     'continue': [dict(world_index=0, loss=30.), dict(world_index=1, loss=50.)]})
    result = partition_saved_bank(events, event, gate, 100.)
    assert result['count'] == 2 and result['unresolved_pairs'] == 1
    assert result['partitions']['01']['contribution_bounds'] == [5., 5.]
    assert result['partitions']['10']['contribution_bounds'] == [-25., 20.]
    assert result['overall_difference_bounds'] == [-20., 25.]
    assert np.sum([p['contribution_bounds'] for p in result['partitions'].values()], axis=0).tolist() == result['overall_difference_bounds']


def test_public_groups_use_selected_root_forecast_and_usage_uses_recorded_actions():
    graph, _, _, _, event, _, _ = case()
    history = [dict(trip='J', evidence='forecast', emission_selected=False, cid=3, receipt_seconds=9),
               dict(trip='J', evidence='forecast', emission_selected=True, cid=5, receipt_seconds=-400)]
    features = public_event_features(graph, event, history)
    assert features['selected_forecast_age_seconds'] == 410
    assert features['groups']['forecast_age'] == 'over_300s'
    assert features['groups']['mct_difference'] == 'nonpositive'
    receipt = dict(loss_cap=90, actions=[dict(kind='onboard', cid=0, clock=10, choice='continue')],
        outcome=dict(loss=None, status='unresolved', trace=[]))
    usage = archived_policy_usage(receipt, graph, event)
    assert usage['boarded_named_at_hub'] is None and usage['loss_bounds'] == [0., 90]
    receipt['actions'].append(dict(kind='platform', stop=1, clock=40, choice=3))
    assert archived_policy_usage(receipt, graph, event)['boarded_named_upstream'] is True


def test_group_summary_retains_unknown_scores_and_deduplicates_shared_events(tmp_path):
    from experiments.decision_gap_v1.event_run import _descriptive_groups
    from experiments.decision_gap_v1.panel import write
    graph, _, _, _, event, metadata, _ = case()
    metadata[5]['departure'] = dict(status='MISSING')
    label = label_event(metadata, event, 100.)
    # Upstream is false: up-only is known false even while hub is unknown.
    model = dict(status='resolved', partitions={'00': .1, '01': .1, '10': .4, '11': .4})
    registered = dict(roots=[dict(root_index=i, day='D', rows=[event]) for i in (0, 1)])
    evaluated = [dict(root_index=i, rows=[dict(event_index=0, model=model, actual=label)]) for i in (0, 1)]
    features = public_event_features(graph, event, [])
    write(tmp_path/'events/group_registration.json.gz', dict(roots=[dict(root_index=i,
        rows=[dict(event_index=0, **features)]) for i in (0, 1)]))
    result = {}
    _descriptive_groups(tmp_path, registered, evaluated, result)
    grouped = result['descriptive_groups']
    assert grouped['all_registered_opportunities']['overall']['events'] == 2
    unique = grouped['distinct_operational_events']['overall']
    assert unique['events'] == 1 and unique['known_labels'] == 1
    assert unique['brier_bounds'] == pytest.approx([.16, .16])
