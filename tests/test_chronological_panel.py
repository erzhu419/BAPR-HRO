"""Provenance and receipt checks for the route-ID experiment runner."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v3.run_chronological_panel import (
    FITTERS, HOLDOUT_ODS, factorial_effects, freeze_protocol, load_network, record_online_decisions,
)
from src.pmf import PMF
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.transit_graph import Connection, Stop, StopLabel, TransitGraph


def test_resume_rejects_changed_source_or_used_input_values(tmp_path):
    output = tmp_path / 'panel.json'
    inputs = {'data': str(tmp_path / 'daily.pkl'), 'world_inputs': {'day': {'mean': 1.}}}
    frozen = freeze_protocol(output, inputs)
    original = frozen.read_text()
    assert freeze_protocol(output, inputs) == frozen
    with pytest.raises(ValueError, match='Inputs or protocol changed'):
        freeze_protocol(output, inputs | {'data': str(tmp_path / 'another.pkl')})
    with pytest.raises(ValueError, match='Inputs or protocol changed'):
        freeze_protocol(output, inputs | {'world_inputs': {'day': {'mean': 3.}}})
    assert frozen.read_text() == original


def test_old_receipts_cannot_be_relabeled_by_a_new_protocol(tmp_path):
    output = tmp_path / 'panel.json'
    directory = tmp_path / 'panel_cases'
    directory.mkdir()
    (directory / 'case_0000.json.gz').write_bytes(b'old receipt')
    with pytest.raises(ValueError, match='lack a frozen input protocol'):
        freeze_protocol(output, {'data': 'new daily data'})
    assert (directory / 'case_0000.json.gz').read_bytes() == b'old receipt'


def test_holdout_freezes_the_connected_stettbach_platform():
    graph, _ = load_network()
    _, source, _, target = HOLDOUT_ODS[1]
    same_name = [sid for sid, stop in graph.stops.items() if stop.name == 'Stettbach, Bahnhof']
    assert len(same_name) > 1
    assert source == 505872492
    legs = [graph.connections[cid] for cid in (91, 92, 93, 94)]
    assert legs[0].dep_stop == source
    assert legs[-1].arr_stop == target
    assert len({leg.trip_id for leg in legs}) == 1
    assert all(left.arr_stop == right.dep_stop for left, right in zip(legs, legs[1:]))


def test_actual_executor_receipt_keeps_missed_attempt_wait_and_poll_times():
    graph = TransitGraph()
    for sid in (0, 1):
        graph.add_stop(Stop(sid, str(sid)))
    graph.add_connection(Connection(0, 'A', 'early', 0, 1, 15, 18))
    graph.add_connection(Connection(1, 'B', 'later', 0, 1, 24, 28))
    labels = [StopLabel(cid, PMF.deterministic(c.arr_time), c.arr_time, 1.)
              for cid, c in enumerate(graph.connections)]
    plan = SimpleNamespace(stop_labels={0: labels, 1: []})
    choices = iter((0, None, 1))
    old_feedback = []

    def select(*_args, **_kwargs):
        cid = next(choices)
        return None if cid is None else (labels[cid], labels[cid].mean_dest_arrival)

    router = SimpleNamespace(route=lambda *_: plan, select_connection=select,
                             observe_delay=lambda *_: None, observe_cancel=lambda *_, **__: None,
                             observe_attempt_outcome=lambda *args: old_feedback.append(args))
    decisions, polls, attempts = record_online_decisions(router)
    scene = JourneyScenario({0: VehicleEvent(-4, False, 11, 14),
                             1: VehicleEvent(0, False, 24, 28)})
    result = simulate_bandit_journey(graph, router, 0, 1, 22, RegimeSchedule([(0, 'fixed')]),
                                    np.random.default_rng(0), max_time=30, scenario=scene)
    assert result.arrival_time == 28
    assert [(d['time'], d['action'], d['connection_id']) for d in decisions] == [
        (22, 'select', 0), (22, 'wait', None), (23, 'select', 1)]
    assert [poll['received_at'] for poll in polls] == [22, 22, 23]
    assert [(a['decision_index'], a['outcome']) for a in attempts if a['outcome'] != 'wait'] == [
        (0, 'missed'), (2, 'board')]
    assert old_feedback[0] == (0, 'missed', 22, 22)


def test_factorial_averages_each_toggle_over_the_other_factors():
    rows = []
    for method, options in FITTERS.items():
        m, c, p = options['online_mean'], options['online_cancel'], bool(options['beta'])
        rows.append({'method': method, 'loss': 10 + m - 3 * c + 2 * p + 4 * m * c})
    effects = factorial_effects([{'spec': ['2023-10-12', 1, False, 120], 'rows': rows}], FITTERS)
    assert effects['online_mean_on_minus_off']['mean_difference'] == 3
    assert effects['online_cancel_on_minus_off']['mean_difference'] == -1
    assert effects['beta_on_minus_off']['mean_difference'] == 2
