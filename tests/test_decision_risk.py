import pytest

from src.committed_transit import CommittedState, build_committed_model
from src.decision_risk import realized_macro_risk, summarize_day, calibrate_daily_groups, assess_day
from src.decision_risk import binomial_failure_upper, service_guard, required_zero_failure_blocks
from src.information_planner import ExactPlanner
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


@pytest.mark.parametrize('departure,arrival,canceled', [(0, 60, False), (0, 200, False), (0, 60, True)])
def test_surrogate_bounds_worldwise_oracle_regret_under_transfer_contract(departure, arrival, canceled):
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'onboard', min_transfer_time=120))
    graph.add_stop(Stop(1, 'goal', min_transfer_time=120))
    graph.add_connection(Connection(0, 'R', 'A', 0, 1, 0, 60))
    graph.add_connection(Connection(1, 'R', 'B', 0, 1, 120, 150))
    scene = JourneyScenario({0: VehicleEvent(999 if canceled else 0, canceled, departure, arrival),
                             1: VehicleEvent(0, False, 120, 150)}, minute=60, publications={})
    state = CommittedState(0, 0, 'A', 0, initialized=True)
    model = build_committed_model(graph, [(1., scene)], 0, 1, 0, 300, initial_state=state)
    first = model.transition(model.scenarios[0], state, ('select', 0))
    macro = first.cost + model.terminal_cost(model.scenarios[0], first.state)
    oracle = ExactPlanner(model).solve().value
    risk = realized_macro_risk(0, 120, 300, departure, arrival, canceled=canceled)
    assert macro - oracle <= risk.upper + 1e-12


def test_missing_target_is_kept_as_risk_interval_and_known_miss_is_failure():
    value = realized_macro_risk(10, 130, 3610, None, None)
    assert (value.lower, value.upper, value.outcome) == (0., 3480., 'unresolved')
    failed = realized_macro_risk(10, 130, 3610, 9, None)
    assert failed.lower == failed.upper == 3480


def test_nine_date_calibration_retains_empty_group_and_censor_bound():
    def row(lower, upper):
        return dict(selected=True, group='ordinary', risk_lower=lower, risk_upper=upper,
                    model_regret_bound=.1, outcome='unresolved', transfer_source='assumed120',
                    zero_buffer_risk_lower=lower, zero_buffer_risk_upper=upper)
    days = {str(i): summarize_day([row(i, i + 3)]) for i in range(9)}
    calibration = calibrate_daily_groups(days)
    assert calibration['risk_upper_seconds'] == 11
    assert calibration['nonempty_dates_by_group'] == {'ordinary': 9, 'late': 0}
    assert assess_day(summarize_day([row(2, 4)]), 3)['ordinary'] == 'unidentified_from_missing_measurements'
    assert assess_day(summarize_day([row(4, 4)]), 3)['ordinary'] == 'verified_above_bound'
    assert assess_day(days['0'], 11)['late'] == 'no_selected_context'


def test_future_realization_changes_score_but_not_gate_selection():
    from copy import deepcopy
    from experiments.reconstruction_v6.decision_risk_run import evaluate_day

    day = '2023-10-03'
    def leg(cid, u, v, departure, arrival):
        return dict(id=cid, trip_id='A', route_id='R', dep_stop=str(u), arr_stop=str(v),
            pickup_allowed=True, dropoff_allowed=True, scheduled_departure_seconds=departure,
            scheduled_arrival_seconds=arrival, outcome='MEASURED',
            departure=dict(status='REAL', time_seconds=departure, official_journey='A'),
            arrival=dict(status='REAL', time_seconds=arrival, official_journey='A'))
    data = dict(stops={str(i): dict(name=str(i), latitude=0., longitude=0.) for i in range(3)},
        dates={day: dict(connections=[leg(10, 0, 1, 0, 60), leg(11, 1, 2, 60, 120)], coverage={})},
        station_transfer_bounds={'1': dict(conservative_max_stated_seconds=120, minimum_stated_seconds=120)})
    fit = dict(route_priors={'R': dict(mean=0., std=.5, cancel_rate=0.),
                             'pooled': dict(mean=0., std=.5, cancel_rate=0.)},
        increments=dict(travel_sd=.1, dwell_sd=.1, gap_sd=.1,
                        travel_drift=0., dwell_drift=0., gap_drift=0.))
    first = evaluate_day(data, day, fit)['rows'][0]
    changed = deepcopy(data)
    changed['dates'][day]['connections'][1]['departure']['status'] = 'CANCELED'
    changed['dates'][day]['connections'][1]['outcome'] = 'CANCELED'
    second = evaluate_day(changed, day, fit)['rows'][0]
    assert first['selected'] and second['selected']
    assert first['model_regret_bound'] == second['model_regret_bound']
    assert first['risk_upper'] == 0.
    assert second['outcome'] == 'failed_canceled' and second['risk_lower'] == 3480.
    changed = deepcopy(data)
    changed['dates'][day]['connections'][1]['arrival']['status'] = 'CANCELED'
    changed['dates'][day]['connections'][1]['outcome'] = 'UNRESOLVED'
    third = evaluate_day(changed, day, fit)['rows'][0]
    assert third['model_regret_bound'] == first['model_regret_bound']
    assert third['outcome'] == 'failed_canceled' and third['risk_lower'] == 3480.
    changed = deepcopy(data)
    changed['dates'][day]['connections'][1]['arrival']['official_journey'] = 'B'
    changed['dates'][day]['connections'][1]['outcome'] = 'UNRESOLVED'
    fourth = evaluate_day(changed, day, fit)['rows'][0]
    assert fourth['model_regret_bound'] == first['model_regret_bound']
    assert fourth['outcome'] == 'unresolved' and fourth['risk_upper'] == 3480.


def test_service_guard_declines_q_zero_claim_when_only_two_training_dates_exist():
    upper = binomial_failure_upper(0, 2)
    assert upper == pytest.approx(1. - .1 ** .5)
    assert binomial_failure_upper(0, 0) == 1.
    assert binomial_failure_upper(2, 2) == 1.
    assert binomial_failure_upper(1, 4) == pytest.approx(.6795394163)
    assert not service_guard(0., upper, 3180.)['authorized']
    n = required_zero_failure_blocks(3180.)
    assert 3180 * binomial_failure_upper(0, n) <= 1
    assert 3180 * binomial_failure_upper(0, n - 1) > 1
