from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v34.forecast import MessageHistory
from experiments.reconstruction_v35.validate import distribution_scores, predict, query_history, summarize_rows
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def test_discrete_proper_score_and_conditional_operating_brier():
    score = distribution_scores([0, 10], [.2, .2], 5, cancel_probability=.6, threshold=0)
    assert score['CRPS_seconds'] == pytest.approx(2.5)
    assert score['central_90_width_seconds'] == 10
    assert score['conditional_operating_Brier'] == pytest.approx(.25)
    assert score['cancel_probability'] == .6


def chain_history():
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop)))
    graph.add_connection(Connection(0, 'R', 'trip', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'R', 'trip', 1, 2, 11, 15))
    times = [np.array([0.]), np.array([10., 20.]), np.array([11., 21.]), np.array([15., 25.])]
    transitions = [np.log(np.array([[.5,.5]])), np.array([[0.,-np.inf],[-np.inf,0.]]),
                   np.array([[0.,-np.inf],[-np.inf,0.]])]
    law = _ChainLaw(times, np.array([0.]), lambda i:transitions[i],
        [np.zeros(len(t)) for t in times], [np.zeros(len(t),bool) for t in times], (True,True,True))
    positions = {(0,'departure'):('trip',0),(0,'arrival'):('trip',1),
                 (1,'departure'):('trip',2),(1,'arrival'):('trip',3)}
    model = SimpleNamespace(graph=graph, _positions=positions, _law=lambda trip,p:law)
    frame = dict(receipt_seconds=21, trip_statuses=[],endpoint_updates=[[1,'arrival','FORECAST',15]])
    return MessageHistory(model, {'R':{'std':1}}, frames=[frame])


def test_onboard_prediction_conditions_joint_chain_without_target_or_future_receipt():
    master = chain_history()
    query = dict(clock=20,cid=1,kind='arrival',actual=15,threshold=20,
        own_observations=[dict(cid=0,kind='departure',clock=0),dict(cid=0,kind='arrival',clock=20)])
    history = query_history(master, query)
    first = predict(history, query)
    second = predict(history, dict(query, actual=25))
    assert first['clocks_seconds'] == second['clocks_seconds'] == [25.]
    assert first['conditional_operating_probabilities'] == [1.]
    assert first['CRPS_seconds'] == 10 and second['CRPS_seconds'] == 0
    assert history.frame_index == 0
    assert not master._records


def test_nonfinite_prediction_is_not_hidden_as_an_unsupported_query():
    with pytest.raises(FloatingPointError):
        distribution_scores([0,10],[np.nan,.5],5)


def test_common_summary_keeps_unsupported_in_the_denominator():
    good = distribution_scores([0],[1],0)
    rows = [dict(target_has_forecast=True,predictions={m:dict(good) for m in
        ('own_only','legacy_messages','fitted_messages')}) for _ in range(2)]
    rows[1]['predictions']['fitted_messages'] = dict(status='unsupported',reason='outside support')
    summary = summarize_rows(rows)
    assert summary['eligible_queries'] == 2 and summary['common_supported'] == 1
    assert summary['models']['fitted_messages']['unsupported'] == 1
    assert summary['fitted_new_unsupported_vs_legacy'] == 1
