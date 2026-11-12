import numpy as np
import pytest

from experiments.reconstruction_v36.validate import age_bin, count_bin, joint_event_scores
from src.evolving_trip_belief import _ChainLaw


def test_joint_event_conditions_on_both_operating_and_keeps_dependence():
    clocks = [np.array([0.,10.,0.,10.]),np.array([1.,11.,1.,11.])]
    initial = np.log([.4,.4,.1,.1])
    transition = np.full((4,4),-np.inf)
    np.fill_diagonal(transition,0.)
    law = _ChainLaw(clocks,initial,lambda i:transition,[np.zeros(4),np.zeros(4)],
        [np.array([False,False,True,True])]*2,(True,))
    score = joint_event_scores(law,0,scheduled_departure=-295,scheduled_arrival=-294,
                              actual_departure=0,actual_arrival=11)
    assert score['operating_pair_probability'] == pytest.approx(.8)
    assert score['events']['J1']['probability'] == pytest.approx(.5)
    assert score['events']['J1']['product_probability'] == pytest.approx(.25)
    assert score['events']['J1']['Brier'] == pytest.approx(.25)


def test_joint_score_retains_future_emissions_and_runtime_dependence():
    times = [np.array([0.,100.]),np.array([10.,110.]),np.array([20.,120.])]
    diagonal = np.array([[0.,-np.inf],[-np.inf,0.]])
    law = _ChainLaw(times,np.log([.5,.5]),lambda i:diagonal,
        [np.zeros(2),np.zeros(2),np.log([.8,.2])],[np.zeros(2,bool)]*3,(True,True))
    score = joint_event_scores(law,0,scheduled_departure=-250,scheduled_arrival=-240,
                              actual_departure=0,actual_arrival=10)
    assert score['events']['J1']['probability'] == pytest.approx(.8)
    assert score['events']['J1']['product_probability'] == pytest.approx(.64)
    assert score['events']['J2']['probability'] == pytest.approx(1.)
    assert score['events']['J2']['product_probability'] == pytest.approx(.84)
    assert score['product_impossible_order_probability'] == pytest.approx(.16)


def test_truth_changes_only_scores_not_the_joint_prediction():
    law = _ChainLaw([np.array([0.]),np.array([10.])],np.array([0.]),
        lambda i:np.array([[0.]]),[np.zeros(1)]*2,[np.zeros(1,bool)]*2,(True,))
    kwargs = dict(scheduled_departure=0,scheduled_arrival=10)
    first = joint_event_scores(law,0,actual_departure=0,actual_arrival=10,**kwargs)
    second = joint_event_scores(law,0,actual_departure=0,actual_arrival=400,**kwargs)
    for event in ('J1','J2'):
        assert first['events'][event]['probability'] == second['events'][event]['probability']
        assert first['events'][event]['Brier'] != second['events'][event]['Brier']


def test_predeclared_strata_boundaries_and_missing_target_message():
    assert [age_bin(x) for x in (None,600,601,1800,1801)] == [
        'no_target_forecast','<=600','(600,1800]','(600,1800]','>1800']
    assert [count_bin(x) for x in (0,1,2,4,5)] == ['0','1','2-4','2-4','5+']
