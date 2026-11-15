"""The frozen calibration gate must retain clustering and missingness."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from experiments.decision_calibration_v2.calibration_summary import (
    PARTITIONS, event_bounds, incidents, summarize,
)


def panel(rows_per_day=1):
    dates = [str(date(2023,11,5)+timedelta(days=i)) for i in range(14)]
    return dict(selected_dates=dates, dates_frozen=True, days=[dict(day=day, available=True,
        denominator_known=True, expected_opportunities=rows_per_day) for day in dates])


def event(day, incoming='A', outgoing='B', actual='10', probabilities=None):
    return dict(day=day, incoming_trip=incoming, outgoing_trip=outgoing,
        model=dict(status='resolved',partitions=probabilities or {key:float(key == actual) for key in PARTITIONS}),
        actual=dict(status='resolved',actual_partition=actual,possible_partitions=[actual]),
        groups=dict(forecast_age='none',mct_difference='over_180s',downstream_options=2))


def test_empty_or_missing_dates_cannot_pass_even_when_represented_rows_fit():
    empty = summarize([],dict(selected_dates=[],dates_frozen=False,days=[]))
    assert empty['verdict']=='unresolved' and empty['declared_dates']==0
    manifest=panel()
    rows=[event(day) for day in manifest['selected_dates'][:-1]]
    result=summarize(rows,manifest)
    assert result['verdict']=='unresolved'
    assert result['incomplete_dates']==[manifest['selected_dates'][-1]]
    assert result['days'][-1]['bias_bounds']['10']==[-1.,1.]
    assert result['opportunity_weighted']['represented_event_rows']==13


def test_incidents_join_transitively_across_both_vehicle_roles():
    rows=[event('2023-11-05','A','B'),event('2023-11-05','C','B'),
          event('2023-11-05','C','D'),event('2023-11-05','X','Y')]
    assert incidents(rows)==[[0,1,2],[3]]
    manifest=panel(4)
    events=[]
    for day in manifest['selected_dates']:
        for row in rows:
            value=deepcopy(row); value['day']=day
            if value['incoming_trip']!='X':
                value['model']['partitions']={'00':.4,'01':0.,'10':.6,'11':0.}
            events.append(value)
    result=summarize(events,manifest)
    assert result['primary']['bias_bounds']['10']==pytest.approx([.2,.2])
    assert result['opportunity_weighted']['bias_bounds']['10']==pytest.approx([.3,.3])
    assert all(len(row['incidents'])==2 for row in result['days'])


def test_known_systematic_bias_fails_all_required_dates_and_is_reproducible():
    manifest=panel()
    rows=[event(day,probabilities={'00':.1,'01':0.,'10':.9,'11':0.}) for day in manifest['selected_dates']]
    result=summarize(rows,manifest)
    assert result['verdict']=='fail'
    assert result['primary']['partitions']['10']['interval']==pytest.approx([.1,.1])
    assert result['primary']['partitions']['00']['interval']==pytest.approx([-.1,-.1])
    assert result==summarize(rows,manifest)


def test_equivalence_requires_intervals_inside_margin_not_only_zero_inclusion():
    manifest=panel(100)
    rows=[]
    for day in manifest['selected_dates']:
        for index in range(100):
            rows.append(event(day, actual='10' if index<99 else '00',
                probabilities={'00':.01,'01':0.,'10':.99,'11':0.}))
    result=summarize(rows,manifest)
    assert result['verdict']=='pass'
    assert all(abs(value)<1e-12 for value in result['primary']['bias_bounds']['10'])
    wide=[event(day,actual='10' if i%2 else '00',probabilities={'00':.5,'01':0.,'10':.5,'11':0.})
          for i,day in enumerate(manifest['selected_dates'])]
    another=summarize(wide,panel())
    assert another['verdict']=='unresolved'
    lo,hi=another['primary']['partitions']['10']['interval']
    assert lo<-.02<0.<.02<hi


def test_unknown_probabilities_and_partial_labels_retain_joint_brier_bounds():
    row=event('2023-11-05',probabilities={'00':.1,'01':.2,'10':.3,'11':.4})
    row['actual']=dict(status='label_unresolved',possible_partitions=['10','11'])
    bounds=event_bounds(row)
    assert bounds['residual'][2].tolist()==pytest.approx([-.3,.7])
    assert bounds['brier'].tolist()==pytest.approx([.5,.7])
    row['model']=dict(status='probability_unresolved',reason='missing_history')
    bounds=event_bounds(row)
    assert bounds['residual'][2].tolist()==[-1.,1.]
    assert bounds['residual'][0].tolist()==[-1.,0.]
    assert bounds['brier'].tolist()==[0.,2.]
    manifest=panel()
    rows=[dict(deepcopy(row),day=day) for day in manifest['selected_dates']]
    assert summarize(rows,manifest)['verdict']=='unresolved'


def test_unknown_denominator_or_zero_opportunity_day_overrides_a_numerical_fail():
    manifest=panel()
    rows=[event(day,probabilities={'00':1.,'01':0.,'10':0.,'11':0.}) for day in manifest['selected_dates']]
    manifest['days'][0]['denominator_known']=False
    result=summarize(rows,manifest)
    assert result['verdict']=='unresolved' and result['incomplete_dates']
    manifest=panel()
    manifest['days'][0]['expected_opportunities']=0
    result=summarize(rows[1:],manifest)
    assert result['verdict']=='unresolved'
    assert 'no_structural_opportunity' in result['days'][0]['unresolved_reasons']


def test_missing_whole_rt_archive_keeps_complete_actual_labels_but_cannot_pass():
    manifest=panel()
    # The real Mar09 source shape: no sampled snapshots, but the registered
    # denominator and actual labels can be complete. Values here are synthetic.
    day=manifest['selected_dates'][0]
    manifest['days'][0].update(available=False, messages_available=False, actual_available=True)
    rows=[event(date, actual='11', probabilities={'00':1.,'01':0.,'10':0.,'11':0.})
          for date in manifest['selected_dates']]
    rows[0]['model']=dict(status='probability_unresolved', reason='missing_predecision_message_captures')
    result=summarize(rows,manifest)
    missing=result['days'][0]
    assert missing['status']=='unresolved' and result['incomplete_dates']==[day]
    assert missing['observed_frequency_bounds']['11']==[1.,1.]
    assert missing['observed_frequency_bounds']['00']==[0.,0.]
    assert missing['predicted_probability_bounds']['11']==[0.,1.]
    assert missing['bias_bounds']['11']==[0.,1.]
    assert missing['bias_bounds']['00']==[-1.,0.]
    assert missing['four_class_brier_bounds']==[0.,2.]
    assert result['primary']['partitions']['11']['interval_classification']=='fail'
    assert result['verdict']=='unresolved'
    assert result['opportunity_weighted']['complete_registered_denominator']
    assert not result['opportunity_weighted']['primary_admission_satisfied']

    # Missing an event or its population denominator cannot use these tighter
    # represented-row bounds as a bound for the complete date.
    manifest['days'][0]['expected_opportunities']=2
    assert summarize(rows,manifest)['days'][0]['bias_bounds']['11']==[-1.,1.]
    manifest['days'][0].update(expected_opportunities=1, denominator_known=False)
    assert summarize(rows,manifest)['days'][0]['observed_frequency_bounds']['11']==[0.,1.]


def test_rows_outside_frozen_dates_are_not_silently_added():
    with pytest.raises(ValueError,match='outside'):
        summarize([event('2023-12-01')],panel())
