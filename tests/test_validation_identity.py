"""Frozen validation inputs, common repaired messages, and unchanged aggregation."""
from types import SimpleNamespace

import pytest

from experiments.decision_gap_v1.panel import read, write
from experiments.validation_identity_v1 import validate as wrapper


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    original = tmp_path/'experiments/reconstruction_v35'
    directory = tmp_path/'experiments/validation_identity_v1'
    dates = [f'2023-10-{day:02}' for day in range(3, 12)]
    training_dates = ['2023-10-01','2023-10-02']
    rules = dict(models=list(wrapper.original.MODELS), aggregation='equal-date common support',
                 model_retention_gate='unchanged original five criteria')
    candidate = dict(training_dates=training_dates, route_kind={},
        pooled={kind:dict(bias_seconds=12.,sd_seconds=30.) for kind in ('arrival','departure')})
    train = [dict(date=training_dates[0],split='training',error_eligible=True,forecast_age=25.)]
    events = [dict(date=day,split='validation',source_connection_id=0,kind='arrival',
        trip_id='T',route_id='R',stop_id='B',stop_sequence=2,scheduled=300,
        prediction_clock=0,actual_status='REAL' if i else 'MISSING',
        actual_seconds=310 if i else None,official_journey='train' if i else None)
        for i,day in enumerate(dates)]
    protocol = dict(training_dates=training_dates,validation_dates=dates,validation=rules,
                    candidate_source='experiments/training_identity_v1/candidate_emission.json')
    write(original/'protocol.json',protocol)
    write(original/'events.json.gz',dict(rows=events))
    write(original/'validation_summary.json',dict(original_receipt='preserved'))
    write(tmp_path/protocol['candidate_source'],candidate)
    write(tmp_path/'experiments/training_identity_v1/corrected_training_events.json.gz',dict(rows=train))
    write(directory/'protocol.json',protocol)
    write(directory/'fitted_emission.json',candidate)
    write(directory/'events.json.gz',dict(rows=train+events))
    for day in dates:
        write(directory/'message_days'/f'{day}.json.gz',dict(frames=[dict(source='repaired',receipt_seconds=0)]))
    monkeypatch.setattr(wrapper,'ROOT',tmp_path)
    monkeypatch.setattr(wrapper,'ORIGINAL',original)
    monkeypatch.setattr(wrapper.original,'HERE',original)
    return directory, dates, candidate, train, events


def test_input_freeze_keeps_unresolved_queries_and_exact_repaired_candidate(prepared):
    directory, dates, *_ = prepared
    assert wrapper.check_inputs(directory)['validation_dates'] == dates
    events = read(directory/'events.json.gz')['rows']
    assert len([r for r in events if r['split']=='validation']) == 9
    assert any(r.get('actual_status')=='MISSING' for r in events)


@pytest.mark.parametrize('change', ['candidate','drop_unresolved','label','cutoff','training','gate'])
def test_input_freeze_rejects_scientific_input_changes(prepared, change):
    directory, *_ = prepared
    if change == 'candidate':
        path=directory/'fitted_emission.json'; value=read(path)
        value['pooled']['arrival']['bias_seconds'] += 1
    elif change == 'gate':
        path=directory/'protocol.json'; value=read(path)
        value['validation']['model_retention_gate']='relaxed criterion'
    else:
        path=directory/'events.json.gz'; value=read(path)
        if change == 'drop_unresolved':
            value['rows']=[r for r in value['rows'] if r.get('actual_status')!='MISSING']
        elif change == 'training':
            value['rows'][0]['forecast_age'] += 1
        else:
            value['rows'][2]['actual_seconds' if change=='label' else 'prediction_clock'] += 1
    write(path,value)
    with pytest.raises(ValueError):
        wrapper.check_inputs(directory)


def test_score_day_uses_repaired_frames_for_both_message_arms_and_preserves_unmapped_query(prepared, monkeypatch):
    directory, dates, candidate, *_ = prepared
    day=dates[1]
    endpoint=dict(date=day,split='validation',source_connection_id=0,kind='arrival',
        trip_id='T',route_id='R',stop_id='B',stop_sequence=2,scheduled=300,
        prediction_clock=0,actual_status='REAL',actual_seconds=310,official_journey='train')
    write(directory/'events.json.gz',dict(rows=[endpoint,dict(endpoint,source_connection_id=1)]))
    actuals=directory.parent/'reconstruction_data_audit'
    write(actuals/'rail_calibration_dates_v2.json.gz',dict(dates={day:dict(connections=[])}))
    write(directory.parent/'reconstruction_v5/calibration_frozen_fit.json',dict(
        evolving_fit=dict(increments={},route_priors=dict(pooled=dict(mean=0,std=1,cancel_rate=0)))))
    graph=SimpleNamespace(connections=[SimpleNamespace(route='R')])
    monkeypatch.setattr(wrapper.original,'AUDIT',actuals)
    monkeypatch.setattr(wrapper.original,'context',lambda data,date:(graph,None,None,{0:0},{},None))
    monkeypatch.setattr(wrapper.original,'EvolvingTripBelief',lambda *args,**kwargs:object())
    instances=[]

    class History:
        def __init__(self, model, parameters, *, frames=(), fit=None, source_to_local=None):
            self.frames,self.fit=frames,fit
            self.frame_index=0; self._forecasts={}
            instances.append(self)

        def advance(self, clock):
            self.frame_index=sum(frame['receipt_seconds']<=clock for frame in self.frames)

        def forecast_records(self, trip):
            return ()

    monkeypatch.setattr(wrapper.original,'MessageHistory',History)
    monkeypatch.setattr(wrapper.original,'FittedMessageHistory',History)
    monkeypatch.setattr(wrapper.original,'predict',lambda history,query:
        dict(status='unsupported',reason='unmapped fixed slot') if query['cid'] is None
        else dict(status='ok',CRPS_seconds=1.))
    (directory/'validation_days').mkdir()
    old_here=wrapper.original.HERE
    wrapper.score_day(day,directory)
    assert wrapper.original.HERE==old_here
    for own,legacy,fitted in (instances[:3],instances[3:]):
        assert own.frames==()
        assert legacy.frames is fitted.frames
        assert legacy.frames==[dict(source='repaired',receipt_seconds=0)]
        assert fitted.fit==candidate and legacy.fit is None
    saved=read(directory/'validation_days'/f'{day}.json.gz')
    assert saved['summary']['endpoint']['eligible_queries']==2
    assert saved['summary']['endpoint']['common_supported']==1
    assert all(row['unsupported']==1 for row in saved['summary']['endpoint']['models'].values())


def test_original_module_directory_is_restored_after_scoring_failure(prepared, monkeypatch):
    directory, dates, *_ = prepared
    old_here=wrapper.original.HERE

    def fail(day):
        assert wrapper.original.HERE==directory
        raise FloatingPointError('synthetic nonfinite posterior')

    monkeypatch.setattr(wrapper.original,'run_day',fail)
    with pytest.raises(FloatingPointError,match='synthetic'):
        wrapper.score_day(dates[0],directory)
    assert wrapper.original.HERE==old_here


def test_original_gate_uses_equal_dates_and_macro_coverage_at_new_output(prepared):
    directory, dates, *_ = prepared
    for day_index,day in enumerate(dates):
        count=100 if day_index==0 else 10
        rows=[]
        for index in range(count):
            predictions={}
            for model in wrapper.original.MODELS:
                crps=(8. if day_index==0 else 12.) if model=='fitted_messages' else 10. if model=='legacy_messages' else 12.
                fraction=(.1 if day_index==0 else 1.) if model=='fitted_messages' else .8 if model=='legacy_messages' else .9
                predictions[model]=dict(status='ok',CRPS_seconds=crps,MAE_seconds=crps,
                    central_90_coverage=float(index<count*fraction),central_90_width_seconds=10.,
                    conditional_operating_Brier=.1,cancel_probability=0.)
            rows.append(dict(target_has_forecast=False,predictions=predictions))
        if day_index==0:
            rows.append(dict(target_has_forecast=False,predictions={
                model:dict(status='unsupported',reason='fixed support unavailable') for model in wrapper.original.MODELS}))
        summary=wrapper.original.summarize_rows(rows)
        write(directory/'validation_days'/f'{day}.json.gz',dict(day=day,
            panels=dict(endpoint=rows,onboard=rows),summary=dict(endpoint=summary,onboard=summary),
            source_eligibility={},wall=0.))
    old_here=wrapper.original.HERE
    wrapper.aggregate(directory)
    assert wrapper.original.HERE==old_here
    assert read(old_here/'validation_summary.json')==dict(original_receipt='preserved')
    result=read(directory/'validation_summary.json')
    onboard=result['panels']['onboard']
    assert onboard['eligible_queries']==181 and onboard['common_supported']==180
    assert onboard['models']['fitted_messages']['common_metrics']['CRPS_seconds']<10.
    macro=onboard['equal_date_macro']['fitted_messages']
    assert macro['common_metrics']['CRPS_seconds']==pytest.approx(104/9)
    assert macro['common_metrics']['central_90_coverage']==pytest.approx(.9)
    assert macro['common_metrics_contributing_dates']==9
    # A pooled comparison would reverse CRPS ranking, while averaging each
    # day's absolute coverage error would reverse the coverage decision.
    assert [name for name,passed in result['gate']['criteria'].items() if not passed]==['onboard_CRPS_below_legacy']
    assert not result['gate']['passed']
