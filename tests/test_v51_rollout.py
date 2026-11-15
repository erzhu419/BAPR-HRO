import numpy as np
from experiments.reconstruction_v51.rollout import StableTripWorlds, evaluate
from experiments.reconstruction_v50.execute import ObservationContract
from test_v46_shared import shared_case
from test_v48_frontier import forced_case


def test_trip_tapes_do_not_depend_on_menu_order_or_extra_trips(monkeypatch):
    _,model,params,history,platform,_=shared_case(monkeypatch)
    paths=[path for ride in platform.rides.values() for path in ride.paths()]
    first=StableTripWorlds(history.as_belief(),params,paths,8,99,3,model._trips)
    reordered=StableTripWorlds(history.as_belief(),params,list(reversed(paths)),8,99,3,model._trips)
    subset=StableTripWorlds(history.as_belief(),params,[(2,3)],4,99,3,model._trips)
    for key,values in first.events.items():
        for a,b in zip(values,reordered.events[key],strict=True):assert np.array_equal(a,b)
    for key,values in subset.events.items():
        for a,b in zip(values,first.events[key],strict=True):assert np.array_equal(a,b[:4])


def test_time_budget_retains_every_unfinished_world(monkeypatch):
    graph,model,params,history,node,row=forced_case(monkeypatch)
    result=evaluate(graph,model,params,100,node,row,history,ObservationContract.from_menu(node),
                    root_index=0,count=4,max_process_cpu=0.)
    assert result['completed_pairs']==0
    assert result['paired']['unresolved_pairs']==4
    assert result['paired']['full_bank_mean_gain_interval']==[-(100-row['clock']),100-row['clock']]
    assert all(r['loss'] is None for arm in result['rows'].values() for r in arm)


def test_common_observation_full_rollout_preserves_known_receipts(monkeypatch):
    graph,model,params,history,node,row=forced_case(monkeypatch)
    result=evaluate(graph,model,params,100,node,row,history,ObservationContract.from_menu(node),
                    root_index=0,count=4)
    assert result['completed_pairs']==4
    assert result['future_messages']==0
    assert result['paired']['unresolved_pairs']==0
    assert all(0<=r['loss']<=100-row['clock'] for arm in result['rows'].values() for r in arm)
