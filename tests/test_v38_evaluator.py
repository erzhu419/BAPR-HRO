from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.gated_rollout import conditional_action_check
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v38 import evaluator
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def fixture():
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop,str(stop),min_transfer_time=0))
    rows = [(0,'source',0,1,0,10),(1,'source',1,2,20,30),
            (2,'middle',1,2,12,25),(3,'middle',2,3,35,50),(4,'last',2,3,32,40)]
    for cid,trip,origin,target,departure,arrival in rows:
        graph.add_connection(Connection(cid,'R',trip,origin,target,departure,arrival))
    positions, trips, laws = {}, {}, {}
    for trip,cids,times in [('source',(0,1),[[0],[10],[20],[30]]),
                           ('middle',(2,3),[[12],[25],[35],[50]]),
                           ('last',(4,),[[32,38],[40,48]])]:
        events = [SimpleNamespace(cid=cid,kind=kind) for cid in cids for kind in ('departure','arrival')]
        trips[trip] = events
        positions.update({(event.cid,event.kind):(trip,slot) for slot,event in enumerate(events)})
        arrays = [np.asarray(t) for t in times]
        with np.errstate(divide='ignore'):
            transitions = [np.log(np.eye(len(a))) for a in arrays[:-1]]
            initial = np.log(np.ones(len(arrays[0]))/len(arrays[0]))
        laws[trip] = _ChainLaw(arrays,initial,lambda slot,matrices=transitions: matrices[slot],
            [np.zeros(len(a)) for a in arrays],[np.zeros(len(a),bool) for a in arrays],
            (True,)*(len(arrays)-1))
    model = SimpleNamespace(graph=graph,_positions=positions,_trips=trips,
        _law=lambda trip,parameters:laws[trip])
    parameters = {'R':{'std':1.}}
    fit = dict(pooled={kind:dict(bias_seconds=0.,sd_seconds=30.) for kind in ('departure','arrival')},route_kind={})
    history = SingleMessageHistory(model,parameters,fit=fit,frames=[
        dict(receipt_seconds=5,trip_statuses=[],endpoint_updates=[[4,'arrival','FORECAST',44.]]),
        dict(receipt_seconds=99,trip_statuses=[['last','CANCELED']],endpoint_updates=[])])
    history.advance(10)
    history.observe(0,'departure',0)
    history.observe(0,'arrival',10)
    final = Platform(2,{4:Ride(4,(4,),{4:None})})
    first = Platform(1,{2:Ride(2,(2,3),{2:final,3:None})})
    node = Ride(0,(0,1),{0:first,1:final})
    return graph,model,parameters,node,history


def scientific(result):
    keys = ('actions','paired','selected_action','seed','count','public_history','future_messages','continuation')
    return {**{key:result[key] for key in keys},
        'rows':{label:[{key:row[key] for key in ('loss','status','trace','decisions','world_index')}
                        for row in rows] for label,rows in result['rows'].items()}}


@pytest.mark.parametrize('variant',['legacy','shared','fast'])
def test_complete_replay_preserves_original_worlds_traces_and_gate(variant):
    graph,model,parameters,node,history = fixture()
    public = history.public_state()
    args = (graph,model,parameters,100,node,dict(kind='onboard',cid=0,clock=10),history,'continue','alight')
    original = conditional_action_check(*args,count=8,seed=37050)
    result = evaluator.evaluate_actions(*args,count=8,seed=37050,variant=variant)
    assert scientific(result)==scientific(original)
    assert history.public_state()==public and history.frame_index==1
    assert len(result['rows']['nominal'])==len(result['rows']['combined'])==8
    assert all(row['trace'] for rows in result['rows'].values() for row in rows)
    assert result['cost']['planner_instances']==(2 if variant=='legacy' else 1)
    for unit,total in (('process_cpu','total_process_cpu'),('wall','total_wall')):
        cost = result['cost']
        assert cost[total]>=cost['preparation'][unit]+cost['decision'][unit]+sum(
            action[unit] for action in cost['action_continuations'].values())


def test_sharing_reuses_terminal_compilation_and_stats_are_per_arm_deltas():
    graph,model,parameters,node,history = fixture()
    args = (graph,model,parameters,100,node,dict(kind='onboard',cid=0,clock=10),history,'continue','alight')
    separate = evaluator.evaluate_actions(*args,count=8,seed=37050,variant='legacy')
    shared = evaluator.evaluate_actions(*args,count=8,seed=37050,variant='shared')
    assert shared['cost']['planner_stats_total']['ride_compiles']<separate['cost']['planner_stats_total']['ride_compiles']
    assert scientific(shared)==scientific(separate)
    for key,total in shared['cost']['planner_stats_total'].items():
        assert sum(action['planner_stats'][key] for action in shared['cost']['action_continuations'].values())==pytest.approx(total)


def test_one_bank_keeps_unresolved_mass_and_independent_history_per_action(monkeypatch):
    graph,model,parameters,node,history = fixture()
    calls = []

    def bank(belief,parameters,paths,count,seed):
        calls.append(('bank',count,seed))
        return SimpleNamespace(belief=belief)

    class Continuation:
        def __init__(self,graph,planner,deadline,**kwargs):
            self.planner = planner

        def forced(self,node,row,action,snapshot,world):
            assert snapshot.frames==() and isinstance(snapshot,SingleMessageHistory)
            assert snapshot.public_state()==history.public_state()
            snapshot.observe(4,'departure',32)
            calls.append((action,world.index))
            self.planner.stats['ride_queries']+=1
            loss = None if action=='continue' and world.index==1 else (10. if action=='continue' else 20.)
            return dict(loss=loss,status='unsupported_public_history:test' if loss is None else 'arrived',
                trace=[dict(kind='onboard',cid=4,choice=action)],decisions=dict(onboard=1),
                total_process_cpu=0.,total_wall=0.,compute_wall=0.)

    monkeypatch.setattr(evaluator,'TripWorlds',bank)
    monkeypatch.setattr(evaluator,'SnapshotExecutor',Continuation)
    result = evaluator.evaluate_actions(graph,model,parameters,100,node,
        dict(kind='onboard',cid=0,clock=10),history,'continue','alight',count=3,seed=7,variant='shared')
    assert calls==[('bank',3,7),('continue',0),('continue',1),('continue',2),
                   ('alight',0),('alight',1),('alight',2)]
    assert result['paired']['unresolved_pairs']==1
    assert result['paired']['full_bank_mean_gain_interval']==pytest.approx([-40/3,50/3])
    assert result['selected_action']=='alight'
    assert result['cost']['planner_stats_total']['ride_queries']==6
