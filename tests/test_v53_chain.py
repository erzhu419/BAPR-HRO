"""Whole-tail integration against private-state sums and V52 recursion."""
from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference
from experiments.reconstruction_v53.chain import terminal_ride_value
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from src.transit_graph import TransitGraph, Stop, Connection


def case(*, canceled=False, hidden=False, conditioned=False, deadline=60.):
    graph=TransitGraph()
    for i in range(4):
        graph.add_stop(Stop(i,str(i),min_transfer_time=0))
    for cid,(dep,arr) in enumerate(((10,20),(25,35),(40,50))):
        graph.add_connection(Connection(cid,'R','T',cid,cid+1,dep,arr))
    model=EvolvingTripBelief(graph,public_occurrences=False)
    parameters={'R':dict(mean=0.,std=.1,cancel_rate=0.)}
    fit=dict(route_kind={},pooled={k:dict(bias_seconds=0.,sd_seconds=5.)
                                  for k in ('departure','arrival')})
    history=PhysicalHistory(model,parameters,fit=fit)
    history.message_clock=0.
    if conditioned:
        history.observe(0,'arrival',21.)
    reference=CausalReference(graph,model,parameters,deadline,history,
        ObservationContract({}),max_states=None,max_process_cpu=None)
    times=[np.arange(t,t+3,dtype=float) for t in (10,20,25,35,40,50)]
    matrix=np.log(np.array([[.8,.15,.05],[.1,.7,.2],[.05,.15,.8]]))
    emissions=[np.zeros(3) for _ in times]
    emissions[3]=np.log([.2,.5,1.])
    emissions[5]=np.log([1.,.7,.3])
    cancel=[np.zeros(3,dtype=bool) for _ in times]
    if canceled:
        cancel[2][2]=True
        cancel[4][1]=True
    if hidden:
        cancel[2][:]=True
        cancel[3][:]=True
    law=_ChainLaw(times,np.log([.2,.3,.5]),lambda slot:matrix,emissions,cancel,(True,)*5)
    reference.root_law.roots['T']=law
    path=(0,2) if hidden else (0,1,2)
    ride=Ride(0,path,{2:None})
    return reference,history,ride


def enumerate_value(reference,history,ride,position,clock):
    law=reference.root_law.law('T',history)
    weighted=total=0.
    for indices in product(*(range(len(t)) for t in law.times)):
        logweight=law.initial[indices[0]]+sum(e[i] for e,i in zip(law.emissions,indices))
        logweight+=sum(law._transition(j)[indices[j],indices[j+1]]
                       for j in range(len(indices)-1))
        mass=float(np.exp(logweight))
        if not mass:
            continue
        total+=mass
        previous=clock
        value=reference.deadline
        failed=False
        for cid in ride.path[position:]:
            for kind in ('departure','arrival'):
                slot=reference.model._positions[cid,kind][1]
                when=law.times[slot][indices[slot]]
                if law.canceled[slot][indices[slot]] or when<previous or when>=reference.deadline:
                    failed=True
                    break
                previous=when
            if failed:
                break
            if cid in ride.exits:
                assert ride.exits[cid] is None
                value=previous
                break
        weighted+=mass*value
    return weighted/total


@pytest.mark.parametrize('options,position,clock', [
    ({},0,0.),
    ({'canceled':True},0,0.),
    ({'hidden':True},0,0.),
    ({'conditioned':True},1,21.),
    ({'deadline':51.},0,11.),
    ({'deadline':36.},0,0.),
])
def test_whole_tail_matches_enumeration_and_chronological_recursion(options,position,clock):
    reference,history,ride=case(**options)
    expected=enumerate_value(reference,history,ride,position,clock)
    integrated=terminal_ride_value(reference.root_law,history,ride,position,clock,reference.deadline)
    recursive=reference._ride(ride,position,history,clock,-1)
    assert recursive[2]
    assert integrated==pytest.approx(expected,abs=1e-9)
    assert integrated==pytest.approx(recursive[0],abs=1e-9)
    if options.get('hidden'):
        assert integrated<reference.deadline  # Hidden canceled slots are not boarded.


def test_nonterminal_first_exit_cannot_be_integrated_away():
    reference,history,ride=case()
    ride.exits[1]=Platform(2,{})
    assert terminal_ride_value(reference.root_law,history,ride,0,0.,60.) is None
    # The same route after that retained choice has a forced terminal suffix.
    assert terminal_ride_value(reference.root_law,history,ride,2,37.,60.) is not None


def test_first_absorbing_exit_ignores_later_route_entries():
    reference,history,ride=case()
    ride.exits[1]=None
    expected=enumerate_value(reference,history,ride,0,0.)
    assert terminal_ride_value(reference.root_law,history,ride,0,0.,60.)==pytest.approx(expected,abs=1e-9)
    assert expected<40.
