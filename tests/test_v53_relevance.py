"""Trip projection preserves causal values and the actual scorer's inputs."""
import json
from types import SimpleNamespace

import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference
from experiments.reconstruction_v53.relevance import RemainingReadSets, history_key
from src.evolving_trip_belief import EvolvingTripBelief
from src.transit_graph import Connection
from test_v52_reference import frozen_chain, information_case


def dependency_case():
    trips = {0:'X', 1:'X', 2:'Y', 3:'Y', 4:'Z', 5:'W', 6:'W', 7:'Q'}
    model = SimpleNamespace(_positions={(cid, kind):(trip, 2*cid+(kind=='arrival'))
        for cid, trip in trips.items() for kind in ('departure', 'arrival')})
    z = Platform(2, {4:Ride(4, (4,), {4:None})})
    w = Platform(3, {5:Ride(5, (5,), {5:None})})
    # Y reaches the destination at 2, but the unchanged scorer also reads W
    # through its later exit. Physical reachability alone would miss W.
    y = Ride(2, (2,3), {2:None, 3:w})
    x = Ride(0, (0,1), {0:z, 1:None})
    platform = Platform(0, {0:x, 2:y})
    contract = ObservationContract({0:(0,2,6,7), 2:(4,), 3:(5,)})
    return model, platform, contract


def test_suffix_liveness_preserves_post_destination_scores_and_same_trip_sight():
    model, platform, contract = dependency_case()
    index = RemainingReadSets(model, contract)
    original = contract.describe()
    assert index.platform_trips(platform) == frozenset(('X','Y','Z','W'))
    assert index.ride_trips(platform.rides[0], 1) == frozenset(('X',))
    assert index.platform_trips(platform, (2,)) == frozenset(('Y','W'))
    # cid6 is not an action or a path input, but observes the live W trip.
    assert index.contract_for(platform, remaining_cids=(2,)).cids_at(0) == (2,6)
    assert index.contract_for(platform, remaining_cids=()).cids_at(0) == ()
    assert contract.describe() == original


def test_current_visible_inputs_expire_only_after_wait_and_keep_ordered_evidence():
    model, platform, contract = dependency_case()
    index = RemainingReadSets(model, contract)
    left = SimpleNamespace(_records={
        'X':((0,'departure','observed',10.), (1,'arrival','observed',20.)),
        'Y':((2,'departure','unseen',((0.,10.),)),),
        'W':((6,'departure','observed',8.),),
        'Q':((7,'departure','observed',6.),)})
    right = SimpleNamespace(_records=dict(left._records, X=((0,'departure','observed',11.),
        (1,'arrival','observed',20.)), Q=((7,'departure','observed',9.),)))
    comparison = index.platform_trips(platform, (0,2))
    after_wait = index.platform_trips(platform, (2,))
    assert history_key(left, comparison) != history_key(right, comparison)
    assert history_key(left, after_wait) == history_key(right, after_wait)
    assert dict(history_key(left, after_wait))['W'] == left._records['W']
    assert 'Q' in left._records  # Projection has not edited public evidence.


def reference_with_nuisance(projected, mct=0):
    base, platform, initial = information_case(observed=True, mct=mct,
                                               implementation='mass_terminal')
    base.graph.add_connection(Connection(3, 'R', 'C', 0, 1, 3, 4))
    model = EvolvingTripBelief(base.graph, public_occurrences=False)
    history = PhysicalHistory(model, base.parameters, fit=initial.fit)
    history.message_clock = 0.
    contract = ObservationContract({0:(0,1,2,3)})
    index = RemainingReadSets(model, contract)
    observation = index.contract_for(platform) if projected else contract
    reference = CausalReference(base.graph, model, base.parameters, 60, history,
        observation, max_states=100000, max_process_cpu=None, keep_decisions=True,
        implementation='mass_terminal')
    reference.root_law.roots.update(base.root_law.roots)
    reference.root_law.roots['C'] = frozen_chain([[3,8,12],[4,9,13]], [1/3]*3)
    return reference, platform


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('mct', [0,11])
def test_independent_observation_integrates_out_for_all_five_policies(mode, mct):
    results=[]
    for projected in (False, True):
        reference, platform = reference_with_nuisance(projected, mct)
        result = reference.solve(platform, dict(kind='platform_start',clock=0.,entry=0.), mode=mode)
        assert result['value']['complete']
        results.append(result)
    full, reduced = results
    assert reduced['value']['mean'] == pytest.approx(full['value']['mean'], abs=1e-8)
    assert reduced['counters']['states'] < full['counters']['states']
    assert all(d['clock'] >= mct for d in reduced['decisions'])
    assert full['future_public_frames'] == reduced['future_public_frames'] == 0
    def choices(result):
        return {(d['clock'], tuple(d.get('visible', ())), str(d['selected']), d['optimized'],
                 json.dumps([r for r in d['public_history'] if r['trip'] != 'C'], sort_keys=True))
                for d in result['decisions']}
    assert choices(reduced) == choices(full)
    # The informative same-trip cid1 is retained although only cid0/cid2
    # are boardable. Dropping it would change the causal optimum 27.5→30.
    assert reduced['observation_contract']['station_to_cids'][0]['cids'] == [0,1,2]
    if mode == 'optimal' and mct == 0:
        assert reduced['value']['mean'] == pytest.approx(27.5)
