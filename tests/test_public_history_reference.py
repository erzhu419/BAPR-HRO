import pytest
import numpy as np

from src.public_history_reference import FinitePublicModel, PublicHistoryReference, PublicTransition


def hidden_choice_model(reveal=False):
    def actions(history):
        return ('observe',) if len(history) == 1 and reveal else ('left', 'right')

    def transition(state, action):
        if action == 'observe':
            yield PublicTransition(1., state, ('signal', state))
        else:
            yield PublicTransition(1., None, 'done', 10. if action == state else 90.)
    return FinitePublicModel({'left': .5, 'right': .5}, 'same public prefix',
                             actions, transition, lambda history: (0., 100.))


def test_hidden_world_identifier_cannot_select_its_own_future_action():
    result = PublicHistoryReference(hidden_choice_model()).solve()
    assert result.exact
    assert result.lower == result.upper == 50.
    assert result.lower > 10.  # The forbidden per-world hindsight minimum.
    assert result.decisions[-1]['private_support'] == 2


def test_a_public_signal_changes_the_next_action_after_it_is_observed():
    result = PublicHistoryReference(hidden_choice_model(reveal=True)).solve()
    assert result.exact and result.lower == result.upper == 10.
    children = [d for d in result.decisions if len(d['history']) == 2]
    assert {d['action'] for d in children} == {'left', 'right'}
    assert all(d['action'] == d['history'][-1][1][1] for d in children)


def test_partial_enumeration_returns_valid_bounds_and_unresolved_mass():
    result = PublicHistoryReference(hidden_choice_model(reveal=True), max_transition_terms=1).solve()
    assert not result.exact and result.budget_exhausted
    assert result.lower <= 10. <= result.upper
    assert result.unresolved_mass == 1.


def test_depth_limit_is_not_mislabeled_exact():
    result = PublicHistoryReference(hidden_choice_model(reveal=True), max_depth=1).solve()
    assert (result.lower, result.upper) == (0., 100.)
    assert result.unresolved_mass == 1.
    assert not result.exact


def test_shared_latent_state_is_retained_across_public_observations():
    def transition(state, action):
        if action == 'advance':
            yield PublicTransition(1., state, 'identical receipt')
        else:
            yield PublicTransition(1., None, 'done', 20. if state else 80.)
    model = FinitePublicModel({False: .25, True: .75}, 'entry',
        lambda h: ('advance',) if len(h) == 1 else ('board',), transition, lambda h: (0., 100.))
    result = PublicHistoryReference(model).solve()
    assert result.exact and result.lower == 35.
    assert all(d['private_support'] == 2 for d in result.decisions)


def test_missing_probability_mass_is_an_error_not_free_success():
    model = FinitePublicModel({0: 1.}, 'entry', lambda h: ('board',),
        lambda s, a: [PublicTransition(.9, None, 'done', 10.)], lambda h: (0., 100.))
    with pytest.raises(ValueError, match='retain all probability'):
        PublicHistoryReference(model).solve()


def test_sampled_saa_bank_cannot_be_certified_as_the_original_law():
    model = hidden_choice_model()
    sampled = FinitePublicModel(model.initial_mass, model.initial_observation,
        model.actions, model.transition, model.bounds, provenance='sampled_bank')
    with pytest.raises(ValueError, match='sampled bank'):
        PublicHistoryReference(sampled)


@pytest.mark.parametrize('ready,expected', [(0, 33.5), (7, 68.5)])
def test_structured_shared_trip_reference_matches_complete_public_history_tree(ready, expected):
    from experiments.reconstruction_v33.reference import SharedSlice, structured_reference
    from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
    from src.transit_graph import Connection, Stop, TransitGraph

    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=5))
    for cid, trip, dep, arr, t, u in ((0, 'A', 0, 1, 0, 10), (1, 'A', 1, 2, 11, 20),
                                     (2, 'B', 0, 1, 5, 15), (3, 'B', 1, 2, 16, 25),
                                     (4, 'B', 2, 3, 26, 30)):
        graph.add_connection(Connection(cid, trip, trip, dep, arr, t, u))
    def law(clocks, delay):
        times = [np.array([t, t+delay]) for t in clocks]
        with np.errstate(divide='ignore'):
            transition = np.log(np.eye(2))
        return _ChainLaw(times, np.log([.5, .5]), lambda slot: transition,
            [np.zeros(2) for _ in times], [np.zeros(2, bool) for _ in times],
            (True,) * (len(times)-1))
    laws = {'A': law([0, 10, 11, 20], 6), 'B': law([5, 15, 16, 25, 26, 30], 7)}
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    belief._law = lambda trip, parameters: laws[trip]
    spec = dict(a=0, b=2, entry=0, ready=ready, deadline=100,
        direct=dict(first_path=[2, 3, 4]), variants=[
            dict(first_path=[0], transfer_stop=1, tail_paths=[[3, 4]]),
            dict(first_path=[0, 1], transfer_stop=2, tail_paths=[[4]])])
    adapter = SharedSlice(graph, belief, {}, spec)
    tree = PublicHistoryReference(adapter.model()).solve()
    structured = structured_reference(adapter)
    assert tree.exact and tree.lower == pytest.approx(expected)
    assert structured['exact']
    assert structured['value'] == pytest.approx(tree.lower)
    assert structured['first_event_mass_error'] < 1e-12
    for name in ('conditional_single_chain_stop', 'conditional_two_target_chain_stop'):
        control = structured['controls'][name]
        assert control['expected_arrival'] == pytest.approx(tree.lower)
        assert control['gap_to_exact'] == control['root_disagreement_probability'] == 0.
        assert control['onboard_disagreement_visits'] == 0.
    for control in structured['controls'].values():
        assert control['attribution_error'] < 1e-10
    for decision in structured['decisions']:
        if decision['clock'] < ready:
            assert all(c['action'] == 'wait' for c in decision['controls'].values())
