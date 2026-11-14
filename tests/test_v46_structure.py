from copy import deepcopy
from types import SimpleNamespace

from experiments.reconstruction_v46.structure import boardings, executed_boardings, grouped_platforms, inspect_platform


def model():
    return SimpleNamespace(_positions={
        (0, 'departure'): ('A', 0), (0, 'arrival'): ('A', 1),
        (1, 'departure'): ('B', 2), (1, 'arrival'): ('B', 3),
        (2, 'departure'): ('B', 0), (2, 'arrival'): ('B', 1),
        (3, 'departure'): ('B', 4), (3, 'arrival'): ('B', 5),
    })


def platform(tail_cid):
    leaf = ('platform', 1, ((tail_cid, ('ride', tail_cid, (tail_cid,), ((tail_cid, None),))),))
    return ('platform', 0, ((0, ('ride', 0, (0,), ((0, leaf),))),
                            (1, ('ride', 1, (1,), ((1, None),)))))


def graph():
    return SimpleNamespace(stops={0: SimpleNamespace(name='root'), 1: SimpleNamespace(name='tail')})


def test_shared_trip_interface_preserves_upstream_queries_and_original_menu():
    node = platform(2); before = deepcopy(node)
    result = inspect_platform(node, model(), graph())
    assert node == before
    assert boardings(node) == result['max_remaining_boardings'] == 2
    assert result['distinct_root_trips'] and not result['pure_terminal_platform']
    link = result['root_to_future_links'][0]
    assert (link['observed_root_cid'], link['later_candidate_cid']) == (1, 2)
    assert (link['root_departure_slot'], link['later_departure_slot']) == (2, 0)
    assert link['slot_direction'] == 'backward'
    assert result['required_observation_interface'][0]['backward_conditioning_required']


def test_shared_trip_forward_is_distinguished_from_backward():
    result = inspect_platform(platform(3), model(), graph())
    assert result['root_to_future_links'][0]['slot_direction'] == 'forward'
    assert not result['required_observation_interface'][0]['backward_conditioning_required']


def test_all_positive_first_platform_mass_is_retained_including_tiny_branch():
    node = platform(2)
    atoms = [dict(kind='platform', probability=1.-1e-12, entry=10., node_structure=node),
             dict(kind='platform', probability=3.969529108641934e-18, entry=11., node_structure=node),
             dict(kind='terminal', probability=1e-12, entry=None)]
    result = grouped_platforms(atoms, model(), graph())
    assert len(result) == 1 and result[0]['atom_indices'] == [0, 1]
    assert result[0]['entries'][1]['probability'] == atoms[1]['probability']
    assert result[0]['probability'] == sum(a['probability'] for a in atoms[:2])


def test_early_destination_stops_physical_path_but_keeps_original_scoring_dependencies():
    physical_model = model()
    physical_model._positions.update({(4, 'departure'): ('A', 2), (4, 'arrival'): ('A', 3)})
    leaf = ('platform', 1, ((1, ('ride', 1, (1,), ((1, None),))),))
    ride = ('ride', 0, (0, 4), ((0, None), (4, leaf)))
    node = ('platform', 0, ((0, ride), (1, leaf[2][0][1])))
    before = deepcopy(node)
    result = inspect_platform(node, physical_model, graph())
    candidate = result['candidates'][0]
    assert node == before
    assert candidate['first_destination_cid'] == 0
    assert candidate['physical_executable_prefix'] == [0]
    assert candidate['early_destination']
    assert candidate['exits_after_first_destination'] == [dict(cid=4, has_tail=True, tail_trips=['B'])]
    assert candidate['policy_dependencies_after_first_destination'] == candidate['future_trips'] == ['B']
    assert candidate['pure_physical_terminal'] and not candidate['pure_terminal']
    assert result['pure_physical_terminal_platform'] and not result['pure_terminal_platform']
    assert boardings(node) == result['max_remaining_boardings'] == 2
    assert executed_boardings(node) == result['max_executable_boardings'] == 1
    assert len(result['root_to_future_links']) == 1
    assert not result['root_to_future_links'][0]['physically_reachable']
    assert result['physically_reachable_root_to_future_links'] == []
    assert result['required_observation_interface'][0]['trip'] == 'B'


def test_transfer_before_first_destination_remains_physically_executable():
    physical_model = model()
    physical_model._positions.update({(4, 'departure'): ('A', 2), (4, 'arrival'): ('A', 3)})
    leaf = ('platform', 1, ((1, ('ride', 1, (1,), ((1, None),))),))
    ride = ('ride', 0, (0, 4), ((0, leaf), (4, None)))
    result = inspect_platform(('platform', 0, ((0, ride),)), physical_model, graph())
    candidate = result['candidates'][0]
    assert candidate['physical_executable_prefix'] == [0, 4]
    assert candidate['first_destination_cid'] == 4
    assert not candidate['pure_physical_terminal'] and not candidate['early_destination']
    assert candidate['max_executable_boardings'] == 2
