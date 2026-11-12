from copy import deepcopy

import pytest

from experiments.reconstruction_v30.selection import actual_option, build_day_histories
from tests.test_v25_execution import endpoints, graph


def root(options):
    return dict(root_id='d_0_origin', day='d', root_kind='origin', start=0,
                deadline=100, stop=0, prefix=[], full_options=options)


def option(cid, tails=()):
    return dict(first_cid=cid, first_path=[cid], tail_paths=[[tail] for tail in tails])


def test_fixed_target_does_not_recover_preentry_or_publicly_departed_vehicle():
    net = graph([(0, 0, 1, 10, 30), (1, 0, 1, 40, 60)])
    metadata = {0: endpoints(0, 10, 30), 1: endpoints(1, 40, 60)}
    r = root([option(0), option(1)])
    old = actual_option(net, metadata, r, 0, 20, 30, 40, 'target')
    assert old['cost'] == 100 and old['status'] == 'target_departed_before_entry'
    assert old['root_departure'] is old['first_arrival'] is None
    with pytest.raises(ValueError, match='already appeared'):
        actual_option(net, metadata, r, 1, 20, 30, 40, 'target')


def test_labels_preserve_tail_uncertainty_mct_ties_and_physical_late_arrival():
    net = graph([(0, 0, 1, 30, 40), (1, 1, 2, 44, 50),
                 (2, 1, 2, 50, 80), (3, 1, 2, 50, 70)], {1: 5})
    metadata = {cid: endpoints(cid, dep, arr) for cid, dep, arr in
                [(0, 30, 40), (1, 44, 50), (2, 50, 80), (3, 50, 105)]}
    r = root([option(0, (1, 2, 3))])
    label = actual_option(net, metadata, r, 0, 0, 30, 30)
    assert label['first_path_completed'] and label['first_arrival'] == 40
    assert label['tail_cid'] == 3 and label['tail_wait'] == 10
    assert label['final_arrival'] == 105 and label['cost'] == 100 and not label['arrived']
    metadata[1]['departure']['status'] = 'UNKNOWN'
    missing = actual_option(net, metadata, r, 0, 0, 30, 30)
    assert missing['first_arrival'] == 40 and missing['cost'] is None
    assert missing['tail_offer_known'] is False and missing['tail_offer_observed'] is None


def test_first_offer_selection_keeps_empty_root_and_deduplicates_v29_case():
    net = graph([(0, 0, 1, 30, 40), (1, 0, 1, 50, 60)])
    metadata = {0: endpoints(0, 30, 40), 1: endpoints(1, 50, 60)}
    r = root([option(0), option(1)])
    empty = dict(r, root_id='empty', full_options=[])
    case = dict(root=deepcopy(r), entry=0, ready=30, clock=30, visible=[0], seen=[0],
                current_cid=0, target_cid=1)
    histories, coverage = build_day_histories(net, metadata, [r, empty], [case])
    assert len(histories) == 1 and histories[0]['tags'] == ['first_offer', 'v29_disagreement']
    assert [row['status'] for row in coverage] == ['offer', 'empty_menu']
    assert histories[0]['seen'] == [0] and histories[0]['frozen_target_cid'] == 1


def test_known_deadline_cost_survives_missing_later_physical_endpoint():
    net = graph([(0, 0, 1, 30, 105), (1, 1, 2, 110, 120)])
    metadata = {0: endpoints(0, 30, 105), 1: endpoints(1, 110, None)}
    r = root([dict(first_cid=0, first_path=[0, 1], tail_paths=[])])
    label = actual_option(net, metadata, r, 0, 0, 30, 30)
    assert label['cost'] == 100
    assert label['first_arrival'] is None and label['final_arrival'] is None
    assert label['missing_reason'] == 'unresolved_arrival'
