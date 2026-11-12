from copy import deepcopy

from experiments.reconstruction_v29.run import gate


def rows(law='route_empirical20'):
    return [dict(root_id=f'2023-07-17_0_{start}_origin', law=law,
                 scores=dict(status='supported', nominal=dict(action='wait_target'),
                             integrated=dict(action='wait_target')))
            for start in range(27600, 32101, 900)]


def test_stop_requires_all_six_supported_primary_rankings_unchanged():
    result = gate(rows())
    assert result['status'] == 'stop_exact_scoring_rescue'
    assert result['primary_cells'] == result['supported_cells'] == 6
    assert result['changed_roots'] == []


def test_primary_flip_or_unsupported_or_incomplete_population_cannot_stop():
    changed = rows()
    changed[2]['scores']['integrated']['action'] = 'board_current'
    result = gate(changed)
    assert result['status'] == 'ranking_change_requires_separate_policy_evidence'
    assert result['changed_roots'] == [changed[2]['root_id']]
    unsupported = rows()
    unsupported[2]['scores'].update(status='unsupported', nominal=None, integrated=None)
    for sample in (unsupported, rows()[:-1]):
        result = gate(sample)
        assert result['status'] == 'unresolved_support'
        assert result['supported_cells'] == 5


def test_original_law_sensitivity_remains_separate_from_primary_gate():
    primary, sensitivity = rows(), rows('original20')
    sensitivity[0]['scores']['integrated']['action'] = 'board_current'
    sensitivity[1]['scores'].update(status='unsupported', nominal=None, integrated=None)
    sample = primary+sensitivity
    saved = deepcopy(sample)
    assert gate(sample) == gate(primary)
    assert sample == saved  # Original flips/unsupported evidence remains available to by-law reporting.
    primary[4]['scores']['integrated']['action'] = 'board_current'
    result = gate(primary+rows('original20'))
    assert result['status'] == 'ranking_change_requires_separate_policy_evidence'
    assert result['changed_roots'] == [primary[4]['root_id']]
