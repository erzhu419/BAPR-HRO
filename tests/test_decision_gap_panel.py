"""A new task panel cannot silently inherit old states or certificates."""
from copy import deepcopy
from pathlib import Path

from experiments.decision_gap_v1.panel import (
    NEW_LABELS, build_panel, legacy_extension, public_state, read, state_identity,
)
from test_decision_gap_summary import bank


def public(action='alight'):
    return dict(kind='onboard',root_index=0,cid=1,clock=10.,node={'path':[1,2]},
        public_history={'seen':[]},selected_action=action,base_action='continue',gate_index=0)


def test_new_panel_merges_action_sources_but_keeps_information_and_menu_distinct():
    rows = bank()
    rows['old_old_cheap']['states'] = [dict(public(),cid=999)]
    rows['old_new_cheap']['states'] = [dict(public(),cid=998)]
    for label in NEW_LABELS:
        rows[label]['states'] = [public()]
    rows['new_full128']['states'][0]['selected_action'] = 'continue'
    changed_information = public()
    changed_information['public_history']['seen'] = [2]
    rows['new_full128']['states'].append(changed_information)
    changed_menu = public()
    changed_menu['node']['path'].append(3)
    rows['new_full128']['states'].append(changed_menu)
    panel = build_panel([rows])
    assert panel['summary']['unique_states'] == 3
    assert panel['summary']['total_occurrences'] == 10
    assert panel['summary']['action_disagreement_states'] == 1
    assert panel['states'][0]['selected_actions'] == ['alight','continue']
    assert set(panel['states'][0]['policies']) == set(NEW_LABELS)
    assert all(row['state']['cid'] == 1 for row in panel['states'])
    assert all('selected_action' not in row['state'] for row in panel['states'])


def test_root_frozen_observation_context_prevents_cross_root_state_merging():
    first,second = bank(0),bank(1)
    for label in NEW_LABELS:
        first[label]['states'] = [public()]
        second[label]['states'] = [dict(public(),root_index=1)]
    result = build_panel([first,second])
    assert result['summary']['unique_states'] == 2
    assert result['summary']['roots'] == [0,1]


def test_legacy_supplement_recovers_exact_raw_gate_without_changing_88_panel():
    experiments = Path(__file__).resolve().parents[1]/'experiments'
    original = read(experiments/'reconstruction_v52/panel.json.gz')
    before = deepcopy(original)
    result = legacy_extension(experiments)
    extra = result['additional_rows'][0]
    assert result['legacy_queries'] == 88
    assert result['total_queries'] == 89
    assert (extra['source_state_index'],extra['state']['root_index'],
            extra['state']['cid'],extra['state']['clock']) == (118,15,127,28844)
    assert (extra['base_action'],extra['selected_action']) == ('continue','alight')
    raw = read(experiments/'reconstruction_v51/roots/015_full128.json.gz')
    gate = raw['gates'][extra['gate_index']]
    assert extra['state'] == public_state(raw['states'][gate['state_index']],raw['day'])
    assert all(state_identity(row['state']) != state_identity(extra['state']) for row in original['rows'])
    assert read(experiments/'reconstruction_v52/panel.json.gz') == before
