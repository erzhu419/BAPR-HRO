"""Public-state union for the new domain and a separate legacy-panel supplement."""
from collections import Counter
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_LABELS = ('new_cheap', 'new_mean', 'new_single', 'new_pair', 'new_one_step',
              'new_snapshot_csa', 'new_matched128', 'new_full128')


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as handle:
        return json.load(handle)


def write(path, value):
    """Derived summaries are reproducible; frozen source receipts are never written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(path, 'wt') if path.suffix == '.gz' else path.open('w')) as handle:
        json.dump(value, handle, allow_nan=False, indent=None if path.suffix == '.gz' else 2)
        handle.write('\n')


def public_state(state, day):
    return dict({key:value for key,value in state.items()
                 if key not in ('selected_action', 'base_action', 'gate_index')}, day=day)


def state_identity(state):
    """Exact V51 canonical equality, without a digest or action-dependent key."""
    return json.dumps(state, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def build_panel(banks):
    states, state_by_key, contexts = [], {}, []
    for rows in banks:
        cheap = rows['new_cheap']
        index = cheap['root_index']
        contexts.append(dict(root_index=index, day=cheap['day'], root_id=cheap['root_id'],
            origin=cheap['origin'], goal=cheap['goal'], menu_source=cheap['menu_source'],
            observation_contract=cheap['observation'], loss_cap=cheap['loss_cap']))
        for label in NEW_LABELS:
            row = rows[label]
            if row['root_index'] != index or row['observation'] != cheap['observation']:
                raise AssertionError(f'New-domain root/observation mismatch: {index}/{label}')
            for sequence, state in enumerate(row['states']):
                public = public_state(state, row['day'])
                key = state_identity(public)
                if key not in state_by_key:
                    state_by_key[key] = len(states)
                    states.append(dict(index=len(states), state=public, occurrences=[],
                                       policies=[], selected_actions=[]))
                saved = states[state_by_key[key]]
                occurrence = dict(source=f'roots/{index:03}_{label}.json.gz#/states/{sequence}',
                    root_index=index, policy=label, state_index=sequence,
                    selected_action=state['selected_action'])
                if 'gate_index' in state:
                    occurrence['gate_index'] = state['gate_index']
                saved['occurrences'].append(occurrence)
                if label not in saved['policies']:
                    saved['policies'].append(label)
                if state['selected_action'] not in saved['selected_actions']:
                    saved['selected_actions'].append(state['selected_action'])
    for state in states:
        state['observed_action_disagreement'] = len(state['selected_actions']) > 1
    summary = dict(unique_states=len(states),
        total_occurrences=sum(len(row['occurrences']) for row in states),
        counts_by_kind=dict(Counter(row['state']['kind'] for row in states)),
        roots=sorted({row['state']['root_index'] for row in states}),
        action_disagreement_states=sum(row['observed_action_disagreement'] for row in states),
        action_disagreement_roots=sorted({row['state']['root_index'] for row in states
                                         if row['observed_action_disagreement']}))
    return dict(stage='decision_gap_v1', contexts=contexts, summary=summary, states=states,
        policies=list(NEW_LABELS),
        identity='Full raw canonical JSON equality; root identity retains the frozen observation contract',
        selection='Every visited public decision of all eight new-domain policies, including unresolved journeys',
        scope='Observed archive histories; not all counterfactual histories or an independent evaluation set')


def legacy_extension(experiments=HERE.parent):
    """Reference the unchanged 88 states and recover the omitted changed gate."""
    experiments = Path(experiments)
    original = read(experiments/'reconstruction_v52/panel.json.gz')
    all_states = read(experiments/'reconstruction_v51/states.json.gz')
    bank = read(experiments/'reconstruction_v51/roots/015_full128.json.gz')
    matches = [(i, gate) for i,gate in enumerate(bank['gates'])
               if gate['cid'] == 127 and gate['clock'] == 28844]
    if len(original['rows']) != 88 or len(matches) != 1:
        raise AssertionError('Frozen legacy panel or missing-gate identity changed')
    gate_index, gate = matches[0]
    state_index = gate['state_index']
    state = public_state(bank['states'][state_index], bank['day'])
    identity = state_identity(state)
    if gate['base_action'] == gate['selected_action']:
        raise AssertionError('Legacy supplement must remain the omitted changed gate')
    if any(state_identity(row['state']) == identity for row in original['rows']):
        raise AssertionError('Legacy supplement already belongs to the original panel')
    sources = [row for row in all_states['states'] if state_identity(row['state']) == identity]
    if len(sources) != 1:
        raise AssertionError('Missing gate has no unique original public-state source')
    context = next(row for row in all_states['contexts'] if row['root_index'] == 15)
    extra = dict(query_index=88, source_state_index=sources[0]['index'], state=state,
        context=context, occurrences=sources[0]['occurrences'],
        reason='Full128 changed gate missing from the 88-state V52 selection',
        raw_source=f'../reconstruction_v51/roots/015_full128.json.gz#/states/{state_index}',
        gate_index=gate_index, base_action=gate['base_action'], selected_action=gate['selected_action'])
    return dict(stage='decision_gap_v1_legacy_extension',
        legacy_panel_reference='../reconstruction_v52/panel.json.gz',
        legacy_queries=88, additional_queries=1, total_queries=89, additional_rows=[extra],
        scope='Original V51/V50 menu and information domain only; old certificates stay with their original inputs')


if __name__ == '__main__':
    write(HERE/'legacy_panel_extension.json.gz', legacy_extension())
