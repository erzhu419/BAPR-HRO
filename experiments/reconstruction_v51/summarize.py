"""Same-task quality/cost and the full new public decision-state panel."""
from collections import Counter
import csv
import json
from math import isfinite
from pathlib import Path

import numpy as np

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import save
from experiments.reconstruction_v44.matched import choose
from experiments.reconstruction_v50.summarize import bounds, comparison, first_divergence
from experiments.reconstruction_v51.run import LABELS

HERE = Path(__file__).resolve().parent


def public_state(state, day):
    """An action or its computational source cannot define a public state."""
    return dict({key:value for key,value in state.items()
                 if key not in ('selected_action', 'base_action', 'gate_index')}, day=day)


def state_identity(state):
    return json.dumps(state, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    protocol = read(HERE/'contract.json')
    if tuple(protocol['labels']) != LABELS:
        raise ValueError('Policy panel differs from frozen contract')
    banks, records, flat = [], [], []
    states, contexts, state_by_key = [], [], {}
    same_full_gates = {}
    gate_comparisons, total_gates = 0, 0
    gate_modes, gate_fallbacks = Counter(), Counter()
    incomplete_gates = Counter()
    for index in range(protocol['roots']):
        rows = {label:read(HERE/'roots'/f'{index:03}_{label}.json.gz') for label in LABELS}
        banks.append(rows)
        common = read(HERE.parent/'reconstruction_v50/roots'/f'{index:03}_wide_common.json.gz')
        cheap = rows['cheap']
        if (cheap['outcome']['loss'], cheap['outcome']['status'], cheap['actions']) != (
                common['outcome']['loss'], common['outcome']['status'], common['actions']):
            raise AssertionError(f'Common-wide cheap changed: root{index}')
        if not all(cheap['v50_cheap_validation'].values()):
            raise AssertionError('Cheap reproduction metadata disagrees')
        contexts.append(dict(root_index=index, day=cheap['day'], root_id=cheap['root_id'],
            origin=cheap['origin'], goal=cheap['goal'], menu_source=cheap['menu_source'],
            observation_contract=cheap['observation'], loss_cap=cheap['loss_cap']))
        old_cheap = read(HERE.parent/'reconstruction_v50/roots'/f'{index:03}_narrow_native.json.gz')
        old_rollout = read(HERE.parent/'reconstruction_v45/roots'/f'{index:03}_full128.json.gz')
        record = dict(root_index=index, day=cheap['day'], origin=cheap['origin'], goal=cheap['goal'],
            results={}, old_capped_context=dict(cheap_loss=old_cheap['outcome']['loss'],
                full128_loss=old_rollout['outcome']['loss'], same_task_comparison=False))
        for label, row in rows.items():
            if (row['root_index'],row['root_id'],row['day'],row['observation']) != (
                    index,cheap['root_id'],cheap['day'],cheap['observation']):
                raise AssertionError(f'Root or observation contract changed: {index}/{label}')
            cost = row['cost']
            for metric in ('process_cpu', 'wall'):
                expected = cost[metric]+cost['shared_menu'][metric]
                if abs(expected-cost[metric+'_including_menu']) > 1e-9:
                    raise AssertionError('Menu preparation missing from total query cost')
            result = dict(loss=row['outcome']['loss'], status=row['outcome']['status'],
                loss_bounds=bounds(row), cost=cost, states=len(row['states']), gates=len(row['gates']),
                first_divergence_from_cheap=first_divergence(cheap['actions'],row['actions']),
                control_stats=row['control_stats'])
            record['results'][label] = result
            flat.append(dict(root_index=index,day=row['day'],origin=row['origin'],goal=row['goal'],
                label=label,loss=result['loss'],status=result['status'],loss_lower=result['loss_bounds'][0],
                loss_upper=result['loss_bounds'][1],cpu=cost['process_cpu'],wall=cost['wall'],
                cpu_including_menu=cost['process_cpu_including_menu'],
                wall_including_menu=cost['wall_including_menu'],states=result['states'],gates=result['gates']))
            for sequence, state in enumerate(row['states']):
                public = public_state(state, row['day'])
                key = state_identity(public)
                if key not in state_by_key:
                    state_by_key[key] = len(states)
                    states.append(dict(index=len(states), state=public, occurrences=[], policies=[], selected_actions=[]))
                saved = states[state_by_key[key]]
                occurrence = dict(source=f'roots/{index:03}_{label}.json.gz#/states/{sequence}',
                    root_index=index, policy=label, state_index=sequence, selected_action=state['selected_action'])
                if 'gate_index' in state:
                    occurrence['gate_index'] = state['gate_index']
                saved['occurrences'].append(occurrence)
                if label not in saved['policies']:
                    saved['policies'].append(label)
                if state['selected_action'] not in saved['selected_actions']:
                    saved['selected_actions'].append(state['selected_action'])
            for sequence, gate in enumerate(row['gates']):
                total_gates += 1
                if gate['future_messages'] != 0 or gate['count'] != protocol['rollout_limits']['worlds_per_query']:
                    raise AssertionError('Rollout information or sample count changed')
                if set(gate['rows']) != {'alight','continue'} or any(
                        len(values)!=gate['count'] for values in gate['rows'].values()):
                    raise AssertionError('Unresolved sample rows were dropped')
                low, high = gate['paired']['full_bank_mean_gain_interval']
                if not all(isfinite(value) for value in (low,high)) or low > high+1e-9:
                    raise AssertionError('Invalid finite-bank action interval')
                if gate['selected_action'] != choose(gate,gate['base_action']):
                    raise AssertionError('Rollout action differs from declared sign/fallback rule')
                state = row['states'][gate['state_index']]
                if state['gate_index'] != sequence or state['selected_action'] != gate['selected_action']:
                    raise AssertionError('Gate is not connected to its actual public action')
                gate_modes[label,gate['used_kernel']] += 1
                if gate['fallback'] is not None:
                    gate_fallbacks[gate['fallback']] += 1
                if gate['completed_pairs'] < gate['count']:
                    incomplete_gates[label] += 1
                if gate['used_kernel']=='full_common_observation_pi' and gate['completed_pairs']==gate['count']:
                    identity = state_identity(public_state(state,row['day']))
                    science = dict(rows=gate['rows'], paired=gate['paired'], common_tape=gate['common_tape'])
                    if identity in same_full_gates:
                        if same_full_gates[identity] != science:
                            raise AssertionError('Same public-state full128 bank changed across policies')
                        gate_comparisons += 1
                    else:
                        same_full_gates[identity] = science
        records.append(record)
    aggregate = {}
    for label in LABELS:
        rows = [r[label] for r in banks]
        limits = np.array([bounds(row) for row in rows])
        losses = [row['outcome']['loss'] for row in rows if row['outcome']['loss'] is not None]
        aggregate[label] = dict(roots=len(rows),complete=len(losses),unresolved=len(rows)-len(losses),
            all_root_mean_loss_bounds=np.mean(limits,axis=0).tolist(),
            complete_mean_loss=None if not losses else float(np.mean(losses)),
            mean_cpu=float(np.mean([r['cost']['process_cpu_including_menu'] for r in rows])),
            mean_wall=float(np.mean([r['cost']['wall_including_menu'] for r in rows])),
            total_states=sum(len(r['states']) for r in rows),total_gates=sum(len(r['gates']) for r in rows),
            versus_common_wide_cheap=comparison(banks,'cheap',label))
    for state in states:
        state['observed_action_disagreement'] = len(state['selected_actions']) > 1
    state_summary = dict(unique_states=len(states),
        total_occurrences=sum(len(state['occurrences']) for state in states),
        counts_by_kind=dict(Counter(state['state']['kind'] for state in states)),
        roots=sorted({state['state']['root_index'] for state in states}),
        action_disagreement_states=sum(state['observed_action_disagreement'] for state in states),
        action_disagreement_roots=sorted({state['state']['root_index'] for state in states
                                         if state['observed_action_disagreement']}),
        source='All actual visited onboard/platform decisions from every V51 policy; no outcome selection')
    panel = dict(stage='V51', contexts=contexts, summary=state_summary, states=states,
        identity='Full raw canonical JSON equality; root identity retains the root-frozen future observation contract',
        scope='Observed actual histories, not all stochastic counterfactual states and not an independent evaluation set')
    panel_path = HERE/'states.json.gz'
    if panel_path.exists():
        if read(panel_path) != panel:
            raise AssertionError('Saved state panel differs from current complete receipts')
    else:
        save(panel_path,panel)
    absorbed = [r['root_index'] for r in records
        if r['old_capped_context']['cheap_loss'] is not None
        and r['old_capped_context']['full128_loss'] is not None
        and r['old_capped_context']['cheap_loss'] > r['old_capped_context']['full128_loss']
        and r['results']['cheap']['loss'] == r['old_capped_context']['full128_loss']]
    result = dict(stage='V51', archive_calls=protocol['archive_calls'], aggregate=aggregate, records=records,
        v50_cheap_reproduced=len(records),state_panel=state_summary,
        gate_audit=dict(total=total_gates,same_state_full128_comparisons=gate_comparisons,
            kernels=[dict(policy=label,kernel=kernel,count=count) for (label,kernel),count in gate_modes.items()],
            fallback_counts=dict(gate_fallbacks),cpu_unfinished_by_policy=dict(incomplete_gates)),
        old_capped_context=dict(scope='Cross-menu descriptive context only, not same-task algorithm evidence',
            old_rollout_positive_roots_absorbed_by_common_wide_cheap=absorbed,
            mean_old_cheap=float(np.mean([r['old_capped_context']['cheap_loss'] for r in records])),
            mean_old_full128=float(np.mean([r['old_capped_context']['full128_loss'] for r in records]))),
        g2_passed=False,new_solver_admitted=False,
        next='V52 ordinary bounded causal optimization on an input-declared panel from this complete new state union',
        server_transfer_bytes=0)
    save(HERE/'summary.json',result)
    write_csv(HERE/'paired.csv',flat)
    plot(aggregate)
    print(json.dumps(dict(aggregate=aggregate,state_panel=state_summary,gate_audit=result['gate_audit']),indent=2))
    return result


def plot(aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    offsets = dict(cheap=(18,-20), single=(-8,18), mean=(12,6),
        pair=(-8,18), one_step=(10,-23), matched128=(-24,25), full128=(-20,-25))
    for i,label in enumerate(LABELS):
        row = aggregate[label]
        low,high = row['all_root_mean_loss_bounds']
        x, color = row['mean_cpu'], f'C{i}'
        if row['unresolved']:
            # A missing-outcome range has no observed all-root mean point.
            ax.vlines(x,low,high,color=color,linewidth=1.8)
            ax.plot([x,x],[low,high],linestyle='none',marker='_',markersize=10,color=color)
            ax.annotate(f"{label}\n{row['unresolved']} missing outcome; bounds only",(x,low),
                xytext=(12,10),textcoords='offset points',fontsize=9,color=color)
            continue
        ax.plot(x,low,marker='s' if label=='full128' else 'o',linestyle='none',
            color=color,markerfacecolor='none' if label=='full128' else color)
        ax.annotate(label,(x,low),xytext=offsets[label],textcoords='offset points',
            ha='right' if label in ('single','pair','matched128','full128') else 'left',
            fontsize=9,color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.7))
    ax.set(xlabel='Mean whole-journey CPU including menu construction (s)',
           ylabel='Mean truncated actual loss over all72 roots (s)',
           title='Common wide actions and observations; lower is better')
    ax.set_xscale('log')
    ax.margins(x=.12,y=.14)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(HERE/'quality_time.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    summarize()
