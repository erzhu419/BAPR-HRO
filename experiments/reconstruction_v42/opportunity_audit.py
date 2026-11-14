"""Read saved V41 traces; count structural opportunities without policy replay."""
from collections import Counter
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'reconstruction_v41'


def key(node):
    if node['kind'] == 'ride':
        return ('ride', tuple(node['path']), tuple((x['cid'], None if x['next'] is None else key(x['next']))
                for x in node['exits']))
    return ('platform', node['stop'], tuple((x['cid'], key(x['next'])) for x in node['rides']))


def walk(state, arm, row):
    ride = state['node']
    platform = None
    entries = []
    decision_geometry = []
    if arm == 'alight':
        platform = next(x['next'] for x in ride['exits'] if x['cid'] == state['cid'])
        entries.append(('platform', key(platform), state['clock'], 0))
    else:
        remaining = ride['path'][ride['path'].index(state['cid'])+1:]
        ride = dict(kind='ride', path=remaining,
                    exits=[x for x in ride['exits'] if x['cid'] in remaining])
        entries.append(('ride', key(ride), state['clock'], 0))
    terminal = []
    for i, event in enumerate(row['trace']):
        if event['kind'] == 'onboard':
            assert event['cid'] in [x['cid'] for x in ride['exits']]
            remaining = ride['path'][ride['path'].index(event['cid']):]
            future = dict(kind='ride', path=remaining,
                          exits=[x for x in ride['exits'] if x['cid'] in remaining])
            decision_geometry.append(('onboard', key(future), event['cid'], event['clock'], i))
            if event['choice'] == 'alight':
                platform = next(x['next'] for x in ride['exits'] if x['cid'] == event['cid'])
                entries.append(('platform', key(platform), event['clock'], i+1))
        elif event['kind'] == 'platform':
            assert platform is not None and platform['stop'] == event['stop']
            decision_geometry.append(('platform', key(platform), event['stop'], event['clock'], i))
            if event['choice'] is not None:
                ride = next(x['next'] for x in platform['rides'] if x['cid'] == event['choice'])
                entries.append(('ride', key(ride), event['clock'], i+1))
                platform = None
                if all(x['next'] is None for x in ride['exits']):
                    terminal.append(dict(cid=event['choice'], clock=event['clock'],
                                         segments=len(ride['path']), decisions_after=len(row['trace'])-i-1))
        else:
            raise ValueError(event)
    return entries, terminal, decision_geometry


def family_audit(family):
    total = Counter()
    states, witnesses, event_witnesses, divergent_witnesses = [], [], [], []
    for index in range(148):
        path = SOURCE/'banks'/f'{index:03}_{family}.json.gz'
        data = json.loads(gzip.decompress(path.read_bytes()))
        state = data['state']
        local = Counter()
        local['bank_process_cpu'] = data['cost']['total_process_cpu']
        local['continue_process_cpu'] = data['cost']['continuations']['continue']['process_cpu']
        for left, right in zip(data['rows']['alight'], data['rows']['continue'], strict=True):
            ae, at, ag = walk(state, 'alight', left)
            be, bt, bg = walk(state, 'continue', right)
            amap = {entry[:3]: entry[3] for entry in ae}
            candidates = [(amap[e[:3]], e[3]) for e in be if e[:3] in amap]
            local['pairs'] += 1
            local['entry_checks'] += len(ae)+len(be)
            local['decisions'] += len(left['trace'])+len(right['trace'])
            local['terminal_rides'] += len(at)+len(bt)
            local['terminal_segments'] += sum(x['segments'] for x in at+bt)
            local['terminal_followed_by_decision'] += sum(x['decisions_after'] > 0 for x in at+bt)
            local['equal_final_losses'] += left['loss'] == right['loss']
            local['pairs_with_matching_entry_geometry'] += bool(candidates)
            if candidates:
                i,j = min(candidates, key=lambda pair:pair[1])
                local['optimistic_second_arm_decisions_saved'] += len(right['trace'])-j
                identical = left['trace'][i:] == right['trace'][j:] and left['loss'] == right['loss']
                local['matching_entry_equal_science_suffix'] += identical
                if len(witnesses) < 6:
                    witnesses.append(dict(index=index, root_index=state['root_index'],
                        world_index=left['world_index'], matched_trace_indices=[i,j],
                        equal_science_suffix=identical, left_trace=left['trace'],right_trace=right['trace'],
                        losses=[left['loss'],right['loss']]))
            if at and bt and at[-1]['cid'] == bt[-1]['cid'] and at[-1]['clock'] == bt[-1]['clock']:
                local['same_final_direct_boarding'] += 1
            amap = {entry[:4]: entry[4] for entry in ag}
            event_matches = [(amap[e[:4]], e[4]) for e in bg if e[:4] in amap]
            local['pairs_with_matching_decision_geometry'] += bool(event_matches)
            if event_matches:
                i,j=min(event_matches,key=lambda pair:pair[1])
                local['optimistic_event_second_arm_decisions_saved'] += len(right['trace'])-j
                local['event_second_arm_whole_cpu_ceiling'] += right['total_process_cpu']
                equal=left['trace'][i:]==right['trace'][j:] and left['loss']==right['loss']
                local['matching_decision_equal_science_suffix'] += equal
                witness=dict(index=index,root_index=state['root_index'],world_index=left['world_index'],
                    matched_trace_indices=[i,j],equal_science_suffix=equal,
                    left_trace=left['trace'],right_trace=right['trace'],losses=[left['loss'],right['loss']])
                if len(event_witnesses)<6:
                    event_witnesses.append(witness)
                if not equal and len(divergent_witnesses)<6:
                    divergent_witnesses.append(witness)
        total.update(local)
        states.append(dict(index=index, root_index=state['root_index'], **local))
    return dict(total=dict(total), states=states, witnesses=witnesses,
        event_witnesses=event_witnesses,divergent_witnesses=divergent_witnesses,
        states_with_matching_decision_geometry=sum(s.get('pairs_with_matching_decision_geometry',0)>0 for s in states),
        states_with_matching_entry_geometry=sum(s.get('pairs_with_matching_entry_geometry',0)>0 for s in states))


def main():
    result = dict(scope='Saved V41 traces only; no policy or conditional-world replay. Matching geometry ignores history and is an opportunity screen, not a meeting certificate.',
                  physical=family_audit('physical'), shallow=family_audit('shallow'))
    (HERE/'opportunities.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(total=v['total'],states_with_matching_entry_geometry=v['states_with_matching_entry_geometry'])
                      for k,v in result.items() if k!='scope'},indent=2))


if __name__ == '__main__':
    main()
