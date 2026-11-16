"""Check causal command clocks from saved receipts without new policy calls."""
import argparse
from collections import Counter
from pathlib import Path

from experiments.decision_gap_v1.panel import read,write

HERE=Path(__file__).resolve().parent


def audit(directory=HERE):
    directory=Path(directory)
    contract=read(directory/'contract.json')
    counts=Counter();failures=[]
    def check(condition,rule,row,**detail):
        counts[rule]+=1
        if not condition:
            failures.append(dict(root_index=row['root_index'],arm=row['label'],rule=rule,**detail))
    for i in range(contract['roots']):
        for arm in contract['arms']:
            row=read(directory/'roots'/f"{i:03}_{arm['label']}.json.gz")
            counts['core_receipts']+=1
            trace=row['outcome']['trace']
            for state in row['states']:
                for evidence in state['public_history']:
                    if evidence.get('evidence')=='forecast':
                        check(evidence['receipt_seconds']<=state['clock'],
                            'forecast_received_before_decision',row,clock=state['clock'],
                            receipt_seconds=evidence['receipt_seconds'])
            if arm['clock_contract']!='async':
                continue
            for event in trace:
                if event['kind']=='platform' and event.get('choice') is not None:
                    command=event['command'];cid=event['choice']
                    ready=command['ready_by_cid'][str(cid)]
                    check(command['issued_at']<event['clock'],'platform_command_precedes_departure',row,
                        clock=event['clock'],issued_at=command['issued_at'],cid=cid)
                    check(ready<=event['clock'],'platform_ready_before_departure',row,
                        clock=event['clock'],ready=ready,cid=cid)
            for position,event in enumerate(trace):
                if event['kind']!='onboard' or event.get('receipt',{}).get('method')!='mandatory_menu_exit':
                    continue
                arrival=event['arrival_clock']
                check(event['consumed_at']==arrival,'mandatory_exit_at_arrival',row,
                    cid=event['cid'],consumed_at=event['consumed_at'],arrival=arrival)
                following=next((e for e in trace[position+1:] if e['kind'] in ('platform_command','platform')),None)
                if following is not None:
                    check(following['entry']==arrival,'mandatory_exit_equals_platform_entry',row,
                        cid=event['cid'],arrival=arrival,entry=following['entry'])
            states=[s for s in row['states'] if s['kind']=='onboard'
                    and s.get('decision_type')!='mandatory_menu_exit']
            pubs=[p for p in row['publications'] if p['kind']=='onboard_publication']
            check(len(states)==len(pubs),'onboard_query_publication_pairing',row,
                states=len(states),publications=len(pubs))
            for state,pub in zip(states,pubs):
                check(state['clock']==pub['opens_at'],'onboard_clock_matches_publication',row,
                    clock=state['clock'],opens_at=pub['opens_at'])
                if state['selected_action']!='alight':
                    continue
                consumed=pub['consumed']['consumed_wall']
                check(consumed<pub['closes_at'],'onboard_alight_precedes_departure',row,
                    cid=state['cid'],consumed_at=consumed,closes_at=pub['closes_at'])
                position=next(j for j,event in enumerate(trace)
                    if event['kind']=='onboard' and event['cid']==state['cid']
                    and event['clock']==state['clock'])
                following=next((event for event in trace[position+1:]
                    if event['kind'] in ('platform_command','platform')),None)
                if following is not None:
                    check(following['entry']==consumed,'alight_consumption_equals_platform_entry',row,
                        cid=state['cid'],consumed_at=consumed,entry=following['entry'])
                else:
                    counts['alight_with_no_later_platform_record']+=1
    result=dict(stage='trc_revision_v1',status='fail' if failures else 'pass',
        checked=dict(counts),failures=failures,
        scope='Saved core receipts only; no new control calls. Forecast receipt clocks checked, not actual endpoint timestamps. Departure expiry applies to discretionary onboard queries; mandatory terminal menu exits require no query or fictitious dwell window.')
    write(directory/'execution_audit.json',result)
    if failures:
        raise ValueError(f'{len(failures)} causal receipt checks failed; inspect execution_audit.json')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=HERE)
    audit(p.parse_args().directory)
