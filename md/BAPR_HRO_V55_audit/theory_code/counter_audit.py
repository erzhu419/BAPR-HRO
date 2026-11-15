"""Classify archived V55 b10 work; never edits experiment receipts."""
import gzip
import json
import os
from pathlib import Path
from statistics import mean

BASE = Path(os.environ.get('BAPR_HRO_REPO', '/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-a2b82be9')) / 'experiments'
OUT = Path(__file__).resolve().parent

def read(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)

result = {'commit':'a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491', 'budget':'b10',
    'scope':'Archived unmodified V55 priority vs V53 priority receipts. Counters are work quantities, not CPU attribution.',
    'groups':{}, 'critical_working':[], 'optimal_root_bound_only':{}}
for mode in ('optimal','cheap_p0','cheap_working','one_step_p0','two_step_p0'):
    rows = []
    for path in sorted((BASE/'reconstruction_v55/priority/natural').glob(f'*_b10_{mode}.json.gz')):
        new = read(path)
        old = read(BASE/'reconstruction_v53/natural_priority'/path.name)
        nwidth = new['value']['upper']-new['value']['lower']
        owidth = old['value']['upper']-old['value']['lower']
        counters = new['counters']
        row = dict(query=new['query_index'], root=new['root_index'],
            delta=nwidth-owidth, old_width=owidth, new_width=nwidth,
            admitted=counters['terminal_dp_admitted'],
            states=counters['terminal_dp_states'], calls=counters['terminal_dp_calls'])
        rows.append(row)
        if mode == 'cheap_working' and new['query_index'] in (5,8,23,39,71,76,77):
            result['critical_working'].append(dict(row, cheap_action=new['cheap_action'],
                cpu=new['complete_cost']['process_cpu'], counters=counters))
    groups = {}
    for label,condition in (
        ('none',lambda row:row['admitted']==0),
        ('singleton_only',lambda row:row['admitted']>0 and row['states']==0),
        ('subset',lambda row:row['states']>0)):
        selected = [row for row in rows if condition(row)]
        if selected:
            groups[label] = dict(n=len(selected),
                wider=sum(row['delta']>1e-6 for row in selected),
                narrower=sum(row['delta']<-1e-6 for row in selected),
                equal=sum(abs(row['delta'])<=1e-6 for row in selected),
                mean_width_delta=mean(row['delta'] for row in selected))
    result['groups'][mode] = groups

for label, directory in (('V53_impact','reconstruction_v53/natural_priority'),
                         ('V55_priority','reconstruction_v55/priority/natural')):
    identical = []
    improvements = []
    initial_certificates = final_certificates = 0
    for path in sorted((BASE/directory).glob('*_b10_optimal.json.gz')):
        row = read(path)
        bounds, clock = row['root_bounds'], row['clock']
        lower = min(value['lower'] for value in bounds.values())-clock
        upper = min(value['upper'] for value in bounds.values())-clock
        best = min(bounds, key=lambda action:bounds[action]['upper'])
        others = [value['lower'] for action,value in bounds.items() if action != best]
        initial_certificates += bool(others and bounds[best]['upper'] < min(others)-1e-9)
        final_certificates += row['certified_action'] is not None
        final = row['value']
        if abs(lower-final['lower'])<1e-6 and abs(upper-final['upper'])<1e-6:
            identical.append(row['query_index'])
        else:
            improvements.append(dict(query=row['query_index'], root=row['root_index'],
                initial_width=upper-lower, final_width=final['upper']-final['lower'],
                lower_gain=final['lower']-lower, upper_gain=upper-final['upper']))
    result['optimal_root_bound_only'][label] = dict(identical_value_intervals=len(identical),
        initial_action_certificates=initial_certificates,
        final_action_certificates=final_certificates, new_value_information=improvements)

(OUT/'counter_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result['groups'], indent=2))
