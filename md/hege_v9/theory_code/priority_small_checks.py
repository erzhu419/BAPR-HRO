"""Check diagnostic scheduling against independent original-executor enumeration."""
import sys,json
from pathlib import Path
import numpy as np
from priority_probe import PriorityOracle,REPO,OUT
sys.path.insert(0,str(REPO/'tests'))
import pytest
from test_v46_oracle import case
from test_v46_shared import shared_case
from experiments.reconstruction_v46.small_oracle import evaluate

rows=[]
for name,options in [
    ('mixed_plain',dict(forecasts=False,mct=0,cancel=0.)),
    ('mixed_forecast_cancel',dict(forecasts=True,mct=0,cancel=.1)),
    ('mixed_ready',dict(forecasts=True,mct=3,cancel=0.)),
    ('shared_plain',dict(reverse=False,forecasts=False,cancel=0.)),
    ('shared_forecast_cancel',dict(reverse=False,forecasts=True,cancel=.05)),
    ('shared_reverse',dict(reverse=True,forecasts=True,cancel=0.)),
]:
    with pytest.MonkeyPatch.context() as mp:
        if name.startswith('mixed'):
            graph,model,params,history,platform=case(mp,**options);entry=10
        else:
            graph,model,params,history,platform,entry=shared_case(mp,**options)
        exact=evaluate(graph,model,params,60,platform,
            dict(kind='platform_start',entry=entry,clock=entry),history,action='policy',max_evaluations=100000)
        truth=exact['mean']+entry
        solver=PriorityOracle(graph,model,params,60,history,max_compiles=10000)
        result=solver.platform(platform,entry,checkpoints=tuple(range(1,501)))
        assert result['mean'] is not None
        assert abs(result['mean']-truth)<1e-8
        old=(-np.inf,np.inf)
        for snap in [*result['checkpoints'],result]:
            v=snap['actions']['policy']
            assert v['lower']<=truth+1e-8 and v['upper']>=truth-1e-8
            assert v['lower']>=old[0]-1e-8 and v['upper']<=old[1]+1e-8
            old=(v['lower'],v['upper'])
        rows.append(dict(case=name,independent_truth=truth,priority_value=result['mean'],
            checkpoints_checked=len(result['checkpoints'])+1,
            compiled=result['counters']['charged_compiles'],
            full_probability_count=exact.get('evaluations')))
        print(name,'PASS',truth,result['counters']['charged_compiles'],flush=True)
(OUT/'small_checks.json').write_text(json.dumps(dict(all_passed=True,cases=rows),indent=2)+'\n')
