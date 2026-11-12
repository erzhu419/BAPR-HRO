"""Cold same-menu scoring attribution with an alternating paired order."""

import argparse
import json
import os
from pathlib import Path
from statistics import median

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v15.run_attribution_banks import prepare
from experiments.reconstruction_v20.run_policy_pair import audit_menu, compile_direct
from experiments.reconstruction_v21.run_policy_pair import compile_rollout


HERE = Path(__file__).resolve().parent
REPEATS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nominal', action='store_true', help='Compare V20 and same-menu zero-sample compilation.')
    nominal = parser.parse_args().nominal
    methods = ('contraction', 'direct_nominal' if nominal else 'per_clock')
    cpu = max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    output = dict(status='running', repeats_per_root=REPEATS, cpu=cpu, roots=[])
    for case in CASES:
        rows = []
        for i in range(REPEATS):
            order = methods if i % 2 == 0 else methods[::-1]
            compiled = {}
            for method in order:
                cost = {} if nominal else None
                graph, belief, root, goal, deadline, plan, params = prepare(case, conditioning_cost=cost)
                if method == 'direct_nominal':
                    compiled[method] = compile_rollout('direct_rollout_0', graph, belief, params, root, goal,
                                                        deadline, plan, 0)
                else:
                    compiled[method] = compile_direct(graph, belief, params, root, goal, deadline, plan,
                                                       contract=method == 'contraction')
                if nominal:
                    compiled[method]['construction_seconds'] += cost['wall']
            if nominal:
                direct_paths = {cid: p.primary for cid, p in compiled['contraction']['controller'].scores.paths.items()}
                assert compiled['direct_nominal']['paths'] == direct_paths
                audit = dict(same_paths=len(direct_paths))
                fields = ('construction_seconds', 'screening_wall')
            else:
                audit = audit_menu(compiled['per_clock']['controller'], compiled['contraction']['controller'], graph)
                fields = ('construction_seconds', 'screening_wall', 'scoring_wall')
            rows.append(dict(order=order, audit=audit, timing={m: {k: compiled[m][k] for k in fields}
                                                              for m in methods}))
        summary = {m: {k: median(r['timing'][m][k] for r in rows) for k in rows[0]['timing'][m]}
                   for m in methods}
        output['roots'].append(dict(case=case, summary=summary, rows=rows))
        print(case, summary, flush=True)
    output['status'] = 'complete'
    name = 'nominal_cost_profile.json' if nominal else 'contraction_profile.json'
    (HERE/name).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
