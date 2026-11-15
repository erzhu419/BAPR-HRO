"""Read-only fixed-budget frontier instrumentation; no new control or timing claim."""
import gzip
import json
import os
from pathlib import Path
import sys
from collections import defaultdict

REPO = Path(os.environ.get('BAPR_HRO_REPO', '/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-6e46267'))
sys.path.insert(0, str(REPO))
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v41.run import restored
from experiments.reconstruction_v47.bulk import BulkIncrementalOracle, BulkDPNode

OUT = Path(__file__).resolve().parent / 'diagnostic_replay'
OUT.mkdir(exist_ok=True)

def describe(key, node):
    info = dict(key_kind=key[0], cost_kind=node.kind,
                physical_lower=node.lower, physical_upper=node.upper)
    if key[0] in ('event', 'policy'):
        info.update(event_index=key[2], unseen_mask=key[3])
        if key[0] == 'policy':
            info.update(departed_mask=key[4], interface=list(key[5]))
        else:
            info.update(interface=list(key[4]))
    elif key[0] == 'ride':
        info.update(slot=key[-2], state=key[-1])
    return info

def frontier(root, inverse):
    """Linear root path coefficients. Reject unresolved custom-reducer DAGs."""
    order, visited = [], set()
    def visit(n):
        if n in visited:
            return
        visited.add(n)
        if isinstance(n, BulkDPNode):
            raise ValueError('Cannot identify reducer dependencies as physical probabilities')
        if n.expanded:
            for p, child in n.children:
                if p > 0:
                    visit(child)
        order.append(n)
    visit(root)
    coefficients = defaultdict(float, {root: 1.0})
    for n in reversed(order):
        if n.expanded:
            for p, child in n.children:
                if p > 0:
                    coefficients[child] += coefficients[n] * p
    leaves = []
    for n in order:
        if not n.expanded:
            w = coefficients[n]
            leaves.append(dict(**describe(inverse[n], n), probability=w,
                               width_contribution=w * (n.upper - n.lower)))
    leaves.sort(key=lambda x: -x['width_contribution'])
    return dict(reachable_nodes=len(order), frontier_nodes=len(leaves),
                unresolved_probability=sum(x['probability'] for x in leaves),
                additive_width=sum(x['width_contribution'] for x in leaves),
                frontier=leaves), coefficients

class TracedOracle(BulkIncrementalOracle):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.frontiers = []
        self.expansion_log = []
    def _expand(self, node):
        if not node.expanded and node.build is not None and node.kind is not None:
            self.expansion_log.append((node, self.stats['charged_compiles'] + 1))
        return super()._expand(node)
    def _snapshot(self, phase):
        result = super()._snapshot(phase)
        inverse = {node:key for key,node in self.nodes.items()}
        record = dict(phase=phase, count=self.stats['charged_compiles'], arms={})
        for arm, root in self._roots.items():
            if result['actions'][arm]['mean'] is not None:
                record['arms'][arm] = {'exact': True}
                continue
            try:
                detail, _ = frontier(root, inverse)
                observed = result['actions'][arm]
                detail['reported_unresolved_probability'] = observed['unresolved_probability']
                detail['reported_width'] = observed['upper'] - observed['lower']
                detail['parent_clipping_width_difference'] = detail['additive_width'] - detail['reported_width']
                record['arms'][arm] = detail
            except ValueError as exc:
                record['arms'][arm] = {'not_linear': str(exc)}
        self.frontiers.append(record)
        return result

manifest = read(REPO/'experiments/reconstruction_v38/seed_manifest.json')
data = read(REPO/'experiments/reconstruction_v34/data.json.gz')
fit = read(REPO/'experiments/reconstruction_v34/model_fit.json')
emission = read(REPO/'experiments/reconstruction_v35/fitted_emission.json')
for index in (120, 25):
    entry = manifest[index]
    state = entry['state']
    graph = context(data, state['day'])[0]
    model, parameters = setup(graph, fit)
    node = restore_node(state['node'])
    history = restored('shallow', model, parameters, emission, state)
    oracle = TracedOracle(graph, model, parameters, state['deadline'], history,
                          max_compiles=2000, max_process_cpu=120.)
    result = oracle.forced(node, {key:state[key] for key in ('kind','cid','clock')},
                           schedule='pooled', checkpoints=(500,1000,1500,2000))
    original = read(REPO/'experiments/reconstruction_v47/natural'/f'{index:03}_bulk_recycle.json.gz')
    for arm in ('alight','continue'):
        for key in ('lower','upper','unresolved_probability'):
            assert abs(result['actions'][arm][key]-original['actions'][arm][key]) < 1e-8
    for k in ('charged_compiles','onboard_compiles','shared_policy_compiles','board_compiles','bulk_platforms'):
        assert result['counters'][k] == original['counters'][k]
    inverse = {n:key for key,n in oracle.nodes.items()}
    coefficients = {}
    for arm,root in oracle._roots.items():
        try:
            _, coefficients[arm] = frontier(root, inverse)
        except ValueError:
            pass
    expansions = []
    for n, count in oracle.expansion_log:
        expansions.append(dict(**describe(inverse[n], n), count=count,
             final_root_coefficients={a:co.get(n,0.) for a,co in coefficients.items()}))
    out = dict(index=index,root_index=state['root_index'], archived_values_reproduced=True,
        scope='Same fixed kernel, order, policy and 2000 charged entries. Instrumented read-only diagnosis; no timing or new policy evidence.',
        original_actions=original['actions'], original_difference=original['difference_interval'],
        checkpoints=oracle.frontiers, expansions=expansions)
    path=OUT/f'root{state["root_index"]}_frontier.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(index,state['root_index'],result['difference_interval'],path,flush=True)
