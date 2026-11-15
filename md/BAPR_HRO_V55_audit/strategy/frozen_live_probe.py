"""Small archival V51 diagnostic: five roots x two forced actions x two streams.

Only the first onboard action is forced. Thereafter the exact V51 cheap W
controller is used. 'frozen' keeps all already received information and local
observations, removes future remote frames, and never advances their clock.
The full original ideal logical publication/execution contract remains.

No fitting, no new stochastic worlds, no claim of population performance.
"""
import gzip
import json
import os
import sys
from pathlib import Path
from time import process_time, perf_counter

REPO = Path(os.environ['BAPR_HRO_REPO']).resolve()
sys.path.insert(0, str(REPO))
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v37.controller import has_legal_continue
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v50.run import actions
from experiments.reconstruction_v51.run import PolicyExecutor

HERE = Path(__file__).resolve().parent
ROOTS = (4, 6, 18, 32, 62)


class ForcedFirstCheap(PolicyExecutor):
    def __init__(self, *args, force_first, stream, **kwargs):
        super().__init__(*args, label='cheap', rollout_limits={}, **kwargs)
        self.force_first, self.stream = force_first, stream
        self.first_done = False
        self.first_public = None
        self.first_clock = None

    def decision_history(self, history, clock):
        if self.stream == 'frozen' and self.first_done:
            assert not history.frames
            return dict(message_receipt=dict(clock=float(clock), frames_consumed=0,
                        receipts=[], first_frame_index=0, next_frame_index=0),
                        public_history=history.public_state(),
                        message_state=dict(frame_index=0, clock=history.message_clock))
        return super().decision_history(history, clock)

    def choose_onboard(self, node, cid, clock, history):
        state = self.state(node, history, kind='onboard', cid=cid, clock=float(clock),
            legal_actions=['alight', 'continue'], has_continuation=has_legal_continue(node, cid))
        alight, receipt = self.control.onboard(node, cid, clock, history)
        base = 'alight' if alight else 'continue'
        selected = self.force_first if not self.first_done else base
        if not self.first_done:
            assert has_legal_continue(node, cid)
            self.first_public = state['public_history']
            self.first_clock = float(clock)
            self.first_done = True
            if self.stream == 'frozen':
                # These are the same publication fields cleared by freeze_snapshot.
                # Keep law caches: this experiment changes messages, not cost setup.
                history.frames, history.frame_index = (), 0
                history.message_clock = float(clock)
        selected = self.publish(clock, ('alight', 'continue'), selected)
        self.done('onboard', clock)
        state.update(selected_action=selected, base_action=base)
        return selected == 'alight', dict(policy='forced_first_then_cheap',
            base=base, selected=selected, scoring=receipt)


def event_signature(outcome):
    keys = ('kind','cid','stop','entry','ready','clock','choice',
            'visible','observed','observed_visible','observation_cids','public_history')
    # Stored JSON uses lists/string dictionary keys; new traces use tuples/int
    # keys. Canonicalize representation only before exact structural equality.
    return json.loads(json.dumps([{k:t[k] for k in keys if k in t} for t in outcome['trace']]))


def main():
    tick, cpu = perf_counter(), process_time()
    _, data, messages, fit, _ = inputs()
    emission = read(REPO/'experiments/reconstruction_v35/fitted_emission.json')
    preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick,
        scope='Shared resident input loading; dated graph timings separately recorded.')
    graphs, rows, graph_costs = {}, [], {}
    target = HERE/'frozen_live_receipts'
    target.mkdir(exist_ok=True)
    for root_index in ROOTS:
        old = {lab:read(REPO/f'experiments/reconstruction_v51/roots/{root_index:03}_{lab}.json.gz')
               for lab in ('cheap','full128')}
        menu = read(REPO/f'experiments/reconstruction_v50/menus/{root_index:03}.json.gz')
        root = menu['wide_root']
        if root['day'] not in graphs:
            tick, cpu = perf_counter(), process_time()
            graphs[root['day']] = context(data, root['day'])
            graph_costs[root['day']] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        graph, _, _, mapping, metadata, _ = graphs[root['day']]
        forced = {label:old[label]['actions'][0]['choice'] for label in old}
        assert forced['cheap'] != forced['full128']
        assert all(g['selected_action']==g['base_action'] for g in old['full128']['gates'][1:])
        arms = {}
        # Validate both live arms before doing this root's retrospective ablation.
        for stream in ('live','frozen'):
            for label in ('cheap','full128'):
                tick, cpu = perf_counter(), process_time()
                model, parameters = setup(graph,fit)
                node = root_ride(graph, root)
                observations = ObservationContract.from_menu(node)
                frames = relevant_frames(messages['dates'][root['day']]['frames'], node.trips(model),mapping)
                history = PhysicalHistory(model,parameters,fit=emission,frames=frames,source_to_local=mapping)
                planner = MatchedContinuationPlanner(graph,model,parameters,root['deadline'])
                executor = ForcedFirstCheap(graph,planner,root['deadline'],
                    force_first=forced[label],stream=stream,root_index=root_index,
                    cpu_start=cpu,observation_contract=observations)
                outcome = executor.run(node,root['start'],history,ArchivedWorld(metadata))
                cost = dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick,
                    cached_menu_cost=menu['cost'], graph_preparation=graph_costs[root['day']])
                first_equal = executor.first_public == old['cheap']['states'][0]['public_history']
                assert first_equal
                validation = dict(first_public_history_equal=first_equal)
                if stream == 'live':
                    validation.update(loss_equal=outcome['loss']==old[label]['outcome']['loss'],
                        status_equal=outcome['status']==old[label]['outcome']['status'],
                        action_sequence_equal=actions(outcome)==old[label]['actions'],
                        all_physical_and_public_trace_fields_equal=event_signature(outcome)==event_signature(old[label]['outcome']))
                    if not all(validation.values()):
                        raise AssertionError(f'Live replay mismatch {root_index}/{label}: {validation}')
                key = f'{stream}_{label}'
                receipt = dict(root_index=root_index,day=root['day'],label=key,
                    first_action=forced[label],first_clock=executor.first_clock,
                    outcome=outcome,actions=actions(outcome),states=executor.states,
                    publications=executor.publications,decision_costs=executor.decision_costs,
                    cost=cost,validation=validation,
                    scope='First-action intervention followed by V51 cheap W. Ideal logical clocks; CPU does not delay actions. Actual archival physical events only.')
                outfile = target/f'{root_index:03}_{key}.json.gz'
                with gzip.open(outfile,'wt') as f:json.dump(receipt,f,ensure_ascii=False)
                arms[key] = dict(first_action=forced[label],loss=outcome['loss'],status=outcome['status'],
                    cost=cost,validation=validation,source=str(outfile.relative_to(HERE)),
                    actions=actions(outcome))
                print(root_index,key,outcome['loss'],round(cost['process_cpu'],5),flush=True)
        live_gain = arms['live_cheap']['loss']-arms['live_full128']['loss']
        frozen_gain = arms['frozen_cheap']['loss']-arms['frozen_full128']['loss']
        rows.append(dict(root_index=root_index,day=root['day'],origin=root['origin_name'],goal=root['goal_name'],
            arms=arms,live_alternative_gain=live_gain,frozen_alternative_gain=frozen_gain,
            message_interaction=live_gain-frozen_gain,
            live_vs_frozen={label:dict(loss_equal=arms[f'live_{label}']['loss']==arms[f'frozen_{label}']['loss'],
                actions_equal=arms[f'live_{label}']['actions']==arms[f'frozen_{label}']['actions'])
                for label in ('cheap','full128')}))
    result = dict(commit='a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491',
        scope='Five selected existing development roots x first action x future remote stream; all continuations cheap W. No stochastic model/parameter change, no new out-of-sample dates, no expectation or calibration claim.',
        actual_stream_definition='Consume only frames published by each physical event clock.',
        frozen_stream_definition='Keep all information received before the first onboard action plus all future local physical observations; remove only future remote publications.',
        execution_contract='Original ideal logical publisher; CPU/wall costs recorded, never shifted physical clocks.',
        preparation=preparation,graph_preparation=graph_costs,queries=20,
        live_validation='All 10 live arms reproduce original loss, status, actions, every saved physical/public trace field.',rows=rows)
    (HERE/'frozen_live_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(root_index=r['root_index'],live=r['live_alternative_gain'],
        frozen=r['frozen_alternative_gain'],message_interaction=r['message_interaction'],
        equality=r['live_vs_frozen']) for r in rows],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
