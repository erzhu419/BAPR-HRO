from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v34.conditional_reference import summarize_pairs
from experiments.reconstruction_v38.run import bank_science
from experiments.reconstruction_v40 import run


def test_seed_assignment_preserves_complete_old_state_and_uses_only_public_event_for_new_states():
    state = dict(root_index=4, cid=31, clock=250., count=128,
        public_history=[dict(cid=32, evidence='forecast', predicted_time=300.)],
        actions=dict(base='alight', alternative='continue'))
    manifest = [dict(state=deepcopy(state), seed=37051, source='reference_cases/004_001.json.gz')]
    assert run.seed_for(state, manifest) == dict(seed=37051,
        source='reference_cases/004_001.json.gz', seed_origin='original_manifest')
    revised = deepcopy(state)
    revised['public_history'][0]['predicted_time'] = 320.
    expected = int(np.random.SeedSequence([40050, 4, 31]).generate_state(1)[0])
    assert run.seed_for(revised, manifest) == dict(seed=expected, source=None, seed_origin='public_event_rule')
    revised['clock'] = 270.
    revised['actions']['base'] = 'continue'
    assert run.seed_for(revised, manifest)['seed'] == expected
    revised['cid'] = 33
    assert run.seed_for(revised, manifest)['seed'] != expected
    assert manifest[0]['state'] == state


def test_complete_control_cpu_partitions_cold_incoming_decision_and_final_work(monkeypatch):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(run, 'process_time', lambda: clock.now)
    monkeypatch.setattr(run, 'perf_counter', lambda: clock.now)

    def setup(graph, fit):
        clock.now += 2.  # Charged cold construction before the first decision.
        return SimpleNamespace(), {}

    class Engine:
        stats = dict(requests=0)

        def onboard(self, *args, **kwargs):
            clock.now += 4.
            self.stats['requests'] += 1
            return True, dict(method='test')

        def platform(self, *args, **kwargs):
            clock.now += 2.
            self.stats['requests'] += 1
            return 6, dict(method='test')

    def execute(self, ride, start, history, world):
        clock.now += 1.  # First observed physical prefix.
        assert self.choose_onboard(ride, 5, 100., history)[0]
        clock.now += 5.  # Previous trace plus incoming observations/messages.
        assert self.choose_platform(SimpleNamespace(stop=2), 100., 102., history,
                                    (6,), {6: 110.}, 110.)[0] == 6
        clock.now += 3.  # Final ride and arrival after the last action.
        return dict(loss=20., status='arrived', trace=[])

    monkeypatch.setattr(run, 'setup', setup)
    monkeypatch.setattr(run, 'root_ride', lambda *args: SimpleNamespace(trips=lambda model: set()))
    monkeypatch.setattr(run, 'FastPlanner', lambda *args: SimpleNamespace(stats={}))
    monkeypatch.setattr(run, 'SingleMessageHistory', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(run, 'relevant_frames', lambda *args: [])
    monkeypatch.setattr(run, 'engine_for', lambda *args: Engine())
    monkeypatch.setattr(run.TimedExecutor, 'run', execute)
    result = run.one(0, 'cheap_frontier__combined_rolling', None, {}, {},
        dict(deadline=200., start=0.), {}, {}, [],
        dict(cheap_methods={'combined_rolling': ['rolling_fixed', 'combined']}))
    assert [row['process_cpu'] for row in result['decision_costs']] == [7., 7.]
    assert result['cost']['final_execution_cpu'] == 3.
    assert result['cost']['total_process_cpu'] == 17.
    assert result['cost']['decision_cpu_sum']+result['cost']['final_execution_cpu'] == 17.
    assert result['saved_answers_reused'] == 0


def test_decision_interval_reports_actual_overrun_without_resetting_away_incoming_work(monkeypatch):
    clock = SimpleNamespace(now=1.2)
    monkeypatch.setattr(run, 'process_time', lambda: clock.now)
    executor = run.TimedExecutor(None, None, 100., engine=None, cpu_start=1., cpu_budget=.15)
    executor.finish_decision('onboard', 5, 20., gate=True)
    first = executor.decision_costs[0]
    assert first['process_cpu'] == pytest.approx(.2)
    assert not first['within_budget'] and first['overrun'] == pytest.approx(.05)
    assert first['gate'] and first['cid'] == 5
    clock.now = 1.3
    executor.finish_decision('platform', 2, 22.)
    assert executor.decision_costs[1]['process_cpu'] == pytest.approx(.1)
    assert executor.decision_costs[1]['within_budget']


def test_prefix_science_uses_first_128_rows_and_recomputes_their_gate_without_scheduler_metadata():
    rows = {arm: [dict(loss=(1. if i < 128 else 99.) if arm == 'alternative' else 10.,
        status='arrived', trace=[dict(kind='onboard', cid=3, choice=arm)],
        decisions=dict(onboard=1), world_index=i, block_index=i//128, committed=True)
        for i in range(512)] for arm in ('alternative', 'base')}
    rows['alternative'][7].update(loss=None, status='projection_incompatible:test')
    result = dict(rows=rows, state=dict(clock=0., deadline=100.), seed=37050, count=512,
        actions=dict(base='alight', alternative='continue'), selected_action='alight',
        public_history=[dict(cid=3, evidence='observed', clock=0.)])
    saved = deepcopy(result)
    paired = summarize_pairs([r['loss'] for r in rows['alternative'][:128]],
                             [r['loss'] for r in rows['base'][:128]], 100.)
    paired['orientation'] = 'alternative action loss minus base action loss; negative favors alternative'
    expected = bank_science(dict(result, rows={arm: values[:128] for arm, values in rows.items()},
        paired=paired, count=128, selected_action='continue'))
    assert run.prefix_science(result, 128) == expected
    assert expected['paired']['unresolved_pairs'] == 1
    assert result == saved
