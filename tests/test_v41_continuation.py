from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.execute import DrawnWorld
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v39.state import DecisionEngine
from experiments.reconstruction_v41.continuation import (
    MeetingExecutor, empirical_pair_summary, evaluate, science,
)
from test_v38_evaluator import fixture


def test_single_pair_uncertainty_is_not_reported_as_zero():
    result = empirical_pair_summary([10.], [20.], 100.)
    assert result['complete_pair_statistics']['se'] is None
    assert result['complete_pair_statistics']['ci95'] is None
    assert result['full_bank_mean_gain_interval'] == [-10., -10.]


def test_ordinary_suffix_reuse_preserves_all_paired_outcomes():
    graph, model, parameters, node, history = fixture()
    args = (graph, model, parameters, 100, node, dict(kind='onboard', cid=0, clock=10), history)
    plain = evaluate(*args, planner_type=FastPlanner, count=16, seed=7)
    meeting = evaluate(*args, planner_type=FastPlanner, count=16, seed=7, meeting=True)
    assert science(plain) == science(meeting)
    assert meeting['meeting_stats']['checks'] > 0


def test_identical_full_entry_reuses_suffix_but_retains_trace_and_counts():
    graph, model, parameters, node, history = fixture()
    planner = FastPlanner(graph, model, parameters, 100)
    engine = DecisionEngine(planner)
    bank = TripWorlds(history.as_belief(), parameters, node.paths(), 128, 7)
    suffixes = {}
    stats = dict(checks=0, hits=0, avoided_decisions=0, check_cpu=0.)
    results = []
    for _ in range(2):
        executor = MeetingExecutor(graph, planner, 100, engine=engine,
            suffixes=suffixes, meeting_stats=stats)
        results.append(executor.forced(node, dict(kind='onboard', cid=0, clock=10),
            'alight', history.clone(), DrawnWorld(bank, 0)))
    assert stats['hits'] > 0
    for key in ('loss', 'status', 'trace', 'decisions'):
        assert results[0][key] == results[1][key]
