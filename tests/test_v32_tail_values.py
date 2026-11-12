import numpy as np
import pytest

from experiments.reconstruction_v27.evaluate import integrate as independent_integrate
from experiments.reconstruction_v32.tail_values import integrate
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import (SchedulePriority, SingleTargetOpportunity,
                                      CombinedWaitingOpportunity, TwoTargetOpportunity)


class Scores:
    def __init__(self, laws, costs):
        self.laws, self.costs = laws, costs

    def departure_distribution(self, cid):
        times, mass, cancel = self.laws[cid]
        return np.array(times), np.array(mass), cancel

    def clock_nominal_score(self, cid, clock):
        return self.costs[cid]+clock/10


@pytest.mark.parametrize('entry,ready', [(0, 0), (5, 15), (10, 11), (22, 22)])
def test_integrated_controls_match_separate_public_branch_evaluator(entry, ready):
    laws = {5: ([0, 10, 20], [.2, .4, .3], .1),
            8: ([8, 20, 35], [.2, .5, .2], .1),
            19: ([20, 30, 60], [.4, .3, .2], .1)}
    reference = FullMenuOpportunity(Scores(laws, {5: 30., 8: 40., 19: 20.}),
        {5: (3, 5), 8: (1, 8), 19: (2, 19)}, entry, ready, 50)
    schedules = {5: 25, 8: 30, 19: 35}
    report, table = integrate(reference, schedules)
    single = SingleTargetOpportunity(reference)
    combined = CombinedWaitingOpportunity(single)
    controls = dict(one_step=reference, single=single, combined=combined,
        pair=TwoTargetOpportunity(combined), exact=table['dp'], nominal=SchedulePriority(schedules))
    dp = table['dp']
    independent_report, independent_tables = independent_integrate(reference, controls,
        dict(times=dp.times, supported=dp.supported, dp=dp.values))
    for method, controller in controls.items():
        assert report['expected_arrival'][method] == pytest.approx(independent_report['expected_arrival'][method])
        assert np.allclose(table['values'][method][dp.supported], independent_tables[method][dp.supported])
        for j, clock in enumerate(dp.times):
            for remaining, present in np.argwhere(table['choices'][method][j] != -2):
                index = table['choices'][method][j, remaining, present]
                visible = tuple(cid for i, cid in enumerate(reference.cids) if present & (1 << i))
                seen = tuple(cid for i, cid in enumerate(reference.cids) if not remaining & (1 << i))
                action = controller.choose(visible, seen, clock)[0] if visible and clock >= ready else None
                assert action == (None if index < 0 else reference.cids[index])


def test_pair_integrator_keeps_the_complementary_refusal_example():
    laws = {0: ([5], [1.], 0.), 1: ([20], [.5], .5), 2: ([10, 30], [.5, .5], 0.)}
    reference = FullMenuOpportunity(Scores(laws, {0: 55., 1: 18., 2: 68.}),
                                   {cid: (cid, cid) for cid in laws}, 0, 0, 100)
    report, table = integrate(reference)
    assert report['expected_arrival']['pair'] == pytest.approx(report['expected_arrival']['exact'])
    assert report['expected_arrival']['combined'] > report['expected_arrival']['pair']
    assert table['choices']['pair'][0, 6, 1] == -1
    assert table['choices']['combined'][0, 6, 1] == 0


def test_empty_and_exhausted_menu_keep_deadline_values():
    for laws in ({}, {1: ([5], [1.], 0.)}):
        reference = FullMenuOpportunity(Scores(laws, {1: 10.}),
                                       {cid: (cid, cid) for cid in laws}, 10, 10, 50)
        summary, tables = integrate(reference)
        assert set(summary['expected_arrival'].values()) == {50.}
        assert len(tables['times']) == 0
