"""The first-offer formula must exclude every earlier legal competitor."""

import numpy as np
import pytest

from experiments.bounded_gate_5d_v1.pair_contract import PairPlatform, PairMatchedPlanner
from test_v46_shared import shared_case


@pytest.mark.parametrize("clocks,expected", [((20., 10.), 110.), ((20., 20.), 100.)])
def test_first_offer_includes_lower_ranked_earlier_departure(monkeypatch, clocks, expected):
    graph, model, params, history, platform, entry = shared_case(
        monkeypatch, forecasts=False, cancel=0.0
    )
    ids = sorted(platform.rides)[:2]
    platform = PairPlatform(
        platform.stop, {cid: platform.rides[cid] for cid in ids},
        transfer_times={cid: 0. for cid in ids},
    )
    planner = PairMatchedPlanner(graph, model, params, 300.)

    class Compiler:
        def departure_distribution(self, cid):
            return np.array([clocks[ids.index(cid)]]), np.ones(1), 0.

        def clock_nominal_scores(self, cid, times):
            return np.full_like(times, 100. + 10. * ids.index(cid))

    monkeypatch.setattr(planner, "platform_compiler", lambda *args: Compiler())
    monkeypatch.setattr(planner, "ranks", lambda platform: dict(zip(ids, (0, 1))))
    result = planner.first_offer_entry_values(platform, history, np.array([0.]))
    assert result.tolist() == [expected]
