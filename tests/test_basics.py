"""Basic tests for data structures and synthetic network."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.pmf import PMF, componentwise_sum, convolve_pmfs, prob_reachable
from src.transit_graph import TransitGraph, Stop, Connection
from src.synthetic_network import create_bus_story_network, create_regime_distributions


def test_pmf_basic():
    p = PMF.deterministic(480)
    assert abs(p.mean() - 480.0) < 1e-6
    assert abs(p.total_prob - 1.0) < 1e-6
    assert p.prob_le(479) == 0.0
    assert abs(p.prob_le(480) - 1.0) < 1e-6
    print("[PASS] PMF basic")


def test_pmf_from_delays():
    delays = np.array([0.1, 0.3, 0.4, 0.2])
    p = PMF.from_delays(scheduled=480, delay_probs=delays, delay_offset=-1)
    assert p.offset == 479
    assert abs(p.mean() - (479 * 0.1 + 480 * 0.3 + 481 * 0.4 + 482 * 0.2)) < 1e-6
    print("[PASS] PMF from_delays")


def test_componentwise_sum():
    a = PMF(probs=np.array([0.3, 0.7]), offset=10)
    b = PMF(probs=np.array([0.5, 0.5]), offset=11)
    c = componentwise_sum(a, b)
    assert c.offset == 10
    assert len(c.probs) == 3
    assert abs(c.probs[0] - 0.3) < 1e-6   # t=10: only a
    assert abs(c.probs[1] - 1.2) < 1e-6   # t=11: 0.7 + 0.5
    assert abs(c.probs[2] - 0.5) < 1e-6   # t=12: only b
    print("[PASS] componentwise_sum")


def test_convolve():
    a = PMF.deterministic(10)
    b = PMF.deterministic(5)
    c = convolve_pmfs(a, b)
    assert abs(c.mean() - 15.0) < 1e-6
    print("[PASS] convolve_pmfs")


def test_prob_reachable():
    arr = PMF.deterministic(100)  # arrive at t=100
    dep = PMF.deterministic(105)  # departs at t=105
    # transfer_time=3: need arr+3 <= dep → 103 <= 105 → reachable
    assert abs(prob_reachable(arr, dep, 3) - 1.0) < 1e-6
    # transfer_time=6: need 106 <= 105 → not reachable
    assert abs(prob_reachable(arr, dep, 6) - 0.0) < 1e-6
    print("[PASS] prob_reachable")


def test_synthetic_network():
    g = create_bus_story_network()
    print(f"  {g.summary()}")
    # Check structure
    assert len(g.stops) == 19
    assert len(g.connections) > 0
    # Check transfer stops
    transfers = g.get_transfer_stops()
    transfer_names = [g.stops[s].name for s in transfers]
    assert 0 in transfers, f"Stop A should be a transfer stop, got {transfer_names}"
    assert 3 in transfers, f"Stop B should be a transfer stop"
    assert 6 in transfers, f"Stop C should be a transfer stop"
    assert 9 in transfers, f"Stop D should be a transfer stop"
    # Check routes at transfer stop B
    routes_at_b = g.get_routes_at_stop(3)
    assert "402" in routes_at_b
    assert "102" in routes_at_b
    assert "311" in routes_at_b
    assert "317" in routes_at_b
    # Check distributions are assigned
    for c in g.connections[:5]:
        assert c.dep_distribution is not None
        assert c.arr_distribution is not None
    print("[PASS] synthetic_network")


def test_regime_distributions():
    for regime in ["normal", "disrupted_402", "rush_hour", "weather"]:
        dists = create_regime_distributions(regime)
        assert "402" in dists
        assert "102" in dists
        for route, info in dists.items():
            assert abs(info["delay_probs"].sum() - 1.0) < 0.01 or regime == "disrupted_402"
    # Check 402 disruption
    d402 = create_regime_distributions("disrupted_402")
    assert d402["402"].get("cancel_prob", 0) > 0.5
    print("[PASS] regime_distributions")


if __name__ == "__main__":
    test_pmf_basic()
    test_pmf_from_delays()
    test_componentwise_sum()
    test_convolve()
    test_prob_reachable()
    test_synthetic_network()
    test_regime_distributions()
    print("\nAll tests passed!")
