"""Test Durner's TopoCSA on synthetic network."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.synthetic_network import create_bus_story_network, create_regime_distributions
from src.durner.topocsa import topocsa


def test_basic_routing():
    """Test A→D routing on normal regime."""
    g = create_bus_story_network()
    print(f"Network: {g.summary()}")

    # Route from A(0) to D(9), departing at 8:00 (480 min)
    t0 = time.time()
    result = topocsa(g, s_source=0, s_dest=9, t_source=480)
    elapsed = time.time() - t0

    print(f"\nA→D, depart 8:00, normal regime:")
    print(f"  Mean arrival: {result.mean_arrival:.1f} min ({int(result.mean_arrival)//60}:{int(result.mean_arrival)%60:02d})")
    print(f"  Hyperpath connections: {len(result.hyperpath_connections)}")
    print(f"  Connections processed: {result.n_connections_processed}")
    print(f"  Cycles cut: {len(result.cuts)}")
    print(f"  Runtime: {elapsed*1000:.1f} ms")

    assert result.mean_arrival < float('inf'), "Should find a route A→D"
    # Direct 402 takes ~45 min from 8:00 → ~8:45 (525 min)
    # Stochastic mean is higher than deterministic due to delay distributions
    # 402 direct: ~525 deterministic + delay margin → expect ~530-580
    assert result.mean_arrival < 580, f"Mean arrival too late: {result.mean_arrival}"

    # Check alternatives at source stop A
    src_labels = result.stop_labels.get(0, [])
    print(f"\n  Alternatives at A (stop 0): {len(src_labels)} options")
    for lab in src_labels[-5:]:  # show best 5
        c = g.connections[lab.connection_id]
        print(f"    Route {c.route}, dep {c.dep_time}, mean arr {lab.mean_dest_arrival:.1f}")

    # Check alternatives at transfer stop B
    b_labels = result.stop_labels.get(3, [])
    print(f"\n  Alternatives at B (stop 3): {len(b_labels)} options")
    for lab in b_labels[-5:]:
        c = g.connections[lab.connection_id]
        print(f"    Route {c.route}, dep {c.dep_time}, mean arr {lab.mean_dest_arrival:.1f}")

    print("\n[PASS] basic_routing")


def test_disrupted_402():
    """Test routing when 402 is disrupted — should prefer transfer routes."""
    g = create_bus_story_network()

    # Apply disrupted_402 regime
    dists = create_regime_distributions("disrupted_402")
    g.assign_distributions(dists)
    # Set cancel prob on 402 connections
    for c in g.connections:
        if c.route == "402":
            c.cancel_prob = 0.7

    result_disrupted = topocsa(g, s_source=0, s_dest=9, t_source=480)

    # Compare with normal regime
    g2 = create_bus_story_network()
    result_normal = topocsa(g2, s_source=0, s_dest=9, t_source=480)

    print(f"\nNormal regime:    mean arrival = {result_normal.mean_arrival:.1f}")
    print(f"Disrupted 402:    mean arrival = {result_disrupted.mean_arrival:.1f}")

    # Under disruption, transfer routes should still find a way
    assert result_disrupted.mean_arrival < float('inf'), "Should still find a route"

    # Check that transfer routes are used at stop A
    src_labels = result_disrupted.stop_labels.get(0, [])
    routes_used = set()
    for lab in src_labels:
        c = g.connections[lab.connection_id]
        routes_used.add(c.route)
    print(f"  Routes in hyperpath at A: {routes_used}")
    assert "102" in routes_used, "102 should be in hyperpath when 402 is disrupted"

    print("[PASS] disrupted_402")


def test_different_departure_times():
    """Test routing at different times of day."""
    g = create_bus_story_network()

    for t_dep in [400, 480, 600, 720]:
        result = topocsa(g, s_source=0, s_dest=9, t_source=t_dep)
        h, m = divmod(t_dep, 60)
        print(f"  Depart {h}:{m:02d} → mean arrival {result.mean_arrival:.1f} "
              f"({len(result.hyperpath_connections)} conns in hyperpath)")
        if t_dep <= 900:  # within service hours
            assert result.mean_arrival < float('inf')

    print("[PASS] different_departure_times")


if __name__ == "__main__":
    test_basic_routing()
    test_disrupted_402()
    test_different_departure_times()
    print("\nAll TopoCSA tests passed!")
