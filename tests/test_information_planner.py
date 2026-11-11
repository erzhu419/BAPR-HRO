"""Exact causal-reference regressions with analytically known journey costs."""
from dataclasses import replace

import pytest

from src.information_planner import ExactPlanner, FiniteScenarioModel, Outcome, Scenario


def timing_model(relationship, early):
    pairs = {"opposite": [(10, 50), (50, 10)],
             "independent": [(10, 10), (10, 50), (50, 10), (50, 50)],
             "shared": [(10, 10), (50, 50)]}[relationship]
    scenarios = tuple(Scenario(str(i), 1 / len(pairs), {"A": a, "B": b})
                      for i, (a, b) in enumerate(pairs))

    def actions(state):
        return {"origin": ("direct", "hub"), "hub": ("A", "B")}.get(state, ())

    def transition(world, state, action):
        if state == "origin":
            if action == "direct":
                return Outcome("destination", None, 25, True)
            message = tuple(world.factors.values()) if early else None
            return Outcome("hub", message, 2)
        return Outcome("destination", None, world.factors[action], True)

    return FiniteScenarioModel(scenarios, "origin", actions, transition, 2)


@pytest.mark.parametrize("relationship,early,expected", [
    ("opposite", True, 12), ("independent", True, 22),
    ("shared", True, 25), ("opposite", False, 25),
])
def test_correlation_and_announcement_timing(relationship, early, expected):
    model = timing_model(relationship, early)
    full = ExactPlanner(model)
    restricted = ExactPlanner(model, lambda node: ("A",) if node.state == "hub"
                              else model.actions(node.state))
    assert full.solve().value == pytest.approx(expected)
    assert restricted.solve().value == 25
    certificate = restricted.certificate(restricted.optimal_action)
    assert certificate.support_loss == pytest.approx(25 - expected)
    assert certificate.within_candidate_loss == 0
    assert restricted.decision_loss(restricted.optimal_action) == 0
    direct_policy = lambda node: "direct" if node.state == "origin" else "A"
    assert full.evaluate(direct_policy) == 25
    assert full.decision_loss(direct_policy) == pytest.approx(25 - expected)


def test_within_candidate_loss_uses_restricted_continuation():
    edges = {"origin": {"u": (1, "u"), "v": (2, "v")},
             "u": {"fast": (1, "destination"), "slow": (5, "destination")},
             "v": {"fast": (1, "destination"), "slow": (4, "destination")}}
    model = FiniteScenarioModel((Scenario("fixed", 1),), "origin",
                                lambda state: tuple(edges.get(state, {})),
                                lambda world, state, action:
                                Outcome(edges[state][action][1], None,
                                        edges[state][action][0]), 2)
    restricted = ExactPlanner(model, lambda node: ("slow",) if node.state in ("u", "v")
                              else model.actions(node.state))
    policy = lambda node: "v" if node.state == "origin" else "slow"
    certificate = restricted.certificate(policy)
    assert (certificate.full_optimum, certificate.restricted_optimum,
            certificate.policy_cost) == (2, 6, 6)
    assert certificate.support_loss == 4
    assert certificate.within_candidate_loss == restricted.decision_loss(policy) == 0


def test_hidden_future_is_not_available_to_policy_and_information_bound_is_distinct():
    planner = ExactPlanner(timing_model("opposite", False))
    assert planner.solve().value == 25
    assert planner.clairvoyant_lower_bound() == 12
    # A causal policy can use the message only after it is returned by hub entry.
    assert planner.root.history == ()
    assert planner.root.belief == (0.5, 0.5)
    assert planner.evaluate(planner.optimal_action) == 25


def test_incremental_dependency_closure_matches_fresh_solve_and_reuses_other_branch():
    factors = {"left": 1, "right": 10}
    model = FiniteScenarioModel((Scenario("world", 1, factors),), "origin",
                                lambda state: ("left", "right") if state == "origin"
                                else (("finish",) if state in factors else ()),
                                lambda world, state, action:
                                Outcome(action, None, 0) if state == "origin" else
                                Outcome("destination", None, world.factors[state], True),
                                2, dependencies=lambda state, action:
                                {state} if state in factors else set(),
                                terminal_dependencies=lambda state: set())
    planner = ExactPlanner(model)
    assert planner.solve().value == 1
    factors["left"] = 20
    report = planner.incremental_update({"left"})
    assert report["retained_nodes"] > 0
    assert planner.solve().value == ExactPlanner(model).solve().value == 10


def test_equivalent_action_compression_and_later_broken_equivalence():
    factors = {"A": 1, "B": 1}
    model = FiniteScenarioModel((Scenario("world", 1, factors),), "origin",
                                lambda state: ("A", "B") if state == "origin" else (),
                                lambda world, state, action:
                                Outcome("destination", None, world.factors[action], True),
                                1, dependencies=lambda state, action: {action},
                                terminal_dependencies=lambda state: set())
    compressed = ExactPlanner(model)
    assert compressed.solve_compressed().value == ExactPlanner(model).solve().value == 1
    assert compressed.stats["equivalent_actions_merged"] == 1
    factors["B"] = 0
    compressed.incremental_update({"B"})
    assert compressed.solve_compressed().action == "B"
    assert compressed.solve().value == ExactPlanner(model).solve().value == 0


def test_changed_priors_require_rebuilding_root_posterior():
    model = timing_model("opposite", False)
    planner = ExactPlanner(model)
    assert planner.solve().value == 25
    model.scenarios = (replace(model.scenarios[0], probability=0.9),
                       replace(model.scenarios[1], probability=0.1))
    planner.invalidate_all()
    assert planner.root.belief == (0.9, 0.1)
    assert planner.solve().value == pytest.approx(16)


def test_rejects_noncausal_candidate_menu():
    model = timing_model("opposite", False)
    with pytest.raises(ValueError, match="nonempty physical-action subset"):
        ExactPlanner(model, lambda node: ("teleport",)).solve()
