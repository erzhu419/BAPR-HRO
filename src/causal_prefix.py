"""A finite-bank causal prefix followed by a supplied clairvoyant lower tail."""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import inf


@dataclass(frozen=True)
class CausalPrefixValue:
    value: float
    clairvoyant_root_mean: float
    elapsed_horizon: int
    bank_size: int
    information_nodes: int
    transition_evaluations: int
    oracle_evaluations: int
    boundary_support_mass: dict
    boundary_kind_mass: dict
    singleton_boundary_probability: float
    singleton_before_boundary_probability: float
    expected_boundary_elapsed: float
    factor_order: tuple = ()
    boundary_factor_support_mass: dict = field(default_factory=dict)
    factor_singleton_boundary_probability: dict = field(default_factory=dict)
    factor_singleton_before_boundary_probability: dict = field(default_factory=dict)
    action_branches_pruned: int = 0


class PrefixBudgetExceeded(RuntimeError):
    """The exact prefix solve exhausted its declared information-node budget."""

    def __init__(self, limit, nodes, transitions, action_branches_pruned=0):
        self.limit, self.nodes, self.transitions = limit, nodes, transitions
        self.action_branches_pruned = action_branches_pruned
        super().__init__(f'Causal prefix exceeded {limit} nodes ({transitions} transitions)')


def causal_prefix_lower(model, oracle_terminal, *, elapsed_horizon, max_nodes=None,
                        stop_when=None, factor_labels=None, trace=None, action_outcomes=None,
                        singleton_exact=None, prune_with_oracle=False):
    """Solve every platform action until the first ride or a fixed clock limit.

    ``oracle_terminal(world_index, platform_state)`` returns remaining loss of
    a valid clairvoyant relaxation from that physical state. The clock limit
    is fixed before drawing the bank. The model is the V11 integer-clock
    platform model; action restrictions and history-dependent candidate rules
    are absent. Bank worlds have deterministic transitions and observations.

    State plus surviving world indices is sufficient: posterior probabilities
    are the original probabilities renormalized on those indices. Before the
    first ride, OBSERVE and ALIGHT occur at most once, and other actions advance
    time. Thus no action-stage budget or full history is needed in the cache.

    The result is exact for this finite-bank prefix relaxation. Over IID banks
    its expectation is a lower bound for the population causal optimum; it is
    not the population optimum or an exact information-gap decomposition.

    ``stop_when`` may replace first-ride stopping with another predicate of the
    public physical state. With more rides allowed, zero-time rides strictly
    grow used_connections and ALIGHT changes onboard state; other nonterminal
    actions advance time. The finite used-edge set still excludes state cycles.
    Optional factor labels record which empirical factor atoms remain possible.
    An optional ``trace(state, indices, action_values, boundary_kind)`` callback
    sees each information node reached by the selected policy. Action values
    follow model action order; ties select the first minimum. At a boundary
    the action values are empty and ``boundary_kind`` identifies the stop.
    A model-specific ``action_outcomes(state, indices, action)`` may provide
    support-aware transitions, yielding one (index, outcome) per supported world.
    ``singleton_exact(index, state)`` may certify an exact continuation value,
    or return None to keep expanding the only remaining world.
    ``prune_with_oracle`` orders actions by a lower bound on their value and
    skips those that cannot beat an evaluated action. The oracle must be a
    lower bound for this objective at every child state, including the
    declared prefix contract. Tracing always computes all action rankings.
    """
    if not isinstance(elapsed_horizon, int) or elapsed_horizon < 1:
        raise ValueError('Use a positive integer clock horizon fixed before sampling')
    root = model.initial_state
    cutoff = root.time + elapsed_horizon
    root_used = root.used_connections
    probabilities = tuple(world.probability for world in model.scenarios)
    support = tuple(i for i, p in enumerate(probabilities) if p > 0.)
    factor_labels = {} if factor_labels is None else factor_labels
    factor_order = tuple(factor_labels)
    transitions = nodes = pruned = 0
    prune_with_oracle = prune_with_oracle and trace is None
    searches = {}
    branches = {}
    boundary_kind = {}
    exact_values = {}
    trace_values = {} if trace is not None else None

    @lru_cache(None)
    def tail(index, state):
        return float(oracle_terminal(index, state))

    def weighted_tail(state, indices):
        mass = sum(probabilities[i] for i in indices)
        return sum(probabilities[i] * tail(i, state) for i in indices) / mass

    def expand(state, indices):
        nonlocal transitions, nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise PrefixBudgetExceeded(max_nodes, nodes, transitions, pruned)
        key = state, indices
        stopped = stop_when(state) if stop_when is not None else state.used_connections != root_used
        actions = model.actions(state)
        if stopped:
            kind = 'public_boundary' if stop_when is not None else 'first_ride'
        elif state.time >= cutoff:
            kind = 'clock_limit'
        elif not actions:
            kind = 'terminal'
        else:
            kind = None
        if kind is not None:
            boundary_kind[key] = kind
            return None
        if singleton_exact is not None and len(indices) == 1:
            exact = singleton_exact(indices[0], state)
            if exact is not None:
                boundary_kind[key] = 'exact_continuation'
                exact_values[key] = float(exact)
                return None

        mass = sum(probabilities[i] for i in indices)
        choices = []
        for action in actions:
            groups = defaultdict(list)
            if action_outcomes is None:
                for i in indices:
                    outcome = model.transition(model.scenarios[i], state, action)
                    groups[outcome].append(i)
            else:
                for i, outcome in action_outcomes(state, indices, action):
                    groups[outcome].append(i)
            transitions += len(indices)
            children = []
            for outcome, members in groups.items():
                child_indices = tuple(members)
                probability = sum(probabilities[i] for i in members) / mass
                children.append((probability, outcome.cost, (outcome.state, child_indices)))
            choices.append(tuple(children))
        return tuple(choices)

    # Explicit postorder avoids Python recursion limits for a 720-second wait.
    # It evaluates the same DAG and preserves every original action/observation.
    root_key = root, support
    values, pending, stack = {}, {}, [root_key]

    def search_actions(key, choices):
        nonlocal pruned
        if key in searches:
            order, cursor, best = searches[key]
        else:
            order = sorted((sum(p * (cost + (values[child] if child in values else weighted_tail(*child)))
                                for p, cost, child in choice), i)
                           for i, choice in enumerate(choices))
            cursor, best = 0, (inf, len(choices))
        while cursor < len(order):
            lower, index = order[cursor]
            # Compare both value and original action order to preserve ties.
            if (lower, index) >= best:
                pruned += len(order)-cursor
                break
            choice = choices[index]
            child = next((child for _, _, child in choice if child not in values), None)
            if child is not None:
                searches[key] = order, cursor, best
                return child, None
            cost = sum(p * (cost + values[child]) for p, cost, child in choice)
            best = min(best, (cost, index))
            cursor += 1
        searches.pop(key, None)
        return None, best

    while stack:
        key = stack[-1]
        if key in values:
            stack.pop()
            continue
        if key not in pending:
            choices = expand(*key)
            if choices is None:
                values[key] = exact_values.pop(key) if key in exact_values else weighted_tail(*key)
                stack.pop()
                continue
            pending[key] = choices
        choices = pending[key]
        if prune_with_oracle and len(choices) > 1:
            missing, best = search_actions(key, choices)
            if missing is not None:
                stack.append(missing)
                continue
            value, selected = best
        else:
            missing = next((child for choice in choices for _, _, child in choice if child not in values), None)
            if missing is not None:
                stack.append(missing)
                continue
            costs = [sum(p * (cost + values[child]) for p, cost, child in choice) for choice in choices]
            selected = min(range(len(costs)), key=costs.__getitem__)
            value = costs[selected]
        values[key] = value
        branches[key] = tuple((p, *child) for p, _, child in choices[selected])
        if trace_values is not None:
            trace_values[key] = tuple(zip(model.actions(key[0]), costs))
        del pending[key]
        stack.pop()

    value = values[root_key]

    if trace is not None:
        visited, pending_trace = set(), [root_key]
        while pending_trace:
            key = pending_trace.pop()
            if key in visited:
                continue
            visited.add(key)
            trace(*key, trace_values.get(key, ()), boundary_kind.get(key))
            pending_trace.extend((child, indices) for _, child, indices in branches.get(key, ()))

    @lru_cache(None)
    def factor_counts(indices):
        return tuple(len({factor_labels[name][i] for i in indices}) for name in factor_order)

    def boundary_stats(state, indices, completed):
        key = state, indices
        if key in boundary_kind:
            return ({len(indices): 1.}, {boundary_kind[key]: 1.}, 0., state.time - root.time,
                    {factor_counts(indices): 1.}, (0.,) * len(factor_order))
        sizes, kinds = defaultdict(float), defaultdict(float)
        profiles = defaultdict(float)
        factor_singletons = [0.] * len(factor_order)
        singleton, elapsed = 0., 0.
        for probability, child, child_indices in branches[key]:
            child_sizes, child_kinds, child_singleton, child_elapsed, child_profiles, child_factors = completed[child, child_indices]
            for size, mass in child_sizes.items():
                sizes[size] += probability * mass
            for kind, mass in child_kinds.items():
                kinds[kind] += probability * mass
            singleton += probability * child_singleton
            elapsed += probability * child_elapsed
            for profile, mass in child_profiles.items():
                profiles[profile] += probability * mass
            for i, child_factor in enumerate(child_factors):
                factor_singletons[i] += probability * child_factor
        # A singleton reached only at the boundary is recorded separately.
        factor_singletons = tuple(1. if count == 1 else value
                                  for count, value in zip(factor_counts(indices), factor_singletons))
        return (dict(sizes), dict(kinds), 1. if len(indices) == 1 else singleton, elapsed,
                dict(profiles), factor_singletons)

    completed = {}
    for key in values:  # Each child's value was inserted before its parents.
        completed[key] = boundary_stats(*key, completed)
    sizes, kinds, singleton, elapsed, profiles, factor_singletons = completed[root_key]
    root_mean = weighted_tail(root, support)
    return CausalPrefixValue(value, root_mean, elapsed_horizon, len(support),
        len(values), transitions, tail.cache_info().currsize,
        sizes, kinds, sizes.get(1, 0.), singleton, elapsed, factor_order,
        {'x'.join(map(str, profile)): mass for profile, mass in profiles.items()} if factor_order else {},
        {name: sum(mass for profile, mass in profiles.items() if profile[i] == 1)
         for i, name in enumerate(factor_order)}, dict(zip(factor_order, factor_singletons)), pruned)
