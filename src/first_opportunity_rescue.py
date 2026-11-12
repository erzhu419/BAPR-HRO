"""Board the first publicly legal opportunity in a fixed route menu."""

from .background_policy import IssuedPolicy
from .interruptible_recourse import RecourseCursor
from .local_transfer_rescue import compile_local_transfer_rescue
from .platform_transit import OBSERVE, WAIT
from .recourse_itinerary_router import build_trip_blocks, trip_key
from .tail_rescue import TailRescuePlan


def route_plan(cid, path, issued):
    """Preserve the original A/B continuation clocks where they were issued."""
    if isinstance(path, TailRescuePlan):
        return path
    departures = issued.departures if cid in (issued.primary[:1] + issued.rescue[:1]) else ()
    return TailRescuePlan(tuple(path), departures=departures)


class FirstOpportunityRescue:
    """Use scheduled trip durations solely to break simultaneous offers."""

    def __init__(self, graph, plan, root, goal, deadline, paths):
        self.root = root
        self.deadline = deadline
        self.paths = {cid: route_plan(cid, path, plan) for cid, path in paths.items()}
        connections = graph.connections
        self.ranks = {cid: (connections[path.primary[-1]].arr_time - connections[cid].dep_time, cid)
                      for cid, path in self.paths.items()}
        has_transfer = any(len({trip_key(connections[cid]) for cid in path.primary}) > 1
                           for path in self.paths.values())
        blocks = build_trip_blocks(graph) if has_transfer else None
        self.continuations = {cid: compile_local_transfer_rescue(graph, path, root, goal, deadline,
                                                                trip_blocks=blocks)
                              for cid, path in self.paths.items()}

    def applicable(self, state):
        return (state.stop == self.root.stop and state.current_trip == self.root.current_trip
                and state.last_connection == self.root.last_connection
                and state.used_connections == self.root.used_connections)

    def choice(self, state):
        """Only canonical platform actions may appear in visible_connections."""
        if not state.initialized:
            return OBSERVE
        if state.time >= self.deadline or not self.applicable(state):
            return WAIT
        legal = (cid for cid in state.visible_connections if cid in self.ranks)
        selected = min(legal, key=self.ranks.__getitem__, default=None)
        return WAIT if selected is None else ('board', selected)

    def bind(self, name):
        return _BoundFirstOpportunity(self, name)


class _BoundFirstOpportunity:
    def __init__(self, controller, name):
        self.controller = controller
        self.follow = None
        self.trace, self.stats = [], dict(first_opportunities=0)
        self.policy = IssuedPolicy(self.action, controller.applicable, name)

    def targets(self, state):
        if self.follow is not None:
            return self.follow.targets(state)
        return tuple(self.controller.paths) if self.controller.applicable(state) else ()

    def action(self, state):
        if self.follow is not None:
            return self.follow.action(state)
        action = self.controller.choice(state)
        if action[0] == 'board':
            cid = action[1]
            self.follow = self.controller.continuations[cid].bind(self.policy.name)
            self.follow.cursor = RecourseCursor(self.follow.controller.plan, state.time)
            self.follow.previous = state
            self.stats['first_opportunities'] += 1
            self.trace.append(dict(time=state.time, stop=state.stop, action=action,
                                   branch='first_opportunity',
                                   offers=tuple(i for i in state.visible_connections if i in self.controller.paths)))
        return action
