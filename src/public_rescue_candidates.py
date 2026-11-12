"""A timetable-screened extension of the first-visible platform rule."""

from dataclasses import dataclass
from types import SimpleNamespace

from .background_policy import IssuedPolicy
from .committed_transit import CommittedState
from .interruptible_recourse import InterruptibleRecourse, RecourseCursor
from .recourse_itinerary_router import build_trip_blocks, expand_trip_blocks, nominal_itinerary
from .tail_rescue import TailRescuePlan


@dataclass(frozen=True)
class PublicItinerary:
    path: tuple
    scheduled_arrival: int


def screen_public_itineraries(graph, root, goal, deadline, original, window):
    """Freeze usable first connections before seeing any evaluation world."""
    nominal = {c.id: SimpleNamespace(departure=c.dep_time, arrival=c.arr_time)
               for c in graph.connections}
    trip_blocks = build_trip_blocks(graph)
    ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
    options = {}
    for c in graph.get_connections_from(root.stop):
        if (c.id in original or c.id in root.used_connections or not c.pickup_allowed
                or not ready <= c.dep_time <= ready + window or c.arr_time > deadline):
            continue
        used = root.used_connections | {c.id}
        state = CommittedState(c.arr_stop, c.arr_time, c.trip_id, c.arr_time, used)
        suffix = nominal_itinerary(graph, state, goal, nominal, nominal, deadline, 1500,
                                   trip_blocks=trip_blocks)
        if suffix is None:
            continue
        path = expand_trip_blocks(graph, (c.id,) + suffix, goal, root.used_connections, trip_blocks)
        if path:
            options[c.id] = PublicItinerary(path, graph.connections[path[-1]].arr_time)
    return options


class PublicRescueCandidates:
    """Precompile additional paths. At the root only legally visible rides can win."""

    def __init__(self, graph, plan, root, deadline, options, *, margin):
        self.graph, self.root = graph, root
        self.options, self.margin = options, margin
        self.original = InterruptibleRecourse(graph, plan, root, deadline, mode='first_visible')
        self.original_paths = {path[0]: path for path in (plan.primary, plan.rescue) if path}
        self.suffixes = {
            cid: InterruptibleRecourse(graph, TailRescuePlan(route.path, first_action_now=True),
                                       root, deadline, mode='public_failure')
            for cid, route in options.items()
        }

    def bind(self, name):
        return _BoundPublicRescue(self, name)


class _BoundPublicRescue:
    def __init__(self, controller, name):
        self.controller = controller
        self.original = controller.original.bind(name)
        self.follow = None
        self.trace, self.stats = self.original.trace, self.original.stats
        self.policy = IssuedPolicy(self.action, controller.original.applicable, name)

    def _at_root(self, state):
        root = self.controller.root
        return state.stop == root.stop and state.used_connections == root.used_connections

    def targets(self, state):
        if self.follow is not None:
            return self.follow.targets(state)
        targets = self.original.targets(state)
        if self._at_root(state):
            targets += tuple(self.controller.options)
        return targets

    def action(self, state):
        if self.follow is not None:
            return self.follow.action(state)
        default = self.original.action(state)
        if not self._at_root(state):
            return default
        controller = self.controller
        graph = controller.graph
        candidates = [(controller.options[cid].scheduled_arrival
                       + state.time - graph.connections[cid].dep_time, cid)
                      for cid in state.visible_connections if cid in controller.options]
        if not candidates:
            return default
        score, cid = min(candidates)
        failed = {target for clock, target, kind in state.public_log
                  if kind in ('published_cancel', 'published_departure')
                  or (kind == 'departure' and (clock < state.time or target not in state.visible_connections))}
        competing = ([default[1]] if default[0] == 'board' else
                     [other for other in controller.original_paths if other not in failed])
        reference = min((graph.connections[controller.original_paths[other][-1]].arr_time
                         + max(0, state.time - graph.connections[other].dep_time)
                         for other in competing), default=float('inf'))
        if controller.margin is not None and score + controller.margin >= reference:
            return default
        if controller.margin is None and default[0] == 'board' and score >= reference:
            return default
        self.follow = controller.suffixes[cid].bind(self.policy.name)
        # A visible additional connection is already physically boardable.
        # Keep the root as the suffix interpreter's previous state.
        self.follow.cursor = RecourseCursor(self.follow.controller.plan, state.time)
        self.follow.previous = state
        self.stats['expanded_boardings'] = 1
        self.trace.append(dict(time=state.time, stop=state.stop, action=('board', cid),
                               branch='expanded', reason='public_timetable',
                               score=score, reference=reference if competing else None))
        return ('board', cid)
