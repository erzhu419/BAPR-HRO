"""One public recovery at the first missed transfer of a frozen itinerary."""

from .interruptible_recourse import InterruptibleRecourse, RecourseCursor, _BoundRecourse
from .platform_transit import WAIT
from .recourse_itinerary_router import build_trip_blocks, trip_key
from .tail_rescue import TailRescuePlan


def compile_local_transfer_rescue(graph, plan, root, goal, deadline, *, trip_blocks=None):
    """Freeze direct alternatives at the first transfer platform by timetable.

    Retain one first connection per independent trip, with all of its
    consecutive same-trip segments through the destination. The menu covers
    scheduled departures from the decision root's clock to the deadline;
    actual boardability is determined later by the public platform state.
    """
    connections = graph.connections
    path = plan.primary
    index = next((i for i in range(1, len(path))
                  if trip_key(connections[path[i-1]]) != trip_key(connections[path[i]])), None)
    direct = {}
    if index is not None:
        stop = connections[path[index]].dep_stop
        excluded = {trip_key(connections[cid]) for cid in (*path, *root.used_connections)}
        blocks, positions = build_trip_blocks(graph) if trip_blocks is None else trip_blocks
        choices = sorted(graph.get_connections_from(stop), key=lambda c: (c.dep_time, c.id))
        for first in choices:
            trip = trip_key(first)
            if (not first.pickup_allowed or trip in excluded
                    or not root.time <= first.dep_time < deadline):
                continue
            chain, at = [], stop
            for cid in blocks[trip][positions[first.id]:]:
                connection = connections[cid]
                if connection.dep_stop != at or connection.arr_time > deadline:
                    break
                chain.append(cid)
                at = connection.arr_stop
                if at == goal:
                    direct[first.id] = tuple(chain)
                    excluded.add(trip)
                    break
    direct = dict(sorted(direct.items(), key=lambda item: (connections[item[1][-1]].arr_time,
                                                            item[0])))
    return LocalTransferRescue(graph, plan, root, deadline, index, direct)


class LocalTransferRescue(InterruptibleRecourse):
    """Follow the fixed path until the first transfer failure is public."""

    def __init__(self, graph, plan, root, deadline, transfer_index, fallback_paths):
        super().__init__(graph, plan, root, deadline, mode='public_failure')
        self.transfer_index = transfer_index
        self.target = None if transfer_index is None else plan.primary[transfer_index]
        self.fallback_paths = fallback_paths
        self.fallbacks = {cid: InterruptibleRecourse(graph, TailRescuePlan(path), root,
                                                    deadline, mode='public_failure')
                          for cid, path in fallback_paths.items()}

    def bind(self, name=None):
        return _BoundLocalTransferRescue(self, name or 'local_transfer_rescue')


class _BoundLocalTransferRescue(_BoundRecourse):
    def __init__(self, controller, name):
        super().__init__(controller, name)
        self.miss, self.fallback = None, None
        self.stats.update(local_transfer_misses=0, local_transfer_fallback_boards=0)

    def targets(self, state):
        if self.fallback is not None and self.fallback.controller.plan.primary[0] in state.used_connections:
            return self.fallback.targets(state)
        if self.miss is None or not self.controller.fallback_paths:
            return super().targets(state)
        ready = state.last_arrival + self.controller.graph.stops[state.stop].min_transfer_time
        return tuple(self.controller.fallback_paths) if state.time >= ready else ()

    def action(self, state):
        if self.fallback is not None:
            return self.fallback.action(state)
        controller = self.controller
        if self.miss is None or not controller.fallback_paths:
            before = len(self.trace)
            default = super().action(state)
            if controller.target is not None and self.miss is None and len(self.trace) > before:
                for reason, cid, clock in self.trace[-1]['events']:
                    if cid == controller.target and reason in (
                        'local_departure_unboardable', 'published_departure',
                        'published_cancel', 'attempt_expired',
                    ):
                        self.miss = (reason, cid, clock)
                        self.stats['local_transfer_misses'] += 1
                        break
            if self.miss is None or not controller.fallback_paths:
                return default

        for cid, fallback in controller.fallbacks.items():
            if cid in state.visible_connections:
                self.fallback = fallback.bind(self.policy.name)
                self.fallback.cursor = RecourseCursor(fallback.plan, state.time)
                self.fallback.previous = state
                self.fallback.trace, self.fallback.stats = self.trace, self.stats
                self.stats['local_transfer_fallback_boards'] += 1
                action = ('board', cid)
                self.trace.append(dict(time=state.time, stop=state.stop, action=action,
                                       branch='local_transfer_rescue', reason=self.miss[0],
                                       failed_target=self.miss[1], failure_clock=self.miss[2],
                                       fallback_path=fallback.plan.primary))
                return action
        return WAIT
