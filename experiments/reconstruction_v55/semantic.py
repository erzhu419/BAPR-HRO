"""Goal-dominated scoring suffixes for the frozen rolling-fixed policy.

Only scoring views/read sets change. Physical menus, full trip laws, raw
forecasts, and the original schedule-based ranks remain available unchanged.
"""
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalPlanner
from experiments.reconstruction_v53.relevance import RemainingReadSets


class GoalBarrierViews:
    """Cut after the first physical destination only when its mask is clear.

    The root publication context is fixed. Later hard observations can only
    clear a planning mask, so a clear goal mask remains clear throughout this
    query. Masked goals retain their original scoring suffixes.
    """

    def __init__(self, model, root_history):
        self.model, self.history = model, root_history
        self._views, self._originals = {}, {}
        self.stats = dict(ride_views=0, platform_views=0, goal_barriers=0,
                          dropped_path_positions=0, dropped_exits=0)

    def original(self, node):
        return self._originals.get(node, node)

    def node(self, node):
        if node in self._originals:
            return node
        if node in self._views:
            return self._views[node]
        if isinstance(node, Ride):
            end = len(node.path)
            for position, cid in enumerate(node.path):
                if cid in node.exits and node.exits[cid] is None:
                    if not self.history.announced_unavailable(cid, 'arrival'):
                        end = position+1
                        if end < len(node.path):
                            self.stats['goal_barriers'] += 1
                    break
            path = node.path[:end]
            exits = {cid:None if node.exits[cid] is None else self.platform(node.exits[cid])
                     for cid in path if cid in node.exits}
            changed = (path != node.path or any(exits[cid] is not node.exits[cid] for cid in exits))
            view = Ride(node.first_cid, path, exits) if changed else node
            self.stats['ride_views'] += int(changed)
            self.stats['dropped_path_positions'] += len(node.path)-len(path)
            self.stats['dropped_exits'] += len(node.exits)-len(exits)
        else:
            rides = {cid:self.ride(ride) for cid, ride in node.rides.items()}
            changed = any(rides[cid] is not node.rides[cid] for cid in rides)
            view = Platform(node.stop, rides) if changed else node
            self.stats['platform_views'] += int(changed)
        self._views[node] = view
        self._originals[view] = node
        return view

    def ride(self, ride):
        return self.node(ride)

    def platform(self, platform):
        return self.node(platform)


class BarrierReadSets(RemainingReadSets):
    """Keep full evidence of every semantically live trip, including late slots."""

    def __init__(self, model, observation_contract, views):
        super().__init__(model, observation_contract)
        self.views = views

    def ride_trips(self, ride, position=0):
        return super().ride_trips(self.views.ride(ride), position)

    def platform_trips(self, platform, remaining_cids=None):
        return super().platform_trips(self.views.platform(platform), remaining_cids)

    def node_trips(self, node, *, position=0, remaining_cids=None):
        return super().node_trips(self.views.node(node), position=position,
                                 remaining_cids=remaining_cids)


class _BarrierScoring:
    def structure(self, node):
        # Original rank inputs are fixed metadata, but may distinguish two
        # scoring views. Do not merge their caches merely because trimming
        # leaves the same ride shape.
        return super().structure(self.views.original(node))

    def compile_ride(self, ride, history, method='rolling_fixed'):
        if method != 'rolling_fixed':
            raise ValueError('Goal-barrier scorer is restricted to rolling_fixed')
        return super().compile_ride(self.views.ride(ride), history, method)

    def relevant_trips(self, node):
        return super().relevant_trips(self.views.node(node))

    def scheduled_arrival(self, ride):
        # Ranking is a separate deterministic input to first-opportunity
        # ties. Retain the original calculation, even for a scoring view.
        return super().scheduled_arrival(self.views.original(ride))

    def ranks(self, platform):
        return super().ranks(self.views.original(platform))


class BarrierPhysicalPlanner(_BarrierScoring, PhysicalPlanner):
    def __init__(self, *args, views, **kwargs):
        self.views = views
        super().__init__(*args, **kwargs)


class BarrierPlanner(_BarrierScoring, MatchedContinuationPlanner):
    """The unchanged matched scoring formula on proved goal-dominated views."""

    def __init__(self, *args, views, **kwargs):
        self.views = views
        super().__init__(*args, **kwargs)
        self.tail_base = BarrierPhysicalPlanner(*args, views=views, **kwargs)
