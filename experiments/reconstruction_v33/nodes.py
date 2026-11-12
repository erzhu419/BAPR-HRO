"""The frozen V32 route menu, with every retained on-vehicle exit executable."""

from dataclasses import dataclass, field


@dataclass(eq=False)
class Ride:
    first_cid: int
    path: tuple
    exits: dict = field(default_factory=dict)  # arrival cid -> Platform or destination(None)

    def trips(self, model):
        answer = {model._positions[self.first_cid, 'departure'][0]}
        for platform in self.exits.values():
            if platform is not None:
                answer.update(platform.trips(model))
        return answer

    def paths(self):
        return [self.path, *(path for platform in self.exits.values() if platform is not None
                            for ride in platform.rides.values() for path in ride.paths())]


@dataclass(eq=False)
class Platform:
    stop: int
    rides: dict = field(default_factory=dict)

    def trips(self, model):
        return {trip for ride in self.rides.values() for trip in ride.trips(model)}


def direct_platform(graph, paths):
    rides = {}
    for path in paths:
        path = tuple(path)
        cid = path[0]
        if cid in rides and rides[cid].path != path:
            raise ValueError('The frozen menu has two terminal paths for one departure')
        rides[cid] = Ride(cid, path, {path[-1]: None})
    return Platform(graph.connections[next(iter(rides))].dep_stop, rides)


def platform_from_options(graph, stop, options):
    platform = Platform(stop)
    for row in options:
        cid, path = row['first_cid'], tuple(row['first_path'])
        ride = platform.rides.get(cid)
        if ride is None:
            ride = platform.rides[cid] = Ride(cid, path)
        elif ride.path[:min(len(path), len(ride.path))] != path[:min(len(path), len(ride.path))]:
            raise ValueError('Retained variants must be prefixes of the same physical vehicle')
        if len(path) > len(ride.path):
            ride.path = path
        last = path[-1]
        tail = row['tail_paths']
        if not tail:
            ride.exits[last] = None
        elif last not in ride.exits:
            ride.exits[last] = direct_platform(graph, tail)
        elif ride.exits[last] is not None:
            ride.exits[last].rides.update(direct_platform(graph, tail).rides)
    return platform


def root_ride(graph, root):
    return Ride(root['first_cid'], tuple(root['first_path']), {
        hub['arrival_cid']: platform_from_options(graph, hub['stop'], hub['options'])
        for hub in root['hubs']})
