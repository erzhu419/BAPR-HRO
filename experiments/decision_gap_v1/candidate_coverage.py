"""Public upstream boarding closure of the already declared tail vehicles.

The builder has no observation, endpoint-label or outcome argument. It keeps
the original root, freezes its tail-ride templates, and adds continuous earlier
boarding prefixes at declared platforms. Transfer depth cannot increase.
"""
from collections import Counter

from experiments.reconstruction_v33.nodes import Platform, Ride
from src.recourse_itinerary_router import build_trip_blocks, trip_key


def contains_menu(expanded, original):
    """Every original path and executable exit remains in the expanded menu."""
    if original is None or expanded is None:
        return original is expanded
    if isinstance(original, Ride):
        return (isinstance(expanded, Ride)
                and expanded.first_cid == original.first_cid
                and expanded.path[:len(original.path)] == original.path
                and all(cid in expanded.exits and contains_menu(expanded.exits[cid], child)
                        for cid, child in original.exits.items()))
    return (isinstance(expanded, Platform) and expanded.stop == original.stop
            and all(cid in expanded.rides and contains_menu(expanded.rides[cid], ride)
                    for cid, ride in original.rides.items()))


class _Closure:
    def __init__(self, graph, original, goal):
        self.graph, self.goal = graph, goal
        self.blocks, self.positions = build_trip_blocks(graph)
        self.shapes, self.depths, self.expanded = {}, {}, {}
        self.templates, self.template_by_shape = [], {}
        self.platform_rows, self.candidates = [], []
        # The incoming root ride is not an extra tail template. All catalog
        # entries come from the unchanged menu, before expansion begins.
        self._catalog(original, root=True)

    def shape(self, node):
        if node is None:
            return None
        if node not in self.shapes:
            if isinstance(node, Ride):
                value = ('ride', node.first_cid, tuple(node.path),
                         tuple((cid, self.shape(child)) for cid, child in node.exits.items()))
            else:
                value = ('platform', node.stop,
                         tuple((cid, self.shape(ride)) for cid, ride in node.rides.items()))
            self.shapes[node] = value
        return self.shapes[node]

    def depth(self, node):
        if node is None:
            return 0
        if node not in self.depths:
            if isinstance(node, Ride):
                value = 1+max((self.depth(p) for p in node.exits.values()), default=0)
            else:
                value = max((self.depth(r) for r in node.rides.values()), default=0)
            self.depths[node] = value
        return self.depths[node]

    def _catalog(self, node, *, root=False):
        if isinstance(node, Ride):
            if not root:
                key = self.shape(node)
                if key in self.template_by_shape:
                    self.templates[self.template_by_shape[key]]['source_occurrences'] += 1
                else:
                    index = len(self.templates)
                    self.template_by_shape[key] = index
                    self.templates.append(dict(index=index, ride=node,
                        remaining_rides=self.depth(node), source_occurrences=1))
            for child in node.exits.values():
                if child is not None:
                    self._catalog(child)
        else:
            for ride in node.rides.values():
                self._catalog(ride)

    def merge_platform(self, left, right):
        if left is right:
            return left, False
        if left is None or right is None:
            return None, left is not None
        if left.stop != right.stop:
            raise ValueError('A shared arrival has different transfer stops')
        rides, changed = dict(left.rides), False
        for cid, ride in right.rides.items():
            if cid not in rides:
                rides[cid], changed = ride, True
            else:
                rides[cid], added = self.merge_ride(rides[cid], ride)
                changed |= added
        return (Platform(left.stop, rides) if changed else left), changed

    def merge_ride(self, left, right):
        if left is right:
            return left, False
        n = min(len(left.path), len(right.path))
        if left.first_cid != right.first_cid or left.path[:n] != right.path[:n]:
            raise ValueError('Two routes from one boarding connection are not same-trip prefixes')
        path = left.path if len(left.path) >= len(right.path) else right.path
        exits, changed = dict(left.exits), len(path) > len(left.path)
        for cid, child in right.exits.items():
            if cid not in exits:
                exits[cid], changed = child, True
            else:
                exits[cid], added = self.merge_platform(exits[cid], child)
                changed |= added
        return (Ride(left.first_cid, path, exits) if changed else left), changed

    def ride(self, source, remaining):
        return Ride(source.first_cid, tuple(source.path), {
            cid: None if child is None else self.platform(
                child, min(remaining-1, self.depth(child)))
            for cid, child in source.exits.items()})

    def platform(self, source, remaining):
        key = self.shape(source), remaining
        if key in self.expanded:
            return self.expanded[key]
        index = len(self.platform_rows)
        row = dict(platform_index=index, stop=source.stop, remaining_rides=remaining,
                   old_first_cids=sorted(source.rides))
        self.platform_rows.append(row)
        rides = {cid:self.ride(ride, remaining) for cid, ride in source.rides.items()}
        changed_cids = set()
        for item in self.templates:
            template = item['ride']
            trip = trip_key(self.graph.connections[template.first_cid])
            block = self.blocks[trip]
            join = self.positions[template.first_cid]
            for position, cid in enumerate(block[:join]):
                connection = self.graph.connections[cid]
                if connection.dep_stop != source.stop:
                    continue
                prefix = tuple(block[position:join])
                proposed = prefix+tuple(template.path)
                early_goal = next((c for c in prefix
                                   if self.graph.connections[c].arr_stop == self.goal), None)
                candidate_path = (prefix[:prefix.index(early_goal)+1]
                                  if early_goal is not None else proposed)
                candidate_depth = 1 if early_goal is not None else item['remaining_rides']
                record = dict(platform_index=index, stop=source.stop, remaining_rides=remaining,
                    template_index=item['index'], template_first_cid=template.first_cid,
                    template_path=list(template.path), first_cid=cid, trip=trip,
                    prefix=list(prefix), scheduled_departure=connection.dep_time,
                    was_action=cid in source.rides)
                reason = ('no_pickup' if not connection.pickup_allowed else
                          'remaining_ride_depth' if candidate_depth > remaining else
                          'noncontiguous_same_trip_prefix' if any(
                              trip_key(self.graph.connections[c]) != trip for c in candidate_path)
                          or any(self.graph.connections[a].arr_stop != self.graph.connections[b].dep_stop
                                 for a, b in zip(candidate_path, candidate_path[1:])) else None)
                if reason is not None:
                    self.candidates.append(dict(record, disposition='rejected', reason=reason))
                    continue
                # A newly traversed prefix may itself reach the goal before
                # the old boarding position. It is then a direct legal ride.
                if early_goal is not None:
                    added_ride = Ride(cid, candidate_path, {early_goal:None})
                else:
                    tail = self.ride(template, remaining)
                    added_ride = Ride(cid, proposed, tail.exits)
                if cid not in rides:
                    rides[cid], changed = added_ride, True
                else:
                    rides[cid], changed = self.merge_ride(rides[cid], added_ride)
                if changed:
                    changed_cids.add(cid)
                self.candidates.append(dict(record,
                    disposition='added' if changed else 'already_represented',
                    reason='public_continuous_upstream_prefix', path=list(added_ride.path),
                    first_goal_in_prefix=early_goal))
        result = Platform(source.stop, rides)
        self.expanded[key] = result
        row.update(new_first_cids=sorted(rides),
            added_first_cids=sorted(set(rides)-set(source.rides)),
            augmented_first_cids=sorted(changed_cids & set(source.rides)))
        return result

    def ledger(self):
        return dict(rule='Frozen original tail-template upstream boarding closure; no schedule-ready or probability filter',
            inputs='Public graph, original menu, destination only; no actual endpoint or outcome input',
            scope='Original declared platforms and tail vehicles; same-trip prefixes may exceed the old eight-leg cap; each platform keeps its original remaining ride-depth limit',
            service_conditions='Pickup and graph connectivity checked at generation; original graph supplies dropoff-permitted arrivals. Cancellation, actual entry/ready and deadline are enforced by the unchanged scorer/executor.',
            templates=[dict(template_index=t['index'], first_cid=t['ride'].first_cid,
                path=list(t['ride'].path), remaining_rides=t['remaining_rides'],
                source_occurrences=t['source_occurrences']) for t in self.templates],
            platform_changes=self.platform_rows, candidate_occurrences=self.candidates,
            counts=dict(templates=len(self.templates), platform_contexts=len(self.platform_rows),
                candidate_occurrences=len(self.candidates),
                dispositions=dict(Counter(r['disposition'] for r in self.candidates)),
                rejection_reasons=dict(Counter(r['reason'] for r in self.candidates
                                               if r['disposition']=='rejected')),
                added_first_cids=sum(len(r['added_first_cids']) for r in self.platform_rows),
                augmented_first_cids=sum(len(r['augmented_first_cids']) for r in self.platform_rows)))


def expand_menu(graph, original_node, goal):
    """Return an extended finite menu and its public candidate ledger."""
    builder = _Closure(graph, original_node, goal)
    expanded = (builder.ride(original_node, builder.depth(original_node))
                if isinstance(original_node, Ride) else
                builder.platform(original_node, builder.depth(original_node)))
    if not contains_menu(expanded, original_node):
        raise AssertionError('Upstream closure removed an original path or executable exit')
    return expanded, builder.ledger()


def endpoint_coverage(graph, metadata, node, observations=None):
    """Audit labels after construction; never select candidates or roots."""
    paths = node.paths() if isinstance(node, Ride) else [p for r in node.rides.values() for p in r.paths()]
    edges = sorted({cid for path in paths for cid in path})
    events = {(cid, kind) for cid in edges for kind in ('departure','arrival')}
    if observations is not None:
        events.update((cid, 'departure') for cids in observations.station_to_cids.values() for cid in cids)
    issues = []
    for cid, kind in sorted(events):
        endpoint = metadata[cid][kind]
        if endpoint['status'] not in ('REAL', 'CANCELED'):
            issues.append(dict(cid=cid, kind=kind, reason='missing_or_ambiguous', status=endpoint['status']))
    for cid in edges:
        dep, arr = (metadata[cid][kind] for kind in ('departure','arrival'))
        if dep['status'] == arr['status'] == 'REAL':
            if dep['official_journey'] is None or dep['official_journey'] != arr['official_journey']:
                issues.append(dict(cid=cid, reason='inconsistent_endpoint_identity'))
            if dep['time_seconds'] > arr['time_seconds']:
                issues.append(dict(cid=cid, reason='inconsistent_endpoint_clock'))
    for before, after in sorted({pair for path in paths for pair in zip(path, path[1:])}):
        a, b = metadata[before]['arrival'], metadata[after]['departure']
        if a['status'] == b['status'] == 'REAL':
            if a['official_journey'] != b['official_journey']:
                issues.append(dict(cid=after, before=before, reason='inconsistent_same_vehicle_identity'))
            if a['time_seconds'] > b['time_seconds']:
                issues.append(dict(cid=after, before=before, reason='inconsistent_same_vehicle_clock'))
    return dict(menu_edges=len(edges), observed_events=len(events), issues=issues,
                usage='Post-construction endpoint audit; missing labels remain in the root denominator')
