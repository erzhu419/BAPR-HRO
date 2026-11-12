"""Current decisions on complete or remaining-menu public states.

Each engine belongs to one fixed model, deadline and waiting rule. It caches
only the next action and its scoring receipt. Physical execution, posterior
updates, raw public history and sampled outcomes remain with the executor.
"""

from experiments.reconstruction_v33.nodes import Platform, Ride


def raw_history_key(history, relevant_trips=None):
    """Keep ordered hard evidence and every raw message for the chosen trips.

    SingleMessageHistory.signature contains only the currently selected
    likelihood. Later own observations can select another already received
    forecast, so this state retains all raw forecasts, including past slots.
    """
    if relevant_trips is None:
        trips = set(history._records) | set(history._trip_statuses)
        for source in (history._forecasts, history._endpoint_statuses, history._service_statuses):
            trips.update(history.belief._positions[key][0] for key in source)
    else:
        trips = relevant_trips
    rows, units = [], 0
    for trip in sorted(trips):
        hard = history._records.get(trip, ())
        raw = history.raw_forecast_records(trip)
        trip_status = history._trip_statuses.get(trip)
        endpoint_status = history._status_records(trip)
        service_status = history._service_records(trip)
        rows.append((trip, hard, raw, trip_status, endpoint_status, service_status))
        units += len(hard)+len(raw)+(trip_status is not None)+len(endpoint_status)+len(service_status)
    return tuple(rows), len(rows), units


class DecisionEngine:
    """Share one planner and current-action cache across both action arms.

    ``fullmemo`` retains the original menu and all raw public history in each
    key. ``frontier`` uses the current-and-later ride exits and the complete
    evidence of every trip still present recursively in that menu. A platform
    view removes only seen root offers that are no longer visible. Neither
    variant deletes evidence from the history object or reads a physical world.
    """

    def __init__(self, planner, variant='frontier'):
        if variant not in ('fullmemo', 'frontier'):
            raise ValueError(variant)
        self.planner, self.variant = planner, variant
        self._decisions, self._suffixes, self._platform_views = {}, {}, {}
        self.stats = dict(key_trips=0, key_record_units=0,
                          suffix_views=0, suffix_dropped_exits=0,
                          platform_views=0, platform_dropped_roots=0)
        for kind in ('onboard', 'platform'):
            self.stats.update({f'{kind}_{name}': 0 for name in
                               ('requests', 'hits', 'unique', 'key_trips', 'key_record_units')})

    def _history(self, kind, node, history):
        trips = self.planner.relevant_trips(node) if self.variant == 'frontier' else None
        state, count, units = raw_history_key(history, trips)
        self.stats['key_trips'] += count
        self.stats['key_record_units'] += units
        self.stats[f'{kind}_key_trips'] += count
        self.stats[f'{kind}_key_record_units'] += units
        self.stats[f'{kind}_requests'] += 1
        return state

    def _view(self, ride, cid):
        if self.variant == 'fullmemo':
            return ride
        position = ride.path.index(cid)
        self.stats['suffix_dropped_exits'] += sum(
            previous in ride.exits for previous in ride.path[:position])
        if not position:
            return ride
        key = self.planner.structure(ride), cid
        if key not in self._suffixes:
            path = ride.path[position:]
            self._suffixes[key] = Ride(cid, path,
                {exit_cid: ride.exits[exit_cid] for exit_cid in path if exit_cid in ride.exits})
            self.stats['suffix_views'] += 1
        return self._suffixes[key]

    def _platform_view(self, platform, visible, seen):
        if self.variant == 'fullmemo':
            return platform, seen
        removed = set(seen)-set(visible)
        retained = tuple(cid for cid in platform.rides if cid not in removed)
        self.stats['platform_dropped_roots'] += len(platform.rides)-len(retained)
        if len(retained) == len(platform.rides):
            return platform, seen
        key = self.planner.structure(platform), retained
        if key not in self._platform_views:
            self._platform_views[key] = Platform(platform.stop,
                {cid: platform.rides[cid] for cid in retained})
            self.stats['platform_views'] += 1
        return self._platform_views[key], tuple(cid for cid in seen if cid in retained)

    def onboard(self, ride, cid, clock, history, method='rolling_fixed'):
        view = self._view(ride, cid)
        public = self._history('onboard', view, history)
        key = ('onboard', method, self.planner.structure(view), cid, float(clock), public)
        if key in self._decisions:
            self.stats['onboard_hits'] += 1
        else:
            compiled = self.planner.compile_ride(view, history, method)
            self._decisions[key] = compiled.decision(cid, clock)
            self.stats['onboard_unique'] += 1
        choice, receipt = self._decisions[key]
        receipt = dict(receipt)
        if self.variant == 'frontier':
            receipt['stage'] = tuple(c for c in ride.path if c in ride.exits).index(cid)
        return choice, receipt

    def platform(self, platform, entry, ready, history, visible, observed, clock,
                 choose, method='rolling_fixed'):
        visible, seen = tuple(visible), tuple(observed)
        view, seen = self._platform_view(platform, visible, seen)
        public = self._history('platform', view, history)
        key = ('platform', method, self.planner.structure(view), float(entry),
               float(ready), float(clock), visible, seen, public)
        if key in self._decisions:
            self.stats['platform_hits'] += 1
        else:
            reference = self.planner.platform_reference(view, history, entry, ready, method)
            self._decisions[key] = choose(reference, visible, seen, clock)
            self.stats['platform_unique'] += 1
        choice, receipt = self._decisions[key]
        return choice, dict(receipt)
