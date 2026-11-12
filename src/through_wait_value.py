"""Departure-event waiting value for a fixed menu of through trips."""


class ThroughWaitValue:
    """Integrate the A/B waiting proxy and later through choices.

    The first eligible A/B event is a compulsory offer, with A first on a
    tie. Through departures are optional, and decline irrevocably removes
    them. The clock-conditioned fixed-path scores are the terminal costs.
    This is the value of that reduced, independent first-departure model.
    """

    def __init__(self, scores, through_ids, *, margin=120):
        self.scores = scores
        self.through_ids = tuple(sorted(through_ids))
        self.margin = margin

    def waiting_score(self, state, cursor):
        scores = self.scores
        deadline = scores.deadline
        earliest = max(scores.ready, state.time + 1)
        through = {cid: rows for cid in self.through_ids
                   if (rows := scores.future(cid, state, earliest, deadline))}
        if not through:
            return scores.waiting_score(state, cursor)

        primary, rescue = (), ()
        active = list(through)
        if cursor.plan.primary:
            cid = cursor.plan.primary[0]
            if cid not in scores.rows:
                return None
            latest = deadline if cursor.expiry is None else cursor.expiry
            primary = scores.future(cid, state, max(earliest, cursor.ready or state.time), latest)
            if primary:
                active.append(cid)
            if cursor.plan.failure_index == 0 and cursor.plan.rescue:
                cid = cursor.plan.rescue[0]
                if cid not in scores.rows:
                    return None
                rescue = scores.future(cid, state, earliest, deadline)
                if rescue:
                    active.append(cid)
        trips = [scores.graph.connections[cid].trip_id for cid in active]
        if len(set(trips)) != len(trips) or any(trip is None for trip in trips):
            raise ValueError('The waiting model requires distinct identified first-departure trips')
        return self._solve(primary, rescue, tuple(through.values()))

    @staticmethod
    def _hazards(rows):
        """Convert conditional event masses to hazards at eligible clocks."""
        survival = 1.
        hazards = {}
        for clock, mass, score in sorted(rows):
            if survival <= 0.:
                break
            hazards[clock] = min(1., mass / survival), score
            survival -= mass
        return hazards

    def _solve(self, primary, rescue, through):
        a, b = self._hazards(primary), self._hazards(rescue)
        optional = {}
        for index, rows in enumerate(through):
            for clock, (hazard, score) in self._hazards(rows).items():
                optional.setdefault(clock, []).append((1 << index, hazard, score))

        deadline = self.scores.deadline
        after = [deadline] * (1 << len(through))
        for clock in sorted(a.keys() | b.keys() | optional.keys(), reverse=True):
            pa, sa = a.get(clock, (0., deadline))
            pb, sb = b.get(clock, (0., deadline))
            mandatory = ((pa, sa), ((1. - pa) * pb, sb))
            wait = (1. - pa) * (1. - pb)
            events = optional.get(clock, ())
            before = []
            for available, future in enumerate(after):
                simultaneous = [(0, 1., deadline)]
                for bit, hazard, score in events:
                    if available & bit:
                        simultaneous = [outcome for departed, prob, best in simultaneous
                                        for outcome in ((departed, prob * (1. - hazard), best),
                                                        (departed | bit, prob * hazard, min(best, score)))
                                        if outcome[1] > 0.]
                total = 0.
                for departed, mass, offer in simultaneous:
                    continuation = after[available ^ departed] if departed else future
                    take_below = offer + self.margin
                    at_clock = wait * (offer if take_below < continuation else continuation)
                    at_clock += sum(prob * (offer if take_below < cost else cost)
                                    for prob, cost in mandatory)
                    total += mass * at_clock
                before.append(total)
            after = before
        return after[-1]
