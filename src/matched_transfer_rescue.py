"""Use the same fixed-path transfer continuation after any root boarding."""

from .offgrid_conditional import ActualClockRescue, _BoundActualClockRescue


class MatchedTransferRescue(ActualClockRescue):
    def __init__(self, base, continuations, wait_value, margin):
        super().__init__(base.original.graph, base.original.plan, base.root, base.original.deadline,
                         base.scores, base.extras, margin, base.clock_scores, mode=base.mode,
                         censor_unknown_extras=base.censor_unknown_extras,
                         wait_value=wait_value)
        self.continuations = continuations
        self.suffixes.update({cid: continuations[cid] for cid in self.suffixes.keys() & continuations.keys()})

    def bind(self, name):
        return _BoundMatchedTransferRescue(self, name)


class _BoundMatchedTransferRescue(_BoundActualClockRescue):
    def action(self, state):
        following = self.follow
        action = super().action(state)
        if following is not None:
            self.stats.update({name: count for name, count in following.stats.items()
                               if name.startswith('local_transfer_')})
        elif self.follow is None and action[0] == 'board':
            c = self.controller
            if state.stop == c.root.stop and state.used_connections == c.root.used_connections:
                suffix = c.continuations.get(action[1])
                if suffix is not None:
                    self.start_suffix(state, suffix)
        return action
